/******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 OrthogonalHawk
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 *****************************************************************************/

/******************************************************************************
 *
 * @file     falcon_dsp_multi_rate_channelizer_cuda.cu
 * @author   OrthogonalHawk
 * @date     28-Jan-2020
 *
 * @brief    Signal processing transformation class and functions to implement
 *            a multi-rate channelizer in CUDA.
 *
 * @section  DESCRIPTION
 *
 * Implements a set of signal processing transformation functions and classes that
 *  together implement a multi-rate channelizer and filtering capability.
 *  Implementation uses CUDA to leverage GPU acceleration.
 *
 * @section  HISTORY
 *
 * 28-Jan-2020  OrthogonalHawk  File created.
 * 31-Jan-2020  OrthogonalHawk  Optionally use optimized resampler kernel for
 *                               a single output per thread.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <iostream>
#include <memory>
#include <stdint.h>

#include "transform/falcon_dsp_multi_rate_channelizer_cuda.h"
#include "utilities/falcon_dsp_host_timer.h"

/******************************************************************************
 *                                 CONSTANTS
 *****************************************************************************/

const bool TIMING_LOGS_ENABLED = false;

const uint32_t MAX_NUM_OUTPUT_SAMPLES_PER_THREAD_FOR_RESAMPLER_KERNEL = 1;
const uint32_t MAX_NUM_CUDA_THREADS = 1024;

/******************************************************************************
 *                              ENUMS & TYPEDEFS
 *****************************************************************************/

/******************************************************************************
 *                                  MACROS
 *****************************************************************************/

namespace falcon_dsp
{
    /******************************************************************************
     *                        FUNCTION IMPLEMENTATION
     *****************************************************************************/
    
    /******************************************************************************
     *                           CLASS IMPLEMENTATION
     *****************************************************************************/
    
    falcon_dsp_multi_rate_channelizer_cuda::falcon_dsp_multi_rate_channelizer_cuda(void)
      : m_initialized(false),
        m_input_data(nullptr),
        m_input_data_len(0),
        d_freq_shift_channels(nullptr)
    {
        /* change the shared memory size to 8 bytes per shared memory bank. this is so that we
         *  can better handle complex<float> data, which is natively 8 bytes in size */
        cudaErrChkAssert(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
    }
    
    falcon_dsp_multi_rate_channelizer_cuda::~falcon_dsp_multi_rate_channelizer_cuda(void)
    {
        std::lock_guard<std::mutex> lock(std::mutex);
        
        m_channels.clear();
        
        /* cleanup CUDA memory that was reserved for frequency shift channel information */
        if (d_freq_shift_channels)
        {
            cudaErrChk(cudaFree(d_freq_shift_channels));
            d_freq_shift_channels = nullptr;
        }
    }

    bool falcon_dsp_multi_rate_channelizer_cuda::initialize(uint32_t input_sample_rate,
                                                            std::vector<multi_rate_channelizer_channel_s> channels)
    {
        std::lock_guard<std::mutex> lock(std::mutex);

        /* sanity check the inputs and verify that the class has not already been initialized */
        if (input_sample_rate == 0 ||
            channels.size() == 0 ||
            m_initialized)
        {
            return false;
        }

        /* check each one of the requested channels is achievable */
        for (auto chan_iter : channels)
        {
            /* TODO */
        }

        /* initialize the requested channels */
        for (auto chan_iter : channels)
        {   
            std::unique_ptr<falcon_dsp_channelizer_stream_cuda> new_chan = std::make_unique<falcon_dsp_channelizer_stream_cuda>();
            new_chan->initialize(input_sample_rate,
                                 chan_iter.output_sample_rate_in_sps,
                                 chan_iter.freq_shift_in_hz,
                                 chan_iter.up_rate,
                                 chan_iter.down_rate,
                                 chan_iter.resample_filter_coeffs);

            m_channels.push_back(std::move(new_chan));
        }

        /* allocate CUDA memory for the frequency shift channel information; the master copy
         *  is kept within the m_channels data structure, but it is copied to the device when
         *  the 'apply' method is invoked */
        cudaErrChkAssert(cudaMallocManaged(&d_freq_shift_channels,
                                           m_channels.size() * sizeof(falcon_dsp_freq_shift_params_cuda_s)));

        /* initialization complete */
        m_initialized = true;
        
        return m_initialized;
    }
    
    bool falcon_dsp_multi_rate_channelizer_cuda::apply(std::vector<std::complex<float>>& in,
                                                       std::vector<std::vector<std::complex<float>>>& out)
    {
        std::lock_guard<std::mutex> lock(std::mutex);
        
        /* clear the output data structures and resize so that they can hold
         *  the shifted and resampled data. */
        out.clear();
        out.resize(m_channels.size());
        
        /* allocate CUDA memory for the input samples */
        if (m_input_data_len != in.size())
        {
            if (m_input_data)
            {
                cudaErrChkAssert(cudaFree(m_input_data));
                m_input_data = nullptr;
                m_input_data_len = 0;
            }

            cudaErrChkAssert(cudaMallocManaged(&m_input_data,
                                               in.size() * sizeof(std::complex<float>)));
            m_input_data_len = in.size();
        }

        cudaStream_t main_channelizer_stream;
        cudaErrChkAssert(cudaStreamCreateWithFlags(&main_channelizer_stream, cudaStreamNonBlocking));
        
        /* allocate CUDA memory for the intermediate and output samples */
        for (uint32_t chan_idx = 0; chan_idx < m_channels.size(); ++chan_idx)
        {
            /* resize the outputs based on the actual number of outputs expected
             *  from the resampling kernel. note that by using resize() the vector
             *  size is now equal to the final output size without explicitly
             *  adding data to the vector, which means that we can add data
             *  directly into the vector data buffer without worrying about the 
             *  vector size getting mismatched with the buffer contents (provided
             *  that the promised number of samples are actually copied in...) */
            m_channels[chan_idx]->allocate_memory(in.size());
            out[chan_idx].resize(m_channels[chan_idx]->get_resample_out_data_len());
        }

        /* copy the input data to the GPU */
        cudaErrChkAssert(cudaMemcpyAsync(static_cast<void *>(m_input_data),
                                         static_cast<void *>(in.data()),
                                         m_input_data_len * sizeof(std::complex<float>),
                                         cudaMemcpyHostToDevice, main_channelizer_stream));

        /* copy the frequency shift channel information to the GPU */
        std::vector<falcon_dsp_freq_shift_params_cuda_s> tmp_freq_shift_params(m_channels.size());
        for (uint32_t chan_idx = 0; chan_idx < m_channels.size(); ++chan_idx)
        {
             tmp_freq_shift_params[chan_idx] = m_channels[chan_idx]->get_freq_shift_params();
        }
        
        cudaErrChkAssert(cudaMemcpyAsync(static_cast<void *>(d_freq_shift_channels),
                                         static_cast<void *>(tmp_freq_shift_params.data()),
                                         tmp_freq_shift_params.size() * sizeof(falcon_dsp_freq_shift_params_cuda_s),
                                         cudaMemcpyHostToDevice, main_channelizer_stream));
        
        /* calculate frequency shift kernel parameters */
        uint32_t samples_per_freq_shift_thread_block = MAX_NUM_CUDA_THREADS; /* assumes one output per thread */
        uint32_t num_thread_blocks = (in.size() + samples_per_freq_shift_thread_block - 1) /
                                             samples_per_freq_shift_thread_block;

        uint32_t freq_shift_shared_memory_size_in_bytes = sizeof(falcon_dsp_freq_shift_params_cuda_s) * m_channels.size();
        
        falcon_dsp::falcon_dsp_host_timer timer("FREQ_SHIFT KERNEL", TIMING_LOGS_ENABLED);

        /* run the frequency shift multi-channel kernel on the GPU */
        __freq_shift_multi_chan<<<num_thread_blocks, MAX_NUM_CUDA_THREADS,
                                  freq_shift_shared_memory_size_in_bytes, main_channelizer_stream>>>(
                            d_freq_shift_channels,
                            m_channels.size(),
                            m_input_data,
                            m_input_data_len);
            
        cudaErrChkAssert(cudaPeekAtLastError());
            
        /* wait for GPU to finish frequency shifting */
        cudaErrChkAssert(cudaStreamSynchronize(main_channelizer_stream));
        cudaErrChk(cudaStreamDestroy(main_channelizer_stream));
        
        timer.log_duration("FREQ_SHIFT Kernel Complete");

        /* frequency shifting complete; update trackers */
        for (uint32_t chan_idx = 0; chan_idx < m_channels.size(); ++chan_idx)
        {
            m_channels[chan_idx]->add_freq_shift_samples_handled(in.size());
        }
        
        falcon_dsp::falcon_dsp_host_timer resample_timer("RESAMPLE", TIMING_LOGS_ENABLED);
        
        /* now resample each channel at once using CUDA streams */
        cudaStream_t * cuda_streams = new cudaStream_t[m_channels.size()];
        for (uint32_t chan_idx = 0; chan_idx < m_channels.size(); ++chan_idx)
        {
            cudaErrChkAssert(cudaStreamCreateWithFlags(&cuda_streams[chan_idx], cudaStreamNonBlocking));
            
            if (MAX_NUM_OUTPUT_SAMPLES_PER_THREAD_FOR_RESAMPLER_KERNEL == 1)
            {
                __polyphase_resampler_single_out<<<m_channels[chan_idx]->get_num_resampler_thread_blocks(),
                                                   MAX_NUM_CUDA_THREADS, 0, cuda_streams[chan_idx]>>>(
                             m_channels[chan_idx]->get_resampler_coeffs_ptr(),
                             m_channels[chan_idx]->get_resampler_coeffs_len(),
                             m_channels[chan_idx]->get_resampler_output_params_ptr(),
                             m_channels[chan_idx]->get_resampler_output_params_len(),
                             m_channels[chan_idx]->get_freq_shift_out_data_ptr(true),
                             m_channels[chan_idx]->get_freq_shift_out_data_len(true),
                             m_channels[chan_idx]->get_resample_out_data_ptr(),
                             m_channels[chan_idx]->get_resample_out_data_len(),
                             m_channels[chan_idx]->get_resampler_params().coeffs_per_phase);
            }
            else
            {
                __polyphase_resampler_multi_out<<<m_channels[chan_idx]->get_num_resampler_thread_blocks(),
                                                  MAX_NUM_CUDA_THREADS, 0, cuda_streams[chan_idx]>>>(
                             m_channels[chan_idx]->get_resampler_coeffs_ptr(),
                             m_channels[chan_idx]->get_resampler_coeffs_len(),
                             m_channels[chan_idx]->get_resampler_output_params_ptr(),
                             m_channels[chan_idx]->get_resampler_output_params_len(),
                             m_channels[chan_idx]->get_freq_shift_out_data_ptr(),
                             m_channels[chan_idx]->get_freq_shift_out_data_len(),
                             m_channels[chan_idx]->get_resample_out_data_ptr(),
                             m_channels[chan_idx]->get_resample_out_data_len(),
                             m_channels[chan_idx]->get_resampler_params().coeffs_per_phase,
                             MAX_NUM_OUTPUT_SAMPLES_PER_THREAD_FOR_RESAMPLER_KERNEL);
            }

            cudaErrChkAssert(cudaPeekAtLastError());
            
            /* copy output samples out of CUDA memory. okay to do this without synchronizing first because
             *  each copy operation is scheduled in a specific stream context and the kernel running will
             *  finish before the copy operation starts */
            cudaErrChkAssert(cudaMemcpyAsync(out[chan_idx].data(),
                                             m_channels[chan_idx]->get_resample_out_data_ptr(),
                                             m_channels[chan_idx]->get_resample_out_data_len() * sizeof(std::complex<float>),
                                             cudaMemcpyDeviceToHost, cuda_streams[chan_idx]));
        }

        /* wait for GPU to finish before accessing on host */
        for (uint32_t chan_idx = 0; chan_idx < m_channels.size(); ++chan_idx)
        {
            cudaErrChkAssert(cudaStreamSynchronize(cuda_streams[chan_idx]));
            
            /* finished resampling; now update the resampler state buffer*/
            _manage_resampler_state(chan_idx, in.size());
            
            cudaErrChk(cudaStreamDestroy(cuda_streams[chan_idx]));
        }
        
        resample_timer.log_duration("RESAMPLING Complete");
        
        return out.size() > 0;
    }

    void falcon_dsp_multi_rate_channelizer_cuda::reset_state(void)
    {
        for (uint32_t chan_idx = 0; chan_idx < m_channels.size(); ++chan_idx)
        {
            m_channels[chan_idx]->reset_state();
        }
    }

    void falcon_dsp_multi_rate_channelizer_cuda::_manage_resampler_state(uint32_t chan_idx, uint32_t input_vector_len)
    {
        m_channels[chan_idx]->manage_state(input_vector_len);
    }
}
