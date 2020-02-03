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

const uint32_t MAX_NUM_INPUT_SAMPLES_FOR_MULTI_CHAN_FREQ_SHIFT_KERNEL = 4;
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
    
    falcon_dsp_multi_rate_channelizer_cuda::internal_multi_rate_channelizer_channel_s::internal_multi_rate_channelizer_channel_s(const multi_rate_channelizer_channel_s& other)
      : d_freq_shifted_data(nullptr),
        freq_shifted_data_len(0),
        d_resample_coeffs(nullptr),
        resample_coeffs_len(0),
        d_resample_output_params(nullptr),
        resample_output_params_len(0),
        d_resampled_data(nullptr),
        resampled_data_len(0)
    {
        output_sample_rate_in_sps = other.output_sample_rate_in_sps;
        freq_shift_in_hz = other.freq_shift_in_hz;
        up_rate = other.up_rate;
        down_rate = other.down_rate;
        resample_filter_coeffs = other.resample_filter_coeffs;

        resampler_params.initialize(up_rate, down_rate, resample_filter_coeffs);
            
        /* the resampler coefficients are fixed, so allocate space for them here */
        if (resample_coeffs_len != resample_filter_coeffs.size())
        {
            if (d_resample_coeffs)
            {
                cudaErrChk(cudaFree(d_resample_coeffs));
                d_resample_coeffs = nullptr;
                resample_coeffs_len = 0;
            }

            resample_coeffs_len = resample_filter_coeffs.size();
            cudaErrChkAssert(cudaMallocManaged(&d_resample_coeffs,
                                               resample_coeffs_len * sizeof(std::complex<float>)));
            
            /* copy the coefficients to the GPU */
            cudaErrChkAssert(cudaMemcpy(static_cast<void *>(d_resample_coeffs),
                                        static_cast<void *>(resample_filter_coeffs.data()),
                                        resample_filter_coeffs.size() * sizeof(std::complex<float>),
                                        cudaMemcpyHostToDevice));
        }
    }
            
    falcon_dsp_multi_rate_channelizer_cuda::internal_multi_rate_channelizer_channel_s::~internal_multi_rate_channelizer_channel_s(void)
    {
        cleanup_memory();
    }
    
    uint32_t falcon_dsp_multi_rate_channelizer_cuda::internal_multi_rate_channelizer_channel_s::get_num_outputs_for_input(uint32_t input_vector_len)
    {
        /* compute how many outputs will be generated for input_vector_len inputs */
        uint64_t np = input_vector_len * static_cast<uint64_t>(resampler_params.up_rate);
        uint32_t need = np / resampler_params.down_rate;
        
        if ((resampler_params.coeff_phase + resampler_params.up_rate * resampler_params.xOffset) < (np % resampler_params.down_rate))
        {
            need++;
        }
        
        return need;
    }
    
    uint32_t falcon_dsp_multi_rate_channelizer_cuda::internal_multi_rate_channelizer_channel_s::get_num_resampler_thread_blocks(void)
    {
        return (resample_output_params_len / MAX_NUM_CUDA_THREADS);
    }
    
    uint32_t falcon_dsp_multi_rate_channelizer_cuda::internal_multi_rate_channelizer_channel_s::initialize(uint32_t input_vector_len)
    {
        /* allocate space for the frequency shifted version of the input data. note that this
         *  is also the input data for resampling so it has to take into account the 
         *  resampling state information. therefore, do NOT use the freq_shift_channel_s
         *  allocate_memory method since this will only pay attention to the input vector
         *  length and fail to account for the state information. instead, handle the memory
         *  allocation manually */
        if (freq_shifted_data_len != (input_vector_len + resampler_params.state.size()))
        {
            if (d_freq_shifted_data)
            {
                cudaErrChkAssert(cudaFree(d_freq_shifted_data));
                d_freq_shifted_data = nullptr;
                freq_shifted_data_len = 0;
            }

            freq_shifted_data_len = input_vector_len + resampler_params.state.size();
            cudaErrChkAssert(cudaMallocManaged(&d_freq_shifted_data,
                                               freq_shifted_data_len * sizeof(std::complex<float>)));
        }

        /* frequency shift output memory has been allocated so update the frequency shift
         *  output memory pointer. note that the pointer is set AFTER the state information */
        freq_shift_chan->out_data = d_freq_shifted_data + resampler_params.state.size();
        freq_shift_chan->out_data_len = freq_shifted_data_len - resampler_params.state.size();
        
        /* copy the resampler state vector into CUDA memory */
        cudaErrChkAssert(cudaMemcpy(d_freq_shifted_data,
                                    resampler_params.state.data(),
                                    resampler_params.state.size() * sizeof(std::complex<float>),
                                    cudaMemcpyHostToDevice));
        
        /* calculate the number of thread blocks that will be required for resampling */
        uint32_t expected_num_outputs = get_num_outputs_for_input(input_vector_len);
        int64_t resample_x_idx = resampler_params.xOffset;
        uint32_t num_outputs_per_resampler_thread_block =
                MAX_NUM_CUDA_THREADS * MAX_NUM_OUTPUT_SAMPLES_PER_THREAD_FOR_RESAMPLER_KERNEL;
        uint32_t num_resampler_thread_blocks = expected_num_outputs / num_outputs_per_resampler_thread_block;
        if (expected_num_outputs % num_outputs_per_resampler_thread_block != 0)
        {
            num_resampler_thread_blocks++;
        }
        
        /* pre-compute resample output parameters */
        uint32_t num_outputs_from_thread_blocks = 0;
        uint32_t new_coeff_phase = resampler_params.coeff_phase;
        int64_t new_x_idx = resample_x_idx;
        falcon_dsp::falcon_dsp_polyphase_resampler_cuda::compute_output_params(resampler_params.up_rate,
                                                                               resampler_params.down_rate,
                                                                               resampler_params.state.size(),
                                                                               input_vector_len,
                                                                               resampler_params.coeff_phase,
                                                                               expected_num_outputs,
                                                                               num_outputs_from_thread_blocks,
                                                                               new_coeff_phase,
                                                                               new_x_idx,
                                                                               resample_output_params);
        
        /* update the channel tracking information preemptively, assuming that if the user
         *  calls the initialize method the parameters will actually be used */
        resampler_params.coeff_phase = new_coeff_phase;
        resample_x_idx += new_x_idx;
        resampler_params.xOffset = resample_x_idx - input_vector_len;

        /* allocate space for the output parameters */
        if (resample_output_params_len != (MAX_NUM_CUDA_THREADS * MAX_NUM_OUTPUT_SAMPLES_PER_THREAD_FOR_RESAMPLER_KERNEL * num_resampler_thread_blocks))
        {
            if (d_resample_output_params)
            {
                cudaErrChkAssert(cudaFree(d_resample_output_params));
                d_resample_output_params = nullptr;
                resample_output_params_len = 0;
            }
            
            resample_output_params_len = MAX_NUM_CUDA_THREADS * MAX_NUM_OUTPUT_SAMPLES_PER_THREAD_FOR_RESAMPLER_KERNEL * num_resampler_thread_blocks;
            cudaErrChkAssert(cudaMallocManaged(&d_resample_output_params,
                                               resample_output_params_len *
                                                   sizeof(polyphase_resampler_output_params_s)));
        }
        
        /* copy the output parameters into CUDA memory; these are recomputed each time the kernel runs
         *  although it is hoped that the memory does not need to be reallocated each time... */
        cudaErrChkAssert(cudaMemcpy(d_resample_output_params,
                                    resample_output_params.data(),
                                    resample_output_params.size() * sizeof(polyphase_resampler_output_params_s),
                                    cudaMemcpyHostToDevice));
        
        /* allocate space for the resampled outputs */
        if (resampled_data_len != num_outputs_from_thread_blocks)
        {
            if (d_resampled_data)
            {
                cudaErrChkAssert(cudaFree(d_resampled_data));
                d_resampled_data = nullptr;
                resampled_data_len = 0;
            }
            
            resampled_data_len = num_outputs_from_thread_blocks;
            cudaErrChkAssert(cudaMallocManaged(&d_resampled_data,
                                               resampled_data_len * sizeof(std::complex<float>)));
        }
        
        return num_outputs_from_thread_blocks;
    }
            
    void falcon_dsp_multi_rate_channelizer_cuda::internal_multi_rate_channelizer_channel_s::cleanup_memory(void)
    {        
        if (d_resample_coeffs)
        {
            cudaErrChk(cudaFree(d_resample_coeffs));
            d_resample_coeffs = nullptr;
            resample_coeffs_len = 0;
        }

        if (d_resample_output_params)
        {
            cudaErrChk(cudaFree(d_resample_output_params));
            d_resample_output_params = nullptr;
            resample_output_params_len = 0;
        }

        if (d_resampled_data)
        {
            cudaErrChk(cudaFree(d_resampled_data));
            d_resampled_data = nullptr;
            resampled_data_len = 0;
        }

        /* when the freq_shift_channel_s class destructs it will automatically free any
         *  memory that it still has access to. however, here we're 'lying' to the class
         *  about the true memory pointer because the frequency shift output is the
         *  resampler input and therefore must account for resampler state information.
         *  the frequency shift memory allocation and freeing is therefore the responsibility
         *  of the internal_multi_rate_channelizer_channel_s class */
        if (freq_shift_chan && freq_shift_chan->out_data)
        {
            /* no cudaFree here; handled below */
            freq_shift_chan->out_data = nullptr;
            freq_shift_chan->out_data_len = 0;
        }
        
        /* cleanup the memory allocated for frequency shifted output / resampler input */
        if (d_freq_shifted_data)
        {
            cudaErrChk(cudaFree(d_freq_shifted_data));
            d_freq_shifted_data = nullptr;
            freq_shifted_data_len = 0;
        }
    }
    
    falcon_dsp_multi_rate_channelizer_cuda::falcon_dsp_multi_rate_channelizer_cuda(void)
      : m_initialized(false),
        m_cuda_input_data(nullptr),
        m_max_num_input_samples(0),
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
            std::unique_ptr<internal_multi_rate_channelizer_channel_s> new_chan =
                    std::make_unique<internal_multi_rate_channelizer_channel_s>(chan_iter);
   
            auto freq_shift_params = falcon_dsp_freq_shift::get_freq_shift_params(input_sample_rate,
                                                                                  chan_iter.freq_shift_in_hz);
            
            std::unique_ptr<freq_shift_channel_s> new_freq_shift_chan = std::make_unique<freq_shift_channel_s>();
            new_freq_shift_chan->time_shift_rollover_sample_idx = freq_shift_params.first;
            new_freq_shift_chan->angular_freq = freq_shift_params.second;

            new_chan->freq_shift_chan = std::move(new_freq_shift_chan);
            m_channels.push_back(std::move(new_chan));
        }

        /* allocate CUDA memory for the frequency shift channel information; the master copy
         *  is kept within the m_channels data structure, but it is copied to the device when
         *  the 'apply' method is invoked */
        cudaErrChkAssert(cudaMallocManaged(&d_freq_shift_channels,
                                           m_channels.size() * sizeof(freq_shift_channel_s)));

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
        if (m_max_num_input_samples != in.size())
        {
            if (m_cuda_input_data)
            {
                cudaErrChkAssert(cudaFree(m_cuda_input_data));
                m_cuda_input_data = nullptr;
                m_max_num_input_samples = 0;
            }

            cudaErrChkAssert(cudaMallocManaged(&m_cuda_input_data,
                                               in.size() * sizeof(std::complex<float>)));
            m_max_num_input_samples = in.size();
        }
        
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
            out[chan_idx].resize(m_channels[chan_idx]->initialize(in.size()));
        }

        /* copy the input data to the GPU */
        cudaErrChkAssert(cudaMemcpy(static_cast<void *>(m_cuda_input_data),
                                    static_cast<void *>(in.data()),
                                    in.size() * sizeof(std::complex<float>),
                                    cudaMemcpyHostToDevice));

        /* copy the frequency shift channel information to the GPU */
        for (uint32_t chan_idx = 0; chan_idx < m_channels.size(); ++chan_idx)
        {
            cudaErrChkAssert(cudaMemcpy(static_cast<void *>(&d_freq_shift_channels[chan_idx]),
                                        static_cast<void *>(m_channels[chan_idx]->freq_shift_chan.get()),
                                        sizeof(freq_shift_channel_s),
                                        cudaMemcpyHostToDevice));
        }
        
        /* calculate frequency shift kernel parameters */
        uint32_t num_samples_per_freq_shift_thread = MAX_NUM_INPUT_SAMPLES_FOR_MULTI_CHAN_FREQ_SHIFT_KERNEL;
        uint32_t samples_per_freq_shift_thread_block = num_samples_per_freq_shift_thread * MAX_NUM_CUDA_THREADS;
        uint32_t num_thread_blocks = (in.size() + samples_per_freq_shift_thread_block - 1) /
                                             samples_per_freq_shift_thread_block;

        uint32_t freq_shift_shared_memory_size_in_bytes = sizeof(freq_shift_channel_s) * m_channels.size();
        
        falcon_dsp::falcon_dsp_host_timer timer("FREQ_SHIFT KERNEL", TIMING_LOGS_ENABLED);

        /* run the frequency shift multi-channel kernel on the GPU */
        __freq_shift_multi_chan<<<num_thread_blocks, MAX_NUM_CUDA_THREADS, freq_shift_shared_memory_size_in_bytes>>>(
                            d_freq_shift_channels,
                            m_channels.size(),
                            num_samples_per_freq_shift_thread,
                            m_cuda_input_data,
                            m_max_num_input_samples);
            
        cudaErrChkAssert(cudaPeekAtLastError());
            
        /* wait for GPU to finish frequency shifting */
        cudaErrChkAssert(cudaDeviceSynchronize());

        timer.log_duration("FREQ_SHIFT Kernel Complete");

        /* frequency shifting complete; update the trackers */
        for (uint32_t chan_idx = 0; chan_idx < m_channels.size(); ++chan_idx)
        {
            m_channels[chan_idx]->freq_shift_chan->num_samples_handled += in.size();
            m_channels[chan_idx]->freq_shift_chan->num_samples_handled =
                static_cast<uint32_t>(m_channels[chan_idx]->freq_shift_chan->num_samples_handled) % m_channels[chan_idx]->freq_shift_chan->time_shift_rollover_sample_idx;
        }
            
        /* now resample each channel */
        for (uint32_t chan_idx = 0; chan_idx < m_channels.size(); ++chan_idx)
        {
            std::stringstream resample_timer_name;
            resample_timer_name << "RESAMP KERNEL " << chan_idx;
            falcon_dsp::falcon_dsp_host_timer resample_timer(resample_timer_name.str(), TIMING_LOGS_ENABLED);
        
            if (MAX_NUM_OUTPUT_SAMPLES_PER_THREAD_FOR_RESAMPLER_KERNEL == 1)
            {
                __polyphase_resampler_single_out<<<num_thread_blocks, MAX_NUM_CUDA_THREADS>>>(
                             m_channels[chan_idx]->d_resample_coeffs,
                             m_channels[chan_idx]->resample_coeffs_len,
                             m_channels[chan_idx]->d_resample_output_params,
                             m_channels[chan_idx]->resample_output_params_len,
                             m_channels[chan_idx]->d_freq_shifted_data,
                             m_channels[chan_idx]->freq_shifted_data_len,
                             m_channels[chan_idx]->d_resampled_data,
                             m_channels[chan_idx]->resampled_data_len,
                             m_channels[chan_idx]->resampler_params.coeffs_per_phase);
            }
            else
            {
                __polyphase_resampler_multi_out<<<num_thread_blocks, MAX_NUM_CUDA_THREADS>>>(
                             m_channels[chan_idx]->d_resample_coeffs,
                             m_channels[chan_idx]->resample_coeffs_len,
                             m_channels[chan_idx]->d_resample_output_params,
                             m_channels[chan_idx]->resample_output_params_len,
                             m_channels[chan_idx]->d_freq_shifted_data,
                             m_channels[chan_idx]->freq_shifted_data_len,
                             m_channels[chan_idx]->d_resampled_data,
                             m_channels[chan_idx]->resampled_data_len,
                             m_channels[chan_idx]->resampler_params.coeffs_per_phase,
                             MAX_NUM_OUTPUT_SAMPLES_PER_THREAD_FOR_RESAMPLER_KERNEL);
            }

            cudaErrChkAssert(cudaPeekAtLastError());

            /* wait for GPU to finish before accessing on host */
            cudaErrChkAssert(cudaDeviceSynchronize());
        
            resample_timer.log_duration("Resampling Complete");
            
            /* copy output samples out of CUDA memory */
            cudaErrChkAssert(cudaMemcpy(out[chan_idx].data(),
                                        m_channels[chan_idx]->d_resampled_data,
                                        m_channels[chan_idx]->resampled_data_len * sizeof(std::complex<float>),
                                        cudaMemcpyDeviceToHost));

            /* finished resampling; now update the resampler state buffer*/
            _manage_resampler_state(chan_idx, in.size());
        }

        return out.size() > 0;
    }

    void falcon_dsp_multi_rate_channelizer_cuda::reset_state(void)
    {
        for (uint32_t chan_idx = 0; chan_idx < m_channels.size(); ++chan_idx)
        {
            m_channels[chan_idx]->freq_shift_chan->reset_state();
            m_channels[chan_idx]->resampler_params.reset_state();
        }
    }

    void falcon_dsp_multi_rate_channelizer_cuda::_manage_resampler_state(uint32_t chan_idx, uint32_t input_vector_len)
    {
        /* find number of samples retained in buffer */
        int64_t retain = m_channels[chan_idx]->resampler_params.state.size() - input_vector_len;
        if (retain > 0)
        {
            /* for input_vector_len smaller than state buffer, copy end of buffer to beginning */
            copy(m_channels[chan_idx]->resampler_params.state.end() - retain,
                 m_channels[chan_idx]->resampler_params.state.end(),
                 m_channels[chan_idx]->resampler_params.state.begin());
            
            /* then, copy the entire (short) input to end of buffer */
            uint32_t in_idx = 0;
            for (uint64_t state_copy_idx = retain;
                 state_copy_idx < m_channels[chan_idx]->resampler_params.state.size();
                 ++state_copy_idx)
            {
                /* compute the next index to copy. note that here we need to account for the
                 *  state buffer padding that was added to the resampler input */
                uint32_t next_idx_to_copy = in_idx + m_channels[chan_idx]->resampler_params.state.size();
                
                /* copy over the state information */
                cudaErrChkAssert(cudaMemcpy(m_channels[chan_idx]->resampler_params.state.data() + state_copy_idx,
                                            m_channels[chan_idx]->d_freq_shifted_data + next_idx_to_copy,
                                            sizeof(std::complex<float>),
                                            cudaMemcpyDeviceToHost));
                
                /* keep working through the resampler input buffer */
                in_idx++;
            }
        }
        else
        {
            /* just copy last input samples into state buffer */
            for (uint64_t state_copy_idx = 0;
                 state_copy_idx < m_channels[chan_idx]->resampler_params.state.size();
                 ++state_copy_idx)
            {
                /* compute the next index to copy. note that here we need to account for the
                 *  state buffer padding that was added to the resampler input */
                uint32_t next_idx_to_copy = m_channels[chan_idx]->resampled_data_len -
                                                m_channels[chan_idx]->resampler_params.state.size() +
                                                state_copy_idx;
                
                cudaErrChkAssert(cudaMemcpy(m_channels[chan_idx]->resampler_params.state.data() + state_copy_idx,
                                            m_channels[chan_idx]->d_resampled_data + next_idx_to_copy,
                                            sizeof(std::complex<float>),
                                            cudaMemcpyDeviceToHost));
            }
        }
    }
}
