/******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 OrthogonalHawk
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
 * @file     falcon_dsp_freq_shift_cuda.cu
 * @author   OrthogonalHawk
 * @date     04-Jun-2019
 *
 * @brief    Implements a CUDA-based time series frequency shift operation.
 *
 * @section  DESCRIPTION
 *
 * Implements the CUDA version of a time series frequency shift operation. Both
 *  a standalone function and a class-based tranform object are supported.
 *
 * @section  HISTORY
 *
 * 04-Jun-2019  OrthogonalHawk  File created.
 * 19-Jan-2020  OrthogonalHawk  Refactored to use falcon_dsp_freq_shift_cuda.h
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <iostream>
#include <memory>
#include <stdint.h>

#include "transform/falcon_dsp_freq_shift_cuda.h"
#include "utilities/falcon_dsp_host_timer.h"

/******************************************************************************
 *                                 CONSTANTS
 *****************************************************************************/

const uint32_t DUMMY_SAMPLE_RATE_IN_SPS = 1e6;
const float DUMMY_FREQ_SHIFT_IN_HZ = 0.0;

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
    
    /* @brief CUDA implementation of a frequency shift vector operation.
     * @param[in] in_sample_rate_in_sps - input vector sample rate in samples
     *                                      per second.
     * @param[in] in                    - input vector
     * @param[in] freq_shift_in_hz      - amount to frequency shift in Hz
     * @param[out] out                  - frequency shifted vector
     * @return True if the input vector was frequency shifted as requested;
     *          false otherwise.
     */
    bool freq_shift_cuda(uint32_t in_sample_rate_in_sps, std::vector<std::complex<int16_t>>& in,
                         int32_t freq_shift_in_hz, std::vector<std::complex<int16_t>>& out)
    {
        falcon_dsp_freq_shift_cuda freq_shifter(in_sample_rate_in_sps, freq_shift_in_hz);
        return freq_shifter.apply(in, out);
    }
    
    bool freq_shift_cuda(uint32_t in_sample_rate_in_sps, std::vector<std::complex<float>>& in,
                         int32_t freq_shift_in_hz, std::vector<std::complex<float>>& out)
    {        
        falcon_dsp_freq_shift_cuda freq_shifter(in_sample_rate_in_sps, freq_shift_in_hz);
        return freq_shifter.apply(in, out);
    }
    
    /* @brief CUDA implementation of a multi-channel frequency shift vector operation.
     * @param[in] in_sample_rate_in_sps - input vector sample rate in samples
     *                                      per second.
     * @param[in] in                    - input vector
     * @param[in] freq_shift_in_hz      - vector of requested frequency shifts; each shift value
     *                                     generates a separate output 'channel'
     * @param[out] out                  - vector of frequency shifted vectors
     * @return True if the input vector was frequency shifted as requested;
     *          false otherwise.
     */
    bool freq_shift_cuda(uint32_t in_sample_rate_in_sps, std::vector<std::complex<float>>& in,
                         std::vector<int32_t>& freq_shift_in_hz, std::vector<std::vector<std::complex<float>>>& out)
    {
        falcon_dsp_freq_shift_cuda freq_shifter(in_sample_rate_in_sps, freq_shift_in_hz);
        return freq_shifter.apply(in, out);
    }
    
    /* CUDA kernel function that applies a frequency shift and puts the shifted data
     *  into a (new?) memory location. can either be used to modify the data in-place
     *  or to make a new vector with the shifted data. */
    __global__
    void __freq_shift(uint32_t num_samples_handled_previously,
                      uint32_t time_shift_rollover_sample_idx,
                      double angular_freq,
                      uint32_t num_samples_to_process_per_thread,
                      cuFloatComplex * in_data,
                      uint32_t in_data_len,
                      cuFloatComplex * out_data,
                      uint32_t out_data_len)
    {
        /* retrieve the starting data index that corresponds to this thread */
        uint32_t start_data_index = blockIdx.x * blockDim.x * num_samples_to_process_per_thread +
                                        threadIdx.x * num_samples_to_process_per_thread;
     
        /* catch the case where the output buffer is shorter than
         *  the input buffer */
        if (out_data_len < in_data_len)
        {
            return;
        }
        
        /* catch the case where the input size is not an integer
         *  multiple of the thread block size */
        if (start_data_index > in_data_len ||
            start_data_index > out_data_len)
        {
            return;
        }
        
        /* catch the case where this kernel cannot process the full number of samples
         *  when the end of the input buffer is reached. note that due to the previous
         *  check to make sure that the output buffer is at least as long as the input
         *  buffer it is sufficient to only check the input buffer here */
        uint32_t local_num_samples_to_process = num_samples_to_process_per_thread;
        if ((start_data_index + local_num_samples_to_process) > in_data_len)
        {
            local_num_samples_to_process = in_data_len - start_data_index;
        }
        
        /* compute the time shift index for the current thread */
        uint64_t orig_time_shift_idx = num_samples_handled_previously + start_data_index;
        uint64_t time_shift_idx = orig_time_shift_idx;
        
        float freq_shift_angle = 0.;
        float freq_shift_real = 0.;
        float freq_shift_imag = 0.;
        cuFloatComplex freq_shift;
                    
        for (uint32_t ii = 0;
             ii < local_num_samples_to_process && (start_data_index + ii) < in_data_len && (start_data_index + ii) < out_data_len;
             ++ii)
        {
            time_shift_idx %= time_shift_rollover_sample_idx;
            
            /* compute the frequency shift multiplier value */
            freq_shift_angle = angular_freq * time_shift_idx;
            freq_shift_real = cosf(freq_shift_angle);
            freq_shift_imag = sinf(freq_shift_angle);
            
            /* set a CUDA complex variable to apply the freqency shift */
            freq_shift.x = freq_shift_real;
            freq_shift.y = freq_shift_imag;

            /* apply the frequency shift; may be used to modify data in place or to put
             *  the output into a new vector based on the input arguments */
            out_data[start_data_index + ii] = cuCmulf(in_data[start_data_index + ii], freq_shift);
            
            /* finished with this sample; increment time index */
            time_shift_idx++;
        }
    }
    
    /* CUDA kernel function that supports multi-channel frequency shifting. */
    __global__
    void __freq_shift_multi_chan(uint32_t num_samples_handled_previously,
                                 freq_shift_channel_s * channels,
                                 uint32_t num_channels,
                                 uint32_t num_samples_to_process_per_thread,
                                 cuFloatComplex * in_data,
                                 uint32_t in_data_len)
    {
        /* retrieve the starting data index that corresponds to this thread */
        uint32_t start_data_index = blockIdx.x * blockDim.x * num_samples_to_process_per_thread +
                                        threadIdx.x * num_samples_to_process_per_thread;
     
        /* catch the case where the channel information is not available or where
         *  the output buffer is shorter than the input buffer */
        for (uint32_t chan_idx = 0; chan_idx < num_channels; ++chan_idx)
        {
            if (channels[chan_idx].out_data == nullptr ||
                channels[chan_idx].out_data_len < in_data_len)
            {               
                return;
            }
        }
        
        /* catch the case where the input size is not an integer
         *  multiple of the thread block size */
        for (uint32_t chan_idx = 0; chan_idx < num_channels; ++chan_idx)
        {
            if (start_data_index > in_data_len ||
                start_data_index > channels[chan_idx].out_data_len)
            {
                return;
            }
        }
        
        /* catch the case where this kernel cannot process the full number of samples
         *  when the end of the input buffer is reached. note that due to the previous
         *  check to make sure that the output buffer is at least as long as the input
         *  buffer it is sufficient to only check the input buffer here */
        uint32_t local_num_samples_to_process = num_samples_to_process_per_thread;
        if ((start_data_index + local_num_samples_to_process) > in_data_len)
        {
            local_num_samples_to_process = in_data_len - start_data_index;
        }
        
        /* compute the time shift index for the current thread */
        uint64_t time_shift_idx = num_samples_handled_previously + start_data_index;
        
        float freq_shift_angle = 0.;
        float freq_shift_real = 0.;
        float freq_shift_imag = 0.;
        cuFloatComplex freq_shift;
        cuFloatComplex next_input_sample;
        
        for (uint32_t sample_idx = 0;
             sample_idx < local_num_samples_to_process &&
                 (start_data_index + sample_idx) < in_data_len;
             ++sample_idx)
        {
            next_input_sample = in_data[start_data_index + sample_idx];
            
            for (uint32_t chan_idx = 0; chan_idx < num_channels; ++chan_idx)
            {
                time_shift_idx = num_samples_handled_previously + start_data_index + sample_idx;
                time_shift_idx %= channels[chan_idx].time_shift_rollover_sample_idx;
            
                /* compute the frequency shift multiplier value */
                freq_shift_angle = channels[chan_idx].angular_freq * time_shift_idx;
                freq_shift_real = cosf(freq_shift_angle);
                freq_shift_imag = sinf(freq_shift_angle);
            
                /* set a CUDA complex variable to apply the freqency shift */
                freq_shift.x = freq_shift_real;
                freq_shift.y = freq_shift_imag;

                /* apply the frequency shift; may be used to modify data in place or to put
                 *  the output into a new vector based on the input arguments */
                channels[chan_idx].out_data[start_data_index + sample_idx] = cuCmulf(next_input_sample, freq_shift);
            }
        }
    }
    
    /******************************************************************************
     *                           CLASS IMPLEMENTATION
     *****************************************************************************/
    
    falcon_dsp_freq_shift_cuda::falcon_dsp_freq_shift_cuda(uint32_t input_sample_rate_in_sps, int32_t freq_shift_in_hz)
      : falcon_dsp_freq_shift(DUMMY_SAMPLE_RATE_IN_SPS, DUMMY_FREQ_SHIFT_IN_HZ),
        m_cuda_input_data(nullptr),
        m_max_num_input_samples(0)
    {
        /* user is only requesting a single channel */
        auto chan_params = falcon_dsp_freq_shift::get_freq_shift_params(input_sample_rate_in_sps, freq_shift_in_hz);
        std::unique_ptr<freq_shift_channel_s> new_chan = std::make_unique<freq_shift_channel_s>();
        new_chan->time_shift_rollover_sample_idx = chan_params.first;
        new_chan->angular_freq = chan_params.second;

        m_freq_shift_channels.push_back(std::move(new_chan));
        
        /* allocate CUDA memory for the channel information; the master copy is kept
         *  on the host, but is copied to the device when the 'apply' method is invoked */
        cudaMallocManaged(&d_freq_shift_channels, m_freq_shift_channels.size() * sizeof(freq_shift_channel_s));
    }

    falcon_dsp_freq_shift_cuda::falcon_dsp_freq_shift_cuda(uint32_t input_sample_rate, std::vector<int32_t> freq_shift_in_hz)
      : falcon_dsp_freq_shift(DUMMY_SAMPLE_RATE_IN_SPS, DUMMY_FREQ_SHIFT_IN_HZ),
        m_cuda_input_data(nullptr),
        m_max_num_input_samples(0)
    {
        /* user is requesting multiple channels */
        for (auto freq_shift : freq_shift_in_hz)
        {
            auto chan_params = falcon_dsp_freq_shift::get_freq_shift_params(input_sample_rate, freq_shift);
            std::unique_ptr<freq_shift_channel_s> new_chan = std::make_unique<freq_shift_channel_s>();
            new_chan->time_shift_rollover_sample_idx = chan_params.first;
            new_chan->angular_freq = chan_params.second;

            m_freq_shift_channels.push_back(std::move(new_chan));
        }
          
        /* allocate CUDA memory for the channel information; the master copy is kept
         *  on the host, but is copied to the device when the 'apply' method is invoked */
        cudaMallocManaged(&d_freq_shift_channels, m_freq_shift_channels.size() * sizeof(freq_shift_channel_s));
    }
    
    falcon_dsp_freq_shift_cuda::~falcon_dsp_freq_shift_cuda(void)
    {
        std::lock_guard<std::mutex> lock(std::mutex);
        
        /* the freq_shift_channel_s objects automatically delete CUDA memory if it
         *  was allocated when the object is destroyed */
        m_freq_shift_channels.clear();
        
        /* cleanup the CUDA memory that was reserved in the constructor to house
         *  the channel information when CUDA kernels are running */
        if (d_freq_shift_channels)
        {
            cudaFree(d_freq_shift_channels);
            d_freq_shift_channels = nullptr;
        }
    }

    bool falcon_dsp_freq_shift_cuda::apply(std::vector<std::complex<int16_t>>& in, std::vector<std::complex<int16_t>>& out)
    {       
        /* create another copy of the data and cast to std::complex<float> */
        std::vector<std::complex<float>> tmp_in_vec;
        tmp_in_vec.reserve(in.size());
        for (auto in_iter = in.begin(); in_iter != in.end(); ++in_iter)
        {
            tmp_in_vec.push_back(std::complex<float>((*in_iter).real(), (*in_iter).imag()));
        }
        
        /* frequency shift the input data */
        std::vector<std::complex<float>> tmp_out_vec;
        bool ret = apply(tmp_in_vec, tmp_out_vec);

        /* cast the filtered output back to std::complex<int16_t> */
        for (auto out_iter = tmp_out_vec.begin(); out_iter != tmp_out_vec.end(); ++out_iter)
        {
            out.push_back(std::complex<int16_t>((*out_iter).real(), (*out_iter).imag()));
        }
        
        return ret;
    }
    
    bool falcon_dsp_freq_shift_cuda::apply(std::vector<std::complex<float>>& in, std::vector<std::complex<float>>& out)
    {
        /* sanity check the input and verify that invoking this method make sense. specifically,
         *  if a user configured the frequency shifter for multiple channels then this is not
         *  the right method to call. instead, the user should invoke the method that provides
         *  multiple output channels */
        if (m_freq_shift_channels.size() > 1)
        {
            return false;
        }
        
        std::vector<std::vector<std::complex<float>>> out_data;
        bool ret = apply(in, out_data);
        
        if (out_data.size() > 0)
        {
            /* swap to avoid unnecessary copying */
            out.swap(out_data[0]);
        }
        
        return ret;
    }
    
    
    bool falcon_dsp_freq_shift_cuda::apply(std::vector<std::complex<float>>& in, std::vector<std::vector<std::complex<float>>>& out)
    {
        std::lock_guard<std::mutex> lock(std::mutex);
        
        /* clear the output data structures and resize so that they can hold the shifted
         *  data. note that by using resize() the vector size is now equal to in.size()
         *  even without explicitly adding data to the vector, which means that we can
         *  add data directly into the vector data buffer without worrying about the
         *  vector size getting mismatched with the buffer contents */
        out.clear();
        out.resize(m_freq_shift_channels.size());
        for (uint32_t chan_idx = 0; chan_idx < m_freq_shift_channels.size(); ++chan_idx)
        {
            out[chan_idx].resize(in.size());
        }
        
        /* if there is enough space in previously allocated memory then use it; otherwise
         *  allocate new memory buffers. it is left as a future feature to specify a maximum
         *  memory size and process the data in chunks instead of requiring enough GPU
         *  memory to process the whole vector at once */
        
        /* allocate CUDA memory for the input samples */
        if (m_max_num_input_samples < in.size())
        {
            if (m_cuda_input_data)
            {
                cudaFree(m_cuda_input_data);
                m_cuda_input_data = nullptr;
                m_max_num_input_samples = 0;
            }
            
            cudaMallocManaged(&m_cuda_input_data, in.size() * sizeof(std::complex<float>));
            m_max_num_input_samples = in.size();
        }
        
        /* allocate CUDA memory for the output samples. all channels have the same amount
         *  of output space and at least one channel is guaranteed by the constructor. note
         *  this output memory is NOT needed if there is only a single output channel; in
         *  this case the data can be modified in-place on the GPU */
        if (m_freq_shift_channels.size() > 1 &&
            m_freq_shift_channels[0]->out_data_len < in.size())
        {
            for (uint32_t chan_idx = 0; chan_idx < m_freq_shift_channels.size(); ++chan_idx)
            {
                /* clean up existing memory */
                if (m_freq_shift_channels[chan_idx]->out_data)
                {
                    cudaFree(m_freq_shift_channels[chan_idx]->out_data);
                    m_freq_shift_channels[chan_idx]->out_data = nullptr;
                    m_freq_shift_channels[chan_idx]->out_data_len = 0;
                }
                
                /* allocate CUDA unified memory space for the output data */
                cudaMallocManaged(&(m_freq_shift_channels[chan_idx]->out_data), in.size() * sizeof(std::complex<float>));
                m_freq_shift_channels[chan_idx]->out_data_len = in.size();
            }
        }
        
        /* copy the input data to the GPU */
        cuFloatComplex * cuda_float_complex_input_data = static_cast<cuFloatComplex *>(m_cuda_input_data);        
        cudaMemcpy(static_cast<void *>(cuda_float_complex_input_data),
                   static_cast<void *>(in.data()),
                   in.size() * sizeof(std::complex<float>),
                   cudaMemcpyHostToDevice);
        
        /* run kernel on the GPU */
        uint32_t num_samples_per_thread = 4;
        uint32_t thread_block_size = 256;
        uint32_t samples_per_thread_block = num_samples_per_thread * thread_block_size;
        uint32_t num_thread_blocks = (in.size() + samples_per_thread_block - 1) / samples_per_thread_block;
        
        falcon_dsp::falcon_dsp_host_timer timer("KERNEL");
        if (m_freq_shift_channels.size() == 1)
        {
            __freq_shift<<<num_thread_blocks, thread_block_size>>>(m_freq_shift_channels[0]->num_samples_handled,
                                                                   m_freq_shift_channels[0]->time_shift_rollover_sample_idx,
                                                                   m_freq_shift_channels[0]->angular_freq,
                                                                   num_samples_per_thread,
                                                                   cuda_float_complex_input_data,
                                                                   m_max_num_input_samples,
                                                                   cuda_float_complex_input_data, /* modify in place */
                                                                   m_max_num_input_samples);
        
            /* wait for GPU to finish before accessing on host */
            cudaDeviceSynchronize();
            
            timer.log_duration("Single Chan Kernel Complete");
        
            /* copy output samples out of CUDA memory */
            cudaMemcpy(static_cast<void *>(out[0].data()),
                       static_cast<void *>(cuda_float_complex_input_data),
                       in.size() * sizeof(std::complex<float>),
                       cudaMemcpyDeviceToHost);
        }
        else /* use the multi-channel kernel */
        {            
            /* copy the channel information to the GPU */
            for (uint32_t chan_idx = 0; chan_idx < m_freq_shift_channels.size(); ++chan_idx)
            {
                cudaMemcpy(static_cast<void *>(&d_freq_shift_channels[chan_idx]),
                           static_cast<void *>(m_freq_shift_channels[chan_idx].get()),
                           sizeof(freq_shift_channel_s),
                           cudaMemcpyHostToDevice);
            }

            __freq_shift_multi_chan<<<num_thread_blocks, thread_block_size>>>(m_freq_shift_channels[0]->num_samples_handled,
                                                                              d_freq_shift_channels,
                                                                              m_freq_shift_channels.size(),
                                                                              num_samples_per_thread,
                                                                              cuda_float_complex_input_data,
                                                                              m_max_num_input_samples);
            
            /* wait for GPU to finish before accessing on host */
            cudaDeviceSynchronize();
            
            timer.log_duration("Multi Chan Kernel Complete");

            /* copy output samples out of CUDA memory */
            for (uint32_t chan_idx = 0; chan_idx < m_freq_shift_channels.size(); ++chan_idx)
            {
                cudaMemcpy(static_cast<void *>(out[chan_idx].data()),
                           static_cast<void *>(m_freq_shift_channels[chan_idx]->out_data),
                           in.size() * sizeof(std::complex<float>),
                           cudaMemcpyDeviceToHost);                
            }
        }

        for (auto& chan_iter : m_freq_shift_channels)
        {
            chan_iter->num_samples_handled += in.size();
            chan_iter->num_samples_handled = static_cast<uint32_t>(chan_iter->num_samples_handled) % chan_iter->time_shift_rollover_sample_idx;
        }

        return out.size() > 0;
    }
}
