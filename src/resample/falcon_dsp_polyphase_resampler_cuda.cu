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

/* The polyphase resampler implementation is based on code provided at
 *  https://github.com/staticfloat/upfirdn; which appears to be originally
 *  from https://sourceforge.net/motorola/upfirdn/home/Home/
 *
 * The Resampler interface is somewhat modified for the author's purposes, but
 *  the original copyright is provided here due to the heavy reuse from the
 *  original code.
 */

/*
Copyright (c) 2009, Motorola, Inc
All Rights Reserved.
Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright 
notice, this list of conditions and the following disclaimer in the 
documentation and/or other materials provided with the distribution.

* Neither the name of Motorola nor the names of its contributors may be 
used to endorse or promote products derived from this software without 
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS 
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,  
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR 
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/******************************************************************************
 *
 * @file     falcon_dsp_polyphase_resampler_cuda.cu
 * @author   OrthogonalHawk
 * @date     22-Apr-2019
 *
 * @brief    CUDA implementation of a polyphase resampler.
 *
 * @section  DESCRIPTION
 *
 * Implements CUDA version of a polyphase resampler.
 *
 * @section  HISTORY
 *
 * 22-Apr-2019  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <iostream>
#include <cuComplex.h>
#include <numeric>
#include <stdint.h>

#include "resample/falcon_dsp_polyphase_resampler_cuda.h"
#include "utilities/falcon_dsp_utils.h"

/******************************************************************************
 *                                 CONSTANTS
 *****************************************************************************/

const uint32_t MAX_NUM_INPUT_SAMPLES_PER_CUDA_KERNEL = 16384 * 16;
const uint32_t MAX_NUM_SHARED_MEMORY_COEFFS_PER_THREAD_BLOCK = 256;

const uint32_t MAX_NUM_OUTPUTS_PER_CUDA_THREAD = 32;
const uint32_t MAX_NUM_CUDA_THREADS = 256;

/******************************************************************************
 *                              ENUMS & TYPEDEFS
 *****************************************************************************/

struct output_sample_s
{
    bool             active;
    cuFloatComplex   acc;
    cuFloatComplex * coeff_ptr;
    int64_t          data_start_idx;
    int64_t          data_stop_idx;
};
        
/******************************************************************************
 *                                  MACROS
 *****************************************************************************/

/******************************************************************************
 *                            CLASS IMPLEMENTATION
 *****************************************************************************/

namespace falcon_dsp
{
    template<class T, class C>
    falcon_dsp_polyphase_resampler_cuda<T, C>::falcon_dsp_polyphase_resampler_cuda(uint32_t up_rate, uint32_t down_rate, std::vector<C>& filter_coeffs)
      : falcon_dsp_polyphase_resampler<T, C>(up_rate, down_rate, filter_coeffs)
    {
        /* allocate CUDA unified memory space for filter coefficients */
        cudaMallocManaged(&m_cuda_filter_coeffs, falcon_dsp_polyphase_resampler<T, C>::m_transposed_coeffs.size() * sizeof(C));
                  
        /* allocate CUDA unified memory space for input and output data */
        m_max_num_cuda_input_samples = (MAX_NUM_INPUT_SAMPLES_PER_CUDA_KERNEL +
                                        falcon_dsp_polyphase_resampler<T, C>::m_coeffs_per_phase + 1);
        
        m_max_num_cuda_output_samples = falcon_dsp_polyphase_resampler<T, C>::needed_out_count(MAX_NUM_INPUT_SAMPLES_PER_CUDA_KERNEL);
        
        cudaMallocManaged(&m_cuda_input_samples, m_max_num_cuda_input_samples * sizeof(T));
        cudaMallocManaged(&m_cuda_output_samples, m_max_num_cuda_output_samples * sizeof(T));

        /* calculate the average number of samples that are advanced for
         *  each output sample */
        m_avg_advance_per_output_sample = get_average_advance_in_samples();

        /* calculate the optimal number of outputs per CUDA thread. this value
         *  is chosen to minimize the number of repeated requests for the same
         *  input data from global memory and to prevent memory buffer collisions
         *  between CUDA threads in the same warp */
        m_num_outputs_per_cuda_thread = static_cast<uint32_t>(
                                            std::ceil(static_cast<float>(falcon_dsp::calculate_filter_delay_from_up_down_rates(filter_coeffs.size(), up_rate, down_rate)) /
                                                      static_cast<float>(falcon_dsp_polyphase_resampler_cuda<T, C>::m_avg_advance_per_output_sample)));

        if (m_num_outputs_per_cuda_thread > MAX_NUM_OUTPUTS_PER_CUDA_THREAD)
        {
            m_num_outputs_per_cuda_thread = MAX_NUM_OUTPUTS_PER_CUDA_THREAD;
        }
        
       /* change the shared memory size to 8 bytes per shared memory bank. this is so that we
        *  can better handle complex<float> data, which is natively 8 bytes in size */
       cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    }
    
    template<class T, class C>
    falcon_dsp_polyphase_resampler_cuda<T, C>::~falcon_dsp_polyphase_resampler_cuda(void)
    {
        std::lock_guard<std::mutex> lock(falcon_dsp_polyphase_resampler<T, C>::m_mutex);
        
        if (m_cuda_filter_coeffs)
        {
            cudaFree(m_cuda_filter_coeffs);
            m_cuda_filter_coeffs = nullptr;
        }
        
        if (m_cuda_input_samples)
        {
            cudaFree(m_cuda_input_samples);
            m_cuda_input_samples = nullptr;
        }
        
        if (m_cuda_output_samples)
        {
            cudaFree(m_cuda_output_samples);
            m_cuda_output_samples = nullptr;
        }
    }

    /* CUDA kernel function that resamples the input array */
    __global__
    void _polyphase_resampler_cuda(cuFloatComplex * in, uint32_t in_len,
                                   cuFloatComplex * out, uint32_t out_len,
                                   cuFloatComplex * coeffs, uint32_t coeffs_len,
                                   uint32_t coeffs_per_phase,
                                   uint32_t num_outputs_per_cuda_thread,
                                   int64_t start_x_idx,
                                   uint32_t start_t,
                                   uint32_t start_output_idx,
                                   uint32_t up_rate,
                                   uint32_t down_rate)
    {
        __shared__ cuFloatComplex s_coeffs[MAX_NUM_SHARED_MEMORY_COEFFS_PER_THREAD_BLOCK];
        output_sample_s out_samples[MAX_NUM_OUTPUTS_PER_CUDA_THREAD];

        /* initialize the output sample data structures */
        for (uint32_t ii = 0;
             ii < num_outputs_per_cuda_thread && ii < MAX_NUM_OUTPUTS_PER_CUDA_THREAD;
             ++ii)
        {
            out_samples[ii].active = false;
            out_samples[ii].coeff_ptr = nullptr;
            out_samples[ii].data_start_idx = LONG_MAX;
            out_samples[ii].data_stop_idx = LONG_MAX;
            out_samples[ii].acc.x = 0;
            out_samples[ii].acc.y = 0;
        }

        /* compute the thread index */
        uint32_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;

        /* copy coefficients to shared memory */
        if (threadIdx.x < coeffs_len)
        {
            s_coeffs[threadIdx.x] = coeffs[threadIdx.x];
        }
        __syncthreads();

        /* compute local thread variables */
        int64_t thread_x_idx = start_x_idx;
        uint32_t thread_t = start_t;
        
        int64_t thread_start_output_sample_idx = start_output_idx + (thread_index * num_outputs_per_cuda_thread);
        
        /* verify that this thread has at least one output to compute */
        if (thread_start_output_sample_idx >= out_len)
        {
            return;
        }
        
        for (int64_t out_sample_idx = start_output_idx;
             out_sample_idx < thread_start_output_sample_idx;
             ++out_sample_idx)
        {
            /* compute the next output sample 'cycle' updates */
            thread_t += down_rate;
            thread_x_idx += thread_t / up_rate;
            thread_t %= up_rate;
        }
        
        /* capture the FIRST output sample information */
        out_samples[0].active = true;
        out_samples[0].coeff_ptr = s_coeffs + thread_t * coeffs_per_phase;
        out_samples[0].data_start_idx = thread_x_idx - coeffs_per_phase + 1;
        out_samples[0].data_stop_idx = thread_x_idx;

        int64_t thread_data_start_idx = out_samples[0].data_start_idx;
        int64_t thread_data_stop_idx =  out_samples[0].data_stop_idx;
        
        /* ensure that the output sample is still within the range of the
         *  configured output data; there may be cases where some threads in
         *  a thread block are not needed. */
        for (uint32_t out_sample_idx = 1;
             out_sample_idx < num_outputs_per_cuda_thread && (out_sample_idx + thread_start_output_sample_idx) < out_len;
             ++out_sample_idx)
        {
            /* compute the next output sample 'cycle' updates */
            thread_t += down_rate;
            thread_x_idx += thread_t / up_rate;
            thread_t %= up_rate;
            
            out_samples[out_sample_idx].active = true;
            out_samples[out_sample_idx].coeff_ptr = s_coeffs + thread_t * coeffs_per_phase;
            out_samples[out_sample_idx].data_start_idx = thread_x_idx - coeffs_per_phase + 1;
            out_samples[out_sample_idx].data_stop_idx = thread_x_idx;
            
            thread_data_stop_idx = out_samples[out_sample_idx].data_stop_idx;
        }
        
        uint32_t first_active_out_sample = 0;
        cuFloatComplex next_x_val;
        for (int64_t x_idx = thread_data_start_idx; x_idx < thread_data_stop_idx; ++x_idx)
        {
            next_x_val = in[x_idx];
            for (uint32_t thread_out_sample_idx = first_active_out_sample;
                 thread_out_sample_idx < num_outputs_per_cuda_thread;
                 ++thread_out_sample_idx)
            {               
                /* we don't need to check whether x_idx is less than data_stop_idx
                 *  here because it's checked later and once x_idx is >= data_stop_idx
                 *  this output is 'disabled' and will no longer be assessed */
                if (x_idx >= out_samples[thread_out_sample_idx].data_start_idx)
                {                   
                    out_samples[thread_out_sample_idx].acc = cuCaddf(out_samples[thread_out_sample_idx].acc,
                                                                     cuCmulf(next_x_val, *(out_samples[thread_out_sample_idx].coeff_ptr++)));
                                                              
                    if ((out_samples[thread_out_sample_idx].data_stop_idx - 1) <= x_idx)
                    {
                        /* finished computing acc for this output */
                        first_active_out_sample++;
                    }
                }
                else
                {
                    break;
                }
            }
        }
        
        /* set the global outputs */
        uint64_t global_output_sample_idx_base = start_output_idx + (thread_index * num_outputs_per_cuda_thread);
        for (uint32_t out_sample_idx = 0;
             out_sample_idx < num_outputs_per_cuda_thread && out_samples[out_sample_idx].active;
             ++out_sample_idx)
        {
            out[global_output_sample_idx_base + out_sample_idx] = out_samples[out_sample_idx].acc;
        }
    }
    
    /* specialized implementation */
    template<>
    int32_t falcon_dsp_polyphase_resampler_cuda<std::complex<float>, std::complex<float>>::apply(std::vector<input_type>& in, std::vector<output_type>& out)
    {
        std::lock_guard<std::mutex> lock(falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_mutex);
        
        /* copy the filter coefficients into CUDA memory */
        cudaMemcpy(m_cuda_filter_coeffs,
                   m_transposed_coeffs.data(),
                   m_transposed_coeffs.size() * sizeof(std::complex<float>),
                   cudaMemcpyHostToDevice );
        
        cuFloatComplex * cuda_filter_coeffs = static_cast<cuFloatComplex *>(m_cuda_filter_coeffs);
        cuFloatComplex * cuda_input_data = static_cast<cuFloatComplex *>(m_cuda_input_samples);
        cuFloatComplex * cuda_output_data = static_cast<cuFloatComplex *>(m_cuda_output_samples);
        
        /* clear out the output and allocate space for the resulting data */
        out.clear();
        out.reserve(needed_out_count(in.size()));
        for (uint32_t out_idx = 0;
             out_idx < needed_out_count(in.size());
             ++out_idx)
        {
            out.push_back(std::complex<float>(0, 0));
        }
        uint32_t cur_out_idx = 0;
        
        /* don't bother with running a CUDA/GPU implementation if the input size is not
         *  larger than the state vector */
        if (in.size() <= falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_state.size())
        {
            return falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::apply(in, out);  
        }
        
        /* copy all input samples into CUDA memory */
        for (uint32_t in_data_idx = 0; in_data_idx < in.size(); ++in_data_idx)
        {
            cuda_input_data[in_data_idx] = *(static_cast<cuFloatComplex *>(static_cast<void *>(&in[in_data_idx])));
        }
        
        /* x_idx points to the latest processed input sample */
        int64_t x_idx = falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_xOffset;
        while (static_cast<uint64_t>(x_idx) < in.size())
        {
            bool required_state_array = false;
            
            /* setup variables for the cases where the next sample is handled in the CPU */
            output_type acc = output_type(0);
            typename std::vector<coeff_type>::iterator coeff_iter =
                m_transposed_coeffs.begin() + m_t * m_coeffs_per_phase;
            
            /* need to look back over the previous samples to compute the
             *  current filtered value */
            int64_t x_back_idx = x_idx - m_coeffs_per_phase + 1;
            int64_t offset = 0 - x_back_idx;
            
            /* values toward the beginning of the vector may require samples
             *  from the state array; handle these in C++/CPU */
            if (offset > 0)
            {
                required_state_array = true;
                
                /* need to draw from the state buffer */
                typename std::vector<std::complex<float>>::iterator state_iter = falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_state.end() - offset;
                while (state_iter < falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_state.end())
                {
                    acc += *(state_iter++) * *(coeff_iter++);
                }
                x_back_idx += offset;
            }
            
            /* if possible, handle multiple samples at once using CUDA. check for whether or not
             *  the state array was required as a way to detect samples at the beginning of the
             *  input vector */
            if (!required_state_array &&
                (out.size() - cur_out_idx) > 0 &&
                m_transposed_coeffs.size() < MAX_NUM_SHARED_MEMORY_COEFFS_PER_THREAD_BLOCK)
            {
                uint32_t num_thread_blocks = (out.size() - cur_out_idx) / (MAX_NUM_CUDA_THREADS * m_num_outputs_per_cuda_thread);
                if ((out.size() - cur_out_idx) % (MAX_NUM_CUDA_THREADS * m_num_outputs_per_cuda_thread) != 0)
                {
                    num_thread_blocks++;
                }
                
                printf("Num required thread blocks(%u threads):%u for %u output samples (%u per thread)\n",
                    MAX_NUM_CUDA_THREADS, num_thread_blocks,
                    static_cast<uint32_t>((out.size() - cur_out_idx)), m_num_outputs_per_cuda_thread);

                _polyphase_resampler_cuda<<<num_thread_blocks, MAX_NUM_CUDA_THREADS>>>(
                        cuda_input_data,
                        m_max_num_cuda_input_samples,
                        cuda_output_data,
                        m_max_num_cuda_output_samples,
                        cuda_filter_coeffs,
                        m_transposed_coeffs.size(),
                        m_coeffs_per_phase,
                        m_num_outputs_per_cuda_thread,
                        x_idx, /* x_start_idx */
                        m_t,
                        cur_out_idx,
                        m_up_rate,
                        m_down_rate);

                uint32_t num_outputs_from_thread_blocks, new_t;
                int64_t new_x_idx;
                compute_next_filter_params(x_idx, in.size(), m_t,
                                           (out.size() - cur_out_idx),
                                           num_outputs_from_thread_blocks,
                                           new_t, new_x_idx);

                /* wait for GPU to finish before accessing on host */
                cudaDeviceSynchronize();
                
                /* copy output samples out of CUDA memory */
                cudaMemcpy(out.data() + cur_out_idx,
                           cuda_output_data + cur_out_idx,
                           num_outputs_from_thread_blocks * sizeof(std::complex<float>),
                           cudaMemcpyDeviceToHost);
                
                /* update tracking parameters */
                m_t = new_t;
                x_idx += new_x_idx;
                cur_out_idx += num_outputs_from_thread_blocks;
            }
            else
            {               
                /* either handling data at the beginning of the vector or close to the end;
                 *  just handle here in the CPU to keep things simpler */
                while (x_back_idx <= x_idx)
                {
                    acc += in[x_back_idx++] * *(coeff_iter++);
                }
                
                if (cur_out_idx == 20)
                {
                    printf("acc.re=%f acc.im=%f\n", acc.real(), acc.imag());
                }

                out[cur_out_idx++] = acc;
                falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_t += falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_down_rate;

                int64_t advance_amount = falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_t / falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_up_rate;
                x_idx += advance_amount;

                // which phase of the filter to use
                falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_t %= falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_up_rate;
            }
        }
        
        falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_xOffset = x_idx - in.size();

        /* finished resampling; now update the state buffer so that future (assumed contiguous)
         *  input vectors can be resampled using these old samples.
         * 
         * based on the check at the start of this function, if the input size is less than
         *  the state buffer size the computation was passed over to the C++ implementation
         *  so there is no need to duplicate that handling here. */
        
        /* just copy last input samples into state buffer */
        for (uint64_t state_copy_idx = 0;
             state_copy_idx < falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_state.size();
             ++state_copy_idx)
        {
            falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_state[state_copy_idx] =
                in[in.size() - falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_state.size() + state_copy_idx];   
        }
        
        /* number of samples computed */
        return out.size();
    }
    
    template<class T, class C>
    void falcon_dsp_polyphase_resampler_cuda<T, C>::compute_next_filter_params(int64_t cur_x_idx, size_t in_size, uint32_t cur_t,
                                                                               uint32_t max_out_samples,
                                                                               uint32_t& num_out_samples,
                                                                               uint32_t& new_t,
                                                                               int64_t& new_x_idx)
    {
        uint32_t local_t = cur_t;
        new_x_idx = cur_x_idx;

        num_out_samples = 0;
        new_t = local_t;
        
        while (num_out_samples < max_out_samples && new_x_idx < in_size)
        {
            /* compute the next 'cycle' updates */
            local_t += falcon_dsp_polyphase_resampler<T, C>::m_down_rate;
            int64_t advance_amount = local_t / falcon_dsp_polyphase_resampler<T, C>::m_up_rate;
            new_x_idx += advance_amount;
            local_t %= falcon_dsp_polyphase_resampler<T, C>::m_up_rate;
            
            /* increment trackers; assuming one output sample per thread */
            num_out_samples++;
            new_t = local_t;
        }
    }
    
    template<class T, class C>
    uint32_t falcon_dsp_polyphase_resampler_cuda<T, C>::get_average_advance_in_samples(void)
    {
        const uint32_t NUM_SAMPLES_TO_EVALUATE = 1e6;

        uint32_t local_t = falcon_dsp_polyphase_resampler<T, C>::m_t;
        int64_t  local_x_idx = 0;
        std::vector<uint32_t> local_advances;

        bool reached_limit = false;
        while (!reached_limit && local_x_idx < NUM_SAMPLES_TO_EVALUATE)
        {
            /* compute the next 'cycle' updates */
            local_t += falcon_dsp_polyphase_resampler<T, C>::m_down_rate;
            int64_t advance_amount = local_t / falcon_dsp_polyphase_resampler<T, C>::m_up_rate;
            local_t %= falcon_dsp_polyphase_resampler<T, C>::m_up_rate;
            local_x_idx += advance_amount;

            local_advances.push_back(advance_amount);
        }

        float accum_sum = static_cast<float>(std::accumulate(local_advances.begin(), local_advances.end(), 0));
        float advance_avg = accum_sum / static_cast<float>(local_advances.size());

        return static_cast<uint32_t>(std::ceil(advance_avg));
    }

    /* force instantiation for specific types */
    template class falcon_dsp_polyphase_resampler_cuda<std::complex<float>, std::complex<float>>;
}

/******************************************************************************
 *                            CLASS IMPLEMENTATION
 *****************************************************************************/
