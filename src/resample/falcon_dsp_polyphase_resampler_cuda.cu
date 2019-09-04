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
#include <stdint.h>

#include "resample/falcon_dsp_polyphase_resampler_cuda.h"

/******************************************************************************
 *                                 CONSTANTS
 *****************************************************************************/

const uint32_t MAX_NUM_INPUT_SAMPLES_PER_CUDA_KERNEL = 1024;
const uint32_t MAX_NUM_CUDA_THREADS = 256;

/******************************************************************************
 *                              ENUMS & TYPEDEFS
 *****************************************************************************/

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
        
        std::cout << "Maximum input samples: " << m_max_num_cuda_input_samples << std::endl;
        
        m_max_num_cuda_output_samples = falcon_dsp_polyphase_resampler<T, C>::needed_out_count(MAX_NUM_INPUT_SAMPLES_PER_CUDA_KERNEL);
        
        cudaMallocManaged(&m_cuda_input_samples, m_max_num_cuda_input_samples * sizeof(T));
        cudaMallocManaged(&m_cuda_output_samples, m_max_num_cuda_output_samples * sizeof(T));
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
    void _polyphase_resampler_cuda(cuFloatComplex * in, uint32_t in_len, cuFloatComplex * out, uint32_t out_len,
                                   cuFloatComplex * coeffs, uint32_t coeffs_len, uint32_t coeffs_per_phase,
                                   int64_t start_x_idx, uint32_t start_t,
                                   uint32_t up_rate, uint32_t down_rate)
    {
        /* compute the thread index */
        uint32_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
        
        /* compute local thread variables */
        int64_t thread_x_idx = start_x_idx;
        uint32_t thread_t = start_t;
        
        for (uint32_t ii = 0; ii < thread_index; ++ii)
        {
            /* compute the next 'cycle' updates */
            thread_t += down_rate;
            thread_x_idx += thread_t / up_rate;
            thread_t %= up_rate;
        }
        
        /* apply the polyphase filter */
        cuFloatComplex acc;
        cuFloatComplex * coeff_ptr = coeffs + thread_t * coeffs_per_phase;
        int64_t x_back_idx = thread_x_idx - coeffs_per_phase + 1;
        while (x_back_idx <= thread_x_idx)
        {
            acc = cuCaddf(acc, cuCmulf(in[x_back_idx++], *(coeff_ptr++)));
        }
        
        /* set the output variable */
        out[thread_index] = acc;
    }
    
    /* specialized implementation */
    template<>
    int32_t falcon_dsp_polyphase_resampler_cuda<std::complex<float>, std::complex<float>>::apply(std::vector<input_type>& in, std::vector<output_type>& out)
    {
        std::lock_guard<std::mutex> lock(falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_mutex);
        
        cuFloatComplex * cuda_filter_coeffs = static_cast<cuFloatComplex *>(m_cuda_filter_coeffs);
        cuFloatComplex * cuda_input_data = static_cast<cuFloatComplex *>(m_cuda_input_samples);
        cuFloatComplex * cuda_output_data = static_cast<cuFloatComplex *>(m_cuda_output_samples);
        
        out.clear();
        
        /* copy the filter coefficients into CUDA memory */
        for (uint32_t ii = 0;
             ii < falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_transposed_coeffs.size();
             ++ii)
        {
            cuda_filter_coeffs[ii] =
                *(static_cast<cuFloatComplex *>(
                    static_cast<void *>(&m_transposed_coeffs[ii])));   
        }
        
        /* don't bother with running a CUDA/GPU implementation if the input size is not
         *  larger than the state vector */
        if (in.size() <= falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_state.size())
        {
            return falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::apply(in, out);  
        }
        
        /* x_idx points to the latest processed input sample */
        int64_t x_idx = falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_xOffset;
        while (static_cast<uint64_t>(x_idx) < in.size())
        {
            bool required_state_array = false;
            
            /* setup variables for the cases where the next sample is handled in the CPU */
            output_type acc = output_type(0);
            typename std::vector<coeff_type>::iterator coeff_iter =
                falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_transposed_coeffs.begin() +
                falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_t * falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_coeffs_per_phase;
            
            /* need to look back over the previous samples to compute the
             *  current filtered value */
            int64_t x_back_idx = x_idx - falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_coeffs_per_phase + 1;
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
            uint32_t num_samples, num_threads, new_t;
            if (!required_state_array &&
                falcon_dsp_polyphase_resampler_cuda<std::complex<float>, std::complex<float>>::compute_next_kernel_params(
                        x_idx, in.size(), num_samples, num_threads, new_t))
            {
                /* copy input samples into CUDA memory. note that this includes both the previous
                 *  samples required for filtering NUM_INPUT_SAMPLES_PER_CUDA_KERNEL (one for
                 *  each kernel thread) */
                for (uint32_t ii = 0;
                     ii < num_samples + falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_coeffs_per_phase + 1;
                     ++ii)
                {
                    cuda_input_data[ii] = *(static_cast<cuFloatComplex *>(
                                              static_cast<void *>(&in[x_back_idx + ii])));
                }
                
                uint32_t num_thread_blocks = 1;
                _polyphase_resampler_cuda<<<num_thread_blocks, num_threads>>>(
                        cuda_input_data,
                        m_max_num_cuda_input_samples,
                        cuda_output_data,
                        m_max_num_cuda_output_samples,
                        cuda_filter_coeffs,
                        falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_transposed_coeffs.size(),
                        falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_coeffs_per_phase,
                        falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_coeffs_per_phase + 1,
                        falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_t,
                        falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_up_rate,
                        falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_down_rate);
                
                /* wait for GPU to finish before accessing on host */
                cudaDeviceSynchronize();
                
                /* copy output samples out of CUDA memory */
                for (uint32_t ii = 0; ii < falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::needed_out_count(num_samples); ++ii)
                {
                    out.push_back(*
                        static_cast<std::complex<float> *>(
                            static_cast<void *>(&cuda_output_data[ii])));
                }
                
                /* update tracking parameters */
                falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::m_t = new_t;
                x_idx += num_samples;
            }
            else
            {
                /* either handling data at the beginning of the vector or close to the end;
                 *  just handle here in the CPU to keep things simpler */
                while (x_back_idx <= x_idx)
                {
                    acc += in[x_back_idx++] * *(coeff_iter++);
                }

                out.push_back(acc);
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
    bool falcon_dsp_polyphase_resampler_cuda<T, C>::compute_next_kernel_params(int64_t cur_x_idx, size_t in_size,
                                                                               uint32_t& num_in_samples, uint32_t& num_threads,
                                                                               uint32_t& new_t)
    {
        uint32_t local_t = falcon_dsp_polyphase_resampler<T, C>::m_t;
        int64_t local_x_idx = cur_x_idx;
        
        num_in_samples = 0;
        num_threads = 0;
        new_t = local_t;
        
        bool reached_limit = false;
        while (!reached_limit)
        {
            /* compute the next 'cycle' updates */
            local_t += falcon_dsp_polyphase_resampler<T, C>::m_down_rate;
            int64_t advance_amount = local_t / falcon_dsp_polyphase_resampler<T, C>::m_up_rate;
            local_x_idx += advance_amount;
            local_t %= falcon_dsp_polyphase_resampler<T, C>::m_up_rate;
            
            /* check for sufficient input space */
            if ((local_x_idx - cur_x_idx) < MAX_NUM_INPUT_SAMPLES_PER_CUDA_KERNEL)
            {
                num_in_samples = local_x_idx - cur_x_idx;
                num_threads++;
                new_t = local_t;
            }
            else
            {
                reached_limit = true;
            }
             
            /* check for the thread limit */
            if ((num_threads + 1) > MAX_NUM_CUDA_THREADS)
            {
                reached_limit = true;   
            }
        }
        
        return (num_in_samples > 0 && num_threads > 0);
    }
    
    /* force instantiation for specific types */
    template class falcon_dsp_polyphase_resampler_cuda<std::complex<float>, std::complex<float>>;
}

/******************************************************************************
 *                            CLASS IMPLEMENTATION
 *****************************************************************************/
