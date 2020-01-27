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
 * 24-Jan-2020  OrthogonalHawk  Fixing bugs with implementation and switched to
 *                               fully specified class from templated class.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <iostream>
#include <numeric>
#include <stdint.h>

#include "resample/falcon_dsp_polyphase_resampler_cuda.h"
#include "utilities/falcon_dsp_host_timer.h"
#include "utilities/falcon_dsp_utils.h"
#include "utilities/falcon_dsp_cuda_utils.h"

/******************************************************************************
 *                                 CONSTANTS
 *****************************************************************************/

const bool TIMING_ENABLED = false;

const uint32_t MAX_NUM_OUTPUTS_PER_CUDA_THREAD = 1;
const uint32_t MAX_NUM_CUDA_THREADS = 1024;

/* use CUDA shared memory to hold coefficient data if possible. for the Jetson
 *  Nano there appears to be 49152 bytes of shared memory. let's round this down
 *  to 48 Kbytes. each coefficient is two float values, so 8 bytes each. there
 *  should therefore be room for up to 6000 coefficients in shared memory.
 *
 * [1] https://devtalk.nvidia.com/default/topic/1056895/jetson-nano/about-jetson-nano-device-query/
 */
const uint32_t MAX_NUM_COEFFICIENTS_IN_SHARED_MEMORY = 6000;

/******************************************************************************
 *                              ENUMS & TYPEDEFS
 *****************************************************************************/

struct output_sample_s
{
    __device__ output_sample_s(void)
      : active(false),
        coeff_ptr(nullptr),
        data_start_idx(LONG_MAX),
        data_stop_idx(LONG_MAX)
    {
        acc.x = 0;
        acc.y = 0;
    }
    
    bool             active;
    cuFloatComplex   acc;
    cuFloatComplex * coeff_ptr;
    int64_t          data_start_idx;
    int64_t          data_stop_idx;
};
        
/******************************************************************************
 *                                  MACROS
 *****************************************************************************/

namespace falcon_dsp
{
    /******************************************************************************
     *                         FUNCTION IMPLEMENTATION
     *****************************************************************************/
    
    /* CUDA kernel function that resamples the input array */
    __global__
    void __polyphase_resampler_cuda(cuFloatComplex * coeffs, uint32_t coeffs_len,
                                    kernel_thread_params_s * thread_params, uint32_t params_len,
                                    cuFloatComplex * in, uint32_t in_len,
                                    cuFloatComplex * out, uint32_t out_len,
                                    uint32_t coeffs_per_phase,
                                    uint32_t num_outputs_per_cuda_thread,
                                    int64_t start_x_idx,
                                    uint32_t start_t,
                                    uint32_t start_output_idx,
                                    uint32_t up_rate,
                                    uint32_t down_rate)
    {
        __shared__ cuFloatComplex s_coeffs[MAX_NUM_COEFFICIENTS_IN_SHARED_MEMORY];
        output_sample_s out_samples[MAX_NUM_OUTPUTS_PER_CUDA_THREAD];
        
        /* compute the thread index */
        uint32_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
        
        /* sanity check inputs */
        if (in == nullptr ||
            out == nullptr ||
            coeffs == nullptr ||
            num_outputs_per_cuda_thread > MAX_NUM_OUTPUTS_PER_CUDA_THREAD ||
            thread_index > params_len)
        {
            return;
        }
        
        /* optionally load coefficients into shared memory */
        if (coeffs_len < MAX_NUM_COEFFICIENTS_IN_SHARED_MEMORY)
        {
            uint32_t num_coefficients_per_thread = (coeffs_len / blockDim.x) + 1;
            for (uint32_t coeff_idx = threadIdx.x * num_coefficients_per_thread;
                 coeff_idx < ((threadIdx.x * num_coefficients_per_thread) + num_coefficients_per_thread) &&
                     coeff_idx < coeffs_len;
                 ++coeff_idx)
            {
                s_coeffs[coeff_idx] = coeffs[coeff_idx];
            }
            
            __syncthreads();
        }
        
        /* retrieve local thread variables */
        kernel_thread_params_s params = thread_params[thread_index];
        int64_t thread_x_idx = params.thread_start_x_idx;
        uint32_t thread_t = params.thread_start_t;
        
        /* verify that this thread has at least one output to compute */
        int64_t thread_start_output_sample_idx = start_output_idx + (thread_index * num_outputs_per_cuda_thread);
        if (thread_start_output_sample_idx >= out_len)
        {
            return;
        }
        
        /* capture the FIRST output sample information; based on the previous
         *  checks the thread is responsible for at least ONE output sample */
        out_samples[0].active = true;
        if (coeffs_len < MAX_NUM_COEFFICIENTS_IN_SHARED_MEMORY)
        {
            out_samples[0].coeff_ptr = s_coeffs + thread_t * coeffs_per_phase;
        }
        else
        {
            out_samples[0].coeff_ptr = coeffs + thread_t * coeffs_per_phase;
        }
        out_samples[0].data_start_idx = thread_x_idx - coeffs_per_phase + 1;
        out_samples[0].data_stop_idx = thread_x_idx;

        /* start tracking the input samples that this thread will need to access */
        int64_t thread_data_start_idx = out_samples[0].data_start_idx;
        int64_t thread_data_stop_idx =  out_samples[0].data_stop_idx;
        
        /* ensure that the output sample is still within the range of the
         *  configured output data; there may be cases where some threads in
         *  a thread block are not needed. */
        for (uint32_t out_sample_idx = 1;
             out_sample_idx < num_outputs_per_cuda_thread &&
                 (out_sample_idx + thread_start_output_sample_idx) < out_len;
             ++out_sample_idx)
        {
            /* compute the next output sample 'cycle' updates */
            thread_t += down_rate;
            thread_x_idx += thread_t / up_rate;
            thread_t %= up_rate;
            
            /* store parameters for the next output sample */
            out_samples[out_sample_idx].active = true;
            if (coeffs_len < MAX_NUM_COEFFICIENTS_IN_SHARED_MEMORY)
            {
                out_samples[out_sample_idx].coeff_ptr = s_coeffs + thread_t * coeffs_per_phase;
            }
            else
            {
                out_samples[out_sample_idx].coeff_ptr = coeffs + thread_t * coeffs_per_phase;
            }
            out_samples[out_sample_idx].data_start_idx = thread_x_idx - coeffs_per_phase + 1;
            out_samples[out_sample_idx].data_stop_idx = thread_x_idx;
            
            /* update the thread input sample tracking */
            thread_data_stop_idx = out_samples[out_sample_idx].data_stop_idx;
        }
        
        uint32_t first_active_out_sample = 0;
        cuFloatComplex next_x_val;
        next_x_val.x = 0.0;
        next_x_val.y = 0.0;
        
        /* read samples from the input buffer and process for each one of the outputs
         *  that require an input from each input sample */
        for (int64_t x_idx = thread_data_start_idx;
             x_idx < (thread_data_stop_idx + 1) &&
                 x_idx < in_len;
             ++x_idx)
        {
            next_x_val = in[x_idx];
            
            for (uint32_t thread_out_sample_idx = first_active_out_sample;
                 thread_out_sample_idx < num_outputs_per_cuda_thread &&
                    out_samples[thread_out_sample_idx].active;
                 ++thread_out_sample_idx)
            {
                /* we don't need to check whether x_idx is less than data_stop_idx
                 *  here because it's checked later and once x_idx is >= data_stop_idx
                 *  this output is 'disabled' and will no longer be assessed */
                if (x_idx >= out_samples[thread_out_sample_idx].data_start_idx)
                {
                    out_samples[thread_out_sample_idx].acc =
                            cuCaddf(out_samples[thread_out_sample_idx].acc,
                                    cuCmulf(next_x_val, *(out_samples[thread_out_sample_idx].coeff_ptr++)));

                    if (out_samples[thread_out_sample_idx].data_stop_idx <= x_idx)
                    { 
                        /* finished accumulating for this output; it is ready to 
                         *  be transferred to the output array (performed later) */
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
             out_sample_idx < num_outputs_per_cuda_thread &&
                 out_samples[out_sample_idx].active;
             ++out_sample_idx)
        {
            out[global_output_sample_idx_base + out_sample_idx] = out_samples[out_sample_idx].acc;
        }
    }
    
    
    /******************************************************************************
     *                            CLASS IMPLEMENTATION
     *****************************************************************************/
    
    falcon_dsp_polyphase_resampler_cuda::falcon_dsp_polyphase_resampler_cuda(uint32_t up_rate, uint32_t down_rate,
                                                                             std::vector<std::complex<float>>& filter_coeffs)
      : falcon_dsp_polyphase_resampler(up_rate, down_rate, filter_coeffs),
        m_cuda_input_samples(nullptr),
        m_max_num_cuda_input_samples(0),
        m_kernel_thread_params(nullptr),
        m_num_kernel_thread_params(0),
        m_cuda_output_samples(nullptr),
        m_max_num_cuda_output_samples(0),
        m_cuda_filter_coeffs(nullptr),
        m_avg_advance_per_output_sample(0),
        m_num_outputs_per_cuda_thread(MAX_NUM_OUTPUTS_PER_CUDA_THREAD)
    {
        /* allocate CUDA unified memory space for filter coefficients */
        cudaErrChkAssert(cudaMallocManaged(&m_cuda_filter_coeffs,
                                           m_transposed_coeffs.size() * sizeof(std::complex<float>)));

        /* copy the filter coefficients into CUDA memory */
        cudaErrChkAssert(cudaMemcpy(m_cuda_filter_coeffs,
                                    m_transposed_coeffs.data(),
                                    m_transposed_coeffs.size() * sizeof(std::complex<float>),
                                    cudaMemcpyHostToDevice));

        /* calculate the average number of samples that are advanced for
         *  each output sample */
        m_avg_advance_per_output_sample = get_average_advance_in_samples();
        
       /* change the shared memory size to 8 bytes per shared memory bank. this is so that we
        *  can better handle complex<float> data, which is natively 8 bytes in size */
       cudaErrChkAssert(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
    }
    
    falcon_dsp_polyphase_resampler_cuda::~falcon_dsp_polyphase_resampler_cuda(void)
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_cuda_filter_coeffs)
        {
            cudaErrChk(cudaFree(m_cuda_filter_coeffs));
            m_cuda_filter_coeffs = nullptr;
        }

        if (m_kernel_thread_params)
        {
            cudaErrChk(cudaFree(m_kernel_thread_params));
            m_kernel_thread_params = nullptr;
            m_num_kernel_thread_params = 0;
        }

        if (m_cuda_input_samples)
        {
            cudaErrChk(cudaFree(m_cuda_input_samples));
            m_cuda_input_samples = nullptr;
        }

        if (m_cuda_output_samples)
        {
            cudaErrChk(cudaFree(m_cuda_output_samples));
            m_cuda_output_samples = nullptr;
        }
    }
    
    int32_t falcon_dsp_polyphase_resampler_cuda::apply(std::vector<std::complex<float>>& in,
                                                       std::vector<std::complex<float>>& out)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        /* calculate the number of thread blocks that will be required */
        uint32_t cur_out_idx = 0;
        int64_t x_idx = m_xOffset;
        uint32_t num_thread_blocks = needed_out_count(in.size()) / (MAX_NUM_CUDA_THREADS * m_num_outputs_per_cuda_thread);
        if (needed_out_count(in.size()) % (MAX_NUM_CUDA_THREADS * m_num_outputs_per_cuda_thread) != 0)
        {
            num_thread_blocks++;
        }
        
        /* pre-compute thread parameters */
        uint32_t num_outputs_from_thread_blocks = 0;
        uint32_t new_t = m_t;
        int64_t new_x_idx = x_idx;
        std::vector<kernel_thread_params_s> kernel_thread_params;
        compute_kernel_params(m_up_rate, m_down_rate, m_state.size(), in.size(), m_t,
                              needed_out_count(in.size()), m_num_outputs_per_cuda_thread,
                              num_outputs_from_thread_blocks, new_t, new_x_idx, kernel_thread_params);
        
        /* allocate CUDA unified memory space for input and output data. it is left as a future
         *  optimization to break the input data into chunks to limit the amount of memory */
        if (m_max_num_cuda_input_samples != (in.size() + m_state.size()))
        {
            if (m_cuda_input_samples)
            {
                cudaErrChkAssert(cudaFree(m_cuda_input_samples));
                m_cuda_input_samples = nullptr;
                m_max_num_cuda_input_samples = 0;
            }
            
            m_max_num_cuda_input_samples = in.size() + m_state.size();
            cudaErrChkAssert(cudaMallocManaged(&m_cuda_input_samples,
                                               m_max_num_cuda_input_samples * sizeof(std::complex<float>)));
        }
        
        if (m_max_num_cuda_output_samples != num_outputs_from_thread_blocks)
        {
            if (m_cuda_output_samples)
            {
                cudaErrChkAssert(cudaFree(m_cuda_output_samples));
                m_cuda_output_samples = nullptr;
                m_max_num_cuda_output_samples = 0;
            }
            
            m_max_num_cuda_output_samples = num_outputs_from_thread_blocks;
            cudaErrChkAssert(cudaMallocManaged(&m_cuda_output_samples,
                                               m_max_num_cuda_output_samples * sizeof(std::complex<float>)));
        }
        
        /* clear out the output and allocate space for the resulting data */
        out.clear();
        out.resize(m_max_num_cuda_output_samples, std::complex<float>(0.0, 0.0));

        printf("Num required thread blocks(%u threads):%u for %u output samples (%u allocated) (%u per thread) (%zu params)\n",
            MAX_NUM_CUDA_THREADS, num_thread_blocks,
            num_outputs_from_thread_blocks, m_max_num_cuda_output_samples,
            m_num_outputs_per_cuda_thread,
            kernel_thread_params.size());

        /* allocated CUDA memory for the thread parameters if necessary */
        if (m_num_kernel_thread_params != (MAX_NUM_CUDA_THREADS * num_thread_blocks))
        {
            if (m_kernel_thread_params)
            {
                cudaFree(m_kernel_thread_params);
                m_kernel_thread_params = nullptr;
                m_num_kernel_thread_params = 0;
            }
            
            m_num_kernel_thread_params = MAX_NUM_CUDA_THREADS * num_thread_blocks;
            cudaErrChkAssert(cudaMallocManaged(&m_kernel_thread_params,
                                               m_num_kernel_thread_params * sizeof(kernel_thread_params_s)));
        }
        
        /* copy the state vector into CUDA memory */
        cudaErrChkAssert(cudaMemcpy(m_cuda_input_samples,
                         m_state.data(),
                         m_state.size() * sizeof(std::complex<float>),
                         cudaMemcpyHostToDevice));
        
        /* copy the thread parameters into CUDA memory */
        cudaErrChkAssert(cudaMemcpy(m_kernel_thread_params,
                                    kernel_thread_params.data(),
                                    kernel_thread_params.size() * sizeof(kernel_thread_params_s),
                                    cudaMemcpyHostToDevice));
        
        /* copy the input samples into CUDA memory */
        cudaErrChkAssert(cudaMemcpy(m_cuda_input_samples + m_state.size(),
                                    in.data(),
                                    in.size() * sizeof(std::complex<float>),
                                    cudaMemcpyHostToDevice));
        
        falcon_dsp::falcon_dsp_host_timer timer("KERNEL", TIMING_ENABLED);
        
        __polyphase_resampler_cuda<<<num_thread_blocks, MAX_NUM_CUDA_THREADS>>>(
                         m_cuda_filter_coeffs,
                         m_transposed_coeffs.size(),
                         m_kernel_thread_params,
                         m_num_kernel_thread_params,
                         m_cuda_input_samples,
                         m_max_num_cuda_input_samples,
                         m_cuda_output_samples,
                         m_max_num_cuda_output_samples,
                         m_coeffs_per_phase,
                         m_num_outputs_per_cuda_thread,
                         m_state.size(), /* x_start_idx */
                         m_t,
                         cur_out_idx,
                         m_up_rate,
                         m_down_rate);

        cudaErrChkAssert(cudaPeekAtLastError());

        /* wait for GPU to finish before accessing on host */
        cudaDeviceSynchronize();
        
        timer.log_duration("Filtering");
        
        /* copy output samples out of CUDA memory */
        cudaMemcpy(out.data() + cur_out_idx,
                   m_cuda_output_samples + cur_out_idx,
                   num_outputs_from_thread_blocks * sizeof(std::complex<float>),
                   cudaMemcpyDeviceToHost);
        
        /* update tracking parameters */
        m_t = new_t;
        x_idx += new_x_idx;
        cur_out_idx += num_outputs_from_thread_blocks; 
        m_xOffset = x_idx - in.size();

        /* finished resampling; now update the state buffer*/
        _manage_state(in);
        
        /* number of samples computed */
        return out.size();
    }
    
    bool falcon_dsp_polyphase_resampler_cuda::compute_kernel_params(uint32_t up_rate, uint32_t down_rate,
                                                                    int64_t start_x_idx, size_t in_size, uint32_t start_t,
                                                                    uint32_t max_out_samples,
                                                                    uint32_t max_out_samples_per_thread,
                                                                    uint32_t& num_out_samples,
                                                                    uint32_t& new_t,
                                                                    int64_t& new_x_idx,
                                                                    std::vector<kernel_thread_params_s>& params)
    {
        uint32_t num_samples_in_current_thread = 0;
        
        /* initialize the output variables */
        num_out_samples = 0;
        new_t = start_t;
        new_x_idx = start_x_idx;
        params.clear();
        
        /* sanity check the inputs */
        if (max_out_samples_per_thread == 0)
        {
            return false;
        }
        
        /* always start by pushing back the initial parameters */
        params.push_back(kernel_thread_params_s(new_x_idx, new_t));
        
        /* compute kernel thread parameters */
        while (num_out_samples < max_out_samples && new_x_idx < in_size)
        {
            /* periodically save off the thread params */
            if (num_samples_in_current_thread >= max_out_samples_per_thread)
            {
                params.push_back(kernel_thread_params_s(new_x_idx, new_t));
                num_samples_in_current_thread = 0;
            }
            
            /* compute the next 'cycle' updates */
            new_t += down_rate;
            int64_t advance_amount = new_t / up_rate;
            new_x_idx += advance_amount;
            new_t %= up_rate;
            num_out_samples++;
            num_samples_in_current_thread++;
        }

        return true;
    }
    
    void falcon_dsp_polyphase_resampler_cuda::compute_next_filter_params(int64_t cur_x_idx, size_t in_size, uint32_t cur_t,
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
            local_t += m_down_rate;
            int64_t advance_amount = local_t / m_up_rate;
            new_x_idx += advance_amount;
            local_t %= m_up_rate;
            
            /* increment trackers; assuming one output sample per thread */
            num_out_samples++;
            new_t = local_t;
        }
    }
    
    uint32_t falcon_dsp_polyphase_resampler_cuda::get_average_advance_in_samples(void)
    {
        const uint32_t NUM_SAMPLES_TO_EVALUATE = 1e6;

        uint32_t local_t = m_t;
        int64_t  local_x_idx = 0;
        std::vector<uint32_t> local_advances;

        bool reached_limit = false;
        while (!reached_limit && local_x_idx < NUM_SAMPLES_TO_EVALUATE)
        {
            /* compute the next 'cycle' updates */
            local_t += m_down_rate;
            int64_t advance_amount = local_t / m_up_rate;
            local_t %= m_up_rate;
            local_x_idx += advance_amount;

            local_advances.push_back(advance_amount);
        }

        float accum_sum = static_cast<float>(std::accumulate(local_advances.begin(), local_advances.end(), 0));
        float advance_avg = accum_sum / static_cast<float>(local_advances.size());

        return static_cast<uint32_t>(std::ceil(advance_avg));
    }
}
