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
 * @file     falcon_dsp_fir_filter_cuda.cu
 * @author   OrthogonalHawk
 * @date     24-Jan-2020
 *
 * @brief    Implements a CUDA-based FIR filtering operation.
 *
 * @section  DESCRIPTION
 *
 * Implements the CUDA version of FIR filtering operations. Both a standalone
 *  function and a class-based tranform object are supported.
 *
 * @section  HISTORY
 *
 * 24-Jan-2020  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <iostream>
#include <memory>
#include <stdint.h>

#include "transform/falcon_dsp_fir_filter_cuda.h"
#include "utilities/falcon_dsp_cuda_utils.h"
#include "utilities/falcon_dsp_host_timer.h"

/******************************************************************************
 *                                 CONSTANTS
 *****************************************************************************/

const bool TIMING_LOGS_ENABLED = false;

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
    
    /* @brief CUDA implementation of a linear FIR filter vector operation.
     * @param[in] filter_coeffs         - FIR filter coefficients
     * @param[in] in                    - input vector
     * @param[out] out                  - filtered vector
     * @return True if the input vector was filtered as requested;
     *          false otherwise.
     */
    bool fir_filter_cuda(std::vector<std::complex<float>> &coeffs, std::vector<std::complex<int16_t>>& in,
                         std::vector<std::complex<int16_t>>& out)
    {
        falcon_dsp_fir_filter_cuda filter_obj(coeffs);
        return filter_obj.apply(in, out);
    }

    bool fir_filter_cuda(std::vector<std::complex<float>> &coeffs, std::vector<std::complex<float>>& in,
                         std::vector<std::complex<float>>& out)
    {
        falcon_dsp_fir_filter_cuda filter_obj(coeffs);
        return filter_obj.apply(in, out);
    }
    
    /* CUDA kernel function that applies an FIR filter. this kernel assumes that the caller
     *  pads the input with either zeroes or previous input (state) information. */
    __global__
    void __fir_filter(cuFloatComplex * coeffs,
                      uint32_t coeff_len,
                      uint32_t num_output_samples_to_process_per_thread,
                      cuFloatComplex * in_data,
                      uint32_t in_data_len,
                      cuFloatComplex * out_data,
                      uint32_t out_data_len)
    {
        /* retrieve the starting data index that corresponds to this thread. given the
         *  simplified FIR filter equation:
         * 
         *      y(n) = x(n)h(0) + x(n-1)h(1) + x(n-2)h(2) + ... x(n-m)h(m)
         *
         *        ^^^ y: output
         *            x: input
         *            h: FIR filter coefficients; length 'm'
         *
         * the start_data_index points to the x(n-m) data value, which is the 'oldest' sample
         *  that is required to compute y(n).
         */
        uint32_t start_data_index = (blockIdx.x * blockDim.x * num_output_samples_to_process_per_thread) +
                                        (threadIdx.x * num_output_samples_to_process_per_thread);
        
        /* the same calculation is equivalent to finding the output offset, or the 'n'
         *  used to index the output array */
        uint32_t output_data_offset = start_data_index;
        
        uint32_t num_padding_samples = coeff_len - 1;
        
        /* catch the case where the output buffer + padding is shorter than
         *  the input buffer */
        if ((out_data_len + num_padding_samples) < in_data_len)
        {
            return;
        }
        
        /* catch the case where the input size is not an integer
         *  multiple of the thread block size */
        if ((start_data_index + num_padding_samples) > in_data_len ||
            start_data_index > out_data_len)
        {
            return;
        }
        
        /* the previous checks captured cases where the current thread should do nothing.
         *  now catch the case where this kernel cannot process the full number of samples
         *  when the end of the input buffer is reached. note that due to the previous
         *  check to make sure that the output buffer is at least as long as the input
         *  buffer it is sufficient to only check the input buffer here */
        uint32_t local_num_samples_to_process = num_output_samples_to_process_per_thread;
        if ((start_data_index + num_padding_samples + num_output_samples_to_process_per_thread) > in_data_len)
        {
            local_num_samples_to_process = in_data_len - start_data_index - num_padding_samples;
        }
        
        cuFloatComplex *data_ptr = nullptr;
        cuFloatComplex *coeff_ptr = nullptr;
        cuFloatComplex accum;
        
        /* compute the output values */
        for (uint32_t out_sample_idx = 0;
             out_sample_idx < local_num_samples_to_process &&
                 (start_data_index + num_padding_samples + out_sample_idx) < in_data_len &&
                 (output_data_offset + out_sample_idx) < out_data_len;
             ++out_sample_idx)
        {
            /* reset for each new output. data_ptr is pointed to the oldest data that
             *  is needed for the y(n) output. coeff_ptr is therefore pointed to the
             *  end of the coefficient array */
            data_ptr = &in_data[start_data_index + out_sample_idx];
            coeff_ptr = &coeffs[coeff_len - 1]; /* last coefficient in the array */
            
            accum.x = 0;
            accum.y = 0;
            
            /* go through all coefficients */
            for (uint32_t ii = 0; ii < coeff_len; ++ii)
            {               
                accum = cuCaddf(accum, cuCmulf(*data_ptr++, *coeff_ptr--));
            }
            
            /* output computed; save to the output buffer */
            out_data[output_data_offset + out_sample_idx] = accum;
        }
    }

    
    /******************************************************************************
     *                           CLASS IMPLEMENTATION
     *****************************************************************************/
    
    falcon_dsp_fir_filter_cuda::falcon_dsp_fir_filter_cuda(std::vector<std::complex<float>> &coeffs)
      : falcon_dsp_fir_filter(coeffs),
        m_cuda_coeff_data(nullptr),
        m_cuda_input_data(nullptr),
        m_max_num_input_samples(0),
        m_cuda_output_data(nullptr),
        m_max_num_output_samples(0)
    {        
        /* allocate CUDA memory for the coefficient information; since these are set
         *  when the class is constructed and cannot be changed the amount of data
         *  is known now */
        cudaErrChkAssert(cudaMallocManaged(&m_cuda_coeff_data,
                                           m_coefficients.size() * sizeof(cuFloatComplex)));
            
        /* copy the coefficient information to the GPU */
        cudaErrChkAssert(cudaMemcpy(static_cast<void *>(m_cuda_coeff_data),
                                    static_cast<void *>(m_coefficients.data()),
                                    m_coefficients.size() * sizeof(std::complex<float>),
                                    cudaMemcpyHostToDevice));
        
        m_input_padding_in_samples = m_coefficients.size() - 1;
    }
    
    falcon_dsp_fir_filter_cuda::~falcon_dsp_fir_filter_cuda(void)
    {
        std::lock_guard<std::mutex> lock(std::mutex);
        
        /* cleanup the CUDA memory that was reserved in the constructor to house
         *  the coefficient information */
        if (m_cuda_coeff_data)
        {
            cudaErrChk(cudaFree(m_cuda_coeff_data));
        }
    }

    bool falcon_dsp_fir_filter_cuda::apply(std::vector<std::complex<int16_t>>& in, std::vector<std::complex<int16_t>>& out)
    {       
        /* create another copy of the data and cast to std::complex<float> */
        std::vector<std::complex<float>> tmp_in_vec;
        tmp_in_vec.reserve(in.size());
        for (auto in_iter = in.begin(); in_iter != in.end(); ++in_iter)
        {
            tmp_in_vec.push_back(std::complex<float>((*in_iter).real(), (*in_iter).imag()));
        }
        
        /* filter the input data */
        std::vector<std::complex<float>> tmp_out_vec;
        bool ret = apply(tmp_in_vec, tmp_out_vec);

        /* cast the filtered output back to std::complex<int16_t> */
        for (auto out_iter = tmp_out_vec.begin(); out_iter != tmp_out_vec.end(); ++out_iter)
        {
            out.push_back(std::complex<int16_t>((*out_iter).real(), (*out_iter).imag()));
        }
        
        return ret;
    }
    
    bool falcon_dsp_fir_filter_cuda::apply(std::vector<std::complex<float>>& in, std::vector<std::complex<float>>& out)
    {
        std::lock_guard<std::mutex> lock(std::mutex);
        
        /* sanity check object state */
        if (m_coefficients.size() == 0)
        {
            return false;
        }
        
        /* clear the output data structures and resize so that they can hold the filtered
         *  data. note that by using resize() the vector size is now equal to the requested
         *  size even without explicitly adding data to the vector, which means that we can
         *  add data directly into the vector data buffer without worrying about the
         *  vector size getting mismatched with the buffer contents.
         *
         * note that the output size will always be equal to the input size since the input
         *  data is either padded with zeroes or previous state information. */
        out.clear();
        out.resize(in.size());
        
        /* if there is enough space in previously allocated memory then use it; otherwise
         *  allocate new memory buffers. it is left as a future feature to specify a maximum
         *  memory size and process the data in chunks instead of requiring enough GPU
         *  memory to process the whole vector at once */
        
        /* allocate CUDA memory for the input samples */
        if (m_max_num_input_samples < (in.size() + m_input_padding_in_samples))
        {
            if (m_cuda_input_data)
            {
                cudaErrChkAssert(cudaFree(m_cuda_input_data));
                m_cuda_input_data = nullptr;
                m_max_num_input_samples = 0;
            }
            
            cudaErrChkAssert(cudaMallocManaged(&m_cuda_input_data,
                                               (in.size() + m_input_padding_in_samples) * sizeof(std::complex<float>)));
            m_max_num_input_samples = in.size() + m_input_padding_in_samples;
        }
        
        /* allocate CUDA memory for the output samples */
        if (m_max_num_output_samples < in.size())
        {
            /* clean up existing memory */
            if (m_cuda_output_data)
            {
                cudaErrChkAssert(cudaFree(m_cuda_output_data));
                m_cuda_output_data = nullptr;
                m_max_num_output_samples = 0;
            }
                
            /* allocate CUDA unified memory space for the output data */
            cudaErrChkAssert(cudaMallocManaged(&m_cuda_output_data,
                                               in.size() * sizeof(std::complex<float>)));
            m_max_num_output_samples = in.size();
        }
        
        /* prepare the padding/state information. state information is stored such that the last
         *  element in the m_state container should immediately precede the input data. */
        std::vector<std::complex<float>> prev_data(m_input_padding_in_samples, std::complex<float>(0.0, 0.0));
        auto prev_data_iter = prev_data.rbegin();
        auto state_iter = m_state.rbegin();
        for (uint32_t ii = 0;
             ii < m_input_padding_in_samples &&
                 prev_data_iter != prev_data.rend() &&
                 state_iter != m_state.rend();
             ++ii)
        {
            *(prev_data_iter++) = *(state_iter++);
        }
        
        /* copy the padding/state information to the GPU */
        cudaErrChkAssert(cudaMemcpy(static_cast<void *>(m_cuda_input_data),
                                    static_cast<void *>(prev_data.data()),
                                    prev_data.size() * sizeof(std::complex<float>),
                                    cudaMemcpyHostToDevice));
        
        /* copy the input data to the GPU; note the offset by prev_data.size() to
         *  provide room for the padding/state samples */
        cudaErrChkAssert(cudaMemcpy(static_cast<void *>(m_cuda_input_data + prev_data.size()),
                                    static_cast<void *>(in.data()),
                                    in.size() * sizeof(std::complex<float>),
                                    cudaMemcpyHostToDevice));
        
        /* run kernel on the GPU */
        uint32_t num_samples_per_thread = 1;
        uint32_t samples_per_thread_block = num_samples_per_thread * MAX_NUM_CUDA_THREADS;
        uint32_t num_thread_blocks = (in.size() + samples_per_thread_block - 1) / samples_per_thread_block;
        
        falcon_dsp::falcon_dsp_host_timer timer("KERNEL", TIMING_LOGS_ENABLED);
        
        __fir_filter<<<num_thread_blocks, MAX_NUM_CUDA_THREADS>>>(m_cuda_coeff_data,
                                                                  m_coefficients.size(),
                                                                  num_samples_per_thread,
                                                                  m_cuda_input_data,
                                                                  m_max_num_input_samples,
                                                                  m_cuda_output_data,
                                                                  m_max_num_output_samples);
        
        cudaErrChkAssert(cudaPeekAtLastError());
        
        /* wait for GPU to finish before accessing on host */
        cudaErrChkAssert(cudaDeviceSynchronize());
            
        timer.log_duration("Single Chan Kernel Complete");
        
        /* copy output samples out of CUDA memory */
        cudaErrChkAssert(cudaMemcpy(static_cast<void *>(out.data()),
                                    static_cast<void *>(m_cuda_output_data),
                                    in.size() * sizeof(std::complex<float>),
                                    cudaMemcpyDeviceToHost));

        /* finished handling the current data; now update the state array */
        _update_state(in);
        
        return out.size() > 0;
    }
}
