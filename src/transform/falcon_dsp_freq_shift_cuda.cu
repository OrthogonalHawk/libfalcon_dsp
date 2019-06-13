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
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <iostream>
#include <stdint.h>

#include <cuComplex.h>

#include "transform/falcon_dsp_transform.h"

/******************************************************************************
 *                                 CONSTANTS
 *****************************************************************************/

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
    /* CUDA kernel function that applies a frequency shift */
    __global__
    void _freq_shift(uint32_t num_samples_handled_previously,
                     uint32_t rollover_sample_idx,
                     double   angular_freq,
                     cuFloatComplex * data,
                     uint32_t data_size)
    {
        uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
     
        /* catch the case where the input size is not an integer
         *  multiple of the thread block size */
        if (index > data_size)
        {
            return;
        }
        
        /* compute the sample index for the current thread */
        uint64_t orig_sample_idx = num_samples_handled_previously + index;
        uint64_t sample_idx = orig_sample_idx;
        
        for (uint32_t ii = 0; ii < 1; ++ii)
        {
            sample_idx += ii;
            sample_idx %= rollover_sample_idx;
            if (sample_idx < orig_sample_idx)
            {
                printf("Detected a rollover for input sample %lu (%u %u)\n",
                    orig_sample_idx, num_samples_handled_previously, index);
            }

            if (sample_idx >= 417214 && sample_idx <= 417220)
            {
                printf("cuda input[%u]: (%f,%f)\n", sample_idx, data[sample_idx].x, data[sample_idx].y);
            }
            
            /* compute the frequency shift multiplier value */
            float angle = angular_freq * sample_idx;
            float real = cosf(angle);
            float imag = sinf(angle);
            
            /* create a CUDA complex variable to apply the freqency shift */
            cuFloatComplex shift;
            shift.x = real;
            shift.y = imag;

            if (sample_idx >= 417214 && sample_idx <= 417220)
            {
                printf("shift[%u]: angle=%f (%f,%f)\n", sample_idx, angular_freq * sample_idx, real, imag);
            }
            /* apply the frequency shift in-place */
            data[sample_idx] = cuCmulf(data[sample_idx], shift);
        }
    }
    
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
    
    falcon_dsp_freq_shift_cuda::falcon_dsp_freq_shift_cuda(uint32_t input_sample_rate_in_sps, int32_t freq_shift_in_hz)
      : falcon_dsp_freq_shift(input_sample_rate_in_sps, freq_shift_in_hz),
        m_cuda_data_vector(nullptr),
        m_max_num_cuda_input_samples(0)
    { }
    
    falcon_dsp_freq_shift_cuda::~falcon_dsp_freq_shift_cuda(void)
    {
        std::lock_guard<std::mutex> lock(std::mutex);
        
        /* clean up existing memory */
        if (m_cuda_data_vector != nullptr)
        {
            cudaFree(m_cuda_data_vector);
            m_cuda_data_vector = nullptr;
        }
    }

    bool falcon_dsp_freq_shift_cuda::apply(std::vector<std::complex<int16_t>>& in, std::vector<std::complex<int16_t>>& out)
    {
        std::lock_guard<std::mutex> lock(std::mutex);
        
        /* clear the output data structure */
        out.clear();
        
        /* if there is enough space in previously allocated memory then use it; otherwise
         *  allocate a new memory buffer */
        if (m_max_num_cuda_input_samples < in.size())
        {
            /* clean up existing memory */
            if (m_cuda_data_vector != nullptr)
            {
                cudaFree(m_cuda_data_vector);
                m_cuda_data_vector = nullptr;
            }
            
            /* allocate CUDA unified memory space for the data to be transformed. note that space is
             *  reserved for std::complex<float> because this is what is supported in CUDA. it will
             *  be converted back to std::complex<int16_t> before it is returned to the user */
            cudaMallocManaged(&m_cuda_data_vector, in.size() * sizeof(std::complex<float>));
            m_max_num_cuda_input_samples = in.size();
        }

        cuFloatComplex * cuda_data = static_cast<cuFloatComplex *>(m_cuda_data_vector);
        
        for (uint32_t ii = 0; ii < in.size(); ++ii)
        {
            /* copy input samples into CUDA memory */
            std::complex<float> val(in[ii].real(), in[ii].imag());
            cuda_data[ii] = *(static_cast<cuFloatComplex *>(static_cast<void *>(&val)));
        }
        
        /* run kernel on the GPU */
        uint32_t thread_block_size = 256;
        uint32_t num_thread_blocks = (in.size() + thread_block_size - 1) / thread_block_size;
        _freq_shift<<<num_thread_blocks, thread_block_size>>>(m_samples_handled,
                                                              m_calculated_rollover_sample_idx,
                                                              m_angular_freq,
                                                              cuda_data,
                                                              in.size());
        
        /* wait for GPU to finish before accessing on host */
        cudaDeviceSynchronize();
        
        /* copy output samples out of CUDA memory */
        for (uint32_t ii = 0; ii < in.size(); ++ii)
        {
            void * void_ptr = static_cast<void *>(&cuda_data[ii]);
            if (void_ptr != nullptr)
            {   
                std::complex<float> * complex_float_ptr = static_cast<std::complex<float> *>(void_ptr);
                if (complex_float_ptr != nullptr)
                {
                    out.push_back(*complex_float_ptr);
                }
                else
                {
                    std::cout << "ERROR: Found nullptr complex_float_ptr for ii=" << ii << std::endl;
                }
            }
            else
            {
                std::cout << "ERROR: found nullptr reference to cuda_data[" << ii << "]" << std::endl;
            }
        }
        
        m_samples_handled += in.size();
        m_samples_handled = static_cast<uint32_t>(m_samples_handled) % m_calculated_rollover_sample_idx;
            
        return out.size() > 0;
    }
}

/******************************************************************************
 *                            CLASS IMPLEMENTATION
 *****************************************************************************/
