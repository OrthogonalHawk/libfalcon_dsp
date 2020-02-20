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
 * @file     falcon_dsp_polar_discriminator_cuda.cu
 * @author   OrthogonalHawk
 * @date     19-Feb-2020
 *
 * @brief    Implements CUDA-based polar discriminator operations, such as one
 *            needs for FM demodulation.
 *
 * @section  DESCRIPTION
 *
 * Implements the CUDA version of polar discriminator operations. Both a standalone
 *  function and a class-based tranform object are supported.
 *
 * @section  HISTORY
 *
 * 19-Feb-2020  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <cmath>
#include <iostream>
#include <stdint.h>

#include "transform/falcon_dsp_polar_discriminator_cuda.h"
#include "utilities/falcon_dsp_cuda_utils.h"
#include "utilities/falcon_dsp_host_timer.h"

/******************************************************************************
 *                                 CONSTANTS
 *****************************************************************************/

const bool TIMING_LOGS_ENABLED = true;

const uint32_t MAX_NUM_CUDA_THREADS = 1024;

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
    /******************************************************************************
     *                        FUNCTION IMPLEMENTATION
     *****************************************************************************/
    
    /* @brief CUDA implementation of a polar discriminator vector operation.
     * @param[in] in                    - input vector
     * @param[out] out                  - filtered vector
     * @return True if the output vector was populated with the discriminator output
     *          as requested; false otherwise.
     */
    bool polar_discriminator_cuda(std::vector<std::complex<int16_t>>& in,
                                  std::vector<float>& out)
    {
        falcon_dsp_polar_discriminator_cuda discriminator;
        return discriminator.apply(in, out);
    }

    bool polar_discriminator_cuda(std::vector<std::complex<float>>& in,
                                  std::vector<float>& out)
    {
        falcon_dsp_polar_discriminator_cuda discriminator;
        return discriminator.apply(in, out);
    }
    
    /* CUDA kernel function that performs a polar discriminator operation */
    __global__
    void __polar_discriminator(cuFloatComplex * in_data,
                               uint32_t in_data_len,
                               float * out_data,
                               uint32_t out_data_len)
    {
        __shared__ cuFloatComplex s_input_data[MAX_NUM_CUDA_THREADS + 1];
        
        /* each N output sample requires the N and N-1 input samples. this
         *  start_data_index points at the N-1 input sample. the 0th thread
         *  in a thread block computes output[0] */
        uint32_t start_data_index = blockIdx.x * blockDim.x + threadIdx.x;
        
        /* sanity check inputs */
        if (in_data == nullptr ||
            out_data == nullptr)
        {
            return;
        }
        
        /* copy input data to shared memory; all threads in a thread block
         *  should copy here */
        if (start_data_index < in_data_len)
        {
            s_input_data[threadIdx.x] = in_data[start_data_index];
        }
        
        /* need one more input sample; assuming that the current thread block
         *  is not the last one */
        if (threadIdx.x == 0 && ((start_data_index + blockDim.x + 1) < in_data_len))
        {
            s_input_data[blockDim.x + 1] = in_data[start_data_index + blockDim.x];
        }
        
        /* sync the threads to ensure that all inputs are available */
        __syncthreads();
        
        /* catch the case where the input size is not an integer
         *  multiple of the thread block size */
        if (start_data_index > in_data_len ||
            start_data_index >= out_data_len)
        {
            return;
        }

        cuFloatComplex conjPrevVal = s_input_data[threadIdx.x];
        conjPrevVal.y = -1 * conjPrevVal.y;
        cuFloatComplex compareVal = cuCmulf(s_input_data[threadIdx.x + 1], conjPrevVal);
        
        out_data[start_data_index] = static_cast<float>(atan2(compareVal.y, compareVal.x));
    }    

    /******************************************************************************
     *                           CLASS IMPLEMENTATION
     *****************************************************************************/
    
    falcon_dsp_polar_discriminator_cuda::falcon_dsp_polar_discriminator_cuda(void)
      : falcon_dsp_polar_discriminator(),
        m_input_data(nullptr),
        m_input_data_len(0),
        m_output_data(nullptr),
        m_output_data_len(0)
    { }
    
    falcon_dsp_polar_discriminator_cuda::~falcon_dsp_polar_discriminator_cuda(void)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        if (m_input_data)
        {
            cudaErrChk(cudaFree(m_input_data));
            m_input_data = nullptr;
            m_input_data_len = 0;
        }
        
        if (m_output_data)
        {
            cudaErrChk(cudaFree(m_input_data));
            m_output_data = nullptr;
            m_output_data_len = 0;
        }
    }
    
    void falcon_dsp_polar_discriminator_cuda::reset_state(void)
    {
        /* currently no difference between the base and derived class reset methods */
        falcon_dsp_polar_discriminator::reset_state();
    }

    bool falcon_dsp_polar_discriminator_cuda::apply(std::vector<std::complex<int16_t>>& in, std::vector<float>& out)
    {
        out.clear();
        out.reserve(in.size());
        
        /* create another copy of the data and cast to std::complex<float> */
        std::vector<std::complex<float>> tmp_in_vec;
        tmp_in_vec.reserve(in.size());
        for (auto in_iter = in.begin(); in_iter != in.end(); ++in_iter)
        {
            tmp_in_vec.push_back(std::complex<float>((*in_iter).real(), (*in_iter).imag()));
        }
        
        /* run the discriminator on the input data */
        return apply(tmp_in_vec, out);
    }
    
    bool falcon_dsp_polar_discriminator_cuda::apply(std::vector<std::complex<float>>& in, std::vector<float>& out)
    {
        std::lock_guard<std::mutex> lock(std::mutex);
        
        out.clear();
        
        /* sanity check the inputs */
        if (in.size() == 0)
        {
            return false;
        }
        else if (in.size() == 1 && m_state.size() == 0)
        {
            /* no work to do; store the state value and continue */
            m_state.push_back(*in.rbegin());
            
            return true;
        }
        
        /* verify that sufficient CUDA memory is available for the
         *  discriminator input and output */
        if ((in.size() + m_state.size()) != m_input_data_len)
        {
            if (m_input_data)
            {
                cudaErrChk(cudaFree(m_input_data));
                m_input_data = nullptr;
                m_input_data_len = 0;
            }
            
            m_input_data_len = in.size() + m_state.size();
            cudaErrChkAssert(cudaMallocManaged(&m_input_data,
                                               m_input_data_len * sizeof(cuFloatComplex)));
        }
        
        if (((in.size() + m_state.size()) - 1) != m_output_data_len)
        {
            if (m_output_data)
            {
                cudaErrChk(cudaFree(m_output_data));
                m_output_data = nullptr;
                m_output_data_len = 0;
            }

            m_output_data_len = (in.size() + m_state.size()) - 1;
            cudaErrChkAssert(cudaMallocManaged(&m_output_data,
                                               m_output_data_len * sizeof(float)));
        }
        
        if (m_state.size() > 0)
        {
            /* copy the state data to the GPU */
            cudaErrChkAssert(cudaMemcpy(static_cast<void *>(m_input_data),
                                        static_cast<void *>(m_state.data()),
                                        m_state.size() * sizeof(std::complex<float>),
                                        cudaMemcpyHostToDevice));
        }
                                    
        /* copy the input data to the GPU */
        cuFloatComplex * input_ptr = m_input_data + m_state.size();
        cudaErrChkAssert(cudaMemcpy(static_cast<void *>(m_input_data),
                                    static_cast<void *>(in.data()),
                                    in.size() * sizeof(std::complex<float>),
                                    cudaMemcpyHostToDevice));
        
        uint32_t num_thread_blocks = m_output_data_len / MAX_NUM_CUDA_THREADS;
        if (m_output_data_len % MAX_NUM_CUDA_THREADS != 0 || num_thread_blocks == 0)
        {
            num_thread_blocks++;
        }
        
        falcon_dsp::falcon_dsp_host_timer timer("KERNEL", TIMING_LOGS_ENABLED);
        
        __polar_discriminator<<<num_thread_blocks, MAX_NUM_CUDA_THREADS>>>(m_input_data,
                                                                           m_input_data_len,
                                                                           m_output_data,
                                                                           m_output_data_len);

        cudaErrChkAssert(cudaPeekAtLastError());
        
        /* wait for GPU to finish before accessing on host */
        cudaErrChkAssert(cudaDeviceSynchronize());
            
        timer.log_duration("Polar Discriminator Complete");
        
        /* copy output samples out of CUDA memory */
        out.resize(m_output_data_len);
        cudaErrChkAssert(cudaMemcpy(static_cast<void *>(out.data()),
                                    static_cast<void *>(m_output_data),
                                    m_output_data_len * sizeof(float),
                                    cudaMemcpyDeviceToHost));

        /* update state information; store the very last value in the input vector */
        m_state.clear();
        m_state.push_back(*in.rbegin());
        
        return out.size() > 0;
    }
}
