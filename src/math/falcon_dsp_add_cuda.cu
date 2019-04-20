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
 * @file     falcon_dsp_add_cuda.cu
 * @author   OrthogonalHawk
 * @date     19-Apr-2019
 *
 * @brief    CUDA implementation of basic addition operations.
 *
 * @section  DESCRIPTION
 *
 * Implements CUDA versions of basic addition operations.
 *
 * @section  HISTORY
 *
 * 19-Apr-2019  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <stdint.h>

#include "falcon_dsp_math.h"

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
 *                           FUNCTION IMPLEMENTATION
 *****************************************************************************/

namespace falcon_dsp
{
    /* CUDA kernel function that adds the elements of two arrays */
    template<typename T>
    __global__
    void _add(uint32_t length, T * in_a, T * in_b, T * out)
    {
        for (uint32_t ii = 0; ii < length; ii++)
        {
            out[ii] = in_a[ii] + in_b[ii];
        }
    }
    
    /*
     * @brief Adds two vectors together and places the sums into an output vector
     * @param[in]  in_a  first vector to add
     * @param[in]  in_b  second vector to add
     * @param[out] out   output vector
     * @return None
     */
    template<typename T>
    void add_vector_cuda(std::vector<T>& in_a, std::vector<T>& in_b, std::vector<T>& out)
    {
        out.clear();
        
        if (in_a.size() == in_b.size())
        {
            out.reserve(in_a.size());
            
            /* create pointers for CUDA/device memory */
            T * in_a_cuda_mem = nullptr;
            T * in_b_cuda_mem = nullptr;
            T * out_cuda_mem = nullptr;
            
            /* allocate unified memory, which is accessible from CPU or GPU */
            cudaMallocManaged(&in_a_cuda_mem, in_a.size() * sizeof(T));
            cudaMallocManaged(&in_b_cuda_mem, in_a.size() * sizeof(T));
            cudaMallocManaged(&out_cuda_mem,  in_a.size() * sizeof(T));
            
            /* initialize CUDA memory by copying in the input vectors */
            for (uint32_t ii = 0; ii < in_a.size(); ++ii)
            {
                in_a_cuda_mem[ii] = in_a[ii];
                in_b_cuda_mem[ii] = in_b[ii];   
            }
            
            /* run kernel on the GPU */
            _add<<<1, 1>>>(in_a.size(), in_a_cuda_mem, in_b_cuda_mem, out_cuda_mem);

            /* wait for GPU to finish before accessing on host */
            cudaDeviceSynchronize();
            
            /* populate the caller's output vector */
            for (uint32_t ii = 0; ii < in_a.size(); ++ii)
            {
                out.push_back(out_cuda_mem[ii]);   
            }
            
            /* cleanup the CUDA memory that was allocated previously */
            cudaFree(in_a_cuda_mem);
            cudaFree(in_b_cuda_mem);
            cudaFree(out_cuda_mem);
        }
    }
    
    /* force instantiation for specific types */
    template void add_vector_cuda<uint32_t>(std::vector<uint32_t>& in_a, std::vector<uint32_t>& in_b, std::vector<uint32_t>& out);
}

/******************************************************************************
 *                            CLASS IMPLEMENTATION
 *****************************************************************************/
