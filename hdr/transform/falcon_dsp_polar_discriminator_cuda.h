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
 * @file     falcon_dsp_polar_discriminator_cuda.h
 * @author   OrthogonalHawk
 * @date     21-Jan-2020
 *
 * @brief    Signal processing transformation functions for polar discrimination;
 *            CUDA versions.
 *
 * @section  DESCRIPTION
 *
 * Defines a set of signal processing transformation polar discriminator functions.
 *  Includes CUDA implementations.
 *
 * @section  HISTORY
 *
 * 19-Feb-2020  OrthogonalHawk  Created file.
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_TRANSFORM_POLAR_DISCRIMINATOR_CUDA_H__
#define __FALCON_DSP_TRANSFORM_POLAR_DISCRIMINATOR_CUDA_H__

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <complex>
#include <mutex>
#include <vector>

#include <cuComplex.h>

#include "transform/falcon_dsp_polar_discriminator.h"

/******************************************************************************
 *                                 CONSTANTS
 *****************************************************************************/

/******************************************************************************
 *                              ENUMS & TYPEDEFS
 *****************************************************************************/

/******************************************************************************
 *                                  MACROS
 *****************************************************************************/

namespace falcon_dsp
{
    /******************************************************************************
     *                           FUNCTION DECLARATION
     *****************************************************************************/
 
    /* @brief CUDA implementation of a polar discriminator vector operation.
     * @param[in] in                    - input vector
     * @param[out] out                  - filtered vector
     * @return True if the output vector was populated with the discriminator output
     *          as requested; false otherwise.
     */
    bool polar_discriminator_cuda(std::vector<std::complex<int16_t>>& in,
                                  std::vector<float>& out);
    
    bool polar_discriminator_cuda(std::vector<std::complex<float>>& in,
                                  std::vector<float>& out);
    
    /* CUDA kernel function that performs a polar discriminator operation */
    __global__
    void __polar_discriminator(cuFloatComplex * in_data,
                               uint32_t in_data_len,
                               float * out_data,
                               uint32_t out_data_len);

    /******************************************************************************
     *                            CLASS DECLARATION
     *****************************************************************************/
    
    /* @brief CUDA implementation of a polar discriminator utility class.
     * @description By implementing the polar discriminator utility as a class
     *               interface instead of a simple function the user is able
     *               to process an arbitrarily long input.
     */
    class falcon_dsp_polar_discriminator_cuda : public falcon_dsp_polar_discriminator
    {
    public:

        falcon_dsp_polar_discriminator_cuda(void);
        ~falcon_dsp_polar_discriminator_cuda(void);

        falcon_dsp_polar_discriminator_cuda(const falcon_dsp_polar_discriminator_cuda&) = delete;

        virtual void reset_state(void);
        virtual bool apply(std::vector<std::complex<int16_t>>& in, std::vector<float>& out);
        virtual bool apply(std::vector<std::complex<float>>& in, std::vector<float>& out);
    
    protected:
    
        cuFloatComplex *                 m_input_data;
        uint32_t                         m_input_data_len;
        
        float *                          m_output_data;
        uint32_t                         m_output_data_len;
    };
}

#endif // __FALCON_DSP_TRANSFORM_POLAR_DISCRIMINATOR_CUDA_H__
