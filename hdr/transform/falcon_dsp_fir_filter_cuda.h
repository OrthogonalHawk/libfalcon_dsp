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
 * @file     falcon_dsp_fir_filter_cuda.h
 * @author   OrthogonalHawk
 * @date     24-Jan-2020
 *
 * @brief    Signal processing transformation functions for FIR filtering;
 *            CUDA versions.
 *
 * @section  DESCRIPTION
 *
 * Defines a set of signal processing transformation FIR filter functions.
 *  Includes CUDA implementations.
 *
 * @section  HISTORY
 *
 * 24-Jan-2020  OrthogonalHawk  File created.h
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_TRANSFORM_FIR_FILTER_CUDA_H__
#define __FALCON_DSP_TRANSFORM_FIR_FILTER_CUDA_H__

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <complex>
#include <memory>
#include <mutex>
#include <vector>

#include <cuComplex.h>

#include "transform/falcon_dsp_fir_filter.h"

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
    
    /* @brief CUDA implementation of a linear FIR filter vector operation.
     * @param[in] coeffs                - FIR filter coefficients
     * @param[in] in                    - input vector
     * @param[out] out                  - filtered vector
     * @return True if the input vector was filtered as requested;
     *          false otherwise.
     */
    bool fir_filter_cuda(std::vector<std::complex<float>> &coeffs, std::vector<std::complex<int16_t>>& in,
                         std::vector<std::complex<int16_t>>& out);
    
    bool fir_filter_cuda(std::vector<std::complex<float>> &coeffs, std::vector<std::complex<float>>& in,
                         std::vector<std::complex<float>>& out);
    
    /* CUDA kernel function that applies an FIR filter */
    __global__
    void __fir_filter(cuFloatComplex * coeffs,
                      uint32_t coeff_len,
                      uint32_t num_output_samples_to_process_per_thread,
                      cuFloatComplex * in_data,
                      uint32_t in_data_len,
                      cuFloatComplex * out_data,
                      uint32_t out_data_len);
    
    
    /******************************************************************************
     *                            CLASS DECLARATION
     *****************************************************************************/
    
    /* @brief CUDA implementation of a linear FIR filter utility class.
     * @description By implementing the linear FIR filter utility as a class
     *               interface instead of a simple function the user is able
     *               to filter an arbitrarily long input.
     */
    class falcon_dsp_fir_filter_cuda : public falcon_dsp_fir_filter
    {
    public:
        
        falcon_dsp_fir_filter_cuda(std::vector<std::complex<float>> &coeffs);
        ~falcon_dsp_fir_filter_cuda(void);
        
        falcon_dsp_fir_filter_cuda(void) = delete;
        falcon_dsp_fir_filter_cuda(const falcon_dsp_fir_filter_cuda&) = delete;
        
        bool apply(std::vector<std::complex<int16_t>>& in, std::vector<std::complex<int16_t>>& out) override;
        bool apply(std::vector<std::complex<float>>& in, std::vector<std::complex<float>>& out) override;
    
    private:

        /* variables for input data memory management */
        cuFloatComplex *                                      m_cuda_coeff_data;
        uint32_t                                              m_input_padding_in_samples;
        
        cuFloatComplex *                                      m_cuda_input_data;
        uint32_t                                              m_max_num_input_samples;
        
        /* variables for output data memory management */
        cuFloatComplex *                                      m_cuda_output_data;
        uint32_t                                              m_max_num_output_samples;
    };
}

#endif // __FALCON_DSP_TRANSFORM_FIR_FILTER_CUDA_H__
