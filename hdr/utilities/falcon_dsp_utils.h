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
 * @file     falcon_dsp_utils.h
 * @author   OrthogonalHawk
 * @date     10-May-2019
 *
 * @brief    Digital Signal Processing utility functions; C++ implementation.
 *
 * @section  DESCRIPTION
 *
 * Defines various utility functions supported by the FALCON DSP library.
 *
 * @section  HISTORY
 *
 * 10-May-2019  OrthogonalHawk  File created.
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_UTILS_H__
#define __FALCON_DSP_UTILS_H__

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <vector>

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
 *                           FUNCTION DECLARATION
 *****************************************************************************/

namespace falcon_dsp
{
    /* @brief FIR Low-Pass Filter Coefficient generation
     * @description Source code from http://digitalsoundandmusic.com/download/programmingexercises/Creating_FIR_Filters_in_C++.pdf
     * @param[in]  M      - filter length in number of taps
     * @param[in]  fc     - cutoff frequency in Hz (un-normalized). Note that a normalized
     *                       cutoff frequency can be used, but then fsamp should be set
     *                       to 1 so that is not used to normalize fc again.
     * @param[in]  fsamp  - sampling frequency of the signal to be filtered in Hz. fsamp
     *                       is used to normalize the cutoff frequency.
     * @param[out] coeffs - FIR filter coefficients
     * @return true if the coefficients were generated successfully; false otherwise.
     */
    bool firlpf(uint32_t M, double fc, double fsamp, std::vector<double>& coeffs);
}

/******************************************************************************
 *                            CLASS DECLARATION
 *****************************************************************************/

#endif // __FALCON_DSP_UTILS_H__
