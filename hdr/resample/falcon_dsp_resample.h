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
 * @file     falcon_dsp_polyphase_resampler.h
 * @author   OrthogonalHawk
 * @date     25-Aug-2019
 *
 * @brief    Generic resampler functions.
 *
 * @section  DESCRIPTION
 *
 * Generic resampling functions that rely on more specific implementations in
 *  other files.
 *
 * @section  HISTORY
 *
 * 25-Aug-2019  OrthogonalHawk  File created.
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_RESAMPLER_H__
#define __FALCON_DSP_RESAMPLER_H__

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <complex>
#include <mutex>
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
    /* @brief Generic resampling operation.
     * @param[in] in_sample_rate_in_sps  - input vector sample rate in samples
     *                                      per second.
     * @param[in] in                     - input vector
     * @param[in] coeffs                 - FIR filter coefficients
     * @param[in] out_sample_rate_in_sps - output vector sample rate in samples
     *                                      per second.
     * @param[out] out                   - resampled vector
     * @return True if the input vector was resampled as requested;
     *          false otherwise.
     */
    bool resample(uint32_t in_sample_rate_in_sps, std::vector<std::complex<int16_t>>& in,
                  std::vector<float>& filter_coeffs,
                  uint32_t out_sample_rate_in_sps, std::vector<std::complex<int16_t>>& out);
    
    /* @brief Same implementation as 'resample', just with a different name. The "up"
     *         refers to "upsampling", followed by "fir" or Finite Impulse Response (FIR)
     *         filtering, and then "downsampling" (i.e. "dn").
     */
    bool upfirdn(uint32_t in_sample_rate_in_sps, std::vector<std::complex<int16_t>>& in,
                 std::vector<float>& filter_coeffs,
                 uint32_t out_sample_rate_in_sps, std::vector<std::complex<int16_t>>& out);
}

/******************************************************************************
 *                            CLASS DECLARATION
 *****************************************************************************/

#endif // __FALCON_DSP_RESAMPLER_H__
