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
 * @file     falcon_dsp_resample_cuda.cu
 * @author   OrthogonalHawk
 * @date     03-Sep-2019
 *
 * @brief    CUDA implementation of a polyphase resampler.
 *
 * @section  DESCRIPTION
 *
 * Implements CUDA version of a polyphase resampler.
 *
 * @section  HISTORY
 *
 * 03-Sep-2019  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <iostream>
#include <stdint.h>

#include "resample/falcon_dsp_polyphase_resampler_cuda.h"
#include "utilities/falcon_dsp_utils.h"

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
    /* @brief Generic CUDA resampling operation.
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
    bool resample_cuda(uint32_t in_sample_rate_in_sps, std::vector<std::complex<float>>& in,
                  std::vector<std::complex<float>>& filter_coeffs,
                  uint32_t out_sample_rate_in_sps, std::vector<std::complex<float>>& out)
    {
        /* compute the required up and down sample rates */
        int64_t up_rate, down_rate;
        double decimal = static_cast<double>(out_sample_rate_in_sps) / static_cast<double>(in_sample_rate_in_sps);
        rat_approx(decimal, 1024, up_rate, down_rate);
        
        falcon_dsp_polyphase_resampler_cuda<std::complex<float>, std::complex<float>> resampler(up_rate, down_rate, filter_coeffs);
        
        return resampler.apply(in, out) > 0;
    }
    
    /* @brief Same implementation as 'resample_cuda', just with a different name. The "up"
     *         refers to "upsampling", followed by "fir" or Finite Impulse Response (FIR)
     *         filtering, and then "downsampling" (i.e. "dn").
     */
    bool upfirdn_cuda(uint32_t in_sample_rate_in_sps, std::vector<std::complex<float>>& in,
                      std::vector<std::complex<float>>& filter_coeffs,
                      uint32_t out_sample_rate_in_sps, std::vector<std::complex<float>>& out)
    {
        return resample_cuda(in_sample_rate_in_sps, in, filter_coeffs, out_sample_rate_in_sps, out);
    }
}

/******************************************************************************
 *                            CLASS IMPLEMENTATION
 *****************************************************************************/
