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
 * @file     falcon_dsp_polyphase_resampler.h
 * @author   OrthogonalHawk
 * @date     21-Apr-2019
 *
 * @brief    Polyphase resampler interface that supports arbitrary input and output
 *            sample rates; C++ and CUDA versions.
 *
 * @section  DESCRIPTION
 *
 * Polyphase resampler interface that supports arbitrary input and output sample
 *  rates. Includes C++ implementation.
 *
 * Notes from https://github.com/staticfloat/upfirdn/blob/master/README.txt
 *
 * ALGORITHM DESCRIPTION
 *  A "filter bank with resampling" is an operation on an input signal that 
 *   generates an output signal, consisting of the following 3 steps:
 *  1) upsampling (that is, zero-insertion) of the input signal by an integer 
 *     factor (call it P).
 *  2) applying an FIR (finite-impulse response filter) to the result of 1).
 *  3) downsampling (that is, decimation) of the result of 2) by an integer 
 *     factor (call it Q).
 *
 *  For an input signal with sampling rate T, the generated output signal has 
 *   sampling rate of P/Q*T.  The FIR filter is usually designed to prevent 
 *   aliasing from corrupting the output signal.
 *
 *  An "efficiently implemented, polyphase filter bank with resampling" implements 
 *   these three operations with a minimal amount of computation.
 *
 *  The algorithm is an implementation of the block diagram shown on page 129 of 
 *   the Vaidyanathan text [1] (Figure 4.3-8d).
 *
 *  [1]  P. P. Vaidyanathan, Multirate Systems and Filter Banks, Prentice Hall, 
 *    1993.
 *
 * @section  HISTORY
 *
 * 21-Apr-2019  OrthogonalHawk  File created.
 * 24-Jan-2020  OrthogonalHawk  Switched to fully specified class instead of
 *                               templated class.
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_POLYPHASE_RESAMPLER_H__
#define __FALCON_DSP_POLYPHASE_RESAMPLER_H__

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

struct polyphase_resampler_params_s
{
    void reset_state(void)
    {
        /* reset the state information; an end-user might invoke this function if processing
         *  non-continuous data */
        state.clear();
        state.resize(coeffs_per_phase - 1, std::complex<float>(0.0, 0.0));
        coeff_phase = 0;
        xOffset = 0;
    }
    
    uint32_t                             up_rate;
    uint32_t                             down_rate;
    
    std::vector<std::complex<float>>     transposed_coeffs;
    std::vector<std::complex<float>>     state;
        
    uint32_t                             padded_coeff_count;  // ceil(len(coefs)/upRate)*upRate
    uint32_t                             coeffs_per_phase;    // _paddedCoefCount / upRate

    uint32_t                             coeff_phase;         // "time" (modulo upRate)
    uint32_t                             xOffset;  
};

/******************************************************************************
 *                                  MACROS
 *****************************************************************************/

/******************************************************************************
 *                           FUNCTION DECLARATION
 *****************************************************************************/

namespace falcon_dsp
{
    /******************************************************************************
     *                            CLASS DECLARATION
     *****************************************************************************/
 
    /* @brief C++ implementation of a polyphase resampler
     */
    class falcon_dsp_polyphase_resampler
    {
    public:

        static polyphase_resampler_params_s get_resampler_params(uint32_t up_rate, uint32_t down_rate,
                                                                 const std::vector<std::complex<float>>& filter_coeffs);

        falcon_dsp_polyphase_resampler(uint32_t up_rate, uint32_t down_rate,
                                       const std::vector<std::complex<float>>& filter_coeffs);
        virtual ~falcon_dsp_polyphase_resampler(void);

        falcon_dsp_polyphase_resampler(void) = delete;
        falcon_dsp_polyphase_resampler(const falcon_dsp_polyphase_resampler&) = delete;

        void reset_state(void);
        virtual int32_t apply(std::vector<std::complex<float>>& in, std::vector<std::complex<float>>& out);
        uint32_t        needed_out_count(uint32_t inCount);
        uint32_t        coeffs_per_phase(void) { return m_params.coeffs_per_phase; }

    protected:
    
        void _manage_state(std::vector<std::complex<float>>& in);
        
        std::mutex                           m_mutex;
        polyphase_resampler_params_s         m_params;
    };
}

#endif // __FALCON_DSP_POLYPHASE_RESAMPLER_H__
