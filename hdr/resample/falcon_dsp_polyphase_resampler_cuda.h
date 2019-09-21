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
 * @date     03-Sep-2019
 *
 * @brief    Polyphase resampler interface that supports arbitrary input and output
 *            sample rates; CUDA version.
 *
 * @section  DESCRIPTION
 *
 * Polyphase resampler interface that supports arbitrary input and output sample
 *  rates. Includes CUDA implementations.
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
 * 03-Sep-2019  OrthogonalHawk  File created.
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_POLYPHASE_RESAMPLER_CUDA_H__
#define __FALCON_DSP_POLYPHASE_RESAMPLER_CUDA_H__

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <complex>
#include <mutex>
#include <vector>

#include "resample/falcon_dsp_polyphase_resampler.h"

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
    /* @brief CUDA implementation of a polyphase resampler
     * @description Derives from the C++ version since there is significant overlap
     *               in implementation. CUDA vs. C++ differentiation in the 'apply'
     *               method when the resampling is performed.
     */
    template<class T, class C>
    class falcon_dsp_polyphase_resampler_cuda : public falcon_dsp_polyphase_resampler<T, C>
    {
    public:
        
        typedef    T input_type;
        typedef    T output_type;
        typedef    C coeff_type;
        
        falcon_dsp_polyphase_resampler_cuda(uint32_t up_rate, uint32_t down_rate, std::vector<coeff_type>& filter_coeffs);
        ~falcon_dsp_polyphase_resampler_cuda(void);
        
        int32_t apply(std::vector<input_type>& in, std::vector<output_type>& out) override;
    
    private:
        
        bool compute_kernel_params(int64_t cur_x_idx, size_t in_size,
                                   uint32_t& num_out_samples,
                                   uint32_t& new_t,
                                   int64_t& final_x_idx);
        
        /* variables for CUDA memory management */
        void * m_cuda_input_samples;
        void * m_cuda_output_samples;
        void * m_cuda_filter_coeffs;
        
        uint32_t m_max_num_cuda_input_samples;
        uint32_t m_max_num_cuda_output_samples;
    };
    
    /* specific implementation of this template class */
    template <>
    int32_t falcon_dsp_polyphase_resampler_cuda<std::complex<float>, std::complex<float>>::apply(std::vector<std::complex<float>>& in, std::vector<std::complex<float>>& out);
}

/******************************************************************************
 *                            CLASS DECLARATION
 *****************************************************************************/

#endif // __FALCON_DSP_POLYPHASE_RESAMPLER_CUDA_H__
