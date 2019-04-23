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
 * @file     falcon_dsp_polyphase_resampler.cc
 * @author   OrthogonalHawk
 * @date     21-Apr-2019
 *
 * @brief    C++ implementation of a polyphase resampler.
 *
 * @section  DESCRIPTION
 *
 * Implements C++ version of a polyphase resampler.
 *
 * @section  HISTORY
 *
 * 21-Apr-2019  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <stdint.h>

#include "falcon_dsp_polyphase_resampler.h"

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
    template<class T, class C>
    falcon_dsp_polyphase_resampler<T, C>::falcon_dsp_polyphase_resampler(uint32_t up_rate, uint32_t down_rate, std::vector<C>& filter_coeffs)
      : m_up_rate(up_rate),
        m_down_rate(down_rate),
        m_padded_coeff_count(filter_coeffs.size()),
        m_t(0),
        m_xOffset(0)
    {
        /* The coefficients are copied into local storage in a transposed, flipped
         *  arrangement.  For example, suppose up_rate is 3, and the input number
         *  of coefficients coefCount = 10, represented as h[0], ..., h[9].
         *  Then the internal buffer will look like this:
         *   h[9], h[6], h[3], h[0],   // flipped phase 0 coefs
         *      0, h[7], h[4], h[1],   // flipped phase 1 coefs (zero-padded)
         *      0, h[8], h[5], h[2],   // flipped phase 2 coefs (zero-padded)
         */
        while (m_padded_coeff_count % m_up_rate)
        {
            m_padded_coeff_count++;
        }
        m_coeffs_per_phase = m_padded_coeff_count / m_up_rate;
    
        m_transposed_coeffs.clear();
        m_transposed_coeffs.reserve(m_padded_coeff_count);
        std::fill(m_transposed_coeffs.begin(), m_transposed_coeffs.end(), 0.);

        m_state.clear();
        m_state.reserve(m_coeffs_per_phase);
        std::fill(m_state.begin(), m_state.end(), 0.);

        /* This both transposes, and "flips" each phase, while
         * copying the defined coefficients into local storage.
         * There is probably a faster way to do this */
        for (uint32_t ii = 0; ii < m_up_rate; ++ii)
        {
            for (uint32_t jj = 0; jj < m_coeffs_per_phase; ++jj)
            {
                if ((jj * m_up_rate + ii) < filter_coeffs.size())
                {
                    m_transposed_coeffs[(m_coeffs_per_phase - 1 - jj) + ii * m_coeffs_per_phase] =
                        filter_coeffs[jj * m_up_rate + ii];
                }
            }
        }
    }
    
    template<class T, class C>
    falcon_dsp_polyphase_resampler<T, C>::~falcon_dsp_polyphase_resampler(void)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_state.clear();
        m_transposed_coeffs.clear();
    }
    
    template<class T, class C>
    void falcon_dsp_polyphase_resampler<T, C>::reset_state(void)
    {
        std::lock_guard<std::mutex> locak(m_mutex);
        
        /* reset the state information; an end-user might invoke this function if processing
         *  non-continuous data */
        std::fill(m_state.begin(), m_state.end(), 0.);
        m_t = 0;
        m_xOffset = 0;
    }
    
    template<class T, class C>
    uint32_t falcon_dsp_polyphase_resampler<T, C>::needed_out_count(uint32_t in_count)
    {
        /* compute how many outputs will be generated for in_count inputs */
        uint64_t np = in_count * static_cast<uint64_t>(m_up_rate);
        uint32_t need = np / m_down_rate;
        
        if ((m_t + m_up_rate * m_xOffset) < (np % m_down_rate))
        {
            need++;
        }
        
        return need;
    }

    /* generic implementation */
    template<class T, class C>
    int32_t falcon_dsp_polyphase_resampler<T, C>::apply(std::vector<input_type>& in, std::vector<output_type>& out)
    {
        std::lock_guard<std::mutex> lock(std::mutex);
        
        out.clear();
        
        /* x_idx points to the latest processed input sample */
        int64_t x_idx = m_xOffset;
        while (static_cast<uint64_t>(x_idx) < in.size())
        {
            output_type acc = output_type(0);
            typename std::vector<coeff_type>::iterator coeff_iter = m_transposed_coeffs.begin() + m_t * m_coeffs_per_phase;
            
            /* need to look back over the previous samples to compute the
             *  current filtered value */
            int64_t x_back_idx = x_idx - m_coeffs_per_phase + 1;
            int64_t offset = 0 - x_back_idx;
            
            if (offset > 0)
            {
                /* need to draw from the state buffer */
                typename std::vector<input_type>::iterator state_iter = m_state.end() - offset;
                while (state_iter < m_state.end())
                {
                    acc += *(state_iter++) * *(coeff_iter++);
                }
                x_back_idx += offset;
            }
            
            while (x_back_idx <= x_idx)
            {
                acc += in[x_back_idx++] * *(coeff_iter++);
            }
            
            out.push_back(acc);
            m_t += m_down_rate;
            
            int64_t advance_amount = m_t / m_up_rate;
            x_idx += advance_amount;

            // which phase of the filter to use
            m_t %= m_up_rate;
        }
        
        m_xOffset = x_idx - in.size();

        // manage _state buffer
        // find number of samples retained in buffer:
        int64_t retain = m_state.size() - in.size();
        if (retain > 0)
        {
            // for in.size() smaller than state buffer, copy end of buffer
            // to beginning:
            copy(m_state.end() - retain, m_state.end(), m_state.begin());
            
            // Then, copy the entire (short) input to end of buffer
            uint32_t in_idx = 0;
            for (uint64_t state_copy_idx = retain; state_copy_idx < m_state.size(); ++state_copy_idx)
            {
                m_state[state_copy_idx] = in[in_idx++];   
            }
        }
        else
        {
            // just copy last input samples into state buffer
            for (uint64_t state_copy_idx = 0; state_copy_idx < m_state.size(); ++state_copy_idx)
            {
                m_state[state_copy_idx] = in[in.size() - m_state.size() + state_copy_idx];   
            }
        }
        
        // number of samples computed
        return out.size();
    }
    
    /* specific implementation for complex<float> */
    template<>
    int32_t falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>::apply(std::vector<std::complex<float>>& in, std::vector<std::complex<float>>& out)
    {
        std::lock_guard<std::mutex> lock(std::mutex);
        
        out.clear();
        
        /* x_idx points to the latest processed input sample */
        int64_t x_idx = m_xOffset;
        while (static_cast<uint64_t>(x_idx) < in.size())
        {
            std::complex<float> acc = std::complex<float>(0);
            std::vector<std::complex<float>>::iterator coeff_iter = m_transposed_coeffs.begin() + m_t * m_coeffs_per_phase;
            
            /* need to look back over the previous samples to compute the
             *  current filtered value */
            int64_t x_back_idx = x_idx - m_coeffs_per_phase + 1;
            int64_t offset = 0 - x_back_idx;
            
            if (offset > 0)
            {
                /* need to draw from the state buffer */
                std::vector<std::complex<float>>::iterator state_iter = m_state.end() - offset;
                while (state_iter < m_state.end())
                {
                    /* by assuming that the filter coefficients are only real (symmetric filter) we can
                     *  bypass multiplication by the imaginary filter coefficients, which will be 0 */
                    acc += std::complex<float>(state_iter->real() * coeff_iter->real(), state_iter->imag() * coeff_iter->real());
                    state_iter++; coeff_iter++;
                }
                x_back_idx += offset;
            }
            
            while (x_back_idx <= x_idx)
            {
                /* by assuming that the filter coefficients are only real (symmetric filter) we can
                 *  bypass multiplication by the imaginary filter coefficients, which will be 0 */
                acc += std::complex<float>(in[x_back_idx].real() * coeff_iter->real(), in[x_back_idx].imag() * coeff_iter->real());
                x_back_idx++; coeff_iter++;
            }
            
            out.push_back(acc);
            m_t += m_down_rate;
            
            int64_t advance_amount = m_t / m_up_rate;
            x_idx += advance_amount;

            // which phase of the filter to use
            m_t %= m_up_rate;
        }
        
        m_xOffset = x_idx - in.size();

        // manage _state buffer
        // find number of samples retained in buffer:
        int64_t retain = m_state.size() - in.size();
        if (retain > 0)
        {
            // for in.size() smaller than state buffer, copy end of buffer
            // to beginning:
            copy(m_state.end() - retain, m_state.end(), m_state.begin());
            
            // Then, copy the entire (short) input to end of buffer
            uint32_t in_idx = 0;
            for (uint64_t state_copy_idx = retain; state_copy_idx < m_state.size(); ++state_copy_idx)
            {
                m_state[state_copy_idx] = in[in_idx++];   
            }
        }
        else
        {
            // just copy last input samples into state buffer
            for (uint64_t state_copy_idx = 0; state_copy_idx < m_state.size(); ++state_copy_idx)
            {
                m_state[state_copy_idx] = in[in.size() - m_state.size() + state_copy_idx];   
            }
        }
        
        // number of samples computed
        return out.size();
    }
    
    /* force instantiation for specific types */
    template class falcon_dsp_polyphase_resampler<std::complex<float>, std::complex<float>>;
}

/******************************************************************************
 *                            CLASS IMPLEMENTATION
 *****************************************************************************/
