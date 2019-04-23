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
 * @file     falcon_dsp_polyphase_resampler_cuda.cu
 * @author   OrthogonalHawk
 * @date     22-Apr-2019
 *
 * @brief    CUDA implementation of a polyphase resampler.
 *
 * @section  DESCRIPTION
 *
 * Implements CUDA version of a polyphase resampler.
 *
 * @section  HISTORY
 *
 * 22-Apr-2019  OrthogonalHawk  File created.
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
    falcon_dsp_polyphase_resampler_cuda<T, C>::falcon_dsp_polyphase_resampler_cuda(uint32_t up_rate, uint32_t down_rate, std::vector<C>& filter_coeffs)
      : falcon_dsp_polyphase_resampler<T, C>(up_rate, down_rate, filter_coeffs)
    { }
    
    template<class T, class C>
    falcon_dsp_polyphase_resampler_cuda<T, C>::~falcon_dsp_polyphase_resampler_cuda(void)
    {
        std::lock_guard<std::mutex> lock(falcon_dsp_polyphase_resampler<T, C>::m_mutex);
        falcon_dsp_polyphase_resampler<T, C>::m_state.clear();
        falcon_dsp_polyphase_resampler<T, C>::m_transposed_coeffs.clear();
    }

    /* generic implementation */
    template<class T, class C>
    int32_t falcon_dsp_polyphase_resampler_cuda<T, C>::apply(std::vector<input_type>& in, std::vector<output_type>& out)
    {
        std::lock_guard<std::mutex> lock(falcon_dsp_polyphase_resampler<T, C>::m_mutex);
        
        out.clear();
        
        /* don't bother with running a CUDA/GPU implementation if the input size is is not
         *  larger than the state vector */
        if (in.size() <= falcon_dsp_polyphase_resampler<T, C>::m_state.size())
        {
            return falcon_dsp_polyphase_resampler<T, C>::apply(in, out);  
        }
        
        /* x_idx points to the latest processed input sample */
        int64_t x_idx = falcon_dsp_polyphase_resampler<T, C>::m_xOffset;
        while (static_cast<uint64_t>(x_idx) < in.size())
        {
            output_type acc = output_type(0);
            typename std::vector<coeff_type>::iterator coeff_iter =
                falcon_dsp_polyphase_resampler<T, C>::m_transposed_coeffs.begin() +
                falcon_dsp_polyphase_resampler<T, C>::m_t * falcon_dsp_polyphase_resampler<T, C>::m_coeffs_per_phase;
            
            /* need to look back over the previous samples to compute the
             *  current filtered value */
            int64_t x_back_idx = x_idx - falcon_dsp_polyphase_resampler<T, C>::m_coeffs_per_phase + 1;
            int64_t offset = 0 - x_back_idx;
            
            if (offset > 0)
            {
                /* need to draw from the state buffer */
                typename std::vector<input_type>::iterator state_iter = falcon_dsp_polyphase_resampler<T, C>::m_state.end() - offset;
                while (state_iter < falcon_dsp_polyphase_resampler<T, C>::m_state.end())
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
            falcon_dsp_polyphase_resampler<T, C>::m_t += falcon_dsp_polyphase_resampler<T, C>::m_down_rate;
            
            int64_t advance_amount = falcon_dsp_polyphase_resampler<T, C>::m_t / falcon_dsp_polyphase_resampler<T, C>::m_up_rate;
            x_idx += advance_amount;

            // which phase of the filter to use
            falcon_dsp_polyphase_resampler<T, C>::m_t %= falcon_dsp_polyphase_resampler<T, C>::m_up_rate;
        }
        
        falcon_dsp_polyphase_resampler<T, C>::m_xOffset = x_idx - in.size();

        // manage _state buffer
        // find number of samples retained in buffer:
        int64_t retain = falcon_dsp_polyphase_resampler<T, C>::m_state.size() - in.size();
        if (retain > 0)
        {
            // for in.size() smaller than state buffer, copy end of buffer
            // to beginning:
            copy(falcon_dsp_polyphase_resampler<T, C>::m_state.end() - retain,
                 falcon_dsp_polyphase_resampler<T, C>::m_state.end(),
                 falcon_dsp_polyphase_resampler<T, C>::m_state.begin());
            
            // Then, copy the entire (short) input to end of buffer
            uint32_t in_idx = 0;
            for (uint64_t state_copy_idx = retain;
                 state_copy_idx < falcon_dsp_polyphase_resampler<T, C>::m_state.size();
                 ++state_copy_idx)
            {
                falcon_dsp_polyphase_resampler<T, C>::m_state[state_copy_idx] = in[in_idx++];   
            }
        }
        else
        {
            // just copy last input samples into state buffer
            for (uint64_t state_copy_idx = 0;
                 state_copy_idx < falcon_dsp_polyphase_resampler<T, C>::m_state.size();
                 ++state_copy_idx)
            {
                falcon_dsp_polyphase_resampler<T, C>::m_state[state_copy_idx] =
                    in[in.size() - falcon_dsp_polyphase_resampler<T, C>::m_state.size() + state_copy_idx];   
            }
        }
        
        // number of samples computed
        return out.size();
    }
    
    /* force instantiation for specific types */
    template class falcon_dsp_polyphase_resampler_cuda<std::complex<float>, std::complex<float>>;
}

/******************************************************************************
 *                            CLASS IMPLEMENTATION
 *****************************************************************************/
