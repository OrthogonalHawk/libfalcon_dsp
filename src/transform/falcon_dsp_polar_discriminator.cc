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
 * @file     falcon_dsp_polar_discriminator.cc
 * @author   OrthogonalHawk
 * @date     21-Jan-2020
 *
 * @brief    Implements C++-based polar discriminator operations, such as one
 *            needs for FM demodulation.
 *
 * @section  DESCRIPTION
 *
 * Implements the C++ version of polar discriminator operations. Both a standalone
 *  function and a class-based tranform object are supported.
 *
 * @section  HISTORY
 *
 * 21-Jan-2020  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <cmath>
#include <iostream>
#include <stdint.h>

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

/******************************************************************************
 *                            CLASS IMPLEMENTATION
 *****************************************************************************/

namespace falcon_dsp
{
    /******************************************************************************
     *                        FUNCTION IMPLEMENTATION
     *****************************************************************************/
    
    /* @brief C++ implementation of a polar discriminator vector operation.
     * @param[in] in                    - input vector
     * @param[out] out                  - filtered vector
     * @return True if the output vector was populated with the discriminator output
     *          as requested; false otherwise.
     */
    bool polar_discriminator(const std::vector<std::complex<int16_t>>& in,
                             std::vector<float>& out)
    {
        falcon_dsp_polar_discriminator discriminator;
        return discriminator.apply(in, out);
    }

    bool polar_discriminator(const std::vector<std::complex<float>>& in,
                             std::vector<float>& out)
    {
        falcon_dsp_polar_discriminator discriminator;
        return discriminator.apply(in, out);
    }
    

    /******************************************************************************
     *                           CLASS IMPLEMENTATION
     *****************************************************************************/
    
    falcon_dsp_polar_discriminator::falcon_dsp_polar_discriminator(void)
    {
        m_state.reserve(1);
    }
    
    void falcon_dsp_polar_discriminator::reset_state(void)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        /* reset the state information; an end-user might invoke this function if processing
         *  non-continuous data */
        m_state.clear();
    }

    bool falcon_dsp_polar_discriminator::apply(const std::vector<std::complex<int16_t>>& in, std::vector<float>& out)
    {
        out.clear();
        out.reserve(in.size());
        
        /* create another copy of the data and cast to std::complex<float> */
        std::vector<std::complex<float>> tmp_in_vec;
        tmp_in_vec.reserve(in.size());
        for (auto in_iter = in.begin(); in_iter != in.end(); ++in_iter)
        {
            tmp_in_vec.push_back(std::complex<float>((*in_iter).real(), (*in_iter).imag()));
        }
        
        /* run the discriminator on the input data */
        return apply(tmp_in_vec, out);
    }
    
    bool falcon_dsp_polar_discriminator::apply(const std::vector<std::complex<float>>& in, std::vector<float>& out)
    {
        std::lock_guard<std::mutex> lock(std::mutex);
        
        out.clear();
        out.reserve(in.size());
        
        /* sanity check the inputs */
        if (in.size() == 0)
        {
            return false;
        }
        
        /* start polar discriminator operation */
        std::complex<float> compare_val;
        
        /* special handling for the first sample; moved outside the loop
         *  so that the if statement is not checked for all other samples */
        if (m_state.size() > 0)
        {
            compare_val = in[0] * std::complex<float>(m_state[0].real(), -1 * m_state[0].imag());
            out.push_back(atan2(compare_val.imag(), compare_val.real()));
        }
        
        /* now process the remaining samples */
        for (uint32_t sample_idx = 1; sample_idx < in.size(); ++sample_idx)
        {
            compare_val = in[sample_idx] * std::complex<float>(in[sample_idx - 1].real(), -1 * in[sample_idx - 1].imag());
            out.push_back(atan2(compare_val.imag(), compare_val.real()));
        }
        
        /* finished handling the current data; now update the state. storing the very last
         *  value in the input vector */
        m_state.clear();
        m_state.push_back(*in.rbegin());
        
        return out.size() > 0;
    }
}
