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
 * @file     falcon_dsp_fir_filter.cc
 * @author   OrthogonalHawk
 * @date     20-Jan-2020
 *
 * @brief    Implements C++-based FIR filtering operations.
 *
 * @section  DESCRIPTION
 *
 * Implements the C++ version of FIR filtering operations. Both a standalone
 *  function and a class-based tranform object are supported.
 *
 * @section  HISTORY
 *
 * 20-Jan-2020  OrthogonalHawk  File created.
 * 22-Jan-2020  OrthogonalHawk  Renamed to focus on FIR filtering.
 * 24-Jan-2020  OrthogonalHawk  Added protected function for state management.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <iostream>
#include <stdint.h>

#include "transform/falcon_dsp_fir_filter.h"
#include "transform/falcon_dsp_predefined_fir_filter.h"

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
    
    /* @brief C++ implementation of a linear FIR filter vector operation.
     * @param[in] filter_coeffs         - FIR filter coefficients
     * @param[in] in                    - input vector
     * @param[out] out                  - filtered vector
     * @return True if the input vector was filtered as requested;
     *          false otherwise.
     */
    bool fir_filter(std::vector<std::complex<float>> &coeffs, std::vector<std::complex<int16_t>>& in,
                    std::vector<std::complex<int16_t>>& out)
    {
        falcon_dsp_fir_filter filter_obj(coeffs);
        return filter_obj.apply(in, out);
    }

    bool fir_filter(std::vector<std::complex<float>> &coeffs, std::vector<std::complex<float>>& in,
                    std::vector<std::complex<float>>& out)
    {
        falcon_dsp_fir_filter filter_obj(coeffs);
        return filter_obj.apply(in, out);
    }
    
    /* @brief Provides FIR filter coefficients intended for anti-alias protection
     *         during resampling operations.
     * @param[in] input_sample_rate     - input sample rate in samples per second
     * @param[in] output_sample_rate    - output sample rate in samples per second
     * @param[out] coeffs               - filter coefficients
     * @return True if suitable filter coefficients were found/computed;
     *          false otherwise.
     */
    bool get_resample_fir_coeffs(uint32_t input_sample_rate, uint32_t output_sample_rate,
                                 std::vector<std::complex<float>> &coeffs)
    {
        coeffs.clear();

        auto rate_pair = std::make_pair(input_sample_rate, output_sample_rate);
        if (s_predefined_fir_coeffs.count(rate_pair) > 0)
        {
            coeffs = s_predefined_fir_coeffs[rate_pair];
            return true;
        }
        else
        {
            return false;
        }
    }

    /******************************************************************************
     *                           CLASS IMPLEMENTATION
     *****************************************************************************/
    
    falcon_dsp_fir_filter::falcon_dsp_fir_filter(std::vector<std::complex<float>> &coeffs)
    {
        m_coefficients = coeffs;
        m_state.reserve(coeffs.size());
    }
    
    void falcon_dsp_fir_filter::reset_state(void)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        /* reset the state information; an end-user might invoke this function if processing
         *  non-continuous data */
        m_state.clear();
    }

    bool falcon_dsp_fir_filter::apply(std::vector<std::complex<int16_t>>& in, std::vector<std::complex<int16_t>>& out)
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
        
        /* filter the input data */
        std::vector<std::complex<float>> tmp_out_vec;
        bool ret = apply(tmp_in_vec, tmp_out_vec);
        
        /* cast the filtered output back to std::complex<int16_t> */
        for (auto out_iter = tmp_out_vec.begin(); out_iter != tmp_out_vec.end(); ++out_iter)
        {
            out.push_back(std::complex<int16_t>((*out_iter).real(), (*out_iter).imag()));
        }
        
        return ret;
    }
    
    bool falcon_dsp_fir_filter::apply(std::vector<std::complex<float>>& in, std::vector<std::complex<float>>& out)
    {
        std::lock_guard<std::mutex> lock(std::mutex);
        
        out.clear();
        out.reserve(in.size());
        
        /* sanity check the class configuration and inputs */
        if (m_coefficients.size() == 0 || in.size() == 0)
        {
            return false;
        }
        
        /* start filtering */
        auto coeff_iter = m_coefficients.begin();
        std::complex<float> filter_accum;
        
        for (uint32_t sample_idx = 0; sample_idx < in.size(); ++sample_idx)
        {            
            /* reset for each new output sample */
            coeff_iter = m_coefficients.begin();
            filter_accum = std::complex<float>(0., 0.);
            
            /* handle elements at the 'front' of the vector where we may need
             *  to draw from the state array */
            if (sample_idx < m_coefficients.size())
            {
                /* first, work through data from the input buffer */
                auto data_iter = in.rbegin() + (in.size() - 1 - sample_idx);
                while (data_iter != in.rend())
                {
                    filter_accum += std::complex<float>((*data_iter).real(), (*data_iter).imag()) * (*coeff_iter++);
                    data_iter++;
                }
                
                /* now work through the state buffer */
                auto state_iter = m_state.rbegin();
                while (coeff_iter != m_coefficients.end() &&
                       state_iter != m_state.rend())
                {
                    filter_accum += (*state_iter++) * (*coeff_iter++);
                }
                
                /* finished computing output */
                out.push_back(filter_accum);
            }
            else
            {
                /* there is sufficient data to pull only from the input buffer
                 *  so use all coefficients here */
                auto data_iter = in.rbegin() + (in.size() - 1 - sample_idx);
                while (coeff_iter != m_coefficients.end())
                {
                    filter_accum += std::complex<float>((*data_iter).real(), (*data_iter).imag()) * (*coeff_iter++);
                    data_iter++;
                }
                
                /* finished computing output */
                out.push_back(filter_accum);
            }
        }
        
        /* finished handling the current data; now update the state array */
        _update_state(in);
        
        return out.size() > 0;
    }
    
    void falcon_dsp_fir_filter::_update_state(std::vector<std::complex<float>>& in)
    {
        /* note that the state array is populated in such a way that the last sample
         *  in the state buffer is the sample that is expected to immediately precede
         *  the next sample.
         *
         * Example:
         *   Data:   x1, x2, x3, x4, x5, x6, x7, x8
         *   Coeffs: h1, h2, h3
         *
         * If the first four values, x1-x4, were passed to the apply method then the state
         *  buffer will have x2-x4 when the apply method returns. state[0] = x2, state[1] = x3, state[2] = x4
         */
        if (in.size() >= m_coefficients.size())
        {
            /* input vector is longer than the state array so copy in the last
             *  values from the input */
            m_state.clear();
            for (uint32_t state_idx = 0; state_idx < m_coefficients.size(); ++state_idx)
            {
                auto next_val = *(in.end() - (m_coefficients.size() - state_idx));
                m_state.push_back(std::complex<float>(next_val.real(), next_val.imag()));
            }
        }
        else
        {
            /* input vector is shorter than the state array; append the entire input array
             *  and then chop off the oldest data */
            for (auto input_iter = in.begin(); input_iter != in.end(); ++input_iter)
            {
                m_state.push_back(std::complex<float>((*input_iter).real(), (*input_iter).imag()));
            }

            int32_t num_elements_to_erase = m_state.size() - m_coefficients.size();
            if (num_elements_to_erase > 0)
            {
                m_state.erase(m_state.begin(), m_state.begin() + num_elements_to_erase);
            }
        }
    }
}
