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
 * @file     falcon_dsp_fm_demod.cc
 * @author   OrthogonalHawk
 * @date     11-Feb-2020
 *
 * @brief    Implements C++-based FM Demodulator.
 *
 * @section  DESCRIPTION
 *
 * Defines a set of signal processing routines to demodulate signals that are
 *  modulated using a Frequency Modulation (FM) technique common to public radio
 *  broadcasts in the United States. This implementation is based on the excellent
 *  tutorial provided at: https://witestlab.poly.edu/blog/capture-and-decode-fm-radio/
 *
 * @section  HISTORY
 *
 * 11-Feb-2020  OrthogonalHawk  Created file.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <iostream>
#include <stdint.h>

#include "signals/falcon_dsp_fm_demod.h"
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
    /******************************************************************************
     *                        FUNCTION IMPLEMENTATION
     *****************************************************************************/    

    /******************************************************************************
     *                           CLASS IMPLEMENTATION
     *****************************************************************************/
    
    falcon_dsp_fm_demodulator::falcon_dsp_fm_demodulator(void)
      : m_initialized(false),
        m_freq_shifter(),
        m_signal_isolation_decimator(),
        m_polar_discriminator(),
        m_deemphasis_filter(),
        m_mono_signal_decimator()
    { }
    
    bool falcon_dsp_fm_demodulator::initialize(uint32_t input_sample_rate_in_sps,
                                               int32_t signal_offset_from_dc_in_hz,
                                               float deemphasis_time_constant_in_usecs)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        if (m_initialized)
        {
            /* only let the user call this function once */
            return false;
        }
            
        m_initialized |= m_freq_shifter.initialize(input_sample_rate_in_sps, signal_offset_from_dc_in_hz);
        
        m_initialized &= m_signal_isolation_decimator.initialize()
        
        /* initialization complete */
        m_initialized = true;
        
        return m_initialized;
    }
    
    void falcon_dsp_fm_demodulator::reset_state(void)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        /* reset the state information; an end-user might invoke this function if processing
         *  non-continuous data */
        
        m_freq_shifter.reset_state();
        m_signal_isolation_decimator.reset_state();
        m_polar_discriminator.reset_state();
        m_deemphasis_filter.reset_state();
        m_mono_signal_decimator.reset_state();
    }

    bool falcon_dsp_iir_filter::apply(std::vector<std::complex<int16_t>>& in, std::vector<std::complex<int16_t>>& out)
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
    
    bool falcon_dsp_iir_filter::apply(std::vector<std::complex<float>>& in, std::vector<std::complex<float>>& out)
    {
        /* implementation based off the guide from [1] and comparison with results
         *  from the Python signal.lfilter implementation (see unit tests for more detail).
         *
         * From [1]:
         *  **Filtering with the filter Function**
         *  For IIR filters, the filtering operation is described not by a simple convolution,
         *   but by a difference equation that can be found from the transfer-function relation.
         *   Assume that a(1) = 1, move the denominator to the left side, and take the inverse
         *   Z-transform to obtain:
         *
         *     y(k)+a(2) y(k−1)+…+a(m+1) y(k−m)=b(1) x(k)+b(2) x(k−1)+⋯+b(n+1) x(k−n).
         *
         *  In terms of current and past inputs, and past outputs, y(k) is:
         *
         *     y(k)=b(1) x(k)+b(2) x(k−1)+⋯+b(n+1) x(k−n)−a(2) y(k−1)−⋯−a(m+1) y(k−m),
         *
         *  which is the standard time-domain representation of a digital filter. Starting with
         *   y(1) and assuming a causal system with zero initial conditions, the representation
         *   is equivalent to:
         *
         *     y(1)=b(1) x(1)
         *     y(2)=b(1) x(2)+b(2) x(1)−a(2) y(1)
         *     y(3)=b(1) x(3)+b(2) x(2)+b(3) x(1)−a(2) y(2)−a(3) y(1)
         *       ⋮   
         *     y(n)=b(1) x(n)+⋯+b(n) x(1)−a(2) y(n−1)−⋯−a(n) y(1).
         *
         * [1] https://www.mathworks.com/help/signal/ug/filter-implementation-and-analysis.html
         */
        
        std::lock_guard<std::mutex> lock(std::mutex);
        
        out.clear();
        out.reserve(in.size());
        
        /* sanity check the class configuration and inputs */
        if (m_b_coefficients.size() == 0 ||
            m_a_coefficients.size() == 0 ||
            in.size() == 0)
        {
            return false;
        }
        
        /* start filtering */
        auto b_coeff_iter = m_b_coefficients.begin();
        auto a_coeff_iter = m_a_coefficients.begin() + 1;
        std::complex<float> filter_accum;
        
        for (uint32_t sample_idx = 0; sample_idx < in.size(); ++sample_idx)
        {            
            /* reset for each new output sample */
            b_coeff_iter = m_b_coefficients.begin();
            a_coeff_iter = m_a_coefficients.begin() + 1;
            filter_accum = std::complex<float>(0., 0.);
            
            
            /*****************************************
             * Apply the 'b' Coefficients
             ****************************************/
            
            /* handle elements at the 'front' of the vector where we may need
             *  to draw from the state array to apply the 'b' coefficients */
            if (sample_idx < m_b_coefficients.size())
            {
                /* first, work through data from the input buffer */
                auto data_iter = in.rbegin() + (in.size() - 1 - sample_idx);
                while (data_iter != in.rend())
                {
                    filter_accum += std::complex<float>((*data_iter).real(), (*data_iter).imag()) * (*b_coeff_iter++);
                    data_iter++;
                }
                
                /* now work through the state buffer */
                auto state_iter = m_input_state.rbegin();
                while (b_coeff_iter != m_b_coefficients.end() &&
                       state_iter != m_input_state.rend())
                {
                    filter_accum += (*state_iter++) * (*b_coeff_iter++);
                }
            }
            else
            {
                /* there is sufficient data to pull only from the input buffer
                 *  so use all coefficients here */
                auto data_iter = in.rbegin() + (in.size() - 1 - sample_idx);
                while (b_coeff_iter != m_b_coefficients.end())
                {
                    filter_accum += std::complex<float>((*data_iter).real(), (*data_iter).imag()) * (*b_coeff_iter++);
                    data_iter++;
                }
            }
            
            
            /*****************************************
             * Apply the 'a' Coefficients
             ****************************************/
            
            auto data_iter = m_prev_outputs.rbegin();
            while (a_coeff_iter != m_a_coefficients.end() &&
                   data_iter != m_prev_outputs.rend())
            {
                filter_accum -= (*data_iter++) * (*a_coeff_iter++);
            }
            
            /* finished computing output */
            out.push_back(filter_accum);
            
            /* update the container with the last computed outputs */
            m_prev_outputs.push_back(filter_accum);
            while (m_prev_outputs.size() > (m_a_coefficients.size() - 1))
            {
                m_prev_outputs.erase(m_prev_outputs.begin(), m_prev_outputs.begin() + 1);
            }
        }
        
        /* finished handling the current data; now update the 'b' coefficient state array.
         *  note that the state array is populated in such a way that the last sample in
         *  the state buffer is the sample that is expected to immediately preceed the next
         *  sample.
         *
         * Example:
         *   Data:   x1, x2, x3, x4, x5, x6, x7, x8
         *   Coeffs: h1, h2, h3
         *
         * If the first four values, x1-x4, were passed to the apply method then the state
         *  buffer will have x2-x4 when the apply method returns. state[0] = x2, state[1] = x3, state[2] = x4
         */
        if (in.size() >= m_b_coefficients.size())
        {
            /* input vector is longer than the state array so copy in the last
             *  values from the input */
            m_input_state.clear();
            for (uint32_t state_idx = 0; state_idx < m_b_coefficients.size(); ++state_idx)
            {
                auto next_val = *(in.end() - (m_b_coefficients.size() - state_idx));
                m_input_state.push_back(std::complex<float>(next_val.real(), next_val.imag()));
            }
        }
        else
        {
            /* input vector is shorter than the state array; append the entire input array
             *  and then chop off the oldest data */
            for (auto input_iter = in.begin(); input_iter != in.end(); ++input_iter)
            {
                m_input_state.push_back(std::complex<float>((*input_iter).real(), (*input_iter).imag()));
            }

            int32_t num_elements_to_erase = m_input_state.size() - m_b_coefficients.size();
            if (num_elements_to_erase > 0)
            {
                m_input_state.erase(m_input_state.begin(), m_input_state.begin() + num_elements_to_erase);
            }
        }
        
        return out.size() > 0;
    }
}
