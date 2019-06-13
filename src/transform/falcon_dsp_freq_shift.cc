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
 * @file     falcon_dsp_freq_shift.cc
 * @author   OrthogonalHawk
 * @date     04-Jun-2019
 *
 * @brief    Implements a C++-based time series frequency shift operation.
 *
 * @section  DESCRIPTION
 *
 * Implements the C++ version of a time series frequency shift operation. Both
 *  a standalone function and a class-based tranform object are supported.
 *
 * @section  HISTORY
 *
 * 04-Jun-2019  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <iostream>
#include <stdint.h>

#include "transform/falcon_dsp_transform.h"

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
    /* @brief C++ implementation of a frequency shift vector operation.
     * @param[in] in_sample_rate_in_sps - input vector sample rate in samples
     *                                      per second.
     * @param[in] in                    - input vector
     * @param[in] freq_shift_in_hz      - amount to frequency shift in Hz
     * @param[out] out                  - frequency shifted vector
     * @return True if the input vector was frequency shifted as requested;
     *          false otherwise.
     */
    bool freq_shift(uint32_t in_sample_rate_in_sps, std::vector<std::complex<int16_t>>& in,
                    int32_t freq_shift_in_hz, std::vector<std::complex<int16_t>>& out)
    {
        falcon_dsp_freq_shift freq_shifter(in_sample_rate_in_sps, freq_shift_in_hz);
        return freq_shifter.apply(in, out);
    }
    
    falcon_dsp_freq_shift::falcon_dsp_freq_shift(uint32_t input_sample_rate_in_sps, int32_t freq_shift_in_hz)
      : m_samples_handled(0)
    {
        /* Frequency shift by multiplying by a complex sinusoid: e^(jwt) = cos(wt) + jsin(wt)
         *
         * Determine the "t" that results in one full revolution of the complex sinusoid
         *   i.e t = samplerate/freqshift
         *
         * Then compute a larger "t" value to reduce the number of rollover events and minimize
         *  the discontinuity error when a rollover does occur. Note that it is important to
         *  keep the rollover index value small (relatively) due to the computational cost of
         *  computing the sin of a large radian number; see [1] for a related discussion.
         *
         * [1] https://hackernoon.com/help-my-sin-is-slow-and-my-fpu-is-inaccurate-a2c751106102
         */
        double rollover_idx = static_cast<double>(input_sample_rate_in_sps) / static_cast<double>(abs(freq_shift_in_hz));

        /* 1e7 was chosen after empirical testing on a Jetson Nano as it did
         *  not seem to trigger the high computational cost case */
        while (rollover_idx < 1e7)
        {
            rollover_idx *= 10;
        }
        
        /* convert the double to an unsigned integer to get the actual
         *  rollover index value */
        m_calculated_rollover_sample_idx = static_cast<uint32_t>(rollover_idx);
        
        std::cout << "Computed rollover index: " << m_calculated_rollover_sample_idx << std::endl;
          
        /* compute the angular frequency since it won't change */
        m_angular_freq = (float(freq_shift_in_hz) / float(input_sample_rate_in_sps)) * 2.0 * M_PI;
          
        printf("Computed angular freq: %f using 2*pi=%.16f\n", m_angular_freq, 2.0 * M_PI);
    }
    
    void falcon_dsp_freq_shift::reset_state(void)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        /* reset the state information; an end-user might invoke this function if processing
         *  non-continuous data */
        m_samples_handled = 0;
    }

    bool falcon_dsp_freq_shift::apply(std::vector<std::complex<int16_t>>& in, std::vector<std::complex<int16_t>>& out)
    {
        std::lock_guard<std::mutex> lock(std::mutex);
        
        out.clear();
        
        for (auto it = in.begin(); it != in.end(); ++it)
        {
            float angle = m_angular_freq * m_samples_handled;
            out.push_back(std::complex<float>((*it).real(), (*it).imag()) *
                          std::complex<float>(cos(angle),
                                              sin(angle)));
            
            if (m_samples_handled >= 417214 && m_samples_handled <= 417220)
            {
                std::cout << "cpp input[" << m_samples_handled << "]: " << *it << std::endl;
                printf("cpp shift[%f] angle=%f cos=%f sin=%f\n", m_samples_handled, angle, cos(angle), sin(angle));
                std::cout << "cpp output[" << m_samples_handled << "]: " << out[m_samples_handled] << std::endl;
            }
            m_samples_handled++;            
            
            if (m_samples_handled >= m_calculated_rollover_sample_idx)
            {
                m_samples_handled = 0;
            }
        }
        
        return out.size() > 0;
    }
}

/******************************************************************************
 *                            CLASS IMPLEMENTATION
 *****************************************************************************/