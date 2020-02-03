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
 * 20-Jan-2020  OrthogonalHawk  Update to reflect movement from falcon_dsp_transform.h
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <iostream>
#include <stdint.h>

#include "transform/falcon_dsp_freq_shift.h"

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
    
    
    /******************************************************************************
     *                            CLASS IMPLEMENTATION
     *****************************************************************************/
    
    falcon_dsp_freq_shift::falcon_dsp_freq_shift(uint32_t input_sample_rate_in_sps, int32_t freq_shift_in_hz)
      : m_samples_handled(0)
    {
        auto params = get_freq_shift_params(input_sample_rate_in_sps, freq_shift_in_hz);
        m_calculated_rollover_sample_idx = params.first;
        m_angular_freq = params.second;
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
            double angle = m_angular_freq * m_samples_handled;
            out.push_back(std::complex<float>((*it).real(), (*it).imag()) *
                          std::complex<float>(cos(angle),
                                              sin(angle)));
            
            m_samples_handled++;            
            
            if (m_samples_handled >= m_calculated_rollover_sample_idx)
            {
                m_samples_handled = 0;
            }
        }
        
        return out.size() > 0;
    }
    
    bool falcon_dsp_freq_shift::apply(std::vector<std::complex<float>>& in, std::vector<std::complex<float>>& out)
    {
        std::lock_guard<std::mutex> lock(std::mutex);
        
        out.clear();

        double angle;
        std::complex<float> shift_angle;
        for (auto it = in.begin(); it != in.end(); ++it)
        {
            angle = m_angular_freq * m_samples_handled;
            shift_angle = std::complex<float>(cos(angle), sin(angle));
            out.push_back(std::complex<float>((*it).real(), (*it).imag()) * shift_angle);
            
            m_samples_handled++;
            
            if (m_samples_handled >= m_calculated_rollover_sample_idx)
            {
                m_samples_handled = 0;
            }
        }
        
        return out.size() > 0;
    }
    
    std::pair<uint32_t, double> falcon_dsp_freq_shift::get_freq_shift_params(uint32_t input_sample_rate_in_sps, int32_t freq_shift_in_hz)
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
         * After further testing, there is another reason to keep the "t" values small, and it
         *  appears to be related to Python/numpy floating point precision when computing the
         *  sine and cosine of large values in radians (large "t" values mean large angle values
         *  are passed to these functions). Specifically, since Python/numpy are used as the
         *  reference data source for unit test vectors I've noticed that the sine and cosine
         *  values begin to diverge between the C++/CUDA calculations and the numpy calculations
         *  after the angle value gets too high.
         *
         * [1] https://hackernoon.com/help-my-sin-is-slow-and-my-fpu-is-inaccurate-a2c751106102
         */
        double rollover_idx = static_cast<double>(input_sample_rate_in_sps) / static_cast<double>(abs(freq_shift_in_hz));

        /* 1e4 was chosen after empirical testing on a Jetson Nano as it did
         *  not seem to trigger the high computational cost case and kept the
         *  resulting angle values within a range where the C++/CUDA and
         *  Python/numpy computations agree with one another. */
        while (rollover_idx < 1e5)
        {
            rollover_idx *= 10;
        }
        
        /* convert the double to an unsigned integer to get the actual
         *  rollover index value */
        uint32_t calculated_rollover_sample_idx = static_cast<uint32_t>(round(rollover_idx));
          
        /* compute the angular frequency since it won't change */
        double angular_freq = (double(freq_shift_in_hz) / double(input_sample_rate_in_sps)) * 2.0 * M_PI;
        
        return std::pair<uint32_t, double>(calculated_rollover_sample_idx, angular_freq);
    }
}
