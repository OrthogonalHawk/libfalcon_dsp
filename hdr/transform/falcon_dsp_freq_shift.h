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
 * @file     falcon_dsp_freq_shift.h
 * @author   OrthogonalHawk
 * @date     19-Jan-2020
 *
 * @brief    Signal processing transformation functions for frequency shifting;
 *            C++ versions.
 *
 * @section  DESCRIPTION
 *
 * Defines a set of signal processing transformation frequency shift functions.
 *  Includes C++ implementations.
 *
 * @section  HISTORY
 *
 * 19-Jan-2020  OrthogonalHawk  File broken out from falcon_dsp_transform.h
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_TRANSFORM_FREQ_SHIFT_H__
#define __FALCON_DSP_TRANSFORM_FREQ_SHIFT_H__

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

/******************************************************************************
 *                                  MACROS
 *****************************************************************************/

namespace falcon_dsp
{
    /******************************************************************************
     *                           FUNCTION DECLARATION
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
                    int32_t freq_shift_in_hz, std::vector<std::complex<int16_t>>& out);
    
    
    /******************************************************************************
     *                            CLASS DECLARATION
     *****************************************************************************/
    
    /* @brief C++ implementation of a frequency shift utility class.
     * @description By implementing the frequency shift utility as a class
     *               interface instead of a simple function the user is able
     *               to shift an arbitrarily long input with minimal discontinuities.
     */
    class falcon_dsp_freq_shift
    {
    public:

        static std::pair<uint32_t, float> get_freq_shift_params(uint32_t input_sample_rate_in_sps, int32_t freq_shift_in_hz);

        falcon_dsp_freq_shift(uint32_t input_sample_rate_in_sps, int32_t freq_shift_in_hz);
        virtual ~falcon_dsp_freq_shift(void) = default;

        falcon_dsp_freq_shift(void) = delete;
        falcon_dsp_freq_shift(const falcon_dsp_freq_shift&) = delete;

        void reset_state(void);
        virtual bool apply(std::vector<std::complex<int16_t>>& in, std::vector<std::complex<int16_t>>& out);
        virtual bool apply(std::vector<std::complex<float>>& in, std::vector<std::complex<float>>& out);

    protected:

        std::mutex m_mutex;
        double     m_samples_handled;
        uint32_t   m_calculated_rollover_sample_idx;
        float      m_angular_freq;
    };
}

#endif // __FALCON_DSP_TRANSFORM_FREQ_SHIFT_H__
