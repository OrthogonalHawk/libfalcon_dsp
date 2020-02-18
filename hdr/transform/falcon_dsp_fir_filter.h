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
 * @file     falcon_dsp_fir_filter.h
 * @author   OrthogonalHawk
 * @date     20-Jan-2020
 *
 * @brief    Signal processing transformation functions for FIR filtering;
 *            C++ versions.
 *
 * @section  DESCRIPTION
 *
 * Defines a set of signal processing transformation FIR filtering functions.
 *  Includes C++ implementations.
 *
 * @section  HISTORY
 *
 * 20-Jan-2020  OrthogonalHawk  Created file.
 * 22-Jan-2020  OrthogonalHawk  Renamed to focus on FIR filtering.
 * 24-Jan-2020  OrthogonalHawk  Added protected function for state management.
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_TRANSFORM_FIR_FILTER_H__
#define __FALCON_DSP_TRANSFORM_FIR_FILTER_H__

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <complex>
#include <deque>
#include <mutex>
#include <vector>

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

namespace falcon_dsp
{
    /******************************************************************************
     *                           FUNCTION DECLARATION
     *****************************************************************************/
    
    /* @brief C++ implementation of a linear FIR filter vector operation.
     * @param[in] coeffs                - FIR filter coefficients
     * @param[in] in                    - input vector
     * @param[out] out                  - filtered vector
     * @return True if the input vector was filtered as requested;
     *          false otherwise.
     */
    bool fir_filter(std::vector<std::complex<float>> &coeffs, std::vector<std::complex<int16_t>>& in,
                    std::vector<std::complex<int16_t>>& out);
    
    bool fir_filter(std::vector<std::complex<float>> &coeffs, std::vector<std::complex<float>>& in,
                    std::vector<std::complex<float>>& out);
    
    /* @brief Provides FIR filter parameters intended for anti-alias protection
     *         during resampling operations.
     * @param[in] input_sample_rate     - input sample rate in samples per second
     * @param[in] output_sample_rate    - output sample rate in samples per second
     * @param[in] taps                  - number of taps in the filter
     * @param[in] source                - algorithm used to derive the coefficients
     * @param[out] up_rate              - upsample rate
     * @param[out] down_rate            - downsample rate
     * @param[out] coeffs               - filter coefficients
     * @return True if suitable filter coefficients were found/computed;
     *          false otherwise.
     */
    bool get_resample_fir_params(uint32_t input_sample_rate, uint32_t output_sample_rate,
                                 filter_taps_e taps, filter_source_type_e source,
                                 uint32_t& up_rate, uint32_t& down_rate,
                                 std::vector<std::complex<float>> &coeffs);
    
    /******************************************************************************
     *                            CLASS DECLARATION
     *****************************************************************************/
    
    /* @brief C++ implementation of a linear FIR filter utility class.
     * @description By implementing the linear FIR filter utility as a class
     *               interface instead of a simple function the user is able
     *               to filter an arbitrarily long input.
     */
    class falcon_dsp_fir_filter
    {
    public:

        falcon_dsp_fir_filter(std::vector<std::complex<float>> &coeffs);
        virtual ~falcon_dsp_fir_filter(void) = default;

        falcon_dsp_fir_filter(void) = delete;
        falcon_dsp_fir_filter(const falcon_dsp_fir_filter&) = delete;

        void reset_state(void);
        virtual bool apply(std::vector<std::complex<int16_t>>& in, std::vector<std::complex<int16_t>>& out);
        virtual bool apply(std::vector<std::complex<float>>& in, std::vector<std::complex<float>>& out);

    protected:
    
        void _update_state(std::vector<std::complex<float>>& in);
        
        std::mutex                               m_mutex;
        std::vector<std::complex<float>>         m_coefficients;
        std::vector<std::complex<float>>         m_state;
    };
}

#endif // __FALCON_DSP_TRANSFORM_FIR_FILTER_H__
