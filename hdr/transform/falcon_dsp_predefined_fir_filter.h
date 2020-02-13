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
 * @file     falcon_dsp_predefined_fir_filter.h
 * @author   OrthogonalHawk
 * @date     12-Feb-2020
 *
 * @brief    Predefined FIR filter coefficients.
 *
 * @section  DESCRIPTION
 *
 * Defines a set of predefined/precomputed FIR filter coefficients.
 *
 * @section  HISTORY
 *
 * 12-Feb-2020  OrthogonalHawk  Created file.
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_TRANSFORM_PREDEFINED_FIR_FILTER_H__
#define __FALCON_DSP_TRANSFORM_PREDEFINED_FIR_FILTER_H__

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <complex>
#include <map>
#include <vector>

/******************************************************************************
 *                                 CONSTANTS
 *****************************************************************************/

/******************************************************************************
 *                              ENUMS & TYPEDEFS
 *****************************************************************************/

struct predefined_resample_filter_params_s {
    uint32_t up_rate;
    uint32_t down_rate;
    std::vector<std::complex<float>> coeffs;
};

/******************************************************************************
 *                                  MACROS
 *****************************************************************************/

namespace falcon_dsp
{
    /******************************************************************************
     *                                 CONSTANTS
     *****************************************************************************/

    /* auto-generated set of predefined filter coefficients. these coefficients were
     *  generated by the 'generate_predefined_fir_filter_coeffs.ipynb' Python Jupyter
     *  notebook in the test/utilities folder.
     */
    std::map<std::pair<uint32_t, uint32_t>, predefined_resample_filter_params_s> s_predefined_resample_fir_coeffs = {
        /* INPUT_SAMPLE_RATE:        1140000 sps
         * OUTPUT_SAMPLE_RATE:        228000 sps
         */
        {
            std::make_pair(1140000, 228000),
            {1, /* up_rate */
             5, /* down_rate */
             std::vector<std::complex<float>>{
                 {-0.00000000, 0.00000000}, {-0.00017707, 0.00000000}, {-0.00035966, 0.00000000}, {-0.00044241, 0.00000000},
                 {-0.00033096, 0.00000000}, {+0.00000000, 0.00000000}, {+0.00046708, 0.00000000}, {+0.00088454, 0.00000000},
                 {+0.00102691, 0.00000000}, {+0.00073161, 0.00000000}, {-0.00000000, 0.00000000}, {-0.00095475, 0.00000000},
                 {-0.00175109, 0.00000000}, {-0.00197600, 0.00000000}, {-0.00137251, 0.00000000}, {+0.00000000, 0.00000000},
                 {+0.00171513, 0.00000000}, {+0.00308764, 0.00000000}, {+0.00342585, 0.00000000}, {+0.00234332, 0.00000000},
                 {+0.00000000, 0.00000000}, {-0.00285148, 0.00000000}, {-0.00507502, 0.00000000}, {-0.00557334, 0.00000000},
                 {-0.00377740, 0.00000000}, {+0.00000000, 0.00000000}, {+0.00452745, 0.00000000}, {+0.00800983, 0.00000000},
                 {+0.00875331, 0.00000000}, {+0.00591027, 0.00000000}, {-0.00000000, 0.00000000}, {-0.00705574, 0.00000000},
                 {-0.01248275, 0.00000000}, {-0.01366165, 0.00000000}, {-0.00925344, 0.00000000}, {-0.00000000, 0.00000000},
                 {+0.01118279, 0.00000000}, {+0.01997782, 0.00000000}, {+0.02214552, 0.00000000}, {+0.01524839, 0.00000000},
                 {+0.00000000, 0.00000000}, {-0.01932744, 0.00000000}, {-0.03572688, 0.00000000}, {-0.04138711, 0.00000000},
                 {-0.03019297, 0.00000000}, {-0.00000000, 0.00000000}, {+0.04610970, 0.00000000}, {+0.10010149, 0.00000000},
                 {+0.15082521, 0.00000000}, {+0.18693077, 0.00000000}, {+0.20000000, 0.00000000}, {+0.18693077, 0.00000000},
                 {+0.15082521, 0.00000000}, {+0.10010149, 0.00000000}, {+0.04610970, 0.00000000}, {-0.00000000, 0.00000000},
                 {-0.03019297, 0.00000000}, {-0.04138711, 0.00000000}, {-0.03572688, 0.00000000}, {-0.01932744, 0.00000000},
                 {+0.00000000, 0.00000000}, {+0.01524839, 0.00000000}, {+0.02214552, 0.00000000}, {+0.01997782, 0.00000000},
                 {+0.01118279, 0.00000000}, {-0.00000000, 0.00000000}, {-0.00925344, 0.00000000}, {-0.01366165, 0.00000000},
                 {-0.01248275, 0.00000000}, {-0.00705574, 0.00000000}, {-0.00000000, 0.00000000}, {+0.00591027, 0.00000000},
                 {+0.00875331, 0.00000000}, {+0.00800983, 0.00000000}, {+0.00452745, 0.00000000}, {+0.00000000, 0.00000000},
                 {-0.00377740, 0.00000000}, {-0.00557334, 0.00000000}, {-0.00507502, 0.00000000}, {-0.00285148, 0.00000000},
                 {+0.00000000, 0.00000000}, {+0.00234332, 0.00000000}, {+0.00342585, 0.00000000}, {+0.00308764, 0.00000000},
                 {+0.00171513, 0.00000000}, {+0.00000000, 0.00000000}, {-0.00137251, 0.00000000}, {-0.00197600, 0.00000000},
                 {-0.00175109, 0.00000000}, {-0.00095475, 0.00000000}, {-0.00000000, 0.00000000}, {+0.00073161, 0.00000000},
                 {+0.00102691, 0.00000000}, {+0.00088454, 0.00000000}, {+0.00046708, 0.00000000}, {+0.00000000, 0.00000000},
                 {-0.00033096, 0.00000000}, {-0.00044241, 0.00000000}, {-0.00035966, 0.00000000}, {-0.00017707, 0.00000000},
                 {-0.00000000, 0.00000000}             }
            } /* end of 1140000 sps -> 228000 sps */
        },
        /* INPUT_SAMPLE_RATE:         228000 sps
         * OUTPUT_SAMPLE_RATE:         45600 sps
         */
        {
            std::make_pair(228000, 45600),
            {1, /* up_rate */
             5, /* down_rate */
             std::vector<std::complex<float>>{
                 {-0.00000000, 0.00000000}, {-0.00017707, 0.00000000}, {-0.00035966, 0.00000000}, {-0.00044241, 0.00000000},
                 {-0.00033096, 0.00000000}, {+0.00000000, 0.00000000}, {+0.00046708, 0.00000000}, {+0.00088454, 0.00000000},
                 {+0.00102691, 0.00000000}, {+0.00073161, 0.00000000}, {-0.00000000, 0.00000000}, {-0.00095475, 0.00000000},
                 {-0.00175109, 0.00000000}, {-0.00197600, 0.00000000}, {-0.00137251, 0.00000000}, {+0.00000000, 0.00000000},
                 {+0.00171513, 0.00000000}, {+0.00308764, 0.00000000}, {+0.00342585, 0.00000000}, {+0.00234332, 0.00000000},
                 {+0.00000000, 0.00000000}, {-0.00285148, 0.00000000}, {-0.00507502, 0.00000000}, {-0.00557334, 0.00000000},
                 {-0.00377740, 0.00000000}, {+0.00000000, 0.00000000}, {+0.00452745, 0.00000000}, {+0.00800983, 0.00000000},
                 {+0.00875331, 0.00000000}, {+0.00591027, 0.00000000}, {-0.00000000, 0.00000000}, {-0.00705574, 0.00000000},
                 {-0.01248275, 0.00000000}, {-0.01366165, 0.00000000}, {-0.00925344, 0.00000000}, {-0.00000000, 0.00000000},
                 {+0.01118279, 0.00000000}, {+0.01997782, 0.00000000}, {+0.02214552, 0.00000000}, {+0.01524839, 0.00000000},
                 {+0.00000000, 0.00000000}, {-0.01932744, 0.00000000}, {-0.03572688, 0.00000000}, {-0.04138711, 0.00000000},
                 {-0.03019297, 0.00000000}, {-0.00000000, 0.00000000}, {+0.04610970, 0.00000000}, {+0.10010149, 0.00000000},
                 {+0.15082521, 0.00000000}, {+0.18693077, 0.00000000}, {+0.20000000, 0.00000000}, {+0.18693077, 0.00000000},
                 {+0.15082521, 0.00000000}, {+0.10010149, 0.00000000}, {+0.04610970, 0.00000000}, {-0.00000000, 0.00000000},
                 {-0.03019297, 0.00000000}, {-0.04138711, 0.00000000}, {-0.03572688, 0.00000000}, {-0.01932744, 0.00000000},
                 {+0.00000000, 0.00000000}, {+0.01524839, 0.00000000}, {+0.02214552, 0.00000000}, {+0.01997782, 0.00000000},
                 {+0.01118279, 0.00000000}, {-0.00000000, 0.00000000}, {-0.00925344, 0.00000000}, {-0.01366165, 0.00000000},
                 {-0.01248275, 0.00000000}, {-0.00705574, 0.00000000}, {-0.00000000, 0.00000000}, {+0.00591027, 0.00000000},
                 {+0.00875331, 0.00000000}, {+0.00800983, 0.00000000}, {+0.00452745, 0.00000000}, {+0.00000000, 0.00000000},
                 {-0.00377740, 0.00000000}, {-0.00557334, 0.00000000}, {-0.00507502, 0.00000000}, {-0.00285148, 0.00000000},
                 {+0.00000000, 0.00000000}, {+0.00234332, 0.00000000}, {+0.00342585, 0.00000000}, {+0.00308764, 0.00000000},
                 {+0.00171513, 0.00000000}, {+0.00000000, 0.00000000}, {-0.00137251, 0.00000000}, {-0.00197600, 0.00000000},
                 {-0.00175109, 0.00000000}, {-0.00095475, 0.00000000}, {-0.00000000, 0.00000000}, {+0.00073161, 0.00000000},
                 {+0.00102691, 0.00000000}, {+0.00088454, 0.00000000}, {+0.00046708, 0.00000000}, {+0.00000000, 0.00000000},
                 {-0.00033096, 0.00000000}, {-0.00044241, 0.00000000}, {-0.00035966, 0.00000000}, {-0.00017707, 0.00000000},
                 {-0.00000000, 0.00000000}             }
            } /* end of 228000 sps -> 45600 sps */
        }
    };
}

#endif // __FALCON_DSP_TRANSFORM_PREDEFINED_FIR_FILTER_H__
