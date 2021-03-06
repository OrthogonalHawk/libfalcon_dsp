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
 * @file     falcon_dsp_predefined_fir_filter.cc
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

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <complex>
#include <map>
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
     *                                 CONSTANTS
     *****************************************************************************/

    /* auto-generated set of predefined filter coefficients. these coefficients were
     *  generated by the 'generate_predefined_fir_filter_coeffs.ipynb' Python Jupyter
     *  notebook in the test/utilities folder.
     */
    std::map<predefined_resample_filter_key_s, predefined_resample_filter_params_s> s_predefined_resample_fir_coeffs = {
    AUTO_GENERATED_COEFFICIENTS_HERE
    };
}
