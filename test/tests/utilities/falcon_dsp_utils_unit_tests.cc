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
 * @file     falcon_dsp_utils_unit_tests.cc
 * @author   OrthogonalHawk
 * @date     10-May-2019
 *
 * @brief    Unit tests that exercise various FALCON DSP library utility functions.
 *
 * @section  DESCRIPTION
 *
 * Implements a Google Test Framework based unit test suite for the FALCON DSP
 *  library functions.
 *
 * @section  HISTORY
 *
 * 10-May-2019  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <stdint.h>
#include <vector>

#include <gtest/gtest.h>

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
 *                           FUNCTION IMPLEMENTATION
 *****************************************************************************/

/******************************************************************************
 *                           UNIT TEST IMPLEMENTATION
 *****************************************************************************/

TEST(falcon_dsp_utils, gcd)
{
    uint64_t val = 0;
    
    /* Python 3.6.7 **********************************************************
     * >>> import math
     * >>> math.gcd(200, 100)
     * 100
     *************************************************************************/
    val = falcon_dsp::calculate_gcd(200, 100);
    EXPECT_EQ(val, 100);
    
    /* Python 3.6.7 **********************************************************
     * >>> import math
     * >>> math.gcd(840, 210)
     * 210
     *************************************************************************/
    val = falcon_dsp::calculate_gcd(840, 210);
    EXPECT_EQ(val, 210);
    
    /* Python 3.6.7 **********************************************************
     * >>> import math
     * >>> math.gcd(2857, 97)  # prime numbers
     * 1
     *************************************************************************/
    val = falcon_dsp::calculate_gcd(2857, 97);
    EXPECT_EQ(val, 1);
}
