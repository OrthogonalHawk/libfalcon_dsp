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
 * @file     falcon_dsp_math_unit_tests.cc
 * @author   OrthogonalHawk
 * @date     19-Apr-2019
 *
 * @brief    Unit tests that exercise various FALCON DSP library math functions.
 *
 * @section  DESCRIPTION
 *
 * Implements a Google Test Framework based unit test suite for the FALCON DSP
 *  library functions.
 *
 * @section  HISTORY
 *
 * 19-Apr-2019  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <stdint.h>
#include <vector>

#include <gtest/gtest.h>

#include "math/falcon_dsp_math.h"

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

TEST(falcon_dsp_math_add, cpp_add_func)
{
    const uint32_t NUM_ELEMENTS = 1e6;
    
    std::vector<uint32_t> vec_a;
    std::vector<uint32_t> vec_b;
    vec_a.reserve(NUM_ELEMENTS);
    vec_b.reserve(NUM_ELEMENTS);
    
    std::vector<uint32_t> vec_sum;
    vec_sum.reserve(NUM_ELEMENTS);
    
    for (uint32_t ii = 0; ii < NUM_ELEMENTS; ++ii)
    {
        vec_a.push_back(1);
        vec_b.push_back(2);
    }
    
    falcon_dsp::add_vector<uint32_t>(vec_a, vec_b, vec_sum);
    
    EXPECT_EQ(vec_sum.size(), NUM_ELEMENTS);
    for (uint32_t ii = 0; ii < NUM_ELEMENTS && ii < vec_sum.size(); ++ii)
    {
        EXPECT_EQ(vec_sum[ii], 3);   
    }
}

TEST(falcon_dsp_math_add, cuda_add_func)
{
    const uint32_t NUM_ELEMENTS = 1e6;
    
    std::vector<uint32_t> vec_a;
    std::vector<uint32_t> vec_b;
    vec_a.reserve(NUM_ELEMENTS);
    vec_b.reserve(NUM_ELEMENTS);
    
    std::vector<uint32_t> vec_sum;
    vec_sum.reserve(NUM_ELEMENTS);
    
    for (uint32_t ii = 0; ii < NUM_ELEMENTS; ++ii)
    {
        vec_a.push_back(1);
        vec_b.push_back(2);
    }
    
    falcon_dsp::add_vector_cuda<uint32_t>(vec_a, vec_b, vec_sum);
    
    EXPECT_EQ(vec_sum.size(), NUM_ELEMENTS);
    for (uint32_t ii = 0; ii < NUM_ELEMENTS && ii < vec_sum.size(); ++ii)
    {
        EXPECT_EQ(vec_sum[ii], 3);   
    }
}
