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
 * @file     falcon_dsp_polar_discriminator_cuda_unit_tests.cu
 * @author   OrthogonalHawk
 * @date     20-Feb-2020
 *
 * @brief    Unit tests that exercise the FALCON DSP polar discriminator functions.
 *
 * @section  DESCRIPTION
 *
 * Implements a Google Test Framework based unit test suite for the FALCON DSP
 *  library functions.
 *
 * @section  HISTORY
 *
 * 20-Feb-2020  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <chrono>
#include <stdint.h>
#include <vector>

#include <gtest/gtest.h>

#include "transform/falcon_dsp_polar_discriminator_cuda.h"
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

TEST(falcon_dsp_polar_discriminator, cuda_basic_polar_discriminator_float_000)
{
    /*********************************************************
     * Test Vectors Derived from Python3:
     *
     * >>> a = np.random.randint(-100, 100, 5)
     * >>> b = np.random.randint(-100, 100, 5)
     * >>> c = a + 1j * b
     * >>> discrim = np.angle(c[1:] * np.conj(c[:-1]))
     * >>> print(c)
     * [ 88.-85.j  22.+97.j  54.+91.j   6.-13.j -29.+58.j]
     * >>> print(discrim)
     * [ 2.11582422 -0.31252633 -2.17362758 -3.11035282]
     ********************************************************/
    
    std::vector<std::complex<float>> in_data = { {88.0, -85.0}, { 22.0, 97.0}, {54.0, 91.0},
                                                 { 6.0, -13.0}, {-29.0, 58.0} };
    
    std::vector<float> out_data;
    std::vector<float> expected_out_data = { 2.11582422, -0.31252633, -2.17362758, -3.11035282 };
    
    falcon_dsp::falcon_dsp_polar_discriminator_cuda polar_discrim_obj;
    EXPECT_TRUE(polar_discrim_obj.apply(in_data, out_data));
    EXPECT_EQ(out_data.size(), expected_out_data.size());
    
    for (uint32_t ii = 0; ii < expected_out_data.size() && ii < out_data.size(); ++ii)
    {
        ASSERT_NEAR(out_data[ii], expected_out_data[ii], 0.0001) << " Error at idx " << ii;
    }
}

TEST(falcon_dsp_polar_discriminator, cuda_basic_polar_discriminator_int16_t_000)
{
    /*********************************************************
     * Test Vectors Derived from Python3:
     *
     * >>> a = np.random.randint(-100, 100, 5)
     * >>> b = np.random.randint(-100, 100, 5)
     * >>> c = a + 1j * b
     * >>> discrim = np.angle(c[1:] * np.conj(c[:-1]))
     * >>> print(c)
     * [ 88.-85.j  22.+97.j  54.+91.j   6.-13.j -29.+58.j]
     * >>> print(discrim)
     * [ 2.11582422 -0.31252633 -2.17362758 -3.11035282]
     ********************************************************/
    
    std::vector<std::complex<int16_t>> in_data = { {88, -85}, { 22, 97}, {54, 91},
                                                   { 6, -13}, {-29, 58} };
    
    std::vector<float> out_data;
    std::vector<float> expected_out_data = { 2.11582422, -0.31252633, -2.17362758, -3.11035282 };
    
    falcon_dsp::falcon_dsp_polar_discriminator_cuda polar_discrim_obj;
    EXPECT_TRUE(polar_discrim_obj.apply(in_data, out_data));
    EXPECT_EQ(out_data.size(), expected_out_data.size());
    
    for (uint32_t ii = 0; ii < expected_out_data.size() && ii < out_data.size(); ++ii)
    {
        ASSERT_NEAR(out_data[ii], expected_out_data[ii], 0.0001) << "Error at idx " << ii;
    }
}
