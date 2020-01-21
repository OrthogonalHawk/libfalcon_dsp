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
 * @file     falcon_dsp_polar_discriminator_cpp_unit_tests.cc
 * @author   OrthogonalHawk
 * @date     21-Jan-2020
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
 * 21-Jan-2020  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <chrono>
#include <stdint.h>
#include <vector>

#include <gtest/gtest.h>

#include "transform/falcon_dsp_polar_discriminator.h"
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

void run_cpp_polar_discriminator_test(std::string input_data_file_name,
                                      std::string expected_output_file_name)
{
    /* get the input data; only int16_t reading from file is supported at this time so it will need
     *  to be converted to std::complex<float> */
    std::vector<std::complex<int16_t>> in_int16_data;
    EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(input_data_file_name,
                                                        falcon_dsp::file_type_e::BINARY, in_int16_data));
    std::vector<std::complex<float>> in_data;
    for (auto in_iter = in_int16_data.begin(); in_iter != in_int16_data.end(); ++in_iter)
    {
        in_data.push_back(std::complex<float>((*in_iter).real(), (*in_iter).imag()));
    }
    
    std::cout << "Read " << in_data.size() << " samples from " << input_data_file_name << std::endl;
    
    /* get the expected output data; only int16_t reading from file is supported at this time so
     *  it will need to be converted to std::complex<float> */
    std::vector<int16_t> expected_out_int16_data;
    EXPECT_TRUE(falcon_dsp::read_data_from_file(expected_output_file_name,
                                                falcon_dsp::file_type_e::BINARY, expected_out_int16_data));
    std::vector<float> expected_out_data;
    for (auto out_iter = expected_out_int16_data.begin(); out_iter != expected_out_int16_data.end(); ++out_iter)
    {
        expected_out_data.push_back(static_cast<float>(*out_iter));
    }
    
    EXPECT_EQ(in_data.size(), expected_out_data.size());
    
    auto start = std::chrono::high_resolution_clock::now();
    
    /* now apply the polar discriminator and verify that the calculated output
     *  matches the expected output */
    std::vector<float> out_data;
    EXPECT_TRUE(falcon_dsp::polar_discriminator(in_data, out_data));
    
    auto done = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = done - start;
    
    std::cout << "Elapsed time (in milliseconds): " << duration_ms.count() << std::endl;
    
    EXPECT_EQ(in_data.size(), out_data.size());
    
    for (uint32_t ii = 0; ii < in_data.size() && ii < out_data.size(); ++ii)
    {   
        float max_diff = abs(expected_out_data[ii]) * 0.01;
        if (max_diff < 10)
        {
            max_diff = 10;
        }
        
        ASSERT_NEAR(expected_out_data[ii], out_data[ii], max_diff);
    }
}

TEST(falcon_dsp_polar_discriminator, cpp_basic_polar_discriminator_float_000)
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
    
    falcon_dsp::falcon_dsp_polar_discriminator polar_discrim_obj;
    EXPECT_TRUE(polar_discrim_obj.apply(in_data, out_data));
    EXPECT_EQ(out_data.size(), expected_out_data.size());
    
    for (uint32_t ii = 0; ii < expected_out_data.size() && ii < out_data.size(); ++ii)
    {
        EXPECT_NEAR(out_data[ii], expected_out_data[ii], 0.0001);
    }
}

TEST(falcon_dsp_polar_discriminator, cpp_basic_polar_discriminator_int16_t_000)
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
    
    falcon_dsp::falcon_dsp_polar_discriminator polar_discrim_obj;
    EXPECT_TRUE(polar_discrim_obj.apply(in_data, out_data));
    EXPECT_EQ(out_data.size(), expected_out_data.size());
    
    for (uint32_t ii = 0; ii < expected_out_data.size() && ii < out_data.size(); ++ii)
    {
        EXPECT_NEAR(out_data[ii], expected_out_data[ii], 0.0001);
    }
}

/*
TEST(falcon_dsp_linear_filter, cpp_linear_filter_010)
{
    run_cpp_linear_filter_test("./vectors/test_010_x.bin",
                               "./vectors/test_010.filter_coeffs.txt",
                               "./vectors/test_010_y.bin");
}

TEST(falcon_dsp_linear_filter, cpp_linear_filter_011)
{
    run_cpp_linear_filter_test("./vectors/test_011_x.bin",
                               "./vectors/test_011.filter_coeffs.txt",
                               "./vectors/test_011_y.bin");
} */