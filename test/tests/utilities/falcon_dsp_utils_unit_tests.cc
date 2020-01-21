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
 * 04-Jun-2019  OrthogonalHawk  Added floating-point ASCII file read/write tests.
 * 21-Jan-2020  OrthogonalHawk  Added tests for non-complex int16_t and float
 *                               read/write functions.
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

TEST(falcon_dsp_utils, factorial)
{
    uint64_t val = 0;
    
    /* Python 3.6.7 **********************************************************
     * >>> import math
     * >>> math.factorial(0)
     * 1
     *************************************************************************/
    val = falcon_dsp::factorial(0);
    EXPECT_EQ(val, 1);
    
    /* Python 3.6.7 **********************************************************
     * >>> import math
     * >>> math.factorial(1)
     * 1
     *************************************************************************/
    val = falcon_dsp::factorial(1);
    EXPECT_EQ(val, 1);
    
    /* Python 3.6.7 **********************************************************
     * >>> import math
     * >>> math.factorial(3)
     * 6
     *************************************************************************/
    val = falcon_dsp::factorial(3);
    EXPECT_EQ(val, 6);
    
    /* Python 3.6.7 **********************************************************
     * >>> import math
     * >>> math.factorial(12)
     * 479001600
     *************************************************************************/
    val = falcon_dsp::factorial(12);
    EXPECT_EQ(val, 479001600);
}

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
     * >>> math.gcd(200, 600)
     * 200
     *************************************************************************/
    val = falcon_dsp::calculate_gcd(200, 600);
    EXPECT_EQ(val, 200);
    
    /* Python 3.6.7 **********************************************************
     * >>> import math
     * >>> math.gcd(600, 200)
     * 200
     *************************************************************************/
    val = falcon_dsp::calculate_gcd(600, 200);
    EXPECT_EQ(val, 200);
    
    /* Python 3.6.7 **********************************************************
     * >>> import math
     * >>> math.gcd(2857, 97)  # prime numbers
     * 1
     *************************************************************************/
    val = falcon_dsp::calculate_gcd(2857, 97);
    EXPECT_EQ(val, 1);
}

TEST(falcon_dsp_utils, lcm)
{
    uint64_t val = 0;
    
    val = falcon_dsp::calculate_lcm(382, 164);
    EXPECT_EQ(val, 31324);
    
    val = falcon_dsp::calculate_lcm(600, 200);
    EXPECT_EQ(val, 600);
    
    val = falcon_dsp::calculate_lcm(200, 600);
    EXPECT_EQ(val, 600);
}

TEST(falcon_dsp_utils, rat_approx)
{
    int64_t num, denom;

    /* 'vanilla' case */
    falcon_dsp::rat_approx(1.0, 1, num, denom);
    EXPECT_EQ(num, 1);
    EXPECT_EQ(denom, 1);

    /* check case where max_denom is too small */
    falcon_dsp::rat_approx(0.5, 1, num, denom);
    EXPECT_EQ(num, 0);
    EXPECT_EQ(denom, 1);

    /* check case where max_denom is larger */
    falcon_dsp::rat_approx(0.5, 10, num, denom);
    EXPECT_EQ(num, 1);
    EXPECT_EQ(denom, 2);

    /* 1/3 = 0.33333 */
    falcon_dsp::rat_approx(0.33333, 100, num, denom);
    EXPECT_EQ(num, 1);
    EXPECT_EQ(denom, 3);

    /* 2/7 = 0.28571 */
    falcon_dsp::rat_approx(0.28571, 100, num, denom);
    EXPECT_EQ(num, 2);
    EXPECT_EQ(denom, 7);

    /* 5/8 = 0.625 */
    falcon_dsp::rat_approx(0.625, 100, num, denom);
    EXPECT_EQ(num, 5);
    EXPECT_EQ(denom, 8);

    /* 30720000 / 40000000 = 0.768 */
    falcon_dsp::rat_approx(0.768, 100, num, denom);
    EXPECT_EQ(num, 53);
    EXPECT_EQ(denom, 69);

    /* 40000000 / 30720000 = 1.30208 */
    falcon_dsp::rat_approx(1.30208, 100, num, denom);
    EXPECT_EQ(num, 125);
    EXPECT_EQ(denom, 96);
}

TEST(falcon_dsp_utils, file_write_and_read_00)
{
    std::string TEST_FILE_NAME = "vectors/tmp_file.bin";
    
    std::vector<std::complex<int16_t>> out_data = { {0, 0}, {1, 0}, {0, 1}, {1, 1} };
    EXPECT_TRUE(falcon_dsp::write_complex_data_to_file(TEST_FILE_NAME, falcon_dsp::file_type_e::BINARY, out_data));
    
    std::vector<std::complex<int16_t>> in_data;
    EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(TEST_FILE_NAME, falcon_dsp::file_type_e::BINARY, in_data));
    EXPECT_EQ(out_data.size(), in_data.size());
    
    for (uint32_t ii = 0; ii < out_data.size() && ii < in_data.size(); ++ii)
    {
        EXPECT_EQ(out_data[ii].real(), in_data[ii].real());
        EXPECT_EQ(out_data[ii].imag(), in_data[ii].imag());
    }
}

TEST(falcon_dsp_utils, file_write_and_read_01)
{
    std::string TEST_FILE_NAME = "vectors/tmp_file.bin";
    
    std::vector<std::complex<int16_t>> out_data = { {0, 0}, {-1, 0}, {0, -1}, {-1, -1} };
    EXPECT_TRUE(falcon_dsp::write_complex_data_to_file(TEST_FILE_NAME, falcon_dsp::file_type_e::BINARY, out_data));
    
    std::vector<std::complex<int16_t>> in_data;
    EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(TEST_FILE_NAME, falcon_dsp::file_type_e::BINARY, in_data));
    EXPECT_EQ(out_data.size(), in_data.size());
    
    for (uint32_t ii = 0; ii < out_data.size() && ii < in_data.size(); ++ii)
    {
        EXPECT_EQ(out_data[ii].real(), in_data[ii].real());
        EXPECT_EQ(out_data[ii].imag(), in_data[ii].imag());
    }
}

TEST(falcon_dsp_utils, file_write_and_read_02)
{
    std::string TEST_FILE_NAME = "vectors/tmp_file.txt";
    
    std::vector<std::complex<int16_t>> out_data = { { 0, 0}, {1,  0}, { 0,  1}, {1, 1},
                                                    {-1, 0}, {0, -1}, {-1, -1}, {0, 0} };
    EXPECT_TRUE(falcon_dsp::write_complex_data_to_file(TEST_FILE_NAME, falcon_dsp::file_type_e::ASCII, out_data));
    
    std::vector<std::complex<int16_t>> in_data;
    EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(TEST_FILE_NAME, falcon_dsp::file_type_e::ASCII, in_data));
    EXPECT_EQ(out_data.size(), in_data.size());
    
    for (uint32_t ii = 0; ii < out_data.size() && ii < in_data.size(); ++ii)
    {
        EXPECT_EQ(out_data[ii].real(), in_data[ii].real());
        EXPECT_EQ(out_data[ii].imag(), in_data[ii].imag());
    }
}

TEST(falcon_dsp_utils, file_write_and_read_03)
{
    std::string TEST_FILE_NAME = "vectors/tmp_file.txt";
    
    std::vector<std::complex<float>> out_data = { { 0.1, 0.2}, {1.3,  0.4}, { 0.5,  1.6}, {1.7, 1.8},
                                                  {-1.9, 0.0}, {0.1, -1.2}, {-1.3, -1.3}, {0.4, 0.5} };
    EXPECT_TRUE(falcon_dsp::write_complex_data_to_file(TEST_FILE_NAME,
                                                       falcon_dsp::file_type_e::ASCII, out_data));
    
    std::vector<std::complex<float>> in_data;
    EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(TEST_FILE_NAME,
                                                        falcon_dsp::file_type_e::ASCII, in_data));
    EXPECT_EQ(out_data.size(), in_data.size());
    
    for (uint32_t ii = 0; ii < out_data.size() && ii < in_data.size(); ++ii)
    {
        EXPECT_EQ(out_data[ii].real(), in_data[ii].real());
        EXPECT_EQ(out_data[ii].imag(), in_data[ii].imag());
    }
}

TEST(falcon_dsp_utils, file_write_and_read_04)
{
    std::string TEST_FILE_NAME = "vectors/test_000_x.bin";
    
    std::vector<std::complex<int16_t>> in_data;
    EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(TEST_FILE_NAME,
                                                        falcon_dsp::file_type_e::BINARY, in_data));
    EXPECT_EQ(1e5, in_data.size());
    
    for (uint32_t ii = 0; ii < in_data.size(); ++ii)
    {
        EXPECT_EQ(ii % 128, in_data[ii].real());
        EXPECT_EQ(-1 * (ii % 128), in_data[ii].imag());
    }
}

TEST(falcon_dsp_utils, file_write_and_read_05)
{
    std::string TEST_FILE_NAME = "vectors/test_000_y.bin";
    
    std::vector<std::complex<int16_t>> in_data;
    EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(TEST_FILE_NAME,
                                                        falcon_dsp::file_type_e::BINARY, in_data));
    EXPECT_EQ(10020, in_data.size());
    
    for (uint32_t ii = 0; ii < in_data.size(); ++ii)
    {
        EXPECT_EQ(ii % 128, in_data[ii].real());
        EXPECT_EQ(-1 * (ii % 128), in_data[ii].imag());
    }
}

TEST(falcon_dsp_utils, file_write_and_read_06)
{
    std::string TEST_FILE_NAME = "vectors/file_read_write_test_06.bin";
    
    std::vector<int16_t> data_to_write;
    for (int16_t ii = -128; ii < 128; ++ii)
    {
        data_to_write.push_back(ii);
    }
    
    EXPECT_TRUE(falcon_dsp::write_data_to_file(TEST_FILE_NAME, falcon_dsp::file_type_e::BINARY, data_to_write));
    
    std::vector<int16_t> read_from_file;
    EXPECT_TRUE(falcon_dsp::read_data_from_file(TEST_FILE_NAME, falcon_dsp::file_type_e::BINARY, read_from_file));
    
    EXPECT_EQ(data_to_write.size(), read_from_file.size());
    for (uint32_t ii = 0; ii < data_to_write.size() && ii < read_from_file.size(); ++ii)
    {
        EXPECT_EQ(read_from_file[ii], data_to_write[ii]);
    }
}

TEST(falcon_dsp_utils, file_write_and_read_07)
{
    std::string TEST_FILE_NAME = "vectors/file_read_write_test_07.txt";
    
    std::vector<int16_t> data_to_write;
    for (int16_t ii = -128; ii < 128; ++ii)
    {
        data_to_write.push_back(ii);
    }
    
    EXPECT_TRUE(falcon_dsp::write_data_to_file(TEST_FILE_NAME, falcon_dsp::file_type_e::ASCII, data_to_write));
    
    std::vector<int16_t> read_from_file;
    EXPECT_TRUE(falcon_dsp::read_data_from_file(TEST_FILE_NAME, falcon_dsp::file_type_e::ASCII, read_from_file));
    
    EXPECT_EQ(data_to_write.size(), read_from_file.size());
    for (uint32_t ii = 0; ii < data_to_write.size() && ii < read_from_file.size(); ++ii)
    {
        EXPECT_EQ(read_from_file[ii], data_to_write[ii]);
    }
}

TEST(falcon_dsp_utils, file_write_and_read_08)
{
    std::string TEST_FILE_NAME = "vectors/file_read_write_test_08.txt";
    
    std::vector<float> data_to_write;
    for (int16_t ii = -128.0; ii < 128.0; ++ii)
    {
        data_to_write.push_back(ii + 0.5);
    }
    
    EXPECT_TRUE(falcon_dsp::write_data_to_file(TEST_FILE_NAME, falcon_dsp::file_type_e::ASCII, data_to_write));
    
    std::vector<float> read_from_file;
    EXPECT_TRUE(falcon_dsp::read_data_from_file(TEST_FILE_NAME, falcon_dsp::file_type_e::ASCII, read_from_file));
    
    EXPECT_EQ(data_to_write.size(), read_from_file.size());
    for (uint32_t ii = 0; ii < data_to_write.size() && ii < read_from_file.size(); ++ii)
    {
        EXPECT_EQ(read_from_file[ii], data_to_write[ii]);
    }
}
