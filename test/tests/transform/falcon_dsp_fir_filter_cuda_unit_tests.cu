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
 * @file     falcon_dsp_fir_filter_cuda_unit_tests.cu
 * @author   OrthogonalHawk
 * @date     24-Jan-2020
 *
 * @brief    Unit tests that exercise the FALCON DSP FIR filtering functions.
 *
 * @section  DESCRIPTION
 *
 * Implements a Google Test Framework based unit test suite for the FALCON DSP
 *  library functions.
 *
 * @section  HISTORY
 *
 * 24-Jan-2020  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <chrono>
#include <stdint.h>
#include <vector>

#include <gtest/gtest.h>

#include "transform/falcon_dsp_fir_filter_cuda.h"
#include "utilities/falcon_dsp_host_timer.h"
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

void run_cuda_fir_filter_test(std::string input_data_file_name,
                              std::string input_coeff_file_name,
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

    /* get the input coefficients */
    std::vector<std::complex<float>> coeffs;
    EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(input_coeff_file_name,
                                                        falcon_dsp::file_type_e::ASCII, coeffs));
    
    /* get the expected output data; only int16_t reading from file is supported at this time so
     *  it will need to be converted to std::complex<float> */
    std::vector<std::complex<int16_t>> expected_out_int16_data;
    EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(expected_output_file_name,
                                                        falcon_dsp::file_type_e::BINARY, expected_out_int16_data));
    std::vector<std::complex<float>> expected_out_data;
    for (auto out_iter = expected_out_int16_data.begin(); out_iter != expected_out_int16_data.end(); ++out_iter)
    {
        expected_out_data.push_back(std::complex<float>((*out_iter).real(), (*out_iter).imag()));
    }
    
    EXPECT_EQ(in_data.size(), expected_out_data.size());
    
    falcon_dsp::falcon_dsp_host_timer timer;
    
    /* now filter the input and verify that the calculated output
     *  matches the expected output */
    std::vector<std::complex<float>> out_data;
    EXPECT_TRUE(falcon_dsp::fir_filter_cuda(coeffs, in_data, out_data));
    
    timer.log_duration("Filtering Complete"); timer.reset();
    
    EXPECT_EQ(in_data.size(), out_data.size());
    
    for (uint32_t ii = 0; ii < in_data.size() && ii < out_data.size(); ++ii)
    {   
        float max_real_diff = abs(expected_out_data[ii].real()) * 0.01;
        if (max_real_diff < 10)
        {
            max_real_diff = 10;
        }
        
        float max_imag_diff = abs(expected_out_data[ii].imag()) * 0.01;
        if (max_imag_diff < 10)
        {
            max_imag_diff = 10;
        }
        
        ASSERT_NEAR(expected_out_data[ii].real(), out_data[ii].real(), max_real_diff);
        ASSERT_NEAR(expected_out_data[ii].imag(), out_data[ii].imag(), max_imag_diff);
    }
    
    timer.log_duration("Data Validated");
}

TEST(falcon_dsp_fir_filter, cuda_basic_filter_uint16_t_000)
{
    /*********************************************************
     * Test Vectors Derived from Python3:
     *
     * >>> x = [1, 1, 1]
     * >>> filter_coeffs = [.5, 2, 4]
     * >>> print([int(x) for x in signal.lfilter(filter_coeffs, 1.0, x)])
     * [0, 2, 6]
     ********************************************************/
    
    std::vector<std::complex<float>> coeffs = { {0.5, 0.0}, {2.0, 0.0}, {4.0, 0.0} };
    std::vector<std::complex<int16_t>> in_data(3);
    std::fill(in_data.begin(), in_data.end(), std::complex<int16_t>(1.0, 0.0));
    
    std::vector<std::complex<int16_t>> out_data;
    std::vector<std::complex<int16_t>> expected_out_data = { {0, 0}, {2, 0}, {6, 0} };
    
    falcon_dsp::falcon_dsp_fir_filter_cuda filter_obj(coeffs);
    EXPECT_TRUE(filter_obj.apply(in_data, out_data));
    EXPECT_EQ(out_data.size(), in_data.size());
    
    for (uint32_t ii = 0; ii < expected_out_data.size() && ii < out_data.size(); ++ii)
    {
        EXPECT_EQ(out_data[ii].real(), expected_out_data[ii].real());
        EXPECT_EQ(out_data[ii].imag(), expected_out_data[ii].imag());
    }
}

TEST(falcon_dsp_fir_filter, cuda_basic_filter_float_000)
{
    /*********************************************************
     * Test Vectors Derived from Python3:
     *
     * >>> x = [1, 1, 1]
     * >>> filter_coeffs = [.5, 2, 4]
     * >>> print(signal.lfilter(filter_coeffs, 1.0, x))
     * [ 0.5  2.5  6.5]
     ********************************************************/
    
    std::vector<std::complex<float>> coeffs = { {0.5, 0.0}, {2.0, 0.0}, {4.0, 0.0} };
    std::vector<std::complex<float>> in_data(3);
    std::fill(in_data.begin(), in_data.end(), std::complex<int16_t>(1.0, 0.0));
    
    std::vector<std::complex<float>> out_data;
    std::vector<std::complex<float>> expected_out_data = { {0.5, 0.0}, {2.5, 0.0}, {6.5, 0.0} };
    
    falcon_dsp::falcon_dsp_fir_filter_cuda filter_obj(coeffs);
    EXPECT_TRUE(filter_obj.apply(in_data, out_data));
    EXPECT_EQ(out_data.size(), in_data.size());
    
    for (uint32_t ii = 0; ii < expected_out_data.size() && ii < out_data.size(); ++ii)
    {
        EXPECT_EQ(out_data[ii].real(), expected_out_data[ii].real());
        EXPECT_EQ(out_data[ii].imag(), expected_out_data[ii].imag());
    }
}

TEST(falcon_dsp_fir_filter, cuda_basic_filter_float_001)
{
    /*********************************************************
     * Test Vectors Derived from Python3:
     *
     * >>> x = [1, 1, 1]
     * >>> filter_coeffs = [.5, 2, 4]
     * >>> print(signal.lfilter(filter_coeffs, 1.0, x))
     * [ 0.5  2.5  6.5 6.5 6.5 6.5 6.5 6.5]
     ********************************************************/
    
    std::vector<std::complex<float>> coeffs = { {0.5, 0.0}, {2.0, 0.0}, {4.0, 0.0} };
    std::vector<std::complex<float>> in_data(8);
    std::fill(in_data.begin(), in_data.end(), std::complex<int16_t>(1.0, 0.0));
    
    std::vector<std::complex<float>> out_data;
    std::vector<std::complex<float>> expected_out_data = { {0.5, 0.0}, {2.5, 0.0}, {6.5, 0.0}, {6.5, 0.0},
                                                           {6.5, 0.0}, {6.5, 0.0}, {6.5, 0.0}, {6.5, 0.0} };
    
    falcon_dsp::falcon_dsp_fir_filter_cuda filter_obj(coeffs);
    EXPECT_TRUE(filter_obj.apply(in_data, out_data));
    EXPECT_EQ(out_data.size(), in_data.size());
    
    for (uint32_t ii = 0; ii < expected_out_data.size() && ii < out_data.size(); ++ii)
    {
        EXPECT_EQ(out_data[ii].real(), expected_out_data[ii].real());
        EXPECT_EQ(out_data[ii].imag(), expected_out_data[ii].imag());
    }
}

TEST(falcon_dsp_fir_filter, cuda_basic_filter_float_002)
{
    /*********************************************************
     * Test Vectors Derived from Python3:
     *
     * >>> x = [1, 1, 1]
     * >>> filter_coeffs = [.5, 2, 4]
     * >>> print(signal.lfilter(filter_coeffs, 1.0, x))
     * [ 0.5  2.5  6.5 6.5 6.5 6.5 6.5 6.5]
     ********************************************************/
    
    std::vector<std::complex<float>> coeffs = { {0.5, 0.0}, {2.0, 0.0}, {4.0, 0.0} };
    std::vector<std::complex<float>> in_data(8);
    std::fill(in_data.begin(), in_data.end(), std::complex<int16_t>(1.0, 0.0));
    
    std::vector<std::complex<float>> out_data;
    std::vector<std::complex<float>> expected_out_data = { {0.5, 0.0}, {2.5, 0.0}, {6.5, 0.0}, {6.5, 0.0},
                                                           {6.5, 0.0}, {6.5, 0.0}, {6.5, 0.0}, {6.5, 0.0} };
    
    falcon_dsp::falcon_dsp_fir_filter_cuda filter_obj(coeffs);
    
    /* pass in each input value separately to verify that the filter object
     *  maintains state correctly */
    for (auto in_iter = in_data.begin(); in_iter != in_data.end(); ++in_iter)
    {
        std::vector<std::complex<float>> tmp_in_data = { *in_iter };
        std::vector<std::complex<float>> tmp_out_data;
        EXPECT_TRUE(filter_obj.apply(tmp_in_data, tmp_out_data));
        EXPECT_EQ(tmp_out_data.size(), 1u);
        out_data.push_back(*tmp_out_data.begin());
    }
    
    EXPECT_EQ(out_data.size(), in_data.size());
    
    for (uint32_t ii = 0; ii < expected_out_data.size() && ii < out_data.size(); ++ii)
    {
        EXPECT_EQ(out_data[ii].real(), expected_out_data[ii].real()) << " Bad Sample Idx: " << ii;
        EXPECT_EQ(out_data[ii].imag(), expected_out_data[ii].imag()) << " Bad Sample Idx: " << ii;
    }
}

TEST(falcon_dsp_fir_filter, cuda_basic_filter_float_003)
{
    /*********************************************************
     * Test Vectors Derived from Python3:
     *
     * >>> x = [1, 1, 1, 2, 2, 2, 5, 5, 5, 1, 1]
     * >>> filter_coeffs = [.5, 2, 4]
     * >>> print(signal.lfilter(filter_coeffs, 1.0, x))
     * [  0.5   2.5   6.5   7.    9.   13.   14.5  20.5  32.5  30.5  22.5]
     ********************************************************/
    
    std::vector<std::complex<float>> coeffs = { {0.5, 0.0}, {2.0, 0.0}, {4.0, 0.0} };
    std::vector<std::complex<float>> in_data = { {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {2.0, 0.0},
                                                 {2.0, 0.0}, {2.0, 0.0}, {5.0, 0.0}, {5.0, 0.0},
                                                 {5.0, 0.0}, {1.0, 0.0}, {1.0, 0.0} };
    
    std::vector<std::complex<float>> out_data;
    std::vector<std::complex<float>> expected_out_data = {  {0.5, 0.0},  {2.5, 0.0},  {6.5, 0.0},  {7.0, 0.0},
                                                            {9.0, 0.0}, {13.0, 0.0}, {14.5, 0.0}, {20.5, 0.0},
                                                           {32.5, 0.0}, {30.5, 0.0}, {22.5, 0.0} };
    
    falcon_dsp::falcon_dsp_fir_filter_cuda filter_obj(coeffs);
    EXPECT_TRUE(filter_obj.apply(in_data, out_data));
    EXPECT_EQ(out_data.size(), in_data.size());
    
    for (uint32_t ii = 0; ii < expected_out_data.size() && ii < out_data.size(); ++ii)
    {
        EXPECT_EQ(out_data[ii].real(), expected_out_data[ii].real());
        EXPECT_EQ(out_data[ii].imag(), expected_out_data[ii].imag());
    }
}

TEST(falcon_dsp_fir_filter, cuda_fir_filter_010)
{
    run_cuda_fir_filter_test("./vectors/test_010_x.bin",
                             "./vectors/test_010.filter_coeffs.txt",
                             "./vectors/test_010_y.bin");
}

TEST(falcon_dsp_fir_filter, cuda_fir_filter_011)
{
    run_cuda_fir_filter_test("./vectors/test_011_x.bin",
                             "./vectors/test_011.filter_coeffs.txt",
                             "./vectors/test_011_y.bin");
}
