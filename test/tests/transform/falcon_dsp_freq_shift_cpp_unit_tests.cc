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
 * @file     falcon_dsp_freq_shift_cpp_unit_tests.cc
 * @author   OrthogonalHawk
 * @date     08-Jun-2019
 *
 * @brief    Unit tests that exercise the FALCON DSP frequency shift functions.
 *
 * @section  DESCRIPTION
 *
 * Implements a Google Test Framework based unit test suite for the FALCON DSP
 *  library functions.
 *
 * @section  HISTORY
 *
 * 08-Jun-2019  OrthogonalHawk  File created.
 * 20-Jan-2020  OrthogonalHawk  Renamed file to include 'cpp' substring.
 * 12-Feb-2020  OrthogonalHawk  Updated to use 'initialize' method.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <chrono>
#include <stdint.h>
#include <vector>

#include <gtest/gtest.h>

#include "transform/falcon_dsp_freq_shift.h"
#include "utilities/falcon_dsp_host_timer.h"
#include "utilities/falcon_dsp_utils.h"

/******************************************************************************
 *                                 CONSTANTS
 *****************************************************************************/

const float MIN_ALLOWED_DIFF = 2048.0 * 0.02;

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

void run_cpp_freq_shift_test(std::string input_file_name, std::string expected_output_file_name,
                             uint32_t input_sample_rate_in_sps, int32_t freq_shift_in_hz)
{
    std::vector<std::complex<int16_t>> in_data;
    EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(input_file_name,
                                                        falcon_dsp::file_type_e::BINARY, in_data));
    
    std::cout << "Read " << in_data.size() << " samples from " << input_file_name << std::endl;

    std::vector<std::complex<int16_t>> expected_out_data;
    EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(expected_output_file_name,
                                                        falcon_dsp::file_type_e::BINARY, expected_out_data));
    
    EXPECT_EQ(in_data.size(), expected_out_data.size());
    
    falcon_dsp::falcon_dsp_host_timer timer;
    
    /* now frequency shift the input and verify that the calculated output
     *  matches the expected output */
    std::vector<std::complex<int16_t>> out_data;
    EXPECT_TRUE(falcon_dsp::freq_shift(input_sample_rate_in_sps, in_data,
                freq_shift_in_hz, out_data));
    
    timer.log_duration("Shifting Complete"); timer.reset();
    
    EXPECT_EQ(in_data.size(), out_data.size());
    
    for (uint32_t ii = 0; ii < in_data.size() && ii < out_data.size(); ++ii)
    {        
        ASSERT_NEAR(expected_out_data[ii].real(), out_data[ii].real(), MIN_ALLOWED_DIFF) << " failure at index " << ii;
        ASSERT_NEAR(expected_out_data[ii].imag(), out_data[ii].imag(), MIN_ALLOWED_DIFF) << " failure at index " << ii;
    }
    
    timer.log_duration("Data Validated");
}

TEST(falcon_dsp_freq_shift, cpp_freq_shift_001)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_001_x.bin";
    std::string OUT_TEST_FILE_NAME = "vectors/test_001_y.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const int32_t  FREQ_SHIFT_IN_HZ = 1e5;
    
    run_cpp_freq_shift_test(IN_TEST_FILE_NAME, OUT_TEST_FILE_NAME,
                            INPUT_SAMPLE_RATE_IN_SPS, FREQ_SHIFT_IN_HZ);
}

TEST(falcon_dsp_freq_shift, cpp_freq_shift_002)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_002_x.bin";
    std::string OUT_TEST_FILE_NAME = "vectors/test_002_y.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const int32_t  FREQ_SHIFT_IN_HZ = 1e5;
    
    run_cpp_freq_shift_test(IN_TEST_FILE_NAME, OUT_TEST_FILE_NAME,
                            INPUT_SAMPLE_RATE_IN_SPS, FREQ_SHIFT_IN_HZ);
}

TEST(falcon_dsp_freq_shift, cpp_freq_shift_003)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_003_x.bin";
    std::string OUT_TEST_FILE_NAME = "vectors/test_003_y.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const int32_t  FREQ_SHIFT_IN_HZ = -5000;
    
    run_cpp_freq_shift_test(IN_TEST_FILE_NAME, OUT_TEST_FILE_NAME,
                            INPUT_SAMPLE_RATE_IN_SPS, FREQ_SHIFT_IN_HZ);
}

TEST(falcon_dsp_freq_shift, cpp_freq_shift_012_0)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_012_x.bin";
    std::string OUT_TEST_FILE_NAME = "vectors/test_012_y_shift_23000_hz.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const int32_t  FREQ_SHIFT_IN_HZ = 23000;
    
    run_cpp_freq_shift_test(IN_TEST_FILE_NAME, OUT_TEST_FILE_NAME,
                            INPUT_SAMPLE_RATE_IN_SPS, FREQ_SHIFT_IN_HZ);
}

TEST(falcon_dsp_freq_shift, cpp_freq_shift_012_1)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_012_x.bin";
    std::string OUT_TEST_FILE_NAME = "vectors/test_012_y_shift_-370400_hz.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const int32_t  FREQ_SHIFT_IN_HZ = -370400;
    
    run_cpp_freq_shift_test(IN_TEST_FILE_NAME, OUT_TEST_FILE_NAME,
                            INPUT_SAMPLE_RATE_IN_SPS, FREQ_SHIFT_IN_HZ);
}

TEST(falcon_dsp_freq_shift, cpp_freq_shift_013_0)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_013_x.bin";
    std::string OUT_TEST_FILE_BASE_NAME = "vectors/test_013_y_shift_264200_hz.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const int32_t  FREQ_SHIFT_IN_HZ = 264200;
    
    run_cpp_freq_shift_test(IN_TEST_FILE_NAME, OUT_TEST_FILE_BASE_NAME,
                            INPUT_SAMPLE_RATE_IN_SPS, FREQ_SHIFT_IN_HZ);
}
