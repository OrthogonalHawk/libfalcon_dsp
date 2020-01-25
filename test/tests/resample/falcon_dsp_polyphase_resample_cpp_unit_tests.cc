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
 * @file     falcon_dsp_polyphase_resample_cpp_unit_tests.cc
 * @author   OrthogonalHawk
 * @date     25-Aug-2019
 *
 * @brief    Unit tests that exercise the FALCON DSP polyphase resampling functions.
 *
 * @section  DESCRIPTION
 *
 * Implements a Google Test Framework based unit test suite for the FALCON DSP
 *  library functions.
 *
 * @section  HISTORY
 *
 * 25-Aug-2019  OrthogonalHawk  File created.
 * 20-Jan-2020  OrthogonalHawk  Adding cpp_resample_009 and updating to reflect
 *                               library refactoring. Also renamed to include 'cpp'
 *                               substring in file name.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <chrono>
#include <stdint.h>
#include <vector>

#include <gtest/gtest.h>

#include "resample/falcon_dsp_resample.h"
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

void convert_complex_int16_to_float(std::vector<std::complex<int16_t>> &in,
                                    std::vector<std::complex<float>> &out)
{
    /* clear the output and reserve sufficient size for the new data */
    out.clear();
    out.reserve(in.size());
    
    /* copy in each element */
    for (uint32_t ii = 0; ii < in.size(); ++ii)
    {
        out.push_back(std::complex<float>(in[ii].real(), in[ii].imag()));
    }
}

void run_cpp_resample_test(std::string input_data_file_name, std::string input_filter_coeff_file_name,
                           std::string expected_output_file_name,
                           uint32_t input_sample_rate_in_sps, uint32_t output_sample_rate_in_sps)
{
    /* get the data to resample; note that it must be read from the file as int16_t
     *  and then converted to float */
    std::vector<std::complex<int16_t>> in_data_from_file;
    EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(input_data_file_name,
                                                        falcon_dsp::file_type_e::BINARY, in_data_from_file));
    std::vector<std::complex<float>> in_data;
    convert_complex_int16_to_float(in_data_from_file, in_data);
    
    std::cout << "Read " << in_data.size() << " input samples from " << input_data_file_name << std::endl;
    
    /* get the filter coefficients */
    std::vector<std::complex<float>> filter_coeffs;
    EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(input_filter_coeff_file_name,
                                                        falcon_dsp::file_type_e::ASCII, filter_coeffs));
    
    /* get the expected output data; note that it must be read from the file as int16_t
     *  and then converted to float */
    std::vector<std::complex<int16_t>> expected_out_data_from_file;
    EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(expected_output_file_name,
                                                        falcon_dsp::file_type_e::BINARY, expected_out_data_from_file));
    
    std::vector<std::complex<float>> expected_out_data;
    convert_complex_int16_to_float(expected_out_data_from_file, expected_out_data);
    
    std::cout << "Read " << expected_out_data.size() << " input files from " << expected_output_file_name << std::endl;
    
    uint32_t filter_delay = falcon_dsp::calculate_filter_delay_from_sample_rates(filter_coeffs.size(), input_sample_rate_in_sps, output_sample_rate_in_sps);
    std::cout << "Computed filter delay of " << filter_delay << " samples" << std::endl;
    
    falcon_dsp::falcon_dsp_host_timer timer;
    
    /* now resample the input and verify that the calculated output
     *  matches the expected output */
    std::vector<std::complex<float>> out_data;
    EXPECT_TRUE(falcon_dsp::resample(input_sample_rate_in_sps, in_data, filter_coeffs,
                                     output_sample_rate_in_sps, out_data));
    
    timer.log_duration("Filtering Complete"); timer.reset();
    
    std::cout << "Resampled output has " << out_data.size() << " samples" << std::endl;
    EXPECT_TRUE(filter_delay < expected_out_data.size());
    EXPECT_TRUE(expected_out_data.size() >= out_data.size());
    EXPECT_TRUE(out_data.size() > (expected_out_data.size() - filter_delay * 3));
    
    for (uint32_t ii = filter_delay; ii < expected_out_data.size() && ii < out_data.size(); ++ii)
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

        EXPECT_NEAR(expected_out_data[ii].real(), out_data[ii].real(), max_real_diff);
        EXPECT_NEAR(expected_out_data[ii].imag(), out_data[ii].imag(), max_imag_diff);
    }
    
    timer.log_duration("Data Validated");
}

TEST(falcon_dsp_resample, cpp_resample_004)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_004_x.bin";
    std::string IN_FILT_COEFF_FILE_NAME = "vectors/test_004.filter_coeffs.txt";
    std::string OUT_TEST_FILE_NAME = "vectors/test_004_y.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const uint32_t OUTPUT_SAMPLE_RATE_IN_SPS = 5e5;
    
    run_cpp_resample_test(IN_TEST_FILE_NAME, IN_FILT_COEFF_FILE_NAME, OUT_TEST_FILE_NAME,
                          INPUT_SAMPLE_RATE_IN_SPS, OUTPUT_SAMPLE_RATE_IN_SPS);
}

TEST(falcon_dsp_resample, cpp_resample_005)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_005_x.bin";
    std::string IN_FILT_COEFF_FILE_NAME = "vectors/test_005.filter_coeffs.txt";
    std::string OUT_TEST_FILE_NAME = "vectors/test_005_y.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const uint32_t OUTPUT_SAMPLE_RATE_IN_SPS = 6e5;
    
    run_cpp_resample_test(IN_TEST_FILE_NAME, IN_FILT_COEFF_FILE_NAME, OUT_TEST_FILE_NAME,
                          INPUT_SAMPLE_RATE_IN_SPS, OUTPUT_SAMPLE_RATE_IN_SPS);
}

TEST(falcon_dsp_resample, cpp_resample_006)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_006_x.bin";
    std::string IN_FILT_COEFF_FILE_NAME = "vectors/test_006.filter_coeffs.txt";
    std::string OUT_TEST_FILE_NAME = "vectors/test_006_y.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const uint32_t OUTPUT_SAMPLE_RATE_IN_SPS = 2e6;
    
    run_cpp_resample_test(IN_TEST_FILE_NAME, IN_FILT_COEFF_FILE_NAME, OUT_TEST_FILE_NAME,
                          INPUT_SAMPLE_RATE_IN_SPS, OUTPUT_SAMPLE_RATE_IN_SPS);
}

TEST(falcon_dsp_resample, cpp_resample_007)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_007_x.bin";
    std::string IN_FILT_COEFF_FILE_NAME = "vectors/test_007.filter_coeffs.txt";
    std::string OUT_TEST_FILE_NAME = "vectors/test_007_y.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const uint32_t OUTPUT_SAMPLE_RATE_IN_SPS = 800e3;
    
    run_cpp_resample_test(IN_TEST_FILE_NAME, IN_FILT_COEFF_FILE_NAME, OUT_TEST_FILE_NAME,
                          INPUT_SAMPLE_RATE_IN_SPS, OUTPUT_SAMPLE_RATE_IN_SPS);
}

TEST(falcon_dsp_resample, cpp_resample_008)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_008_x.bin";
    std::string IN_FILT_COEFF_FILE_NAME = "vectors/test_008.filter_coeffs.txt";
    std::string OUT_TEST_FILE_NAME = "vectors/test_008_y.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const uint32_t OUTPUT_SAMPLE_RATE_IN_SPS = 450e3;
    
    run_cpp_resample_test(IN_TEST_FILE_NAME, IN_FILT_COEFF_FILE_NAME, OUT_TEST_FILE_NAME,
                          INPUT_SAMPLE_RATE_IN_SPS, OUTPUT_SAMPLE_RATE_IN_SPS);
}

TEST(falcon_dsp_resample, cpp_resample_009)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_009_x.bin";
    std::string IN_FILT_COEFF_FILE_NAME = "vectors/test_009.filter_coeffs.txt";
    std::string OUT_TEST_FILE_NAME = "vectors/test_009_y.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const uint32_t OUTPUT_SAMPLE_RATE_IN_SPS = 44e3;
    
    run_cpp_resample_test(IN_TEST_FILE_NAME, IN_FILT_COEFF_FILE_NAME, OUT_TEST_FILE_NAME,
                          INPUT_SAMPLE_RATE_IN_SPS, OUTPUT_SAMPLE_RATE_IN_SPS);
}
