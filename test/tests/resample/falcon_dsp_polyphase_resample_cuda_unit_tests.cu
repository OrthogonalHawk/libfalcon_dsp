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
 * @file     falcon_dsp_polyphase_resample_cuda_unit_tests.cc
 * @author   OrthogonalHawk
 * @date     03-Sep-2019
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
 * 03-Sep-2019  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <chrono>
#include <stdint.h>
#include <vector>

#include <gtest/gtest.h>

#include "resample/falcon_dsp_resample_cuda.h"
#include "resample/falcon_dsp_polyphase_resampler_cuda.h"
#include "utilities/falcon_dsp_host_timer.h"
#include "utilities/falcon_dsp_utils.h"

/******************************************************************************
 *                                 CONSTANTS
 *****************************************************************************/

const uint32_t ALLOWED_SHORT_OUT_SAMPLE_MULTIPLIER = 5;

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

void cuda_convert_complex_int16_to_float(std::vector<std::complex<int16_t>> &in,
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

void run_cuda_resample_test(std::string input_data_file_name, std::string input_filter_coeff_file_name,
                            std::string expected_output_file_name,
                            uint32_t input_sample_rate_in_sps, uint32_t output_sample_rate_in_sps)
{
    /* get the data to resample; note that it must be read from the file as int16_t
     *  and then converted to float */
    std::vector<std::complex<int16_t>> in_data_from_file;
    EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(input_data_file_name,
                                                        falcon_dsp::file_type_e::BINARY, in_data_from_file));
    std::vector<std::complex<float>> in_data;
    cuda_convert_complex_int16_to_float(in_data_from_file, in_data);
    
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
    cuda_convert_complex_int16_to_float(expected_out_data_from_file, expected_out_data);
    
    std::cout << "Read " << expected_out_data.size() << " input files from " << expected_output_file_name << std::endl;
    
    uint32_t filter_delay = falcon_dsp::calculate_filter_delay_from_sample_rates(filter_coeffs.size(), input_sample_rate_in_sps, output_sample_rate_in_sps);
    
    falcon_dsp::falcon_dsp_host_timer timer;
    
    /* now resample the input and verify that the calculated output
     *  matches the expected output */
    std::vector<std::complex<float>> out_data;
    EXPECT_TRUE(falcon_dsp::resample_cuda(input_sample_rate_in_sps, in_data, filter_coeffs,
                                          output_sample_rate_in_sps, out_data));
    
    timer.log_duration("Filtering Complete"); timer.reset();
    
    std::cout << "Resampled output has " << out_data.size() << " samples" << std::endl;
    EXPECT_TRUE(filter_delay < expected_out_data.size());
    EXPECT_TRUE(expected_out_data.size() >= out_data.size());
    EXPECT_TRUE(out_data.size() > (expected_out_data.size() - filter_delay * ALLOWED_SHORT_OUT_SAMPLE_MULTIPLIER));
    
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

        ASSERT_NEAR(expected_out_data[ii].real(), out_data[ii].real(), 2.0) << " Error at index: " << ii;
        ASSERT_NEAR(expected_out_data[ii].imag(), out_data[ii].imag(), 2.0) << " Error at index: " << ii;
    }
    
    timer.log_duration("Data Validated");
}

TEST(falcon_dsp_resample_cuda, basic_cuda_resample_001)
{
    /*********************************************************
     * Test Vectors Derived from Python3:
     *
     * >>> x = [1, 1, 1, 1, 1, 1, 1]
     * >>> h = [0.5, 1.0, 2.0]
     * >>> print(signal.upfirdn(h, x, up=1, down=2))
     * [ 0.5  3.5  3.5  3.5  2. ]
     ********************************************************/
    
    std::vector<std::complex<float>> coeffs = { {0.5, 0.0}, {1.0, 0.0}, {2.0, 0.0} };
    falcon_dsp::falcon_dsp_polyphase_resampler_cuda resampler(1, 2, coeffs);
    
    std::vector<std::complex<float>> in_data = { {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0},
                                                 {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0} };
    std::vector<std::complex<float>> expected_out_data = { {0.5, 0.0}, {3.5, 0.0}, {3.5, 0.0}, {3.5, 0.0}, {2.0, 0.0} };
    
    uint32_t filter_delay = falcon_dsp::calculate_filter_delay_from_sample_rates(coeffs.size(), 10, 5);
    std::cout << "Computed filter delay of " << filter_delay << " samples" << std::endl;
    
    std::vector<std::complex<float>> out_data;
    EXPECT_TRUE(resampler.apply(in_data, out_data));
    EXPECT_TRUE(out_data.size() > (expected_out_data.size() - filter_delay * ALLOWED_SHORT_OUT_SAMPLE_MULTIPLIER));
    
    for (uint32_t ii = 0; ii < expected_out_data.size() && ii < out_data.size(); ++ii)
    {
        ASSERT_NEAR(expected_out_data[ii].real(), out_data[ii].real(), 0.1) << " Error at index: " << ii;
        ASSERT_NEAR(expected_out_data[ii].imag(), out_data[ii].imag(), 0.1) << " Error at index: " << ii;
    }
}

TEST(falcon_dsp_resample_cuda, basic_cuda_resample_002)
{
    /*********************************************************
     * Test Vectors Derived from Python3:
     *
     * >>> x = [1] * 1000
     * >>> h = [0.5, 1.0, 2.0]
     * >>> print(signal.upfirdn(h, x, up=1, down=2))
     * [ 0.5  3.5  3.5  3.5 ... 3.5 ]
     ********************************************************/
    
    std::vector<std::complex<float>> coeffs = { {0.5, 0.0}, {1.0, 0.0}, {2.0, 0.0} };
    falcon_dsp::falcon_dsp_polyphase_resampler_cuda resampler(1, 2, coeffs);
    
    std::vector<std::complex<float>> in_data(1000, std::complex<float>(1.0, 0.0));
    std::vector<std::complex<float>> expected_out_data(501, std::complex<float>(3.5, 0.0));
    expected_out_data[0] = std::complex<float>(0.5, 0.0);
    
    uint32_t filter_delay = falcon_dsp::calculate_filter_delay_from_sample_rates(coeffs.size(), 10, 5);
    std::cout << "Computed filter delay of " << filter_delay << " samples" << std::endl;
    
    std::vector<std::complex<float>> out_data;
    EXPECT_TRUE(resampler.apply(in_data, out_data));
    EXPECT_TRUE(out_data.size() > (expected_out_data.size() - filter_delay * ALLOWED_SHORT_OUT_SAMPLE_MULTIPLIER));
    
    for (uint32_t ii = 0; ii < expected_out_data.size() && ii < out_data.size(); ++ii)
    {
        ASSERT_NEAR(expected_out_data[ii].real(), out_data[ii].real(), 0.1) << " Error at index: " << ii;
        ASSERT_NEAR(expected_out_data[ii].imag(), out_data[ii].imag(), 0.1) << " Error at index: " << ii;
    }
}

TEST(falcon_dsp_resample_cuda, basic_cuda_resample_003)
{
    /*********************************************************
     * Test Vectors Derived from Python3:
     *
     * >>> x = [1] * 1000
     * >>> h = [0.5, 0.5, 1.0, 0.3, 0.07, 0.1, 0.4]
     * >>> print(signal.upfirdn(h, x, up=1, down=2))
     * [ 0.5   2.    2.37  2.87  2.87 ... 2.87  2.37  0.87  0.5 ]
     ********************************************************/
    
    std::vector<std::complex<float>> coeffs = { {0.5,  0.0}, {0.5, 0.0}, {1.0, 0.0}, {0.3, 0.0},
                                                {0.07, 0.0}, {0.1, 0.0}, {0.4, 0.0} };
    falcon_dsp::falcon_dsp_polyphase_resampler_cuda resampler(1, 2, coeffs);
    
    std::vector<std::complex<float>> in_data(1000, std::complex<float>(1.0, 0.0));
    std::vector<std::complex<float>> expected_out_data(503, std::complex<float>(2.87, 0.0));
    expected_out_data[0] = std::complex<float>(0.5, 0.0);
    expected_out_data[1] = std::complex<float>(2.0, 0.0);
    expected_out_data[2] = std::complex<float>(2.37, 0.0);
    expected_out_data[501] = std::complex<float>(2.37, 0.0);
    expected_out_data[501] = std::complex<float>(0.87, 0.0);
    expected_out_data[502] = std::complex<float>(0.5, 0.0);
    
    uint32_t filter_delay = falcon_dsp::calculate_filter_delay_from_sample_rates(coeffs.size(), 10, 5);
    std::cout << "Computed filter delay of " << filter_delay << " samples" << std::endl;
    
    std::vector<std::complex<float>> out_data;
    EXPECT_TRUE(resampler.apply(in_data, out_data));
    EXPECT_TRUE(out_data.size() > (expected_out_data.size() - filter_delay * ALLOWED_SHORT_OUT_SAMPLE_MULTIPLIER));
    
    for (uint32_t ii = 0; ii < expected_out_data.size() && ii < out_data.size(); ++ii)
    {
        ASSERT_NEAR(expected_out_data[ii].real(), out_data[ii].real(), 0.1) << " Error at index: " << ii;
        ASSERT_NEAR(expected_out_data[ii].imag(), out_data[ii].imag(), 0.1) << " Error at index: " << ii;
    }
}

TEST(falcon_dsp_resample_cuda, basic_cuda_resample_004)
{
    /*********************************************************
     * Test Vectors Derived from Python3:
     *
     * >>> x = [1] * 10
     * >>> h = [0.5, 1.0, 2.0]
     * >>> print(signal.upfirdn(h, x, up=4, down=5))
     * [ 0.5  1.   2.   0.   0.5  1.   2.   0. ]
     ********************************************************/
    
    std::vector<std::complex<float>> coeffs = { {0.5, 0.0}, {1.0, 0.0}, {2.0, 0.0} };
    falcon_dsp::falcon_dsp_polyphase_resampler_cuda resampler(4, 5, coeffs);
    
    std::vector<std::complex<float>> in_data(10, std::complex<float>(1.0, 0.0));
    std::vector<std::complex<float>> expected_out_data = { {0.5, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {0.0, 0.0},
                                                           {0.5, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {0.0, 0.0} };
    
    uint32_t filter_delay = falcon_dsp::calculate_filter_delay_from_sample_rates(coeffs.size(), 40, 50);
    std::cout << "Computed filter delay of " << filter_delay << " samples" << std::endl;
    
    std::vector<std::complex<float>> out_data;
    EXPECT_TRUE(resampler.apply(in_data, out_data));
    EXPECT_TRUE(out_data.size() >= (expected_out_data.size() - filter_delay * ALLOWED_SHORT_OUT_SAMPLE_MULTIPLIER));
    
    for (uint32_t ii = 0; ii < expected_out_data.size() && ii < out_data.size(); ++ii)
    {
        ASSERT_NEAR(expected_out_data[ii].real(), out_data[ii].real(), 0.1) << " Error at index: " << ii;
        ASSERT_NEAR(expected_out_data[ii].imag(), out_data[ii].imag(), 0.1) << " Error at index: " << ii;
    }
}

TEST(falcon_dsp_resample_cuda, cuda_resample_004)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_004_x.bin";
    std::string IN_FILT_COEFF_FILE_NAME = "vectors/test_004.filter_coeffs.txt";
    std::string OUT_TEST_FILE_NAME = "vectors/test_004_y.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const uint32_t OUTPUT_SAMPLE_RATE_IN_SPS = 5e5;
    
    run_cuda_resample_test(IN_TEST_FILE_NAME, IN_FILT_COEFF_FILE_NAME, OUT_TEST_FILE_NAME,
                           INPUT_SAMPLE_RATE_IN_SPS, OUTPUT_SAMPLE_RATE_IN_SPS);
}

TEST(falcon_dsp_resample_cuda, cuda_resample_005)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_005_x.bin";
    std::string IN_FILT_COEFF_FILE_NAME = "vectors/test_005.filter_coeffs.txt";
    std::string OUT_TEST_FILE_NAME = "vectors/test_005_y.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const uint32_t OUTPUT_SAMPLE_RATE_IN_SPS = 6e5;
    
    run_cuda_resample_test(IN_TEST_FILE_NAME, IN_FILT_COEFF_FILE_NAME, OUT_TEST_FILE_NAME,
                           INPUT_SAMPLE_RATE_IN_SPS, OUTPUT_SAMPLE_RATE_IN_SPS);
}

TEST(falcon_dsp_resample_cuda, cuda_resample_006)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_006_x.bin";
    std::string IN_FILT_COEFF_FILE_NAME = "vectors/test_006.filter_coeffs.txt";
    std::string OUT_TEST_FILE_NAME = "vectors/test_006_y.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const uint32_t OUTPUT_SAMPLE_RATE_IN_SPS = 2e6;
    
    run_cuda_resample_test(IN_TEST_FILE_NAME, IN_FILT_COEFF_FILE_NAME, OUT_TEST_FILE_NAME,
                           INPUT_SAMPLE_RATE_IN_SPS, OUTPUT_SAMPLE_RATE_IN_SPS);
}

TEST(falcon_dsp_resample_cuda, cuda_resample_007)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_007_x.bin";
    std::string IN_FILT_COEFF_FILE_NAME = "vectors/test_007.filter_coeffs.txt";
    std::string OUT_TEST_FILE_NAME = "vectors/test_007_y.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const uint32_t OUTPUT_SAMPLE_RATE_IN_SPS = 800e3;
    
    run_cuda_resample_test(IN_TEST_FILE_NAME, IN_FILT_COEFF_FILE_NAME, OUT_TEST_FILE_NAME,
                           INPUT_SAMPLE_RATE_IN_SPS, OUTPUT_SAMPLE_RATE_IN_SPS);
}

TEST(falcon_dsp_resample_cuda, cuda_resample_008)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_008_x.bin";
    std::string IN_FILT_COEFF_FILE_NAME = "vectors/test_008.filter_coeffs.txt";
    std::string OUT_TEST_FILE_NAME = "vectors/test_008_y.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const uint32_t OUTPUT_SAMPLE_RATE_IN_SPS = 450e3;
    
    run_cuda_resample_test(IN_TEST_FILE_NAME, IN_FILT_COEFF_FILE_NAME, OUT_TEST_FILE_NAME,
                           INPUT_SAMPLE_RATE_IN_SPS, OUTPUT_SAMPLE_RATE_IN_SPS);
}

TEST(falcon_dsp_resample_cuda, cuda_resample_009)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_009_x.bin";
    std::string IN_FILT_COEFF_FILE_NAME = "vectors/test_009.filter_coeffs.txt";
    std::string OUT_TEST_FILE_NAME = "vectors/test_009_y.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const uint32_t OUTPUT_SAMPLE_RATE_IN_SPS = 44e3;
    
    run_cuda_resample_test(IN_TEST_FILE_NAME, IN_FILT_COEFF_FILE_NAME, OUT_TEST_FILE_NAME,
                           INPUT_SAMPLE_RATE_IN_SPS, OUTPUT_SAMPLE_RATE_IN_SPS);
}