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
 * @file     falcon_dsp_freq_shift_cuda_unit_tests.cu
 * @author   OrthogonalHawk
 * @date     09-Jun-2019
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
 * 09-Jun-2019  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <stdint.h>
#include <vector>

#include <gtest/gtest.h>

#include "transform/falcon_dsp_freq_shift_cuda.h"
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

void run_cuda_freq_shift_test(std::string input_file_name, std::string expected_output_file_name,
                              uint32_t input_sample_rate_in_sps, int32_t freq_shift_in_hz)
{
    std::vector<std::complex<int16_t>> tmp_in_data;
    EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(input_file_name,
                                                        falcon_dsp::file_type_e::BINARY, tmp_in_data));
    
    std::vector<std::complex<float>> in_data;
    for (auto in_iter : tmp_in_data)
    {
        in_data.push_back(std::complex<float>(in_iter.real(), in_iter.imag()));
    }
    
    
    std::cout << "Read " << in_data.size() << " samples from " << input_file_name << std::endl;

    std::vector<std::complex<int16_t>> tmp_expected_out_data;
    EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(expected_output_file_name,
                                                        falcon_dsp::file_type_e::BINARY, tmp_expected_out_data));
    std::vector<std::complex<float>> expected_out_data;
    for (auto out_iter : tmp_expected_out_data)
    {
        expected_out_data.push_back(std::complex<float>(out_iter.real(), out_iter.imag()));
    }
    
    EXPECT_EQ(in_data.size(), expected_out_data.size());
    
    falcon_dsp::falcon_dsp_host_timer timer;

    /* now frequency shift the input and verify that the calculated output
     *  matches the expected output */
    std::vector<std::complex<float>> out_data;
    EXPECT_TRUE(falcon_dsp::freq_shift_cuda(input_sample_rate_in_sps, in_data,
                                            freq_shift_in_hz, out_data));
   
    timer.log_duration("Shifting Complete"); timer.reset();

    EXPECT_EQ(in_data.size(), out_data.size());
    
    for (uint32_t ii = 0; ii < in_data.size() && ii < out_data.size(); ++ii)
    {
        ASSERT_NEAR(expected_out_data[ii].real(), out_data[ii].real(), abs(expected_out_data[ii]) * 0.01) << " failure at index " << ii;
        ASSERT_NEAR(expected_out_data[ii].imag(), out_data[ii].imag(), abs(expected_out_data[ii]) * 0.01) << " failure at index " << ii;
    }
    
    timer.log_duration("Data Validated");
}

void run_cuda_multi_chan_freq_shift_test(std::string input_file_name,
                                         std::string expected_output_file_base_name,
                                         uint32_t input_sample_rate_in_sps,
                                         std::vector<int32_t> freq_shift_channels)
{
    /* read the input data from file and convert to std::complex<float> */
    std::vector<std::complex<int16_t>> tmp_in_data;
    EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(input_file_name,
                                                        falcon_dsp::file_type_e::BINARY, tmp_in_data));
    
    std::vector<std::complex<float>> in_data;
    for (auto in_iter : tmp_in_data)
    {
        in_data.push_back(std::complex<float>(in_iter.real(), in_iter.imag()));
    }
    std::cout << "Read " << in_data.size() << " samples from " << input_file_name << std::endl;

    
    /* read in the expected output data file(s) */
    std::vector<std::vector<std::complex<float>>> expected_out_data;
    for (auto freq_shift : freq_shift_channels)
    {
        std::stringstream ss;
        ss << expected_output_file_base_name << freq_shift << "_hz.bin";
        
        std::cout << "Reading expected output data from " << ss.str() << std::endl;
        
        std::vector<std::complex<int16_t>> tmp_expected_out_data;
        EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(ss.str(),
                                                            falcon_dsp::file_type_e::BINARY, tmp_expected_out_data));
        std::vector<std::complex<float>> expected_out_chan_data;
        for (auto out_iter : tmp_expected_out_data)
        {
            expected_out_chan_data.push_back(std::complex<float>(out_iter.real(), out_iter.imag()));
        }
        
        EXPECT_EQ(in_data.size(), expected_out_chan_data.size());
        expected_out_data.push_back(expected_out_chan_data);
    }
    

    /* now frequency shift the input and verify that the calculated output
     *  matches the expected output */
    falcon_dsp::falcon_dsp_host_timer timer;
    std::vector<std::vector<std::complex<float>>> out_data;
    EXPECT_TRUE(falcon_dsp::freq_shift_cuda(input_sample_rate_in_sps, in_data,
                                            freq_shift_channels, out_data));
   
    timer.log_duration("Shifting Complete"); timer.reset();

    for (auto out_iter : out_data)
    {
        EXPECT_EQ(in_data.size(), out_iter.size());
    }
    
    for (uint32_t out_idx = 0; out_idx < expected_out_data.size() && out_idx < out_data.size(); ++out_idx)
    {
        for (uint32_t ii = 0; ii < in_data.size() && ii < expected_out_data[out_idx].size(); ++ii)
        {
            ASSERT_NEAR(expected_out_data[out_idx][ii].real(), expected_out_data[out_idx][ii].real(),
                        abs(expected_out_data[out_idx][ii]) * 0.01) << " chan[" << out_idx << "] failure at index " << ii;
            
            ASSERT_NEAR(expected_out_data[out_idx][ii].imag(), expected_out_data[out_idx][ii].imag(),
                        abs(expected_out_data[out_idx][ii]) * 0.01) << " chan[" << out_idx << "] failure at index " << ii;
        }
    }
    
    timer.log_duration("Data Validated");
}

void run_cuda_multi_chan_freq_shift_by_segment_test(std::string input_file_name,
                                                    std::string expected_output_file_base_name,
                                                    uint32_t input_sample_rate_in_sps,
                                                    std::vector<int32_t> freq_shift_channels,
                                                    uint32_t segment_size_in_samples)
{
    /* read the input data from file and convert to std::complex<float> */
    std::vector<std::complex<int16_t>> tmp_in_data;
    EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(input_file_name,
                                                        falcon_dsp::file_type_e::BINARY, tmp_in_data));
    
    std::vector<std::complex<float>> in_data;
    for (auto in_iter : tmp_in_data)
    {
        in_data.push_back(std::complex<float>(in_iter.real(), in_iter.imag()));
    }
    std::cout << "Read " << in_data.size() << " samples from " << input_file_name << std::endl;

    
    /* read in the expected output data file(s) */
    std::vector<std::vector<std::complex<float>>> expected_out_data;
    for (auto freq_shift : freq_shift_channels)
    {
        std::stringstream ss;
        ss << expected_output_file_base_name << freq_shift << "_hz.bin";
        
        std::cout << "Reading expected output data from " << ss.str() << std::endl;
        
        std::vector<std::complex<int16_t>> tmp_expected_out_data;
        EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(ss.str(),
                                                            falcon_dsp::file_type_e::BINARY, tmp_expected_out_data));
        std::vector<std::complex<float>> expected_out_chan_data;
        for (auto out_iter : tmp_expected_out_data)
        {
            expected_out_chan_data.push_back(std::complex<float>(out_iter.real(), out_iter.imag()));
        }
        
        EXPECT_EQ(in_data.size(), expected_out_chan_data.size());
        expected_out_data.push_back(expected_out_chan_data);
    }
    

    /* now frequency shift the input in multiple segments and verify that the combined
     *  calculated output matches the expected output */
    falcon_dsp::falcon_dsp_freq_shift_cuda shifter(input_sample_rate_in_sps, freq_shift_channels);
    std::vector<std::vector<std::complex<float>>> out_data;
    out_data.resize(freq_shift_channels.size());
    
    falcon_dsp::falcon_dsp_host_timer timer;
    for (uint32_t segment_start_idx = 0; segment_start_idx < in_data.size(); segment_start_idx += segment_size_in_samples)
    {
        std::vector<std::complex<float>> segment_in;
        for (uint32_t in_sample_idx = segment_start_idx;
             in_sample_idx < (segment_start_idx + segment_size_in_samples) && in_sample_idx < in_data.size();
             ++in_sample_idx)
        {
            segment_in.push_back(in_data[in_sample_idx]);   
        }

        std::vector<std::vector<std::complex<float>>> segment_out_data;
        EXPECT_TRUE(shifter.apply(segment_in, segment_out_data));
        
        for (uint32_t out_vec_idx = 0; out_vec_idx < segment_out_data.size(); ++out_vec_idx)
        {
            for (uint32_t out_sample_idx = 0; out_sample_idx < segment_out_data[out_vec_idx].size(); ++out_sample_idx)
            {
                out_data[out_vec_idx].push_back(segment_out_data[out_vec_idx][out_sample_idx]);
            }
        }
    }
   
    timer.log_duration("Shifting Complete"); timer.reset();

    for (auto out_iter : out_data)
    {
        EXPECT_EQ(in_data.size(), out_iter.size());
    }
    
    for (uint32_t out_idx = 0; out_idx < expected_out_data.size() && out_idx < out_data.size(); ++out_idx)
    {
        for (uint32_t ii = 0; ii < in_data.size() && ii < expected_out_data[out_idx].size(); ++ii)
        {
            ASSERT_NEAR(expected_out_data[out_idx][ii].real(), expected_out_data[out_idx][ii].real(),
                        abs(expected_out_data[out_idx][ii]) * 0.01) << " chan[" << out_idx << "] failure at index " << ii;
            
            ASSERT_NEAR(expected_out_data[out_idx][ii].imag(), expected_out_data[out_idx][ii].imag(),
                        abs(expected_out_data[out_idx][ii]) * 0.01) << " chan[" << out_idx << "] failure at index " << ii;
        }
    }
    
    timer.log_duration("Data Validated");
}

TEST(falcon_dsp_freq_shift, cuda_freq_shift_001)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_001_x.bin";
    std::string OUT_TEST_FILE_NAME = "vectors/test_001_y.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const int32_t  FREQ_SHIFT_IN_HZ = 1e5;
    
    run_cuda_freq_shift_test(IN_TEST_FILE_NAME, OUT_TEST_FILE_NAME,
                             INPUT_SAMPLE_RATE_IN_SPS, FREQ_SHIFT_IN_HZ);
}

TEST(falcon_dsp_freq_shift, cuda_freq_shift_002)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_002_x.bin";
    std::string OUT_TEST_FILE_NAME = "vectors/test_002_y.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const int32_t  FREQ_SHIFT_IN_HZ = 1e5;
    
    run_cuda_freq_shift_test(IN_TEST_FILE_NAME, OUT_TEST_FILE_NAME,
                             INPUT_SAMPLE_RATE_IN_SPS, FREQ_SHIFT_IN_HZ);
}

TEST(falcon_dsp_freq_shift, cuda_freq_shift_003)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_003_x.bin";
    std::string OUT_TEST_FILE_NAME = "vectors/test_003_y.bin";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    const int32_t  FREQ_SHIFT_IN_HZ = -5000;
    
    run_cuda_freq_shift_test(IN_TEST_FILE_NAME, OUT_TEST_FILE_NAME,
                             INPUT_SAMPLE_RATE_IN_SPS, FREQ_SHIFT_IN_HZ);
}

TEST(falcon_dsp_freq_shift, cuda_multi_chan_freq_shift_012)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_012_x.bin";
    std::string OUT_TEST_FILE_BASE_NAME = "vectors/test_012_y_shift_";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    std::vector<int32_t> freq_shifts = { -281375, 79297 };
    
    run_cuda_multi_chan_freq_shift_test(IN_TEST_FILE_NAME,
                                        OUT_TEST_FILE_BASE_NAME,
                                        INPUT_SAMPLE_RATE_IN_SPS,
                                        freq_shifts);
}

TEST(falcon_dsp_freq_shift, cuda_multi_chan_freq_shift_013)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_013_x.bin";
    std::string OUT_TEST_FILE_BASE_NAME = "vectors/test_013_y_shift_";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    std::vector<int32_t> freq_shifts = { 57982, 239457, 426534, 499857, 463877 };
    
    run_cuda_multi_chan_freq_shift_test(IN_TEST_FILE_NAME,
                                        OUT_TEST_FILE_BASE_NAME,
                                        INPUT_SAMPLE_RATE_IN_SPS,
                                        freq_shifts);
}

TEST(falcon_dsp_freq_shift, cuda_multi_chan_freq_shift_014)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_014_x.bin";
    std::string OUT_TEST_FILE_BASE_NAME = "vectors/test_014_y_shift_";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    std::vector<int32_t> freq_shifts = { -141451, 307008, 54623, 445497, 71758, 141718, 114510, 162143, 78135, 118135 };
    
    run_cuda_multi_chan_freq_shift_test(IN_TEST_FILE_NAME,
                                        OUT_TEST_FILE_BASE_NAME,
                                        INPUT_SAMPLE_RATE_IN_SPS,
                                        freq_shifts);
}

TEST(falcon_dsp_freq_shift, cuda_multi_chan_freq_shift_by_segment_014)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_014_x.bin";
    std::string OUT_TEST_FILE_BASE_NAME = "vectors/test_014_y_shift_";
    const uint32_t SEGMENT_SIZE_IN_SAMPLES = 250000 * 5;
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    std::vector<int32_t> freq_shifts = { -141451, 307008, 54623, 445497, 71758, 141718, 114510, 162143, 78135, 118135 };
    
    run_cuda_multi_chan_freq_shift_by_segment_test(IN_TEST_FILE_NAME,
                                                   OUT_TEST_FILE_BASE_NAME,
                                                   INPUT_SAMPLE_RATE_IN_SPS,
                                                   freq_shifts, SEGMENT_SIZE_IN_SAMPLES);
}
