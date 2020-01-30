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
 * @file     falcon_dsp_multi_rate_channelizer_cpp_unit_tests.cc
 * @author   OrthogonalHawk
 * @date     29-Jan-2020
 *
 * @brief    Unit tests that exercise the FALCON DSP C++ multi-rate channelizer.
 *
 * @section  DESCRIPTION
 *
 * Implements a Google Test Framework based unit test suite for the FALCON DSP
 *  library functions.
 *
 * @section  HISTORY
 *
 * 29-Jan-2020  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <stdint.h>
#include <vector>

#include <gtest/gtest.h>

#include "transform/falcon_dsp_multi_rate_channelizer.h"
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

void run_cpp_multi_rate_channelizer_test(std::string input_file_name,
                                         std::string resample_coeffs_file_base_name,
                                         std::string expected_output_file_base_name,
                                         uint32_t input_sample_rate_in_sps,
                                         std::vector<uint32_t> sample_rates,
                                         std::vector<std::pair<uint32_t, uint32_t>> up_down_rates,
                                         std::vector<int32_t> freq_shifts)
{
    /* sanity check inputs */
    ASSERT_EQ(sample_rates.size(), freq_shifts.size());
    ASSERT_EQ(sample_rates.size(), up_down_rates.size());
    ASSERT_GT(sample_rates.size(), 0u);
    
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
    for (auto freq_shift : freq_shifts)
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

        expected_out_data.push_back(expected_out_chan_data);
    }
    
    
    /* read in the coefficient files and create the multi-rate channelizer
     *  configuration vector */
    std::vector<multi_rate_channelizer_channel_s> channels;
    for (uint32_t chan_idx = 0; chan_idx < sample_rates.size(); ++chan_idx)
    {
        std::stringstream ss;
        ss << resample_coeffs_file_base_name << freq_shifts[chan_idx] << "_hz.txt";
        
        std::cout << "Reading resample coeffs from " << ss.str() << std::endl;
        
        std::vector<std::complex<float>> filter_coeffs;
        EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(ss.str(),
                                                            falcon_dsp::file_type_e::ASCII, filter_coeffs));
        
        multi_rate_channelizer_channel_s chan_params;
        chan_params.output_sample_rate_in_sps = sample_rates[chan_idx];
        chan_params.freq_shift_in_hz = freq_shifts[chan_idx];
        chan_params.up_rate = up_down_rates[chan_idx].first;
        chan_params.down_rate = up_down_rates[chan_idx].second;
        chan_params.resample_filter_coeffs = filter_coeffs;
        
        channels.push_back(chan_params);
    }
    
    /* now frequency shift and resample the input */
    falcon_dsp::falcon_dsp_multi_rate_channelizer channelizer;
    ASSERT_TRUE(channelizer.initialize(input_sample_rate_in_sps, channels));
    
    falcon_dsp::falcon_dsp_host_timer timer;
    std::vector<std::vector<std::complex<float>>> out_data;
    EXPECT_TRUE(channelizer.apply(in_data, out_data));
    
    timer.log_duration("Channelization Complete"); timer.reset();

    EXPECT_EQ(out_data.size(), expected_out_data.size());
    
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

TEST(falcon_dsp_multi_rate_channelizer, cpp_multi_rate_chan_015)
{
    std::string IN_TEST_FILE_NAME = "vectors/test_015_x.bin";
    std::string IN_RESAMPLE_COEFFS_BASE_FILE_NAME = "vectors/test_015_resamp_coeffs_";
    std::string OUT_TEST_FILE_BASE_NAME = "vectors/test_015_y_shift_";
    
    /* values must match settings in generate_test_vectors.sh */
    const uint32_t INPUT_SAMPLE_RATE_IN_SPS = 1e6;
    std::vector<uint32_t> sample_rates = { 250000, 100000 };
    std::vector<std::pair<uint32_t, uint32_t>> up_down_rates = { {1, 4}, {1, 10} };
    std::vector<int32_t> freq_shifts = { 60000, -350000 };
    
    run_cpp_multi_rate_channelizer_test(IN_TEST_FILE_NAME,
                                        IN_RESAMPLE_COEFFS_BASE_FILE_NAME,
                                        OUT_TEST_FILE_BASE_NAME,
                                        INPUT_SAMPLE_RATE_IN_SPS,
                                        sample_rates,
                                        up_down_rates,
                                        freq_shifts);
}