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
 * @file     falcon_dsp_fm_demod_cuda_unit_tests.cu
 * @author   OrthogonalHawk
 * @date     18-Feb-2020
 *
 * @brief    Unit tests that exercise the FALCON DSP FM Demodulation functions.
 *
 * @section  DESCRIPTION
 *
 * Implements a Google Test Framework based unit test suite for the FALCON DSP
 *  library functions.
 *
 * @section  HISTORY
 *
 * 18-Feb-2020  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <chrono>
#include <stdint.h>
#include <vector>

#include <gtest/gtest.h>

#include "signals/falcon_dsp_fm_demod_cuda.h"
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

void run_cuda_fm_demod_test(std::string input_data_file_name,
                            uint32_t input_sample_rate_in_sps,
                            int32_t signal_offset_from_dc_in_hz,
                            std::string output_data_file_name)
{
    /* get the input data */
    std::vector<std::complex<float>> in_data;
    EXPECT_TRUE(falcon_dsp::read_complex_data_from_file(input_data_file_name,
                                                        falcon_dsp::file_type_e::BINARY, in_data));
    
    std::cout << "Read " << in_data.size() << " samples from " << input_data_file_name << std::endl;
    
    falcon_dsp::falcon_dsp_fm_demodulator_cuda fm_demod;
    EXPECT_TRUE(fm_demod.initialize(input_sample_rate_in_sps, signal_offset_from_dc_in_hz));
    
    falcon_dsp::falcon_dsp_host_timer timer;
    
    /* now demodulate the FM signal */
    std::vector<int16_t> out_data;
    EXPECT_TRUE(fm_demod.demod_mono(in_data, out_data));
    
    timer.log_duration("Demod Complete"); timer.reset();
    
    fm_demod.reset_state();
    EXPECT_TRUE(fm_demod.demod_mono(in_data, out_data));
    
    timer.log_duration("Demod Complete");
    
    EXPECT_GT(out_data.size(), 0u);
    
    /* write the resulting data to file */
    EXPECT_TRUE(falcon_dsp::write_data_to_file(output_data_file_name, falcon_dsp::file_type_e::BINARY, out_data));
}

TEST(falcon_dsp_fm_demod, cuda_demod_ota_fm_radio)
{
    run_cuda_fm_demod_test("./vectors/ota_fm_radio_data.bin",
                           1140000,
                           -250000,
                           "./vectors/ota_fm_radio_signal_cuda.raw");
}
