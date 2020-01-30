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
 * @file     falcon_dsp_multi_rate_channelizer.cu
 * @author   OrthogonalHawk
 * @date     29-Jan-2020
 *
 * @brief    Signal processing transformation class and functions to implement
 *            a multi-rate channelizer in C++.
 *
 * @section  DESCRIPTION
 *
 * Implements a set of signal processing transformation functions and classes that
 *  together implement a multi-rate channelizer and filtering capability.
 *  Implementation uses C++.
 *
 * @section  HISTORY
 *
 * 29-Jan-2020  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <iostream>
#include <stdint.h>

#include "transform/falcon_dsp_multi_rate_channelizer.h"
#include "utilities/falcon_dsp_host_timer.h"

/******************************************************************************
 *                                 CONSTANTS
 *****************************************************************************/

const bool TIMING_LOGS_ENABLED = false;

/******************************************************************************
 *                              ENUMS & TYPEDEFS
 *****************************************************************************/

/******************************************************************************
 *                                  MACROS
 *****************************************************************************/

namespace falcon_dsp
{
    /******************************************************************************
     *                        FUNCTION IMPLEMENTATION
     *****************************************************************************/
    
    /******************************************************************************
     *                           CLASS IMPLEMENTATION
     *****************************************************************************/    
    falcon_dsp_multi_rate_channelizer::internal_multi_rate_cpp_channelizer_channel_s::internal_multi_rate_cpp_channelizer_channel_s(uint32_t input_sample_rate, const multi_rate_channelizer_channel_s& other)
      : freq_shifter(input_sample_rate, other.freq_shift_in_hz),
        resampler(other.up_rate, other.down_rate, other.resample_filter_coeffs)
    {
        output_sample_rate_in_sps = other.output_sample_rate_in_sps;
        freq_shift_in_hz = other.freq_shift_in_hz;
        up_rate = other.up_rate;
        down_rate = other.down_rate;
        resample_filter_coeffs = other.resample_filter_coeffs;
    }
            
    falcon_dsp_multi_rate_channelizer::internal_multi_rate_cpp_channelizer_channel_s::~internal_multi_rate_cpp_channelizer_channel_s(void)
    { }
    
    falcon_dsp_multi_rate_channelizer::falcon_dsp_multi_rate_channelizer(void)
      : m_initialized(false)
    { }
    
    falcon_dsp_multi_rate_channelizer::~falcon_dsp_multi_rate_channelizer(void)
    {
        std::lock_guard<std::mutex> lock(std::mutex);
        
        m_cpp_channels.clear();
    }
    
    bool falcon_dsp_multi_rate_channelizer::initialize(uint32_t input_sample_rate,
                                                       std::vector<multi_rate_channelizer_channel_s> channels)
    {
        std::lock_guard<std::mutex> lock(std::mutex);

        /* sanity check the inputs and verify that the class has not already been initialized */
        if (input_sample_rate == 0 ||
            channels.size() == 0 ||
            m_initialized)
        {
            return false;
        }

        /* check each one of the requested channels is achievable */
        for (auto chan_iter : channels)
        {
            /* TODO */
        }

        /* initialize the requested channels */
        for (auto chan_iter : channels)
        {
            /* ran into compile errors with std::make_unique; g++ Ubuntu/Linaro 7.4.0
             *  so using the less elegant pointer initialization here */
            std::unique_ptr<internal_multi_rate_cpp_channelizer_channel_s> new_chan =
                    std::unique_ptr<internal_multi_rate_cpp_channelizer_channel_s>(
                        new internal_multi_rate_cpp_channelizer_channel_s(input_sample_rate, chan_iter));

            m_cpp_channels.push_back(std::move(new_chan));
        }

        /* initialization complete */
        m_initialized = true;
        
        return m_initialized;
    }
    
    bool falcon_dsp_multi_rate_channelizer::apply(std::vector<std::complex<float>>& in,
                                                  std::vector<std::vector<std::complex<float>>>& out)
    {
        std::lock_guard<std::mutex> lock(std::mutex);
        
        out.clear();
        
        std::vector<std::complex<float>> freq_shifted_output;
        std::vector<std::complex<float>> resampled_output;
        for (uint32_t out_chan_idx = 0; out_chan_idx < m_cpp_channels.size(); ++out_chan_idx)
        {
            m_cpp_channels[out_chan_idx]->freq_shifter.apply(in, freq_shifted_output);
            m_cpp_channels[out_chan_idx]->resampler.apply(freq_shifted_output, resampled_output);
            
            out.push_back(resampled_output);
        }

        return out.size() > 0;
    }
}
