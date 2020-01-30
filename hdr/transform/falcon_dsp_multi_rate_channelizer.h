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
 * @file     falcon_dsp_multi_rate_channelizer.h
 * @author   OrthogonalHawk
 * @date     29-Jan-2020
 *
 * @brief    Signal processing transformation class and functions to implement
 *            a multi-rate channelizer in C++.
 *
 * @section  DESCRIPTION
 *
 * Defines a set of signal processing transformation functions and classes that
 *  together implement a multi-rate channelizer and filtering capability.
 *  Implementation uses C++.
 *
 * @section  HISTORY
 *
 * 29-Jan-2020  OrthogonalHawk  File created.
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_TRANSFORM_MULTI_RATE_CHANNELIZER_H__
#define __FALCON_DSP_TRANSFORM_MULTI_RATE_CHANNELIZER_H__

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <complex>
#include <memory>
#include <mutex>
#include <vector>

#include "transform/falcon_dsp_freq_shift.h"
#include "transform/falcon_dsp_polyphase_resampler.h"

/******************************************************************************
 *                                 CONSTANTS
 *****************************************************************************/

/******************************************************************************
 *                              ENUMS & TYPEDEFS
 *****************************************************************************/

struct multi_rate_channelizer_channel_s
{
    uint32_t                             output_sample_rate_in_sps;
    int64_t                              freq_shift_in_hz;
    uint32_t                             up_rate;
    uint32_t                             down_rate;
    std::vector<std::complex<float>>     resample_filter_coeffs;
};

/******************************************************************************
 *                                  MACROS
 *****************************************************************************/

namespace falcon_dsp
{
    /******************************************************************************
     *                           FUNCTION DECLARATION
     *****************************************************************************/    
    
    /******************************************************************************
     *                            CLASS DECLARATION
     *****************************************************************************/
    
    /* @brief C++ implementation of a multi-rate channelizer class.
     * @description Builds on several separate C++ implementations from the FALCON
     *               DSP library.
     */
    class falcon_dsp_multi_rate_channelizer
    {
    public:
        
        falcon_dsp_multi_rate_channelizer(void);
        ~falcon_dsp_multi_rate_channelizer(void);
        
        falcon_dsp_multi_rate_channelizer(const falcon_dsp_multi_rate_channelizer&) = delete;
        
        bool initialize(uint32_t input_sample_rate, std::vector<multi_rate_channelizer_channel_s> channels);

        bool apply(std::vector<std::complex<float>>& in, std::vector<std::vector<std::complex<float>>>& out);

        void reset_state(void);

    private:

        /* define an internal, class-only structure so that we can add more information */
        struct internal_multi_rate_cpp_channelizer_channel_s : multi_rate_channelizer_channel_s
        {
            internal_multi_rate_cpp_channelizer_channel_s(uint32_t input_sample_rate,
                                                          const multi_rate_channelizer_channel_s& other);            
            ~internal_multi_rate_cpp_channelizer_channel_s(void);
            
            falcon_dsp_freq_shift                             freq_shifter;
            falcon_dsp_polyphase_resampler                    resampler;
            
        private:
        
            internal_multi_rate_cpp_channelizer_channel_s(void) = delete;
        };

        std::mutex                                            m_mutex;
        bool                                                  m_initialized;
        
        /* variables for multi-channel management */
        std::vector<std::unique_ptr<internal_multi_rate_cpp_channelizer_channel_s>> m_cpp_channels;
    };
}

#endif // __FALCON_DSP_TRANSFORM_MULTI_RATE_CHANNELIZER_H__
