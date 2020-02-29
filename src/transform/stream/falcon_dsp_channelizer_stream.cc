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
 * @file     falcon_dsp_channelizer_stream.cc
 * @author   OrthogonalHawk
 * @date     22-Feb-2020
 *
 * @brief    Implements the FALCON DSP channelizer stream class.
 *
 * @section  DESCRIPTION
 *
 * Implements the FALCON DSP channelizer stream class, which is used for the
 *  multi-rate channelizer.
 *
 * @section  HISTORY
 *
 * 22-Feb-2020  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include "transform/stream/falcon_dsp_channelizer_stream.h"

/******************************************************************************
 *                                 CONSTANTS
 *****************************************************************************/

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
    
    falcon_dsp_channelizer_stream::falcon_dsp_channelizer_stream(void)
      : falcon_dsp_stream(),
        m_output_sample_rate_in_sps(0),
        m_freq_shift_in_hz(0),
        m_up_rate(1),
        m_down_rate(1)
    { }

    falcon_dsp_channelizer_stream::falcon_dsp_channelizer_stream(const falcon_dsp_channelizer_stream& other)
      : falcon_dsp_stream()
    {
        m_output_sample_rate_in_sps = other.m_output_sample_rate_in_sps;
        m_freq_shift_in_hz = other.m_freq_shift_in_hz;
        m_up_rate = other.m_up_rate;
        m_down_rate = other.m_down_rate;
        m_resample_filter_coeffs = other.m_resample_filter_coeffs;
    }

    bool falcon_dsp_channelizer_stream::initialize(uint32_t out_sample_rate_in_sps,
                                                   int64_t  freq_shift_in_hz,
                                                   uint32_t up_rate,
                                                   uint32_t down_rate,
                                                   std::vector<std::complex<float>>& resample_coeffs)
    {
        m_mutex.lock();
     
        bool ret = falcon_dsp_stream::initialize();
        
        if (ret)
        {
            m_output_sample_rate_in_sps = out_sample_rate_in_sps;
            m_freq_shift_in_hz = freq_shift_in_hz;
            m_up_rate = up_rate;
            m_down_rate = down_rate;
            m_resample_filter_coeffs = resample_coeffs;
        }
        
        m_mutex.unlock();
        
        return ret;
    }
}
