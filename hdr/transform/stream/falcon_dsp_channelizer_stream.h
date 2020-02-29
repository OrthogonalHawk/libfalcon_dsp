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
 * @file     falcon_dsp_channelizer_stream.h
 * @author   OrthogonalHawk
 * @date     22-Feb-2020
 *
 * @brief    Signal processing channelizer stream class. Helps to manage state
 *            and memory information associated with a particular channelizer
 *            output data stream.
 *
 * @section  DESCRIPTION
 *
 * Defines a signal processing channelizer stream class which helps to manage
 *  state and memory information.
 *
 * @section  HISTORY
 *
 * 22-Feb-2020  OrthogonalHawk  File created.
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_CHANNELIZER_STREAM_H__
#define __FALCON_DSP_CHANNELIZER_STREAM_H__

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <complex>
#include <stdint.h>
#include <vector>

#include "transform/stream/falcon_dsp_stream.h"

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
     *                           FUNCTION DECLARATION
     *****************************************************************************/    
    
    /******************************************************************************
     *                            CLASS DECLARATION
     *****************************************************************************/
    
    /* @brief C++ implementation of a channelizer stream class.
     * @description Implements the basic channelizer stream class, such as might be
     *               used in a multi-rate channelizer.
     */
    class falcon_dsp_channelizer_stream : public falcon_dsp_stream
    {
    public:
        
        falcon_dsp_channelizer_stream(void);
        ~falcon_dsp_channelizer_stream(void) = default;
        
        falcon_dsp_channelizer_stream(const falcon_dsp_channelizer_stream&);
    
        bool initialize(uint32_t out_sample_rate_in_sps,
                        int64_t  freq_shift_in_hz,
                        uint32_t up_rate,
                        uint32_t down_rate,
                        std::vector<std::complex<float>>& resample_coeffs);
        
        uint32_t get_output_sample_rate_in_sps(void) const { return m_output_sample_rate_in_sps; }
        int64_t get_freq_shift_in_hz(void) const { return m_freq_shift_in_hz; }
        uint32_t get_up_rate(void) const { return m_up_rate; }
        uint32_t get_down_rate(void) const { return m_down_rate; }
        std::vector<std::complex<float>> get_resample_coeffs(void) const { return m_resample_filter_coeffs; }
    
    private:

        uint32_t                             m_output_sample_rate_in_sps;
        int64_t                              m_freq_shift_in_hz;
        uint32_t                             m_up_rate;
        uint32_t                             m_down_rate;
        std::vector<std::complex<float>>     m_resample_filter_coeffs;
    };
}

#endif // __FALCON_DSP_CHANNELIZER_STREAM_H__
