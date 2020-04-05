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
 * @file     falcon_dsp_multi_rate_channelizer_cuda.h
 * @author   OrthogonalHawk
 * @date     28-Jan-2020
 *
 * @brief    Signal processing transformation class and functions to implement
 *            a multi-rate channelizer in CUDA.
 *
 * @section  DESCRIPTION
 *
 * Defines a set of signal processing transformation functions and classes that
 *  together implement a multi-rate channelizer and filtering capability.
 *  Implementation uses CUDA to leverage GPU acceleration.
 *
 * @section  HISTORY
 *
 * 28-Jan-2020  OrthogonalHawk  File created.
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_TRANSFORM_MULTI_RATE_CHANNELIZER_CUDA_H__
#define __FALCON_DSP_TRANSFORM_MULTI_RATE_CHANNELIZER_CUDA_H__

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <complex>
#include <memory>
#include <mutex>
#include <vector>

#include <cuComplex.h>

#include "transform/falcon_dsp_freq_shift_cuda.h"
#include "transform/falcon_dsp_polyphase_resampler_cuda.h"
#include "transform/stream/falcon_dsp_channelizer_stream_cuda.h"
#include "utilities/falcon_dsp_cuda_utils.h"

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
    
    /* @brief CUDA implementation of a multi-rate channelizer class.
     * @description Builds on several separate CUDA implementations from the FALCON
     *               DSP library but brings them together in an optimized way where
     *               data is copied to the GPU once and then processed on-device to
     *               minimize expensive host <-> device memory transfers.
     */
    class falcon_dsp_multi_rate_channelizer_cuda
    {
    public:
        
        falcon_dsp_multi_rate_channelizer_cuda(void);
        ~falcon_dsp_multi_rate_channelizer_cuda(void);
        
        falcon_dsp_multi_rate_channelizer_cuda(const falcon_dsp_multi_rate_channelizer_cuda&) = delete;
        
        bool initialize(uint32_t input_sample_rate, std::vector<multi_rate_channelizer_channel_s> channels);

        bool apply(std::vector<std::complex<float>>& in, std::vector<std::vector<std::complex<float>>>& out);

        void reset_state(void);

    private:
    
        void _manage_resampler_state(uint32_t chan_idx, uint32_t input_vector_len);

        std::mutex                                            m_mutex;
        bool                                                  m_initialized;
        
        /* variables for input data memory management */
        cuFloatComplex *                                      m_input_data;
        uint32_t                                              m_input_data_len;
        
        /* variables for multi-channel management */
        std::vector<std::unique_ptr<falcon_dsp_channelizer_stream_cuda>>  m_channels;
        falcon_dsp_freq_shift_params_cuda_s *                             d_freq_shift_channels;
    };
}

#endif // __FALCON_DSP_TRANSFORM_MULTI_RATE_CHANNELIZER_CUDA_H__
