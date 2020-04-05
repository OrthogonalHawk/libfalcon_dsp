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
 * @file     falcon_dsp_channelizer_stream_cuda.h
 * @author   OrthogonalHawk
 * @date     22-Feb-2020
 *
 * @brief    Signal processing channelizer stream class for CUDA. Helps to
 *            manage state and memory information associated with a particular
 *            channelizer output data stream.
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

#ifndef __FALCON_DSP_CHANNELIZER_STREAM_CUDA_H__
#define __FALCON_DSP_CHANNELIZER_STREAM_CUDA_H__

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <complex>
#include <stdint.h>
#include <vector>

#include "transform/falcon_dsp_polyphase_resampler_cuda.h"
#include "transform/stream/falcon_dsp_freq_shift_stream_cuda.h"

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
    
    /* @brief Implementation of a channelizer stream class for CUDA applications.
     * @description Implements the basic channelizer stream class, such as might be
     *               used in a multi-rate channelizer.
     */
    class falcon_dsp_channelizer_stream_cuda : public falcon_dsp_freq_shift_stream_cuda
    {
    public:
        
        falcon_dsp_channelizer_stream_cuda(void);
        ~falcon_dsp_channelizer_stream_cuda(void);

        falcon_dsp_channelizer_stream_cuda(const falcon_dsp_channelizer_stream_cuda&) = delete;
        
        bool initialize(void) { return false; }
        bool initialize(uint32_t in_sample_rate_in_sps,
                        uint32_t out_sample_rate_in_sps,
                        int64_t  freq_shift_in_hz,
                        uint32_t up_rate,
                        uint32_t down_rate,
                        std::vector<std::complex<float>>& resample_coeffs);
        
        bool allocate_memory(uint32_t input_vector_len) override;
        bool manage_state(uint32_t input_vector_len) override;
        bool reset_state(void) override;
    
        cuFloatComplex * get_resampler_coeffs_ptr(void) const { return m_resample_coeffs; }
        uint32_t get_resampler_coeffs_len(void) const { return m_resample_coeffs_len; }
    
        polyphase_resampler_output_params_s * get_resampler_output_params_ptr(void) const { return m_resample_output_params; }
        uint32_t get_resampler_output_params_len(void) const { return m_resample_output_params_len; }

        polyphase_resampler_params_s get_resampler_params(void) const { return m_resampler_params; }
        cuFloatComplex * get_resample_out_data_ptr(void) const { return m_resampled_out_data; }
        uint32_t get_resample_out_data_len(void) const { return m_resampled_out_data_len; }

        uint32_t get_num_resampler_thread_blocks(void) const { return m_num_resampler_thread_blocks; }

    private:

        bool cleanup_memory(void) override;
        uint32_t get_num_resample_outputs_for_input(uint32_t input_vector_len);
    
        cuFloatComplex *                                        m_resample_coeffs;
        uint32_t                                                m_resample_coeffs_len;
            
        polyphase_resampler_output_params_s *                   m_resample_output_params;
        uint32_t                                                m_resample_output_params_len;
            
        cuFloatComplex *                                        m_resampled_out_data;
        uint32_t                                                m_resampled_out_data_len;
            
        polyphase_resampler_params_s                            m_resampler_params;
        uint32_t                                                m_num_resampler_thread_blocks;
    };
}

#endif // __FALCON_DSP_CHANNELIZER_STREAM_CUDA_H__
