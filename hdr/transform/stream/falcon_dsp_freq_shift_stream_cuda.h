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

#ifndef __FALCON_DSP_FREQ_SHIFT_STREAM_CUDA_H__
#define __FALCON_DSP_FREQ_SHIFT_STREAM_CUDA_H__

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <complex>
#include <cuComplex.h>
#include <stdint.h>
#include <vector>

#include "transform/stream/falcon_dsp_stream_cuda.h"

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
    
    struct falcon_dsp_freq_shift_params_cuda_s
    {
        uint64_t                num_samples_handled;
        uint32_t                time_shift_rollover_sample_idx;
        double                  angular_freq;
        cuFloatComplex *        out_data;
        uint32_t                out_data_len;
    };

    /* @brief Implementation of a channelizer stream class for CUDA applications.
     * @description Implements the basic channelizer stream class, such as might be
     *               used in a multi-rate channelizer.
     */
    class falcon_dsp_freq_shift_stream_cuda : public falcon_dsp_stream_cuda
    {
    public:

        falcon_dsp_freq_shift_stream_cuda(void);
        ~falcon_dsp_freq_shift_stream_cuda(void);
        falcon_dsp_freq_shift_stream_cuda(const falcon_dsp_freq_shift_stream_cuda&) = delete;
        
        bool initialize(void) override { return false; }
        bool initialize(uint32_t rollover_sample_idx, double angular_freq);
        bool allocate_memory(uint32_t input_vector_len) override;
        bool manage_state(uint32_t input_vector_len) override;
        bool reset_state(void) override;

        falcon_dsp_freq_shift_params_cuda_s get_freq_shift_params(void);
        cuFloatComplex * get_freq_shift_out_data_ptr(bool adjust_for_prefix = false) const;
        uint32_t get_freq_shift_out_data_len(bool adjust_for_prefix = false) const;

        void add_freq_shift_samples_handled(uint32_t num_samples_handled);

    protected:

        bool allocate_memory(uint32_t input_vector_len, uint32_t extra_output_sample_prefix);

        bool cleanup_memory(void) override;

    private:

        uint64_t                m_num_freq_shift_samples_handled;
        uint32_t                m_time_shift_rollover_sample_idx;
        double                  m_angular_freq;
        cuFloatComplex *        m_freq_shift_out_data;
        uint32_t                m_freq_shift_out_data_len;
        uint32_t                m_freq_shift_out_data_prefix_len;
    };
}

#endif // __FALCON_DSP_FREQ_SHIFT_STREAM_CUDA_H__
