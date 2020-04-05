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
 * @file     falcon_dsp_freq_shift_stream_cuda.cu
 * @author   OrthogonalHawk
 * @date     28-Feb-2020
 *
 * @brief    Implements the FALCON DSP freq shift CUDA stream class.
 *
 * @section  DESCRIPTION
 *
 * Implements the FALCON DSP freq shift CUDA stream class, which is used for the
 *  multi-rate channelizer.
 *
 * @section  HISTORY
 *
 * 28-Feb-2020  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include "transform/stream/falcon_dsp_freq_shift_stream_cuda.h"
#include "utilities/falcon_dsp_cuda_utils.h"

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
    
    falcon_dsp_freq_shift_stream_cuda::falcon_dsp_freq_shift_stream_cuda(void)
      : falcon_dsp_stream_cuda(),
        m_num_freq_shift_samples_handled(0),
        m_time_shift_rollover_sample_idx(1e6),
        m_angular_freq(0.0),
        m_freq_shift_out_data(nullptr),
        m_freq_shift_out_data_len(0),
        m_freq_shift_out_data_prefix_len(0)
    { }

    falcon_dsp_freq_shift_stream_cuda::~falcon_dsp_freq_shift_stream_cuda(void)
    {
        cleanup_memory();
    }

    bool falcon_dsp_freq_shift_stream_cuda::initialize(uint32_t rollover_sample_idx, double angular_freq)
    {
        m_mutex.lock();

        bool ret = falcon_dsp_stream_cuda::initialize();

        m_time_shift_rollover_sample_idx = rollover_sample_idx;
        m_angular_freq = angular_freq;

        m_mutex.unlock();

        return ret;
    }

    bool falcon_dsp_freq_shift_stream_cuda::allocate_memory(uint32_t input_vector_len)
    {
        return allocate_memory(input_vector_len, 0);
    }
    
    bool falcon_dsp_freq_shift_stream_cuda::allocate_memory(uint32_t input_vector_len, uint32_t extra_output_sample_prefix)
    {
        bool ret = falcon_dsp_stream_cuda::allocate_memory(input_vector_len);

        m_mutex.lock();

        if (m_freq_shift_out_data)
        {
            cudaErrChk(cudaFree(m_freq_shift_out_data));
            m_freq_shift_out_data = nullptr;
            m_freq_shift_out_data_len = 0;
        }

        m_freq_shift_out_data_prefix_len = extra_output_sample_prefix;
        m_freq_shift_out_data_len = input_vector_len + m_freq_shift_out_data_prefix_len;
        cudaErrChkAssert(cudaMallocManaged(&m_freq_shift_out_data,
                                           m_freq_shift_out_data_len * sizeof(std::complex<float>)));
        
        m_mutex.unlock();
        
        return ret;
    }

    bool falcon_dsp_freq_shift_stream_cuda::manage_state(uint32_t input_vector_len)
    {
        /* this class does not have state information to manage */
        return falcon_dsp_stream_cuda::manage_state(input_vector_len);
        
    }

    bool falcon_dsp_freq_shift_stream_cuda::reset_state(void)
    {
        m_mutex.lock();
        
        bool ret = falcon_dsp_stream::reset_state();
        m_num_freq_shift_samples_handled = 0;

        m_mutex.unlock();

        return ret;
    }

    cuFloatComplex * falcon_dsp_freq_shift_stream_cuda::get_freq_shift_out_data_ptr(bool adjust_for_prefix) const
    {
        if (adjust_for_prefix)
        {
            return m_freq_shift_out_data;
        }
        else
        {
            /* seems a little odd to 'adjust' in the non-adjust_for_prefix case but the desire
             *  was to keep the behavior here consistent with the get_freq_shift_out_data_len
             *  method: if the adjust_for_prefix value is false then return the pointer and
             *  length as though the prefix does not exist */
            return m_freq_shift_out_data + m_freq_shift_out_data_prefix_len;
        }
    }
    
    uint32_t falcon_dsp_freq_shift_stream_cuda::get_freq_shift_out_data_len(bool adjust_for_prefix) const
    {
        if (adjust_for_prefix)
        {
            return m_freq_shift_out_data_len;
        }
        else
        {
            /* keep this consistent with the get_freq_shift_out_data_ptr method: if the adjust_for_prefix
             *  value is false then return the pointer and length as though the prefix does not exist. */
            return m_freq_shift_out_data_len - m_freq_shift_out_data_prefix_len;
        }
    }
    
    falcon_dsp_freq_shift_params_cuda_s falcon_dsp_freq_shift_stream_cuda::get_freq_shift_params(void)
    {
        m_mutex.lock();

        falcon_dsp_freq_shift_params_cuda_s ret;
        ret.num_samples_handled = m_num_freq_shift_samples_handled;
        ret.time_shift_rollover_sample_idx = m_time_shift_rollover_sample_idx;
        ret.angular_freq = m_angular_freq;
        ret.out_data = get_freq_shift_out_data_ptr();
        ret.out_data_len = get_freq_shift_out_data_len();

        m_mutex.unlock();

        return ret;
    }

    void falcon_dsp_freq_shift_stream_cuda::add_freq_shift_samples_handled(uint32_t num_samples_handled)
    {
        m_mutex.lock();

        m_num_freq_shift_samples_handled += num_samples_handled;
        m_num_freq_shift_samples_handled %= m_time_shift_rollover_sample_idx;
        
        m_mutex.unlock();
    }

    bool falcon_dsp_freq_shift_stream_cuda::cleanup_memory(void)
    {
        bool ret = falcon_dsp_stream_cuda::cleanup_memory();

        m_mutex.lock();

        if (m_freq_shift_out_data)
        {
            cudaErrChk(cudaFree(m_freq_shift_out_data));
            m_freq_shift_out_data = nullptr;
            m_freq_shift_out_data_len = 0;
        }
        
        m_mutex.unlock();

        return ret;
    }
}
