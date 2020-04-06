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
 * @file     falcon_dsp_channelizer_stream_cuda.cu
 * @author   OrthogonalHawk
 * @date     22-Feb-2020
 *
 * @brief    Implements the FALCON DSP CUDA channelizer stream class.
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

#include "transform/falcon_dsp_freq_shift.h"
#include "transform/stream/falcon_dsp_channelizer_stream_cuda.h"
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
    
    falcon_dsp_channelizer_stream_cuda::falcon_dsp_channelizer_stream_cuda(void)
      : falcon_dsp_freq_shift_stream_cuda(),
        m_resample_coeffs(nullptr),
        m_resample_coeffs_len(0),
        m_resample_output_params(nullptr),
        m_resample_output_params_len(0),
        m_resampled_out_data(nullptr),
        m_resampled_out_data_len(0),
        m_num_resampler_thread_blocks(0)
    { }
    
    falcon_dsp_channelizer_stream_cuda::~falcon_dsp_channelizer_stream_cuda(void)
    {
        cleanup_memory();
    }

    bool falcon_dsp_channelizer_stream_cuda::initialize(uint32_t in_sample_rate_in_sps,
                                                        uint32_t out_sample_rate_in_sps,
                                                        int64_t  freq_shift_in_hz,
                                                        uint32_t up_rate,
                                                        uint32_t down_rate,
                                                        std::vector<std::complex<float>>& resample_coeffs)
    {
        m_mutex.lock();
        
        /* compute frequency shift parameters and initialize the base class */
        auto freq_shift_params = falcon_dsp_freq_shift::get_freq_shift_params(in_sample_rate_in_sps,
                                                                              freq_shift_in_hz);
        bool ret = falcon_dsp_freq_shift_stream_cuda::initialize(freq_shift_params.first, freq_shift_params.second);

        /* setup resampler parameters */
        m_resampler_params.initialize(up_rate, down_rate, resample_coeffs);

        /* the resampler coefficients are fixed, so allocate space for them here */
        if (m_resample_coeffs_len != resample_coeffs.size())
        {
            if (m_resample_coeffs)
            {
                cudaErrChk(cudaFree(m_resample_coeffs));
                m_resample_coeffs = nullptr;
                m_resample_coeffs_len = 0;
            }

            m_resample_coeffs_len = m_resampler_params.transposed_coeffs.size();
            cudaErrChkAssert(cudaMallocManaged(&m_resample_coeffs,
                                               m_resample_coeffs_len * sizeof(std::complex<float>)));

            /* copy the coefficients to the GPU */
            cudaErrChkAssert(cudaMemcpy(static_cast<void *>(m_resample_coeffs),
                                        static_cast<void *>(m_resampler_params.transposed_coeffs.data()),
                                        m_resampler_params.transposed_coeffs.size() * sizeof(std::complex<float>),
                                        cudaMemcpyHostToDevice));
        }
        
        m_mutex.unlock();
        
        return ret;
    }

    bool falcon_dsp_channelizer_stream_cuda::allocate_memory(uint32_t input_vector_len)
    {
        /* sanity check the input */
        if (input_vector_len == 0)
        {
            return false;
        }

        /* allocate space for the frequency shifted version of the input data. note that this
         *  is also the input data for resampling so it has to take into account the 
         *  resampling state information */
        bool ret = falcon_dsp_freq_shift_stream_cuda::allocate_memory(input_vector_len, m_resampler_params.state.size());
        
        /* copy the resampler state vector into CUDA memory */
        cudaErrChkAssert(cudaMemcpy(get_freq_shift_out_data_ptr(true),
                                    m_resampler_params.state.data(),
                                    m_resampler_params.state.size() * sizeof(std::complex<float>),
                                    cudaMemcpyHostToDevice));
        
        /* calculate the number of thread blocks that will be required for resampling */
        uint32_t expected_num_outputs = get_num_resample_outputs_for_input(input_vector_len);
        uint32_t num_outputs_per_resampler_thread_block = get_max_num_cuda_threads(); /* assumes one output per thread */
        m_num_resampler_thread_blocks = expected_num_outputs / num_outputs_per_resampler_thread_block;
        if ( (expected_num_outputs % num_outputs_per_resampler_thread_block != 0) ||
             (m_num_resampler_thread_blocks == 0 && expected_num_outputs > 0) )
        {
            m_num_resampler_thread_blocks++;
        }
        
        /* pre-compute resample output parameters */
        uint32_t num_outputs_from_thread_blocks = 0;
        uint32_t new_coeff_phase = m_resampler_params.coeff_phase;
        int64_t new_x_idx = m_resampler_params.xOffset;

        std::vector<polyphase_resampler_output_params_s> resample_output_params;
        falcon_dsp::falcon_dsp_polyphase_resampler_cuda::compute_output_params(m_resampler_params.up_rate,
                                                                               m_resampler_params.down_rate,
                                                                               m_resampler_params.state.size() + m_resampler_params.xOffset,
                                                                               input_vector_len + m_resampler_params.state.size(),
                                                                               m_resampler_params.coeff_phase,
                                                                               expected_num_outputs,
                                                                               num_outputs_from_thread_blocks,
                                                                               new_coeff_phase,
                                                                               new_x_idx,
                                                                               resample_output_params);
        
        /* update the channel tracking information preemptively, assuming that if the user
         *  calls the initialize method the parameters will actually be used */
        m_resampler_params.coeff_phase = new_coeff_phase;
        m_resampler_params.xOffset = new_x_idx - input_vector_len - m_resampler_params.state.size();

        /* allocate space for the output parameters */
        if (m_resample_output_params_len != num_outputs_from_thread_blocks)
        {
            if (m_resample_output_params)
            {
                cudaErrChkAssert(cudaFree(m_resample_output_params));
                m_resample_output_params = nullptr;
                m_resample_output_params_len = 0;
            }
            
            m_resample_output_params_len = num_outputs_from_thread_blocks;
            cudaErrChkAssert(cudaMallocManaged(&m_resample_output_params,
                                               m_resample_output_params_len *
                                                   sizeof(polyphase_resampler_output_params_s)));
        }
        
        /* copy the output parameters into CUDA memory; these are recomputed each time the kernel runs
         *  although it is hoped that the memory does not need to be reallocated each time... */
        cudaErrChkAssert(cudaMemcpy(m_resample_output_params,
                                    resample_output_params.data(),
                                    std::min(resample_output_params.size() * sizeof(polyphase_resampler_output_params_s),
                                             num_outputs_from_thread_blocks * sizeof(polyphase_resampler_output_params_s)),
                                    cudaMemcpyHostToDevice));
        
        /* allocate space for the resampled outputs */
        if (m_resampled_out_data_len != num_outputs_from_thread_blocks)
        {
            if (m_resampled_out_data)
            {
                cudaErrChkAssert(cudaFree(m_resampled_out_data));
                m_resampled_out_data = nullptr;
                m_resampled_out_data_len = 0;
            }
            
            m_resampled_out_data_len = num_outputs_from_thread_blocks;
            cudaErrChkAssert(cudaMallocManaged(&m_resampled_out_data,
                                               m_resampled_out_data_len * sizeof(std::complex<float>)));
        }
        
        return ret;
    }

    bool falcon_dsp_channelizer_stream_cuda::manage_state(uint32_t input_vector_len)
    {
        bool ret = falcon_dsp_freq_shift_stream_cuda::manage_state(input_vector_len);
        
        /* find number of samples retained in buffer */
        int64_t retain = m_resampler_params.state.size() - input_vector_len;

        if (retain > 0)
        {
            /* for input_vector_len smaller than state buffer, copy end of buffer to beginning */
            copy(m_resampler_params.state.end() - retain,
                 m_resampler_params.state.end(),
                 m_resampler_params.state.begin());
            
            /* then, copy the entire (short) input to end of buffer */
            uint32_t in_idx = 0;
            for (uint64_t state_copy_idx = retain;
                 state_copy_idx < m_resampler_params.state.size();
                 ++state_copy_idx)
            {
                /* compute the next index to copy. note that here we need to account for the
                 *  state buffer padding that was added to the resampler input */
                uint32_t next_idx_to_copy = in_idx + m_resampler_params.state.size();
                
                /* copy over the state information */
                cudaErrChkAssert(cudaMemcpy(m_resampler_params.state.data() + state_copy_idx,
                                            get_freq_shift_out_data_ptr() + next_idx_to_copy,
                                            sizeof(std::complex<float>),
                                            cudaMemcpyDeviceToHost));
                
                /* keep working through the resampler input buffer */
                in_idx++;
            }
        }
        else
        {
            cuFloatComplex * copy_start_idx = get_freq_shift_out_data_ptr(false) + 
                                              get_freq_shift_out_data_len(false) -
                                              m_resampler_params.state.size();
                
            /* just copy last input samples into state buffer */
            cudaErrChkAssert(cudaMemcpy(m_resampler_params.state.data(),
                                        copy_start_idx,
                                        m_resampler_params.state.size() * sizeof(std::complex<float>),
                                        cudaMemcpyDeviceToHost));
            
            cudaErrChkAssert(cudaDeviceSynchronize());
        }
        
        return ret;
    }

    bool falcon_dsp_channelizer_stream_cuda::reset_state(void)
    {
        bool ret = falcon_dsp_freq_shift_stream_cuda::reset_state();
        memset(m_resampler_params.state.data(), 0, m_resampler_params.state.size() * sizeof(std::complex<float>));
        
        return ret;
    }
    
    bool falcon_dsp_channelizer_stream_cuda::cleanup_memory(void)
    {
        bool ret = falcon_dsp_freq_shift_stream_cuda::cleanup_memory();

        m_mutex.lock();

        /* inform the base clase to clean up its memory as well */
        falcon_dsp_freq_shift_stream_cuda::cleanup_memory();

        /* now cleanup memory allocated by this class */
        if (m_resample_coeffs)
        {
            cudaErrChk(cudaFree(m_resample_coeffs));
            m_resample_coeffs = nullptr;
            m_resample_coeffs_len = 0;
        }

        if (m_resample_output_params)
        {
            cudaErrChk(cudaFree(m_resample_output_params));
            m_resample_output_params = nullptr;
            m_resample_output_params_len = 0;
        }

        if (m_resampled_out_data)
        {
            cudaErrChk(cudaFree(m_resampled_out_data));
            m_resampled_out_data = nullptr;
            m_resampled_out_data_len = 0;
        }
        
        m_mutex.unlock();

        return ret;
    }
    
    uint32_t falcon_dsp_channelizer_stream_cuda::get_num_resample_outputs_for_input(uint32_t input_vector_len)
    {
        /* compute how many outputs will be generated for input_vector_len inputs */
        uint64_t np = (m_resampler_params.state.size() + input_vector_len) * static_cast<uint64_t>(m_resampler_params.up_rate);
        uint32_t need = np / m_resampler_params.down_rate;
        
        if ((m_resampler_params.coeff_phase + m_resampler_params.up_rate * m_resampler_params.xOffset) < 
            (np % m_resampler_params.down_rate))
        {
            need++;
        }
        
        return need;
    }
}
