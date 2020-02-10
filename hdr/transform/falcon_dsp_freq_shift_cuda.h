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
 * @file     falcon_dsp_freq_shift_cuda.h
 * @author   OrthogonalHawk
 * @date     19-Jan-2020
 *
 * @brief    Signal processing transformation functions for frequency shifting;
 *            CUDA versions.
 *
 * @section  DESCRIPTION
 *
 * Defines a set of signal processing transformation frequency shift functions.
 *  Includes CUDA implementations.
 *
 * @section  HISTORY
 *
 * 19-Jan-2020  OrthogonalHawk  File broken out from falcon_dsp_transform.h
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_TRANSFORM_FREQ_SHIFT_CUDA_H__
#define __FALCON_DSP_TRANSFORM_FREQ_SHIFT_CUDA_H__

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <complex>
#include <memory>
#include <mutex>
#include <vector>

#include <cuComplex.h>

#include "transform/falcon_dsp_freq_shift.h"
#include "utilities/falcon_dsp_cuda_utils.h"

/******************************************************************************
 *                                 CONSTANTS
 *****************************************************************************/

/******************************************************************************
 *                              ENUMS & TYPEDEFS
 *****************************************************************************/

struct freq_shift_channel_s
{
    freq_shift_channel_s(void)
      : num_samples_handled(0),
        time_shift_rollover_sample_idx(1e6),
        angular_freq(0.0),
        out_data(nullptr),
        out_data_len(0)
    { }

    ~freq_shift_channel_s(void)
    {
        cleanup_memory();
    }

    void allocate_memory(uint32_t input_vector_len)
    {
        if (out_data)
        {
            cudaErrChk(cudaFree(out_data));
            out_data = nullptr;
            out_data_len = 0;
        }

        cudaErrChkAssert(cudaMallocManaged(&out_data,
                                           input_vector_len * sizeof(std::complex<float>)));
        out_data_len = input_vector_len;
    }

    void cleanup_memory(void)
    {
        if (out_data)
        {
            cudaErrChk(cudaFree(out_data));
            out_data = nullptr;
            out_data_len = 0;
        }
    }

    void reset_state(void)
    {
        num_samples_handled = 0;
    }

    uint64_t num_samples_handled;
    uint32_t time_shift_rollover_sample_idx;
    double angular_freq;
    cuFloatComplex * out_data;
    uint32_t out_data_len;
};

/******************************************************************************
 *                                  MACROS
 *****************************************************************************/

namespace falcon_dsp
{
    /******************************************************************************
     *                           FUNCTION DECLARATION
     *****************************************************************************/
    
    /* @brief CUDA implementation of a frequency shift vector operation.
     * @param[in] in_sample_rate_in_sps - input vector sample rate in samples
     *                                      per second.
     * @param[in] in                    - input vector
     * @param[in] freq_shift_in_hz      - amount to frequency shift in Hz
     * @param[out] out                  - frequency shifted vector
     * @return True if the input vector was frequency shifted as requested;
     *          false otherwise.
     */
    bool freq_shift_cuda(uint32_t in_sample_rate_in_sps, std::vector<std::complex<float>>& in,
                         int32_t freq_shift_in_hz, std::vector<std::complex<float>>& out);
    
    /* @brief CUDA implementation of a multi-channel frequency shift vector operation.
     * @param[in] in_sample_rate_in_sps - input vector sample rate in samples
     *                                      per second.
     * @param[in] in                    - input vector
     * @param[in] freq_shift_in_hz      - vector of requested frequency shifts; each shift value
     *                                     generates a separate output 'channel'
     * @param[out] out                  - vector of frequency shifted vectors
     * @return True if the input vector was frequency shifted as requested;
     *          false otherwise.
     */
    bool freq_shift_cuda(uint32_t in_sample_rate_in_sps, std::vector<std::complex<float>>& in,
                         std::vector<int32_t>& freq_shift_in_hz, std::vector<std::vector<std::complex<float>>>& out);
    
    /* CUDA kernel function that applies a frequency shift and puts the shifted data
     *  into a (new?) memory location. can either be used to modify the data in-place
     *  or to make a new vector with the shifted data. */
    __global__
    void __freq_shift(uint32_t num_samples_handled_previously,
                      uint32_t time_shift_rollover_sample_idx,
                      double angular_freq,
                      cuFloatComplex * in_data,
                      uint32_t in_data_len,
                      cuFloatComplex * out_data,
                      uint32_t out_data_len);
    
    /* CUDA kernel function that supports multi-channel frequency shifting. */
    __global__
    void __freq_shift_multi_chan(freq_shift_channel_s * channels,
                                 uint32_t num_channels,
                                 cuFloatComplex * in_data,
                                 uint32_t in_data_len);
    
    
    /******************************************************************************
     *                            CLASS DECLARATION
     *****************************************************************************/
    
    /* @brief CUDA implementation of a frequency shift utility class.
     * @description Derives from the C++ version since there is significant overlap
     *               in implementation. CUDA vs. C++ differentiation in the 'apply'
     *               method where the frequency shift is applied.
     */
    class falcon_dsp_freq_shift_cuda : public falcon_dsp_freq_shift
    {
    public:
        
        falcon_dsp_freq_shift_cuda(uint32_t input_sample_rate, int32_t freq_shift_in_hz);
        falcon_dsp_freq_shift_cuda(uint32_t input_sample_rate, std::vector<int32_t> freq_shift_in_hz);

        ~falcon_dsp_freq_shift_cuda(void);
        
        falcon_dsp_freq_shift_cuda(void) = delete;
        falcon_dsp_freq_shift_cuda(const falcon_dsp_freq_shift_cuda&) = delete;
        
        bool apply(std::vector<std::complex<int16_t>>& in, std::vector<std::complex<int16_t>>& out) override;
        bool apply(std::vector<std::complex<float>>& in, std::vector<std::complex<float>>& out);
        bool apply(std::vector<std::complex<float>>& in, std::vector<std::vector<std::complex<float>>>& out);
    
    private:

        /* variables for input data memory management */
        void *                                                m_cuda_input_data;
        uint32_t                                              m_max_num_input_samples;
        
        /* variables for multi-channel management */
        std::vector<std::unique_ptr<freq_shift_channel_s>>    m_freq_shift_channels;
        freq_shift_channel_s *                                d_freq_shift_channels;
    };
}

#endif // __FALCON_DSP_TRANSFORM_FREQ_SHIFT_CUDA_H__
