/******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 OrthogonalHawk
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
 * @file     falcon_dsp_freq_shift_cuda.cu
 * @author   OrthogonalHawk
 * @date     04-Jun-2019
 *
 * @brief    Implements a CUDA-based time series frequency shift operation.
 *
 * @section  DESCRIPTION
 *
 * Implements the CUDA version of a time series frequency shift operation. Both
 *  a standalone function and a class-based tranform object are supported.
 *
 * @section  HISTORY
 *
 * 04-Jun-2019  OrthogonalHawk  File created.
 * 19-Jan-2020  OrthogonalHawk  Refactored to use falcon_dsp_freq_shift_cuda.h
 * 12-Feb-2020  OrthogonalHawk  Updated to use an 'initialize' method.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <iostream>
#include <memory>
#include <stdint.h>

#include "transform/falcon_dsp_freq_shift_cuda.h"
#include "utilities/falcon_dsp_host_timer.h"
#include "utilities/falcon_dsp_cuda_utils.h"

/******************************************************************************
 *                                 CONSTANTS
 *****************************************************************************/

const bool TIMING_LOGS_ENABLED = false;

const uint32_t MAX_NUM_CUDA_THREADS = 1024;

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
    
    /* @brief CUDA implementation of a frequency shift vector operation.
     * @param[in] in_sample_rate_in_sps - input vector sample rate in samples
     *                                      per second.
     * @param[in] in                    - input vector
     * @param[in] freq_shift_in_hz      - amount to frequency shift in Hz
     * @param[out] out                  - frequency shifted vector
     * @return True if the input vector was frequency shifted as requested;
     *          false otherwise.
     */
    bool freq_shift_cuda(uint32_t in_sample_rate_in_sps, std::vector<std::complex<int16_t>>& in,
                         int32_t freq_shift_in_hz, std::vector<std::complex<int16_t>>& out)
    {
        falcon_dsp_freq_shift_cuda freq_shifter;
        return freq_shifter.initialize(in_sample_rate_in_sps, freq_shift_in_hz) && freq_shifter.apply(in, out);
    }
    
    bool freq_shift_cuda(uint32_t in_sample_rate_in_sps, std::vector<std::complex<float>>& in,
                         int32_t freq_shift_in_hz, std::vector<std::complex<float>>& out)
    {        
        falcon_dsp_freq_shift_cuda freq_shifter;
        return freq_shifter.initialize(in_sample_rate_in_sps, freq_shift_in_hz) && freq_shifter.apply(in, out);
    }
    
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
                         std::vector<int32_t>& freq_shift_in_hz, std::vector<std::vector<std::complex<float>>>& out)
    {
        falcon_dsp_freq_shift_cuda freq_shifter;
        return freq_shifter.initialize(in_sample_rate_in_sps, freq_shift_in_hz) && freq_shifter.apply(in, out);
    }
    
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
                      uint32_t out_data_len)
    {
        /* retrieve the starting data index that corresponds to this thread */
        uint32_t start_data_index = blockIdx.x * blockDim.x + threadIdx.x;
     
        /* catch the cases where the output buffer is shorter than
         *  the input buffer and where the input size is not an integer
         *  multiple of the thread block size */
        if ( (out_data_len < in_data_len) ||
             (start_data_index > in_data_len ) ||
             (start_data_index > out_data_len) )
        {
            return;
        }
        
        /* compute the time shift index for the current thread */
        uint64_t time_shift_idx = num_samples_handled_previously + start_data_index;

        cuFloatComplex freq_shift;
        time_shift_idx %= time_shift_rollover_sample_idx;
            
        /* compute the frequency shift multiplier value */
        double freq_shift_angle = angular_freq * static_cast<double>(time_shift_idx);
            
        /* set a CUDA complex variable to apply the freqency shift */
        freq_shift.x = cos(freq_shift_angle);
        freq_shift.y = sin(freq_shift_angle);

        /* apply the frequency shift; may be used to modify data in place or to put
         *  the output into a new vector based on the input arguments */
        out_data[start_data_index] = cuCmulf(in_data[start_data_index], freq_shift);
    }
    
    /* CUDA kernel function that supports multi-channel frequency shifting. */
    __global__
    void __freq_shift_multi_chan(falcon_dsp_freq_shift_params_cuda_s * channels,
                                 uint32_t num_channels,
                                 cuFloatComplex * in_data,
                                 uint32_t in_data_len)
    {
        extern __shared__ falcon_dsp_freq_shift_params_cuda_s s_channels[];
        
        /* retrieve the starting data index that corresponds to this thread */
        uint32_t start_data_index = blockIdx.x * blockDim.x + threadIdx.x;

        /* copy channel information into shared memory */
        if (threadIdx.x < num_channels)
        {
            s_channels[threadIdx.x] = channels[threadIdx.x];
        }
        __syncthreads();

        /* sanity check the inputs */       
        for (uint32_t chan_idx = 0; chan_idx < num_channels; ++chan_idx)
        {
            /* catch the case where the channel information is not available or where
             *  the output buffer is shorter than the input buffer. also catch the case
             *  where the input size is not an integer multiple of the thread block size */
            if (s_channels[chan_idx].out_data == nullptr ||
                s_channels[chan_idx].out_data_len < in_data_len ||
                start_data_index > in_data_len ||
                start_data_index > s_channels[chan_idx].out_data_len)
            {
                return;
            }
        }

        /* now compute outputs for each channel */
        for (uint32_t chan_idx = 0; chan_idx < num_channels; ++chan_idx)
        {
            uint64_t time_shift_idx = s_channels[chan_idx].num_samples_handled + start_data_index;
            time_shift_idx %= s_channels[chan_idx].time_shift_rollover_sample_idx;
            
            /* compute the frequency shift multiplier value */
            double freq_shift_angle = s_channels[chan_idx].angular_freq * static_cast<double>(time_shift_idx);
            
            /* set a CUDA complex variable to apply the freqency shift */
            cuFloatComplex freq_shift;
            freq_shift.x = cos(freq_shift_angle);
            freq_shift.y = sin(freq_shift_angle);

            /* apply the frequency shift; may be used to modify data in place or to put
             *  the output into a new vector based on the input arguments */
            s_channels[chan_idx].out_data[start_data_index] = cuCmulf(in_data[start_data_index], freq_shift);
        }
    }
    
    /******************************************************************************
     *                           CLASS IMPLEMENTATION
     *****************************************************************************/
    
    falcon_dsp_freq_shift_cuda::falcon_dsp_freq_shift_cuda(void)
      : falcon_dsp_freq_shift(),
        m_cuda_input_data(nullptr),
        m_max_num_input_samples(0),
        m_freq_shift_params(nullptr)
    { }
    
    falcon_dsp_freq_shift_cuda::~falcon_dsp_freq_shift_cuda(void)
    {
        std::lock_guard<std::mutex> lock(std::mutex);
        
        /* the falcon_dsp_freq_shift_params_cuda_s objects automatically delete CUDA
         *  memory if it was allocated when the object is destroyed */
        m_freq_shift_streams.clear();
        
        /* cleanup the CUDA memory that was reserved in the constructor to house
         *  the channel information when CUDA kernels are running */
        if (m_freq_shift_params)
        {
            cudaErrChk(cudaFree(m_freq_shift_params));
            m_freq_shift_params = nullptr;
        }
    }

    bool falcon_dsp_freq_shift_cuda::initialize(uint32_t input_sample_rate_in_sps, int32_t freq_shift_in_hz)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        if (m_initialized)
        {
            return false;
        }
        
        /* user is only requesting a single channel */
        auto chan_params = falcon_dsp_freq_shift::get_freq_shift_params(input_sample_rate_in_sps, freq_shift_in_hz);
        std::unique_ptr<falcon_dsp_freq_shift_stream_cuda> new_chan = std::make_unique<falcon_dsp_freq_shift_stream_cuda>();
        new_chan->initialize(chan_params.first, chan_params.second);
        m_freq_shift_streams.push_back(std::move(new_chan));
        
        /* allocate CUDA memory for the stream parameter information; the master copy is kept
         *  on the host, but is copied to the device when the 'apply' method is invoked */
        cudaErrChkAssert(cudaMallocManaged(&m_freq_shift_params,
                                           m_freq_shift_streams.size() * sizeof(falcon_dsp_freq_shift_params_cuda_s)));
        
        /* change the shared memory size to 8 bytes per shared memory bank. this is so that we
         *  can better handle complex<float> data, which is natively 8 bytes in size */
        cudaErrChkAssert(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
        
        m_initialized = true;
        return m_initialized;
    }

    bool falcon_dsp_freq_shift_cuda::initialize(uint32_t input_sample_rate_in_sps, std::vector<int32_t> freq_shift_in_hz)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        if (m_initialized)
        {
            return false;
        }
        
        /* user is requesting multiple channels */
        for (auto freq_shift : freq_shift_in_hz)
        {
            auto chan_params = falcon_dsp_freq_shift::get_freq_shift_params(input_sample_rate_in_sps, freq_shift);
            std::unique_ptr<falcon_dsp_freq_shift_stream_cuda> new_chan = std::make_unique<falcon_dsp_freq_shift_stream_cuda>();
            new_chan->initialize(chan_params.first, chan_params.second);

            m_freq_shift_streams.push_back(std::move(new_chan));
        }
          
        /* allocate CUDA memory for the channel information; the master copy is kept
         *  on the host, but is copied to the device when the 'apply' method is invoked */
        cudaErrChkAssert(cudaMallocManaged(&m_freq_shift_params,
                                           m_freq_shift_streams.size() * sizeof(falcon_dsp_freq_shift_params_cuda_s)));
        
        /* change the shared memory size to 8 bytes per shared memory bank. this is so that we
         *  can better handle complex<float> data, which is natively 8 bytes in size */
        cudaErrChkAssert(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
        
        m_initialized = true;
        return m_initialized;
    }

    bool falcon_dsp_freq_shift_cuda::apply(std::vector<std::complex<int16_t>>& in, std::vector<std::complex<int16_t>>& out)
    {
        /* create another copy of the data and cast to std::complex<float> */
        std::vector<std::complex<float>> tmp_in_vec;
        tmp_in_vec.reserve(in.size());
        for (auto in_iter = in.begin(); in_iter != in.end(); ++in_iter)
        {
            tmp_in_vec.push_back(std::complex<float>((*in_iter).real(), (*in_iter).imag()));
        }

        /* frequency shift the input data */
        std::vector<std::complex<float>> tmp_out_vec;
        bool ret = apply(tmp_in_vec, tmp_out_vec);

        /* cast the filtered output back to std::complex<int16_t> */
        for (auto out_iter = tmp_out_vec.begin(); out_iter != tmp_out_vec.end(); ++out_iter)
        {
            out.push_back(std::complex<int16_t>((*out_iter).real(), (*out_iter).imag()));
        }

        return ret;
    }
    
    bool falcon_dsp_freq_shift_cuda::apply(std::vector<std::complex<float>>& in, std::vector<std::complex<float>>& out)
    {
        /* sanity check the input and verify that invoking this method make sense. specifically,
         *  if a user configured the frequency shifter for multiple channels then this is not
         *  the right method to call. instead, the user should invoke the method that provides
         *  multiple output channels */
        if (m_freq_shift_streams.size() > 1)
        {
            return false;
        }

        std::vector<std::vector<std::complex<float>>> out_data;
        bool ret = apply(in, out_data);

        if (out_data.size() > 0)
        {
            /* swap to avoid unnecessary copying */
            out.swap(out_data[0]);
        }

        return ret;
    }
    
    
    bool falcon_dsp_freq_shift_cuda::apply(std::vector<std::complex<float>>& in, std::vector<std::vector<std::complex<float>>>& out)
    {   
        std::lock_guard<std::mutex> lock(std::mutex);

        out.clear();
        if (!m_initialized)
        {
            return false;
        }

        /* resize the output data structures so that they can hold the shifted
         *  data. note that by using resize() the vector size is now equal to in.size()
         *  even without explicitly adding data to the vector, which means that we can
         *  add data directly into the vector data buffer without worrying about the
         *  vector size getting mismatched with the buffer contents */
        out.resize(m_freq_shift_streams.size());
        for (uint32_t chan_idx = 0; chan_idx < m_freq_shift_streams.size(); ++chan_idx)
        {
            out[chan_idx].resize(in.size());
        }

        /* if there is enough space in previously allocated memory then use it; otherwise
         *  allocate new memory buffers. it is left as a future feature to specify a maximum
         *  memory size and process the data in chunks instead of requiring enough GPU
         *  memory to process the whole vector at once */

        /* allocate CUDA memory for the input samples */
        if (m_max_num_input_samples < in.size())
        {
            if (m_cuda_input_data)
            {
                cudaErrChkAssert(cudaFree(m_cuda_input_data));
                m_cuda_input_data = nullptr;
                m_max_num_input_samples = 0;
            }
            
            cudaErrChkAssert(cudaMallocManaged(&m_cuda_input_data,
                                               in.size() * sizeof(std::complex<float>)));
            m_max_num_input_samples = in.size();
        }
        
        /* allocate CUDA memory for the output samples. all channels have the same amount
         *  of output space and at least one channel is guaranteed by the constructor. note
         *  this output memory is NOT needed if there is only a single output channel; in
         *  this case the data can be modified in-place on the GPU */
        if (m_freq_shift_streams.size() > 1 &&
            m_freq_shift_streams[0]->get_freq_shift_out_data_len() < in.size())
        {
            for (uint32_t chan_idx = 0; chan_idx < m_freq_shift_streams.size(); ++chan_idx)
            {
                /* clean up existing memory (if necessary) and allocate new memory */
                m_freq_shift_streams[chan_idx]->allocate_memory(in.size());
            }
        }

        /* copy the input data to the GPU */
        cuFloatComplex * cuda_float_complex_input_data = static_cast<cuFloatComplex *>(m_cuda_input_data);        
        cudaErrChkAssert(cudaMemcpy(static_cast<void *>(cuda_float_complex_input_data),
                         static_cast<void *>(in.data()),
                         in.size() * sizeof(std::complex<float>),
                         cudaMemcpyHostToDevice));
        
        /* run kernel on the GPU */
        uint32_t samples_per_thread_block = MAX_NUM_CUDA_THREADS; /* assumes one output per thread */
        uint32_t num_thread_blocks = (in.size() + samples_per_thread_block - 1) / samples_per_thread_block;
        
        falcon_dsp::falcon_dsp_host_timer timer("KERNEL", TIMING_LOGS_ENABLED);
        if (m_freq_shift_streams.size() == 1)
        {
            auto params = m_freq_shift_streams[0]->get_freq_shift_params();
            __freq_shift<<<num_thread_blocks, MAX_NUM_CUDA_THREADS>>>(params.num_samples_handled,
                                                                      params.time_shift_rollover_sample_idx,
                                                                      params.angular_freq,
                                                                      cuda_float_complex_input_data,
                                                                      m_max_num_input_samples,
                                                                      cuda_float_complex_input_data, /* modify in place */
                                                                      m_max_num_input_samples);
        
            cudaErrChkAssert(cudaPeekAtLastError());
            
            /* wait for GPU to finish before accessing on host */
            cudaErrChkAssert(cudaDeviceSynchronize());
            
            timer.log_duration("Single Chan Kernel Complete");
        
            /* copy output samples out of CUDA memory */
            cudaErrChkAssert(cudaMemcpy(static_cast<void *>(out[0].data()),
                             static_cast<void *>(cuda_float_complex_input_data),
                             in.size() * sizeof(std::complex<float>),
                             cudaMemcpyDeviceToHost));
        }
        else /* use the multi-channel kernel */
        {
            uint32_t shared_memory_size_in_bytes = sizeof(falcon_dsp_freq_shift_params_cuda_s) * m_freq_shift_streams.size();

            /* copy the channel information to the GPU */
            std::vector<falcon_dsp_freq_shift_params_cuda_s> tmp_params(m_freq_shift_streams.size());
            for (uint32_t chan_idx = 0; chan_idx < m_freq_shift_streams.size(); ++chan_idx)
            {
                tmp_params[chan_idx] = m_freq_shift_streams[chan_idx]->get_freq_shift_params();
            }
            
            cudaErrChkAssert(cudaMemcpy(static_cast<void *>(m_freq_shift_params),
                                        static_cast<void *>(tmp_params.data()),
                                        sizeof(falcon_dsp_freq_shift_params_cuda_s) * m_freq_shift_streams.size(),
                                        cudaMemcpyHostToDevice));

            __freq_shift_multi_chan<<<num_thread_blocks, MAX_NUM_CUDA_THREADS, shared_memory_size_in_bytes>>>(
                    m_freq_shift_params,
                    m_freq_shift_streams.size(),
                    cuda_float_complex_input_data,
                    m_max_num_input_samples);
            
            cudaErrChkAssert(cudaPeekAtLastError());
            
            /* wait for GPU to finish before accessing on host */
            cudaErrChkAssert(cudaDeviceSynchronize());
            
            timer.log_duration("Multi Chan Kernel Complete");

            /* copy output samples out of CUDA memory */
            for (uint32_t chan_idx = 0; chan_idx < m_freq_shift_streams.size(); ++chan_idx)
            {
                cudaErrChkAssert(cudaMemcpy(static_cast<void *>(out[chan_idx].data()),
                                 static_cast<void *>(m_freq_shift_streams[chan_idx]->get_freq_shift_out_data_ptr()),
                                 m_freq_shift_streams[chan_idx]->get_freq_shift_out_data_len() * sizeof(std::complex<float>),
                                 cudaMemcpyDeviceToHost));
            }
        }

        for (auto& chan_iter : m_freq_shift_streams)
        {
            chan_iter->add_freq_shift_samples_handled(in.size());
        }

        return out.size() > 0;
    }
}
