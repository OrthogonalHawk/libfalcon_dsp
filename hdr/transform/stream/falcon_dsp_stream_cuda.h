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
 * @file     falcon_dsp_stream_cuda.h
 * @author   OrthogonalHawk
 * @date     01-Mar-2020
 *
 * @brief    Base class for FALCON DSP library CUDA streams.
 *
 * @section  DESCRIPTION
 *
 * Defines a base class for FALCON DSP library CUDA streams. Streams are a construct
 *  used to manage a chain of related signal processing operations, particularly
 *  when using those operations together is tied to GPU hardware and memory
 *  management.
 *
 * @section  HISTORY
 *
 * 01-Mar-2020  OrthogonalHawk  File created.
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_STREAM_CUDA_H__
#define __FALCON_DSP_STREAM_CUDA_H__

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

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
    
    /* @brief Base CUDA stream class
     * @description Builds on the C++ base stream class and adds CUDA-specific methods.
     */
    class falcon_dsp_stream_cuda : public falcon_dsp_stream
    {
    public:
        
        falcon_dsp_stream_cuda(void);
        ~falcon_dsp_stream_cuda(void);

        falcon_dsp_stream_cuda(const falcon_dsp_stream_cuda&) = default;
        
        bool initialize(void) override;
        virtual bool allocate_memory(uint32_t input_vector_len);
        virtual bool manage_state(uint32_t input_vector_len);
        bool reset_state(void);

        uint32_t get_max_num_cuda_threads(void) const { return m_max_num_cuda_threads; }
        bool set_max_num_cuda_threads(uint32_t max_num_cuda_threads);
        
    protected:

        virtual bool cleanup_memory(void);

        uint32_t                      m_max_num_cuda_threads;
    };
}

#endif // __FALCON_DSP_STREAM_CUDA_H__
