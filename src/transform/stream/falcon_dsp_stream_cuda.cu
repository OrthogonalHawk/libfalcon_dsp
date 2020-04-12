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
 * @file     falcon_dsp_stream_cuda.cu
 * @author   OrthogonalHawk
 * @date     01-Mar-2020
 *
 * @brief    Implements the base FALCON DSP CUDA stream class.
 *
 * @section  DESCRIPTION
 *
 * Implements the base FALCON DSP CUDA stream class, including helper methods that are
 *  common to all derived stream classes.
 *
 * @section  HISTORY
 *
 * 01-Mar-2020  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include "transform/stream/falcon_dsp_stream_cuda.h"
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

    falcon_dsp_stream_cuda::falcon_dsp_stream_cuda(void)
      : falcon_dsp_stream(),
        m_max_num_cuda_threads(1024)
    {
        /* no action required */
    }
    
    falcon_dsp_stream_cuda::~falcon_dsp_stream_cuda(void)
    {
        /* no action required */
    }
    
    bool falcon_dsp_stream_cuda::initialize(void)
    {
        return falcon_dsp_stream::initialize();
    }
    
    bool falcon_dsp_stream_cuda::allocate_memory(uint32_t input_vector_len)
    {
        /* no CUDA memory allocated by this base class */
        return true;
    }

    bool falcon_dsp_stream_cuda::manage_state(uint32_t input_vector_len)
    {
        /* the base class does not have state that needs managing */
        return true;
    }
    
    bool falcon_dsp_stream_cuda::reset_state(void)
    {
        return falcon_dsp_stream::reset_state();
    }
    
    bool falcon_dsp_stream_cuda::set_max_num_cuda_threads(uint32_t max_num_cuda_threads)
    {
        m_max_num_cuda_threads = max_num_cuda_threads;
        return true;
    }
    
    bool falcon_dsp_stream_cuda::cleanup_memory(void)
    {
        /* no CUDA memory allocated by this base class */
        return true;
    }
}
