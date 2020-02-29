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
 * @file     falcon_dsp_stream.h
 * @author   OrthogonalHawk
 * @date     22-Feb-2020
 *
 * @brief    Base class for FALCON DSP library streams.
 *
 * @section  DESCRIPTION
 *
 * Defines a base class for FALCON DSP library streams. Streams are a construct
 *  used to manage a chain of related signal processing operations, particularly
 *  when using those operations together is tied to GPU hardware and memory
 *  management.
 *
 * @section  HISTORY
 *
 * 22-Feb-2020  OrthogonalHawk  File created.
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_STREAM_H__
#define __FALCON_DSP_STREAM_H__

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <mutex>

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
    
    /* @brief C++ implementation of a multi-rate channelizer class.
     * @description Builds on several separate C++ implementations from the FALCON
     *               DSP library.
     */
    class falcon_dsp_stream
    {
    public:
        
        falcon_dsp_stream(void) = default;
        virtual ~falcon_dsp_stream(void) = default;
        
        falcon_dsp_stream(const falcon_dsp_stream&) = default;
        
        virtual bool initialize(void);

    protected:

        std::recursive_mutex                m_mutex;
    };
}

#endif // __FALCON_DSP_STREAM_H__
