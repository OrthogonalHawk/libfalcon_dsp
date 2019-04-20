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
 * @file     falcon_dsp_math.h
 * @author   OrthogonalHawk
 * @date     19-Apr-2019
 *
 * @brief    Basic math functions; C++ and CUDA versions.
 *
 * @section  DESCRIPTION
 *
 * Defines various math functions supported by the FALCON DSP library.
 *
 * @section  HISTORY
 *
 * 19-Apr-2019  OrthogonalHawk  File created.
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_MATH_H__
#define __FALCON_DSP_MATH_H__

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <vector>

/******************************************************************************
 *                                 CONSTANTS
 *****************************************************************************/

/******************************************************************************
 *                              ENUMS & TYPEDEFS
 *****************************************************************************/

/******************************************************************************
 *                                  MACROS
 *****************************************************************************/

/******************************************************************************
 *                           FUNCTION DECLARATION
 *****************************************************************************/

namespace falcon_dsp
{
    template<typename T>
    void add_vector(std::vector<T>& in_a, std::vector<T>& in_b, std::vector<T>& out);
    
    template<typename T>
    void add_vector_cuda(std::vector<T>& in_a, std::vector<T>& in_b, std::vector<T>&out);
}

/******************************************************************************
 *                            CLASS DECLARATION
 *****************************************************************************/

#endif // __FALCON_DSP_MATH_H__
