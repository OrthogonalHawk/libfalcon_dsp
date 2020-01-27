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
 * @file     falcon_dsp_cuda_utils.h
 * @author   OrthogonalHawk
 * @date     27-Jan-2020
 *
 * @brief    Digital Signal Processing utility functions; CUDA implementation.
 *
 * @section  DESCRIPTION
 *
 * Defines various utility functions supported by the FALCON DSP library.
 *
 * @section  HISTORY
 *
 * 27-Jan-2020  OrthogonalHawk  File created.
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_CUDA_UTILS_H__
#define __FALCON_DSP_CUDA_UTILS_H__

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <complex>
#include <vector>

/******************************************************************************
 *                                 CONSTANTS
 *****************************************************************************/

/******************************************************************************
 *                              ENUMS & TYPEDEFS
 *****************************************************************************/

namespace falcon_dsp
{
    /******************************************************************************
     *                           FUNCTION DECLARATION
     *****************************************************************************/

    /* @brief CUDA error checking function
     * @description Checks for errors after invoking a CUDA API function, prints the
     *               result to stderr and optionally aborts execution.
     * @param  code               - error code to check
     * @param  file    - source file where the error check is occurring
     * @param  line    - source file line where the error check is occurring
     * @param  abort   - boolean indicating whether to abort program execution on error
     * @return None
     */
    void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false);
    
    /******************************************************************************
     *                                  MACROS
     *****************************************************************************/
 
    #define cudaErrChk(ans) { falcon_dsp::gpuAssert((ans), __FILE__, __LINE__, false); }
    #define cudaErrChkAssert(ans) { falcon_dsp::gpuAssert((ans), __FILE__, __LINE__, true); }
    
    /******************************************************************************
     *                            CLASS DECLARATION
     *****************************************************************************/
}

#endif // __FALCON_DSP_CUDA_UTILS_H__
