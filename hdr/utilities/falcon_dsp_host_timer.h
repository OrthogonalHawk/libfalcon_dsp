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
 * @file     falcon_dsp_host_timer.h
 * @author   OrthogonalHawk
 * @date     23-Jan-2020
 *
 * @brief    Timer class to help assess performance in host/CPU programs.
 *
 * @section  DESCRIPTION
 *
 * Defines a timer class to help assess performance in the FALCON DSP library.
 *
 * @section  HISTORY
 *
 * 23-Jan-2020  OrthogonalHawk  File created.
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_HOST_TIMER_H__
#define __FALCON_DSP_HOST_TIMER_H__

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <chrono>
#include <complex>
#include <mutex>
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
    /******************************************************************************
     *                            CLASS DECLARATION
     *****************************************************************************/
    class falcon_dsp_host_timer
    {
    public:
    
        falcon_dsp_host_timer(void);
        falcon_dsp_host_timer(std::string timer_name);
        falcon_dsp_host_timer(std::string timer_name, bool logging_enabled);
        
        ~falcon_dsp_host_timer(void);
    
        falcon_dsp_host_timer(const falcon_dsp_host_timer&) = delete;
    
        /* gets the current timer duration in ms */
        float get_duration_in_ms(void);
    
        /* prints a message with the current timer duration */
        void log_duration(const std::string message);
        
        /* resets the timer start time to now */
        void reset(void);
        
        /* stops the timer */
        void stop(void);
        
    private:
    
        std::mutex                                         m_mutex;
        std::string                                        m_timer_name;
        bool                                               m_logging_enabled;
        bool                                               m_running;
        std::chrono::high_resolution_clock::time_point     m_start;
        std::chrono::high_resolution_clock::time_point     m_stop;
    };
}

#endif // __FALCON_DSP_HOST_TIMER_H__
