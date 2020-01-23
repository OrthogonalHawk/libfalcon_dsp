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
 * @file     falcon_dsp_host_timer.cc
 * @author   OrthogonalHawk
 * @date     23-Jan-2020
 *
 * @brief    Implements a timer that is useful for debugging the FALCON DSP library.
 *
 * @section  DESCRIPTION
 *
 * Implements a host/CPU-based timer class that facilitates timing host/CPU
 *  operations within the FALCON DSP library.
 *
 * @section  HISTORY
 *
 * 23-Jan-2020  OrthogonalHawk  File created.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <iostream>
#include <stdint.h>

#include "utilities/falcon_dsp_host_timer.h"

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
 *                            CLASS IMPLEMENTATION
 *****************************************************************************/

namespace falcon_dsp
{
    /******************************************************************************
     *                           CLASS IMPLEMENTATION
     *****************************************************************************/
    
    falcon_dsp_host_timer::falcon_dsp_host_timer(std::string timer_name)
      : m_timer_name(timer_name),
        m_running(true)
    {
        m_start = std::chrono::high_resolution_clock::now();
    }
    
    falcon_dsp_host_timer::~falcon_dsp_host_timer(void)
    { }

    /* gets the current timer duration in ms */
    float falcon_dsp_host_timer::get_duration_in_ms(void)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        std::chrono::duration<double, std::milli> duration_ms = std::chrono::high_resolution_clock::now() - m_start;
        if (!m_running)
        {
            duration_ms = m_stop - m_start;
        }
        
        return duration_ms.count();
    }
    
    /* prints a message with the current timer duration */
    void falcon_dsp_host_timer::log_duration(const std::string message)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        std::chrono::duration<double, std::milli> duration_ms = std::chrono::high_resolution_clock::now() - m_start;
        if (!m_running)
        {
            duration_ms = m_stop - m_start;
        }
        
        std::cout << "[" << m_timer_name << "] Event: " << message << " Elapsed ms: " << duration_ms.count() << std::endl;
    }
        
    /* resets the timer start time to now */
    void falcon_dsp_host_timer::reset(void)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        m_start = std::chrono::high_resolution_clock::now();
        m_running = true;
    }
        
    /* stops the timer */
    void falcon_dsp_host_timer::stop(void)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        m_stop = std::chrono::high_resolution_clock::now();
        m_running = false;
    }
}
