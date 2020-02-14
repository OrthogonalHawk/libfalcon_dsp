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
 * @file     falcon_dsp_fm_demod.h
 * @author   OrthogonalHawk
 * @date     11-Feb-2020
 *
 * @brief    Signal processing transformation functions for Frequency Modulation (FM)
 *            signal demodulation; C++ versions.
 *
 * @section  DESCRIPTION
 *
 * Defines a set of signal processing routines to demodulate signals that are
 *  modulated using a Frequency Modulation (FM) technique common to public radio
 *  broadcasts in the United States. This implementation is based on the excellent
 *  tutorial provided at: https://witestlab.poly.edu/blog/capture-and-decode-fm-radio/
 *
 * @section  HISTORY
 *
 * 11-Feb-2020  OrthogonalHawk  Created file.
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_SIGNALS_FM_DEMOD_H__
#define __FALCON_DSP_SIGNALS_FM_DEMOD_H__

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <complex>
#include <mutex>
#include <vector>

#include "transform/falcon_dsp_freq_shift.h"
#include "transform/falcon_dsp_iir_filter.h"
#include "transform/falcon_dsp_polar_discriminator.h"
#include "transform/falcon_dsp_polyphase_resampler.h"

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
    
    /* @brief C++ implementation of an FM demodulator class.
     * @description By implementing the FM demodulator as a class interface instead
     *               of a simple function the user is able to demodulate an arbitrarily
     *               long input.
     */
    class falcon_dsp_fm_demodulator
    {
    public:

        /* values from https://www.radiomuseum.org/forum/fm_pre_emphasis_and_de_emphasis.html */
        static constexpr float AMERICAS_FM_DEEMPHASIS_TIME_CONSTANT = 75e-6;
        static constexpr float EUROPE_ASIA_FM_DEEMPHASIS_TIME_CONSTANT = 50e-6;

        static const uint32_t FM_RADIO_SAMPLE_RATE_IN_SPS = 200000;
        static const uint32_t FM_RADIO_AUDIO_SAMPLE_RATE_IN_SPS = 44000;

        falcon_dsp_fm_demodulator(void);
        virtual ~falcon_dsp_fm_demodulator(void) = default;

        falcon_dsp_fm_demodulator(const falcon_dsp_fm_demodulator&) = delete;

        bool initialize(uint32_t input_sample_rate_in_sps,
                        int32_t signal_offset_from_dc_in_hz,
                        float deemphasis_time_constant_in_usecs = AMERICAS_FM_DEEMPHASIS_TIME_CONSTANT);

        void reset_state(void);
        virtual bool demod_mono(std::vector<std::complex<float>>& in, std::vector<int16_t>& out);

    protected:
    
        std::mutex                               m_mutex;
        bool                                     m_initialized;
        falcon_dsp_freq_shift                    m_freq_shifter;
        falcon_dsp_polyphase_resampler           m_signal_isolation_decimator;
        falcon_dsp_polar_discriminator           m_polar_discriminator;
        falcon_dsp_iir_filter                    m_deemphasis_filter;
        falcon_dsp_polyphase_resampler           m_mono_signal_decimator;
    };
}

#endif // __FALCON_DSP_SIGNALS_FM_DEMOD_H__
