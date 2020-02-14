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
 * @file     falcon_dsp_fm_demod.cc
 * @author   OrthogonalHawk
 * @date     11-Feb-2020
 *
 * @brief    Implements C++-based FM Demodulator.
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

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <iostream>
#include <stdint.h>

#include "signals/falcon_dsp_fm_demod.h"
#include "transform/falcon_dsp_fir_filter.h"
#include "utilities/falcon_dsp_utils.h"

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
     *                        FUNCTION IMPLEMENTATION
     *****************************************************************************/    

    /******************************************************************************
     *                           CLASS IMPLEMENTATION
     *****************************************************************************/
    
    falcon_dsp_fm_demodulator::falcon_dsp_fm_demodulator(void)
      : m_initialized(false),
        m_freq_shifter(),
        m_signal_isolation_decimator(),
        m_polar_discriminator(),
        m_deemphasis_filter(),
        m_mono_signal_decimator()
    { }
    
    bool falcon_dsp_fm_demodulator::initialize(uint32_t input_sample_rate_in_sps,
                                               int32_t signal_offset_from_dc_in_hz,
                                               float deemphasis_time_constant_in_usecs)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        if (m_initialized)
        {
            /* only let the user call this function once */
            return false;
        }
        
        /* sanity check the inputs */
        if (deemphasis_time_constant_in_usecs != AMERICAS_FM_DEEMPHASIS_TIME_CONSTANT &&
            deemphasis_time_constant_in_usecs != EUROPE_ASIA_FM_DEEMPHASIS_TIME_CONSTANT)
        {
            fprintf(stderr, "ERROR: Unsupported deemphasis time constant\n");
            return false;
        }
            
        m_initialized |= m_freq_shifter.initialize(input_sample_rate_in_sps, signal_offset_from_dc_in_hz);
        if (!m_initialized)
        {
            fprintf(stderr, "ERROR: Failed to initialize frequency shift operation\n");
            return false;
        }
        
        uint32_t up_rate, down_rate;
        std::vector<std::complex<float>> coeffs;
        m_initialized &= falcon_dsp::get_resample_fir_params(input_sample_rate_in_sps, FM_RADIO_SAMPLE_RATE_IN_SPS,
                                                             up_rate, down_rate, coeffs);
        if (!m_initialized)
        {
            fprintf(stderr, "ERROR: Failed to obtain resample filter parameters %u sps -> %u sps\n",
                input_sample_rate_in_sps, FM_RADIO_SAMPLE_RATE_IN_SPS);
            return false;
        }
        
        m_initialized &= m_signal_isolation_decimator.initialize(up_rate, down_rate, coeffs);
        if (!m_initialized)
        {
            fprintf(stderr, "ERROR: Failed to initialize FM signal isolation decimator\n");
            return false;
        }
        
        /* polar discriminator does not need to be initialized */
        
        /* initialize the deemphasis filter based on the provided time constant. note that
         *  the following values assume a 200 kHz downsampled frequency and can be
         *  generated with the following Python code:
         * >>> d = 2000000 * 75e-6
         * >>> x = np.exp(-1/d)
         * >>> b, a = [1-x], [1,-x]
         */
        std::vector<std::complex<float>> deemphasis_b_coeffs;
        std::vector<std::complex<float>> deemphasis_a_coeffs;
        if (deemphasis_time_constant_in_usecs == AMERICAS_FM_DEEMPHASIS_TIME_CONSTANT)
        {
            deemphasis_b_coeffs.push_back(std::complex<float>(0.0066444937449655628, 0));
            deemphasis_a_coeffs.push_back(std::complex<float>(1.0, 0.0));
            deemphasis_a_coeffs.push_back(std::complex<float>(-0.99335550625503444, 0.0));
        }
        else
        {
            deemphasis_b_coeffs.push_back(std::complex<float>(0.0099501662508318933, 0));
            deemphasis_a_coeffs.push_back(std::complex<float>(1.0, 0.0));
            deemphasis_a_coeffs.push_back(std::complex<float>(-0.99004983374916811, 0.0));
        }

        m_initialized &= m_deemphasis_filter.initialize(deemphasis_b_coeffs, deemphasis_a_coeffs);
        if (!m_initialized)
        {
            fprintf(stderr, "ERROR: Failed to initialize FM deemphasis filter\n");
            return false;
        }
        
        uint32_t mono_up_rate, mono_down_rate;
        std::vector<std::complex<float>> mono_coeffs;
        m_initialized &= falcon_dsp::get_resample_fir_params(FM_RADIO_SAMPLE_RATE_IN_SPS, FM_RADIO_AUDIO_SAMPLE_RATE_IN_SPS,
                                                             mono_up_rate, mono_down_rate, mono_coeffs);
        if (!m_initialized)
        {
            fprintf(stderr, "ERROR: Failed to obtain mono decimate filter parameters %u sps -> %u sps\n",
                FM_RADIO_SAMPLE_RATE_IN_SPS, FM_RADIO_AUDIO_SAMPLE_RATE_IN_SPS);
            return false;
        }

        m_initialized &= m_mono_signal_decimator.initialize(mono_up_rate, mono_down_rate, mono_coeffs);
        if (!m_initialized)
        {
            fprintf(stderr, "ERROR: Failed to initialize mono decimator\n");
            return false;
        }
        
        /* initialization complete */
        return m_initialized;
    }
    
    void falcon_dsp_fm_demodulator::reset_state(void)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        /* reset the state information; an end-user might invoke this function if processing
         *  non-continuous data */
        
        m_freq_shifter.reset_state();
        m_signal_isolation_decimator.reset_state();
        m_polar_discriminator.reset_state();
        m_deemphasis_filter.reset_state();
        m_mono_signal_decimator.reset_state();
    }

    bool falcon_dsp_fm_demodulator::demod_mono(std::vector<std::complex<float>>& in, std::vector<int16_t>& out)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        out.clear();
        
        if (!m_initialized)
        {
            return false;
        }
        
        std::vector<std::complex<float>> freq_shifted_data;
        m_freq_shifter.apply(in, freq_shifted_data);
        
        std::vector<std::complex<float>> resampled_at_200khz_data;
        m_signal_isolation_decimator.apply(freq_shifted_data, resampled_at_200khz_data);
        
        std::vector<float> polar_discrim_out_data;
        m_polar_discriminator.apply(resampled_at_200khz_data, polar_discrim_out_data);
        
        std::vector<float> emphasis_filtered_data;
        m_deemphasis_filter.apply(polar_discrim_out_data, emphasis_filtered_data);
        
        std::vector<float> mono_audio_signal_data;
        m_mono_signal_decimator.apply(emphasis_filtered_data, mono_audio_signal_data);
        
        out.reserve(mono_audio_signal_data.size());
        for (auto mono_iter = mono_audio_signal_data.begin();
             mono_iter != mono_audio_signal_data.end();
             ++mono_iter)
        {
            out.push_back(*mono_iter);
        }
        
        return out.size() > 0;
    }
}
