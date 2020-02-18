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
 * @file     falcon_dsp_fm_demod_cuda.h
 * @author   OrthogonalHawk
 * @date     18-Feb-2020
 *
 * @brief    Signal processing transformation functions for Frequency Modulation (FM)
 *            signal demodulation; CUDA version.
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
 * 18-Feb-2020  OrthogonalHawk  Created file.
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_SIGNALS_FM_DEMOD_CUDA_H__
#define __FALCON_DSP_SIGNALS_FM_DEMOD_CUDA_H__

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <complex>
#include <mutex>
#include <vector>

#include "signals/falcon_dsp_fm_demod.h"
#include "transform/falcon_dsp_multi_rate_channelizer_cuda.h"
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
    
    /* @brief CUDA implementation of an FM demodulator class.
     * @description By implementing the FM demodulator as a class interface instead
     *               of a simple function the user is able to demodulate an arbitrarily
     *               long input.
     */
    class falcon_dsp_fm_demodulator_cuda : public falcon_dsp_fm_demodulator
    {
    public:

        falcon_dsp_fm_demodulator_cuda(void);
        virtual ~falcon_dsp_fm_demodulator_cuda(void) = default;

        falcon_dsp_fm_demodulator_cuda(const falcon_dsp_fm_demodulator_cuda&) = delete;

        bool initialize(uint32_t input_sample_rate_in_sps,
                        int32_t signal_offset_from_dc_in_hz,
                        float deemphasis_time_constant_in_usecs = AMERICAS_FM_DEEMPHASIS_TIME_CONSTANT) override;

        void reset_state(void) override;
        virtual bool demod_mono(std::vector<std::complex<float>>& in, std::vector<int16_t>& out) override;

    protected:

        falcon_dsp_multi_rate_channelizer_cuda   m_shift_and_resample;
    };
}

#endif // __FALCON_DSP_SIGNALS_FM_DEMOD_CUDA_H__
