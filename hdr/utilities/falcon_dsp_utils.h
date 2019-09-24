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
 * @file     falcon_dsp_utils.h
 * @author   OrthogonalHawk
 * @date     10-May-2019
 *
 * @brief    Digital Signal Processing utility functions; C++ implementation.
 *
 * @section  DESCRIPTION
 *
 * Defines various utility functions supported by the FALCON DSP library.
 *
 * @section  HISTORY
 *
 * 10-May-2019  OrthogonalHawk  File created.
 * 26-May-2019  OrthogonalHawk  Added binary file read/write functions.
 * 04-Jun-2019  OrthogonalHawk  Added floating-point ASCII file read functions.
 * 24-Aug-2019  OrthogonalHawk  Adding rational fraction approximation function.
 *
 *****************************************************************************/

#ifndef __FALCON_DSP_UTILS_H__
#define __FALCON_DSP_UTILS_H__

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

/******************************************************************************
 *                                  MACROS
 *****************************************************************************/

/******************************************************************************
 *                           FUNCTION DECLARATION
 *****************************************************************************/

namespace falcon_dsp
{
    enum class file_type_e : uint32_t
    {
        BINARY = 0,
        ASCII
    };

    /* @brief Computes filter delay in terms of samples
     * @description Computes the filter delay in samples based on the provided
     *               filter coefficients and resampling ratio.
     * @param[in]  num_coeffs               - number of filter coefficients
     * @param[in]  in_sample_rate_in_sps    - input data sample rate in samples per second
     * @param[int] out_sample_rate_in_sps   - output data sample rate in samples per second
     * @return Filter delay  in samples
     */
    uint32_t calculate_filter_delay(uint32_t num_coeffs, uint32_t in_sample_rate_in_sps, uint32_t out_sample_rate_in_sps);

    /* @brief Computes the greatest common denominator between two numbers
     * @param[in] a - first value to consider
     * @param[in] b - second value to consider
     * @return Returns the greatest common denominator between a and b
     */
    uint64_t calculate_gcd(uint64_t a, uint64_t b);
    
    /* @brief Computes the least common multiple between two numbers
     * @param[in] a - first value to consider
     * @param[in] b - second value to consider
     * @return Returns the least common multiple between a and b
     */
    uint64_t calculate_lcm(uint64_t a, uint64_t b);
    
    /* @brief Returns the factorial of x
     * @param[in] - input value
     * @return Returns the factorial of x
     */
    uint64_t factorial(uint32_t x);

    /* @brief FIR Low-Pass Filter Coefficient generation
     * @description Source code from http://digitalsoundandmusic.com/download/programmingexercises/Creating_FIR_Filters_in_C++.pdf
     * @param[in]  M      - filter length in number of taps
     * @param[in]  fc     - cutoff frequency in Hz (un-normalized). Note that a normalized
     *                       cutoff frequency can be used, but then fsamp should be set
     *                       to 1 so that is not used to normalize fc again.
     * @param[in]  fsamp  - sampling frequency of the signal to be filtered in Hz. fsamp
     *                       is used to normalize the cutoff frequency.
     * @param[out] coeffs - FIR filter coefficients
     * @return true if the coefficients were generated successfully; false otherwise.
     */
    bool firlpf(uint32_t M, double fc, double fsamp, std::vector<double>& coeffs);
    
    /* @brief Rational fraction approximation
     * @param[in] f       - the number to convert
     * @param[in] md      - max denominator value
     * @param[out] num    - computed numerator value
     * @param[out] denom  - computed denominator value
     * @attribution This function was obtained from https://rosettacode.org/wiki/Convert_decimal_number_to_rational#C
     */
    void rat_approx(double f, int64_t md, int64_t &num, int64_t &denom);
    
    /* @brief Writes a complex data vector to a file
     * @param file_name   - Name of the file to write
     * @param data        - Data vector to write
     * @return true if the file was written successfully; false otherwise.
     */
    bool write_complex_data_to_file(std::string file_name, file_type_e type, std::vector<std::complex<int16_t>>& data);
    bool write_complex_data_to_file(std::string file_name, file_type_e type, std::vector<std::complex<float>>& data);
    
    /* @brief Reads complex data vector from a file
     * @param[in] file_name   - Name of the file to read
     * @param[out] data       - Data vector read from the file
     * @return true if the file was read successfully; false otherwise.
     */
    bool read_complex_data_from_file(std::string file_name, file_type_e, std::vector<std::complex<int16_t>>& data);
    bool read_complex_data_from_file(std::string file_name, file_type_e, std::vector<std::complex<float>>& data);
}

/******************************************************************************
 *                            CLASS DECLARATION
 *****************************************************************************/

#endif // __FALCON_DSP_UTILS_H__
