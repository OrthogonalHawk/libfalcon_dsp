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
 * @file     falcon_dsp_utils.cc
 * @author   OrthogonalHawk
 * @date     10-May-2019
 *
 * @brief    C++ implementation of general purpose Digital Signal Processing
 *            utility functions.
 *
 * @section  DESCRIPTION
 *
 * Implements C++ versions of general purpose Digital Signal Processing utility
 *  functions; many are MATLAB clones.
 *
 * @section  HISTORY
 *
 * 10-May-2019  OrthogonalHawk  File created.
 * 26-May-2019  OrthogonalHawk  Added binary file read/write functions.
 * 04-Jun-2019  OrthogonalHawk  Added floating-point ASCII file read functions.
 * 24-Aug-2019  OrthogonalHawk  Adding rational fraction approximation function.
 *
 *****************************************************************************/

/******************************************************************************
 *                               INCLUDE_FILES
 *****************************************************************************/

#include <cmath>
#include <iostream>
#include <fstream>
#include <stdint.h>

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
 *                           FUNCTION IMPLEMENTATION
 *****************************************************************************/

namespace falcon_dsp
{
    /* @brief Computes the greatest common denominator between two numbers
     * @param[in] a - first value to consider
     * @param[in] b - second value to consider
     * @return Returns the greatest common denominator between a and b
     */
    uint64_t calculate_gcd(uint64_t a, uint64_t b)
    {
        return b == 0 ? a : calculate_gcd(b, a % b);    
    }
    
    /* @brief Computes the least common multiple between two numbers
     * @param[in] a - first value to consider
     * @param[in] b - second value to consider
     * @return Returns the least common multiple between a and b
     */
    uint64_t calculate_lcm(uint64_t a, uint64_t b)
    {
        uint64_t gcd = calculate_gcd(a, b);
        return ((a / gcd) * b);
    }
    
    /* @brief Returns the factorial of x
     * @param[in] - input value
     * @return Returns the factorial of x
     */
    uint64_t factorial(uint32_t x)
    {
        uint64_t ret = 1;
        for (uint32_t ii = 1; ii <= x; ++ii)
        {
            ret *= ii;    
        }
        
        return ret;
    }
    
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
    bool firlpf(uint32_t M, double fc, double fsamp, std::vector<double>& coeffs)
    {
        /* sanity check inputs */
        if (M < 1)
        {
            return false;
        }
        
        bool odd = true;
        if (M % 2 == 0)
        {
            odd = false;
        }
        
        coeffs.clear();
        coeffs.reserve(M);
        
        /* normalize fc and w_c so that pi is equal to the Nyquist angular frequency */
        fc = fc / fsamp;
        double w_c = 2 * M_PI * fc;
        
        /* create the low-pass filter */
        int32_t mid = M / 2;
        for (int32_t ii = 0; ii < static_cast<int32_t>(M); ++ii)
        {
            if (!odd)
            {
                if (ii + 1 == mid || ii == mid)
                {
                    coeffs.push_back(2 * fc);   
                }
                else
                {
                    coeffs.push_back(sin(w_c * (ii - mid)) / (M_PI * (ii - mid)));   
                }
            }
            else
            {
                if (ii == mid)
                {
                    coeffs.push_back(2 * fc);   
                }
                else
                {
                    coeffs.push_back(sin(w_c * (ii - mid)) / (M_PI * (ii - mid)));   
                }
            }
        }
        
        return true;
    }
    
    /* @brief Rational fraction approximation
     * @param[in] f       - the number to convert
     * @param[in] md      - max denominator value
     * @param[out] num    - computed numerator value
     * @param[out] denom  - computed denominator value
     * @attribution This function was obtained from https://rosettacode.org/wiki/Convert_decimal_number_to_rational#C
     * @note The maximum denominator value is specified because machine floating
     *        point number has a finite resolution (10e-16 ish for 64 bit double),
     *        so specifying a "best match with minimal error" is often wrong,
     *        because one can always just retrieve the significand and return that
     *        divided by 2**52, which is in a sense accurate, but generally not very
     *        useful:
     *          1.0/7.0 would be "2573485501354569/18014398509481984", for example.
     */
    void rat_approx(double f, int64_t md, int64_t &num, int64_t &denom)
    {
        /*  a: continued fraction coefficients. */
        int64_t a, h[3] = { 0, 1, 0 }, k[3] = { 1, 0, 0 };
        int64_t x, d, n = 1;
        int32_t i, neg = 0;

        /* sanity check inputs */
        if (md <= 1)
        {
            denom = 1;
            num = static_cast<int64_t>(f);
            return;
        }

        /* handle negative numbers as well */
        if (f < 0)
        {
            neg = 1;
            f = -f;
        }

        while (f != floor(f))
        {
            n <<= 1;
            f *= 2;
        }

        d = f;

        /* continued fraction and check denominator each step */
        for (i = 0; i < 64; i++)
        {
            a = n ? d / n : 0;
            if (i && !a)
            {
                break;
            }

            x = d;
            d = n;
            n = x % n;

            x = a;
		    if (k[1] * a + k[0] >= md)
            {
			    x = (md - k[0]) / k[1];
			    if (x * 2 >= a || k[1] >= md)
                {
				    i = 65;
                }
			    else
                {
				    break;
                }
            }

            h[2] = x * h[1] + h[0]; h[0] = h[1]; h[1] = h[2];
            k[2] = x * k[1] + k[0]; k[0] = k[1]; k[1] = k[2];
        }

        denom = k[1];
        num = neg ? -h[1] : h[1];
    }

    /* @brief Writes a complex data vector to a file
     * @param file_name   - Name of the file to write
     * @param type        - File type/format
     * @param data        - Data vector to write
     * @return true if the file was written successfully; false otherwise.
     */
    bool write_complex_data_to_file(std::string file_name, file_type_e type, std::vector<std::complex<int16_t>>& data)
    {
        bool ret = true;
        
        if (type == file_type_e::BINARY)
        {
            /* attempt to open the output file */
            std::ofstream output_file(file_name, std::ios::out | std::ios::binary);
            if (!output_file.is_open())
            {
                return false;
            }

            for (auto iter = data.begin(); iter != data.end(); ++iter)
            {
                int16_t real = iter->real();
                int16_t imag = iter->imag();

                output_file.write(reinterpret_cast<char *>(&real), sizeof(int16_t));
                output_file.write(reinterpret_cast<char *>(&imag), sizeof(int16_t));

                if (!output_file.good())
                {
                    return false;
                }
            }

            output_file.close();
        }
        else if (type == file_type_e::ASCII)
        {
            /* attempt to open the output file */
            std::ofstream output_file(file_name, std::ios::out);
            if (!output_file.is_open())
            {
                return false;
            }

            for (auto iter = data.begin(); iter != data.end(); ++iter)
            {
                int16_t real = iter->real();
                int16_t imag = iter->imag();

                output_file << real << " " << imag << std::endl;
                
                if (!output_file.good())
                {
                    return false;
                }
            }

            output_file.close();
        }
        else
        {
            ret = false;
        }
        
        return ret;
    }
    
    /* @brief Writes a complex data vector to a file
     * @param file_name   - Name of the file to write
     * @param type        - File type/format
     * @param data        - Data vector to write
     * @return true if the file was written successfully; false otherwise.
     */
    bool write_complex_data_to_file(std::string file_name, file_type_e type, std::vector<std::complex<float>>& data)
    {
        bool ret = true;
        
        if (type == file_type_e::BINARY)
        {
            /* currently not supported */
            ret = false;
        }
        else if (type == file_type_e::ASCII)
        {
            /* attempt to open the output file */
            std::ofstream output_file(file_name, std::ios::out);
            if (!output_file.is_open())
            {
                return false;
            }

            for (auto iter = data.begin(); iter != data.end(); ++iter)
            {
                float real = iter->real();
                float imag = iter->imag();

                output_file << real << " " << imag << std::endl;
                
                if (!output_file.good())
                {
                    return false;
                }
            }

            output_file.close();
        }
        else
        {
            ret = false;
        }
        
        return ret;
    }
    
    /* @brief Reads complex data vector from a file
     * @param[in] file_name   - Name of the file to read
     * @param[in] type        - File type/format
     * @param[out] data       - Data vector read from the file
     * @return true if the file was read successfully; false otherwise.
     */
    bool read_complex_data_from_file(std::string file_name, file_type_e type, std::vector<std::complex<int16_t>>& data)
    {
        bool ret = true;
        data.clear();
        
        if (type == file_type_e::BINARY)
        {
            /* attempt to open the input file */
            std::ifstream input_file(file_name, std::ios::in | std::ios::binary);
            if (!input_file.is_open())
            {
                std::cerr << "Unable to open BINARY input file " << file_name << std::endl;
                return false;
            }

            /* get the file size in bytes */
            input_file.seekg(0, std::ios::end);
            auto file_size_in_bytes = input_file.tellg();
            input_file.seekg(0, std::ios::beg);

            uint32_t expected_number_of_samples = file_size_in_bytes / (sizeof(int16_t) * 2);
            for (uint32_t ii = 0; ii < expected_number_of_samples; ++ii)
            {
                char buf[4];
                input_file.read(buf, 4);

                /* note that this code assumes that the binary file was written in a
                 *  little endian format */
                int16_t real = static_cast<int16_t>(((buf[1] << 8) & 0xFF00) | buf[0]);
                int16_t imag = static_cast<int16_t>(((buf[3] << 8) & 0XFF00) | buf[2]);

                data.push_back(std::complex<int16_t>(real, imag));

                if (!input_file.good())
                {
                    data.clear();
                    
                    std::cerr << "Reached unexpected EOF at sample index " << ii
                              << " of " << expected_number_of_samples << std::endl;
                    return false;
                }
            }
        }
        else if (type == file_type_e::ASCII)
        {
            /* attempt to open the input file */
            std::ifstream input_file(file_name, std::ios::in);
            if (!input_file.is_open())
            {
                std::cerr << "Unable to open ASCII input file " << file_name << std::endl;
                return false;
            }

            int16_t real;
            int16_t imag;
            while (input_file >> real >> imag)
            {
                data.push_back(std::complex<int16_t>(real, imag));
            }
        }
        else
        {
            std::cerr << "Unable to open input file " << file_name
                      << " with unsupported type " << static_cast<uint32_t>(type) << std::endl;
            ret = false;
        }
        
        return ret;
    }
    
    /* @brief Reads complex data vector from a file
     * @param[in] file_name   - Name of the file to read
     * @param[in] type        - File type/format
     * @param[out] data       - Data vector read from the file
     * @return true if the file was read successfully; false otherwise.
     */
    bool read_complex_data_from_file(std::string file_name, file_type_e type, std::vector<std::complex<float>>& data)
    {
        bool ret = true;
        data.clear();
        
        if (type == file_type_e::BINARY)
        {
            /* currently not supported */
            ret = false;
        }
        else if (type == file_type_e::ASCII)
        {
            /* attempt to open the input file */
            std::ifstream input_file(file_name, std::ios::in);
            if (!input_file.is_open())
            {
                return false;
            }

            float real;
            float imag;
            while (input_file >> real >> imag)
            {
                data.push_back(std::complex<float>(real, imag));
            }
        }
        else
        {
            ret = false;
        }
        
        return ret;
    }
}

/******************************************************************************
 *                            CLASS IMPLEMENTATION
 *****************************************************************************/
