###############################################################################
#
# MIT License
#
# Copyright (c) 2019 OrthogonalHawk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
###############################################################################

###############################################################################
# Makefile for the FALCON DSP Library Unit Tests
#
#     See ../../falcon_makefiles/Makefile.apps for usage
#
###############################################################################

FALCON_PATH = $(realpath $(CURDIR)/../..)
PLATFORM_BUILD=1
export PLATFORM_BUILD
#BUILD_OBJS_DIR = build$(LIB_SUFFIX)

###############################################################################
# LIBRARY
###############################################################################

EXE = bin/falcon_dsp_unit_tests

###############################################################################
# SOURCES
###############################################################################

CC_SOURCES = \
    ../src/math/falcon_dsp_add.cc \
    ../src/resample/falcon_dsp_resample.cc \
    ../src/resample/falcon_dsp_polyphase_resampler.cc \
    ../src/transform/falcon_dsp_freq_shift.cc \
    ../src/transform/falcon_dsp_linear_filter.cc \
    ../src/transform/falcon_dsp_polar_discriminator.cc \
    ../src/utilities/falcon_dsp_utils.cc \
    tests/math/falcon_dsp_math_unit_tests.cc \
    tests/resample/falcon_dsp_polyphase_resample_cpp_unit_tests.cc \
    tests/transform/falcon_dsp_freq_shift_cpp_unit_tests.cc \
    tests/transform/falcon_dsp_iir_filter_cpp_unit_tests.cc \
    tests/transform/falcon_dsp_linear_filter_cpp_unit_tests.cc \
    tests/transform/falcon_dsp_polar_discriminator_cpp_unit_tests.cc \
    tests/utilities/falcon_dsp_utils_unit_tests.cc \
    /usr/src/googletest/googletest/src/gtest-all.cc \
    /usr/src/googletest/googlemock/src/gmock-all.cc \
    unit_test_main.cc \
    
CUDA_SOURCES = \
    ../src/math/falcon_dsp_add_cuda.cu \
    ../src/resample/falcon_dsp_polyphase_resampler_cuda.cu \
    ../src/resample/falcon_dsp_resample_cuda.cu \
    ../src/transform/falcon_dsp_freq_shift_cuda.cu \
    tests/resample/falcon_dsp_polyphase_resample_cuda_unit_tests.cu \
    tests/transform/falcon_dsp_freq_shift_cuda_unit_tests.cu \
    unit_test_cuda_init.cu \
    
###############################################################################
# Include ../../falcon_makefiles/Makefile.apps for rules
###############################################################################

include ../../falcon_makefiles/Makefile.apps

###############################################################################
# Adjust *FLAGS and paths as necessary
###############################################################################

CPPFLAGS += -Werror -Wall -Wextra -Wcast-align -Wno-type-limits
CPPFLAGS += -std=c++11 -O3

INC_PATH += \
    -I../hdr \
    -I../hdr/math \
    -I../hdr/resample \
    -I/usr/src/googletest/googletest \
    -I/usr/src/googletest/googlemock \

NVCCFLAGS += $(INC_PATH)
