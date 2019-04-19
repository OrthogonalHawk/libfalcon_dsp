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

BUILD_OBJS_DIR = build$(LIB_SUFFIX)

###############################################################################
# LIBRARY
###############################################################################

EXE = bin/falcon_dsp_unit_tests

###############################################################################
# SOURCES
###############################################################################

CC_SOURCES = \
    /usr/src/googletest/googletest/src/gtest-all.cc \
    /usr/src/googletest/googlemock/src/gmock-all.cc \
    unit_test_main.cc \
    
CUDA_SOURCES = \
    
###############################################################################
# Include ../../falcon_makefiles/Makefile.apps for rules
###############################################################################

include ../../falcon_makefiles/Makefile.apps

###############################################################################
# Adjust *FLAGS and paths as necessary
###############################################################################

CPPFLAGS += -Werror -Wall -Wextra -Wcast-align -Wno-type-limits
CPPFLAGS += -std=c++11 -O3

INC_PATH += -I/usr/src/googletest/googletest \
            -I/usr/src/googletest/googlemock \