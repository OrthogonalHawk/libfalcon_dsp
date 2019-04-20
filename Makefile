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
# Makefile for the FALCON DSP Library
#
#     See ../falcon_makefiles/Makefile.libs for usage
#
###############################################################################

FALCON_PATH = $(realpath $(CURDIR)/..)
PLATFORM_BUILD=1
export PLATFORM_BUILD
BUILD_OBJS_DIR = build$(LIB_SUFFIX)

###############################################################################
# LIBRARY
###############################################################################

LIB = lib/libfalcon_dsp$(LIB_SUFFIX).a

###############################################################################
# SOURCES
###############################################################################

CC_SOURCES = \
    src/math/falcon_dsp_add.cc \
    
CUDA_SOURCES = \
    src/add.cu \
    
###############################################################################
# Include ../falcon_makefiles/Makefile.libs for rules
###############################################################################

include ../falcon_makefiles/Makefile.libs

###############################################################################
# Adjust *FLAGS and paths as necessary
###############################################################################

CPPFLAGS += -Werror -Wall -Wextra -Wcast-align -Wno-type-limits
CPPFLAGS += -std=c++11 -O3

INC_PATH += \
    -I./hdr \
    -I./hdr/math \
