# FALCON Digital Signal Processing (DSP) Library

## Overview
The **libfalcon\_dsp** library provides various Digital Signal Processing routines
 that I have found useful. In general, I've tried to include both C/C++ and CUDA
 versions of these routines along with unit tests for each. The primary development
 environment for this library has been NVIDIA Jetson embedded GPU boards, such as
 the TX2 and the Nano. These environments allow for native development, build,
 and test in a CUDA-compatible environment which reduces the need for containerization
 and cross-compilation.

## Dependencies
The **libfalcon\_dsp** library depends on several other FALCON libraries. The following
 table lists the libraries and the currently supported version/tag/sha of each library.
 Note that other library versions may work as well, but the versions from this table
 are used in the current unit tests and builds.

|LIBRARY|TAG|
|:-----:|:---:|
|[falcon_makefiles](https://github.com/OrthogonalHawk/falcon_makefiles)|master|   <!-- REQUIRED_BUILD_DEPENDENCY -->

## Build Instructions
The following instructions will build the library on an NVIDIA Jetson Nano development kit:

```
$ mkdir libfalcon_dev_env
$ cd libfalcon_dev_env
$ git clone git@github.com:OrthogonalHawk/libfalcon_dsp.git
$ cd libfalcon_dsp
$ ./manage_dependencies.sh --download
$ TARGET=NATIVE make
```
The following instructions will build and run the library unit tests on an NVIDIA Jetson Nano
 development kit. Note that generating the unit test vectors may require the user to install
 additional Python packages. These are currently not documented (TODO for another time) and a
 user is responsible for interpreting any error messages related to missing packages and
 installing the missing dependencies as necessary.

The unit tests are built using the Google Test (GTest) framework and include a variety of 
 command-line options to control unit test execution. The following example is the 'vanilla'
 invocation of the unit test executable. Use the `--help` argument to see additional options.
```
$ mkdir libfalcon_dev_env
$ cd libfalcon_dev_env
$ git clone git@github.com:OrthogonalHawk/libfalcon_dsp.git
$ cd libfalcon_dsp
$ ./manage_dependencies.sh --download
$ cd test
$ ./generate_test_vectors.sh
$ TARGET=NATIVE make
$ ./bin/falcon_dsp_unit_tests_aarch64_linux_gnu
```

## Issues & Versions
Library development is managed through the built-in GitHub tools at: [https://github.com/OrthogonalHawk/libfalcon_dsp](https://github.com/OrthogonalHawk/libfalcon_dsp)
