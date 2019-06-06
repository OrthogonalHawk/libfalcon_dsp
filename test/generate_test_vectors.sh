#!/bin/bash

cd utilities
mkdir tmp_notebooks

# 000 - generate a ramp pattern to verify proper C/C++ file parsing
export FREQ_SHIFT=0
export INPUT_SAMPLE_RATE=1000000
export OUTPUT_SAMPLE_RATE=100000
export NUM_OUTPUT_SAMPLES=10000
export OUT_FILE_NAME=../vectors/test_000
export RAMP_OUTPUT=1

jupyter nbconvert --to notebook --execute generate_polyphase_test_vectors.ipynb --output tmp_notebooks/generate_polyphase_test_vectors_000.ipynb

cd ../