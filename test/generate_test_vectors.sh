#!/bin/bash

cd utilities
mkdir -p tmp_notebooks

# 000 - generate a ramp pattern to verify proper C/C++ file parsing
export FREQ_SHIFT=0
export INPUT_SAMPLE_RATE=1000000
export OUTPUT_SAMPLE_RATE=100000
export NUM_OUTPUT_SAMPLES=10000
export OUT_FILE_NAME=../vectors/test_000
export RAMP_OUTPUT=1

jupyter nbconvert --to notebook --execute generate_polyphase_test_vectors.ipynb --output tmp_notebooks/generate_polyphase_test_vectors_000.ipynb


# 001 - positive frequency shift test vector, no resampling
export FREQ_SHIFT=100000
export INPUT_SAMPLE_RATE=1000000
export OUTPUT_SAMPLE_RATE=1000000
export NUM_OUTPUT_SAMPLES=1000000
export OUT_FILE_NAME=../vectors/test_001
export RAMP_OUTPUT=0

jupyter nbconvert --to notebook --execute generate_polyphase_test_vectors.ipynb --output tmp_notebooks/generate_polyphase_test_vectors_001.ipynb


# 002 - positive frequency shift test vector, no resampling, long file
export FREQ_SHIFT=100000
export INPUT_SAMPLE_RATE=1000000
export OUTPUT_SAMPLE_RATE=1000000
export NUM_OUTPUT_SAMPLES=10000000
export OUT_FILE_NAME=../vectors/test_002
export RAMP_OUTPUT=0

jupyter nbconvert --ExecutePreprocessor.timeout=300 --to notebook --execute generate_polyphase_test_vectors.ipynb --output tmp_notebooks/generate_polyphase_test_vectors_002.ipynb


# 003 - negative frequency shift test vector, no resampling
export FREQ_SHIFT=-5000
export INPUT_SAMPLE_RATE=1000000
export OUTPUT_SAMPLE_RATE=1000000
export NUM_OUTPUT_SAMPLES=10000
export OUT_FILE_NAME=../vectors/test_003
export RAMP_OUTPUT=0

jupyter nbconvert --to notebook --execute generate_polyphase_test_vectors.ipynb --output tmp_notebooks/generate_polyphase_test_vectors_003.ipynb


# 004 - no frequency shift; simple 2x decimation
export FREQ_SHIFT=0
export INPUT_SAMPLE_RATE=1000000
export OUTPUT_SAMPLE_RATE=500000
export NUM_OUTPUT_SAMPLES=100000
export OUT_FILE_NAME=../vectors/test_004
export RAMP_OUTPUT=0

jupyter nbconvert --to notebook --execute generate_polyphase_test_vectors.ipynb --output tmp_notebooks/generate_polyphase_test_vectors_004.ipynb


# 005 - no frequency shift; 1 MHz -> 600 kHz
export FREQ_SHIFT=0
export INPUT_SAMPLE_RATE=1000000
export OUTPUT_SAMPLE_RATE=600000
export NUM_OUTPUT_SAMPLES=1000000
export OUT_FILE_NAME=../vectors/test_005
export RAMP_OUTPUT=0

jupyter nbconvert --to notebook --execute generate_polyphase_test_vectors.ipynb --output tmp_notebooks/generate_polyphase_test_vectors_005.ipynb


# 006 - no frequency shift; 1 MHz -> 2 MHz
export FREQ_SHIFT=0
export INPUT_SAMPLE_RATE=1000000
export OUTPUT_SAMPLE_RATE=2000000
export NUM_OUTPUT_SAMPLES=10000
export OUT_FILE_NAME=../vectors/test_006
export RAMP_OUTPUT=0

jupyter nbconvert --to notebook --execute generate_polyphase_test_vectors.ipynb --output tmp_notebooks/generate_polyphase_test_vectors_006.ipynb


# 007 - no frequency shift; 1 MHz -> 800 kHz
export FREQ_SHIFT=0
export INPUT_SAMPLE_RATE=1000000
export OUTPUT_SAMPLE_RATE=800000
export NUM_OUTPUT_SAMPLES=10000
export OUT_FILE_NAME=../vectors/test_007
export RAMP_OUTPUT=0

jupyter nbconvert --to notebook --execute generate_polyphase_test_vectors.ipynb --output tmp_notebooks/generate_polyphase_test_vectors_007.ipynb


# 008 - no frequency shift; 1 MHz -> 450 kHz
export FREQ_SHIFT=0
export INPUT_SAMPLE_RATE=1000000
export OUTPUT_SAMPLE_RATE=450000
export NUM_OUTPUT_SAMPLES=10000
export OUT_FILE_NAME=../vectors/test_008
export RAMP_OUTPUT=0

jupyter nbconvert --to notebook --execute generate_polyphase_test_vectors.ipynb --output tmp_notebooks/generate_polyphase_test_vectors_008.ipynb


# 009 - no frequency shift; 1 MHz -> 44 kHz
export FREQ_SHIFT=0
export INPUT_SAMPLE_RATE=1000000
export OUTPUT_SAMPLE_RATE=44000
export NUM_OUTPUT_SAMPLES=100000
export OUT_FILE_NAME=../vectors/test_009
export RAMP_OUTPUT=0

jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=3600 --execute generate_polyphase_test_vectors.ipynb --output tmp_notebooks/generate_polyphase_test_vectors_009.ipynb


# 010 - no frequency shift; FIR filtering
export FILTER_ORDER=128
export NUM_TONES=4
export INPUT_SAMPLE_RATE=1000000
export NUM_OUTPUT_SAMPLES=100000
export OUT_FILE_NAME=../vectors/test_010

jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=3600 --execute generate_fir_filter_test_vectors.ipynb --output tmp_notebooks/generate_fir_filter_test_vectors_010.ipynb


# 011 - no frequency shift; FIR filtering
export FILTER_ORDER=128
export NUM_TONES=5
export INPUT_SAMPLE_RATE=1000000
export NUM_OUTPUT_SAMPLES=1000000
export OUT_FILE_NAME=../vectors/test_011

jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=3600 --execute generate_fir_filter_test_vectors.ipynb --output tmp_notebooks/generate_fir_filter_test_vectors_011.ipynb


# 012 - multi-channel freq shift
export NUM_TONES=2
export INPUT_SAMPLE_RATE=1000000
export NUM_OUTPUT_SAMPLES=1000000
export OUT_FILE_NAME=../vectors/test_012
export SEED=3758

jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=3600 --execute generate_multi_chan_freq_shift_test_vectors.ipynb --output tmp_notebooks/generate_multi_chan_freq_shift_test_vectors_012.ipynb


# 013 - multi-channel freq shift
export NUM_TONES=5
export INPUT_SAMPLE_RATE=1000000
export NUM_OUTPUT_SAMPLES=1000000
export OUT_FILE_NAME=../vectors/test_013
export SEED=6277

jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=3600 --execute generate_multi_chan_freq_shift_test_vectors.ipynb --output tmp_notebooks/generate_multi_chan_freq_shift_test_vectors_013.ipynb


# 014 - multi-channel freq shift
export NUM_TONES=10
export INPUT_SAMPLE_RATE=1000000
export NUM_OUTPUT_SAMPLES=1000000
export OUT_FILE_NAME=../vectors/test_014
export SEED=3999

jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=3600 --execute generate_multi_chan_freq_shift_test_vectors.ipynb --output tmp_notebooks/generate_multi_chan_freq_shift_test_vectors_014.ipynb


# 015 - multi-rate channelizer
export INPUT_SAMPLE_RATE=1000000
export CHANNEL_SAMPLE_RATES=100000,250000
export FREQ_SHIFTS=-350000,60000
export NUM_INPUT_SAMPLES=2000000
export OUT_FILE_NAME=../vectors/test_015
export SEED=2020

jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=3600 --execute generate_multi_rate_channelizer_test_vectors.ipynb --output tmp_notebooks/generate_multi_rate_channelizer_test_vectors_015.ipynb


# 016 - multi-rate channelizer
export INPUT_SAMPLE_RATE=2000000
export CHANNEL_SAMPLE_RATES=300000,125000,45000,770000,25000
export FREQ_SHIFTS=-400000,-50000,10000,27000,334000
export NUM_INPUT_SAMPLES=1000000
export OUT_FILE_NAME=../vectors/test_016
export SEED=2021

jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=3600 --execute generate_multi_rate_channelizer_test_vectors.ipynb --output tmp_notebooks/generate_multi_rate_channelizer_test_vectors_016.ipynb


# 017 - multi-rate channelizer
export INPUT_SAMPLE_RATE=30720000
export CHANNEL_SAMPLE_RATES=1920000,1920000,1920000,1920000,1920000,1920000,1920000,1920000,1920000,1920000,1920000
export FREQ_SHIFTS=-5000000,-4000000,-3000000,-2000000,-1000000,1000,1000000,2000000,3000000,4000000,5000000
export NUM_INPUT_SAMPLES=3072000
export OUT_FILE_NAME=../vectors/test_017
export SEED=2022

jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=3600 --execute generate_multi_rate_channelizer_test_vectors.ipynb --output tmp_notebooks/generate_multi_rate_channelizer_test_vectors_017.ipynb


# 018 - multi-rate channelizer
export INPUT_SAMPLE_RATE=30720000
export CHANNEL_SAMPLE_RATES=3686400,3686400,3686400,3686400,3686400,3686400,3686400,3686400,3686400,3686400,3686400
export FREQ_SHIFTS=-5000000,-4000000,-3000000,-2000000,-1000000,1000,1000000,2000000,3000000,4000000,5000000
export NUM_INPUT_SAMPLES=3072000
export OUT_FILE_NAME=../vectors/test_018
export SEED=2023

jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=3600 --execute generate_multi_rate_channelizer_test_vectors.ipynb --output tmp_notebooks/generate_multi_rate_channelizer_test_vectors_018.ipynb


cd ../
