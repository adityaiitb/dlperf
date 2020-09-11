#!/usr/bin/env bash

mkdir -p phase1 phase2

# Phase1
python ../create_pretraining_data.py \
	--vocab_file ../vocab/vocab \
	--input_file synthetic.txt \
	--output_file phase1/training.h5 \
	--max_seq_length 128 \
	--dupe_factor 5 \
	--max_predictions_per_seq 20 \
	--masked_lm_prob 0.15 \

# Phase 2
python ../create_pretraining_data.py \
	--vocab_file ../vocab/vocab \
	--input_file synthetic.txt \
	--output_file phase2/training.h5 \
	--max_seq_length 512 \
	--dupe_factor 5 \
	--max_predictions_per_seq 80 \
	--masked_lm_prob 0.15 \

# View the HDF5 file
# h5dump phase1/training.h5
# h5dump phase2/training.h5
