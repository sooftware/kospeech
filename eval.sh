# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

DATASET_PATH='your_dataset_path'
MODEL_PATH='set_model_path'
DATA_LIST_PATH='../data/data_list/except_outlier_test_list.csv'
BATCH_SIZE=32
NUM_WORKERS=4
SAMPLE_RATE=16000
FRAME_LENGTH=20
FRAME_SHIFT=10
N_MELS=80
DECODE='greedy'
TRANSFORM_METHOD='fbank'
FEATURE_EXTRACT_BY='kaldi'
PRINT_EVERY=10
K=5                # if use beam search
MODE='eval'

# shellcheck disable=SC2164
cd bin

python ./eval.py --sample_rate $SAMPLE_RATE --frame_length $FRAME_LENGTH --frame_shift $FRAME_SHIFT --n_mels $N_MELS \
--normalize --del_silence --feature_extract_by $FEATURE_EXTRACT_BY  \
--num_workers $NUM_WORKERS --use_cuda --batch_size $BATCH_SIZE --k $K  --decode $DECODE \
--print_every $PRINT_EVERY --mode $MODE --dataset_path $DATASET_PATH --data_list_path $DATA_LIST_PATH \
--model_path $MODEL_PATH --transform_method $TRANSFORM_METHOD --input_reverse
