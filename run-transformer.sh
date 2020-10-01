# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.
# It has not yet been fully implemented yet

ARCHITECTURE='transformer'
DATASET_PATH='E:/AIHub/'
DATA_LIST_PATH='../data/data_list/except_outlier_train_list.csv'
BATCH_SIZE=32
NUM_WORKERS=4
NUM_EPOCHS=20
LABEL_SMOOTHING=0.1
REDUCTION='mean'
NUM_CLASSES=2038
D_MODEL=512
NUM_HEADS=8
NUM_ENCODER_LAYERS=3
NUM_DECODER_LAYERS=2
DROPOUT=0.3
FFNET_STYLE='ff'
INIT_LR=1e-06
PEAK_LR=1e-04
FINAL_LR=1e-06
INIT_LR_SCALE=0.01
FINAL_LR_SCALE=0.05
WARMUP_STEPS=2000
FRAME_LENGTH=20
FRAME_SHIFT=10
SAMPLE_RATE=16000
N_MELS=80
FEATURE_EXTRACT_BY='kaldi'      # You can set 'torchaudio'
TRANSFORM_METHOD='fbank'          # Support feature : spech, mel, mfcc
FREQ_MASK_PARA=18
TIME_MASK_NUM=4
FREQ_MASK_NUM=2
SAVE_RESULT_EVERY=1000
CHECKPOINT_EVERY=5000
PRINT_EVERY=10
NOISE_LEVEL=0.7
NOISESET_SIZE=1000
MODE='train'



# shellcheck disable=SC2164
cd bin
python ./main.py --architecture $ARCHITECTURE --num_classes $NUM_CLASSES --d_model $D_MODEL --reduction $REDUCTION \
--num_heads $NUM_HEADS --num_encoder_layers $NUM_ENCODER_LAYERS --num_decoder_layers $NUM_DECODER_LAYERS \
--dropout $DROPOUT --ffnet_style $FFNET_STYLE --dataset_path $DATASET_PATH --data_list_path $DATA_LIST_PATH \
--batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --num_epochs $NUM_EPOCHS --label_smoothing $LABEL_SMOOTHING \
--init_lr $INIT_LR --final_lr $FINAL_LR --peak_lr $PEAK_LR --init_lr_scale $INIT_LR_SCALE --final_lr_scale $FINAL_LR_SCALE \
--frame_length $FRAME_LENGTH \
--sample_rate $SAMPLE_RATE --frame_shift $FRAME_SHIFT --n_mels $N_MELS --feature_extract_by $FEATURE_EXTRACT_BY \
--transform_method $TRANSFORM_METHOD \
--freq_mask_para $FREQ_MASK_PARA --time_mask_num $TIME_MASK_NUM --freq_mask_num $FREQ_MASK_NUM \
--save_result_every $SAVE_RESULT_EVERY --checkpoint_every $CHECKPOINT_EVERY --print_every $PRINT_EVERY \
--noise_level $NOISE_LEVEL --noiseset_size $NOISESET_SIZE --mode $MODE  --del_silence --normalize #--use_cuda
