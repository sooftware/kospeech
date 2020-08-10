# It has not yet been fully implemented yet

DATASET_PATH='your_dataset_path'
DATA_LIST_PATH='../data/data_list/except_outlier_train_list.csv'
BATCH_SIZE=32
NUM_WORKERS=4
NUM_EPOCHS=20
LABEL_SMOOTHING=0.1
ARCHITECTURE='transformer'
NUM_CLASSES=2038
D_MODEL=512
NUM_HEADS=8
NUM_ENCODER_LAYERS=6
NUM_DECODER_LAYERS=6
DROPOUT=0.3
FFNET_STYLE='ff'
INIT_LR=3e-04
HIGH_PLATEAU_LR=3e-04
LOW_PLATEAU_LR=1e-05
RAMPUP_PERIOD=0
DECAY_THRESHOLD=0.02
EXP_DECAY_PERIOD=120000
FRAME_LENGTH=20
SAMPLE_RATE=16000
FRAME_SHIFT=10
N_MELS=80
FEATURE_EXTRACT_BY='librosa'      # You can set 'torchaudio'
TRANSFORM_METHOD='mel'          # Support feature : spech, mel, mfcc
TIME_MASK_PARA=40
FREQ_MASK_PARA=12
TIME_MASK_NUM=2
FREQ_MASK_NUM=2
SAVE_RESULT_EVERY=1000
CHECKPOINT_EVERY=5000
PRINT_EVERY=10
NOISE_LEVEL=0.7
NOISESET_SIZE=1000
MODE='train'



# shellcheck disable=SC2164
cd bin
python ./main.py --architecture $ARCHITECTURE --num_classes $NUM_CLASSES --d_model $D_MODEL \
--num_heads $NUM_HEADS --num_encoder_layers $NUM_ENCODER_LAYERS --num_decoder_layers $NUM_DECODER_LAYERS \
--dropout $DROPOUT --ffnet_style $FFNET_STYLE --dataset_path $DATASET_PATH --data_list_path $DATA_LIST_PATH \
--batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --num_epochs $NUM_EPOCHS --label_smoothing $LABEL_SMOOTHING \
--init_lr $INIT_LR --high_plateau_lr $HIGH_PLATEAU_LR --low_plateau_lr $LOW_PLATEAU_LR --rampup_period $RAMPUP_PERIOD \
--decay_threshold $DECAY_THRESHOLD --exp_decay_period $EXP_DECAY_PERIOD --frame_length $FRAME_LENGTH \
--sample_rate $SAMPLE_RATE --frame_shift $FRAME_SHIFT --n_mels $N_MELS --feature_extract_by $FEATURE_EXTRACT_BY \
--transform_method $TRANSFORM_METHOD --time_mask_para $TIME_MASK_PARA \
--freq_mask_para $FREQ_MASK_PARA --time_mask_num $TIME_MASK_NUM --freq_mask_num $FREQ_MASK_NUM \
--save_result_every $SAVE_RESULT_EVERY --checkpoint_every $CHECKPOINT_EVERY --print_every $PRINT_EVERY \
--noise_level $NOISE_LEVEL --noiseset_size $NOISESET_SIZE --mode $MODE --use_cuda --del_silence --normalize
