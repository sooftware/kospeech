#  End-to-end Speech Recognition
#  @source_code{
#      title={End-to-end Speech Recognition},
#      author={Soohwan Kim, Seyoung Bae, Cheolhwang Won},
#      link={https://github.com/sooftware/End-to-End-Korean-Speech-Recognition},
#      year={2020}
#  }

DATASET_PATH='/data1/'
DATA_LIST_PATH='./data/data_list/filter_train_list.csv'
BATCH_SIZE=32
NUM_WORKERS=4
NUM_EPOCHS=20
HIDDEN_DIM=256
DROPOUT=0.3
NUM_HEADS=8
LABEL_SMOOTHING=0.1
LISTENER_LAYER_SIZE=5
SPELLER_LAYER_SIZE=3
RNN_TYPE='gru'
TEACHER_FORCING_RATIO=0.99
VALID_RATIO=0.003
MAX_LEN=71
MAX_GRAD_NORM=400
INIT_LR=1e-15
HIGH_PLATEAU_LR=3e-03
LOW_PLATEAU_LR=1e-05
RAMPUP_PERIOD=1000
DECAY_START_IMPROVEMENT=0.02
EXP_DECAY_PERIOD=160000
WINDOW_SIZE=20
SAMPLE_RATE=16000
STRIDE=10
N_MELS=80
FEATURE_EXTRACT_BY='librosa'  # You can set 'torchaudio'
TIME_MASK_PARA=50
FREQ_MASK_PARA=12
TIME_MASK_NUM=2
FREQ_MASK_NUM=2
SAVE_RESULT_EVERY=1000
CHECKPOINT_EVERY=5000
PRINT_EVERY=10
NOISESET_SIZE=5000
MODE='train'


python ./main.py --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --num_epochs $NUM_EPOCHS --use_bidirectional \
--input_reverse --spec_augment --noise_augment --use_cuda --hidden_dim $HIDDEN_DIM \
--dropout $DROPOUT --num_heads $NUM_HEADS --label_smoothing $LABEL_SMOOTHING \
--listener_layer_size $LISTENER_LAYER_SIZE --speller_layer_size $SPELLER_LAYER_SIZE --rnn_type $RNN_TYPE \
--high_plateau_lr $HIGH_PLATEAU_LR --teacher_forcing_ratio $TEACHER_FORCING_RATIO --valid_ratio $VALID_RATIO \
--sample_rate $SAMPLE_RATE --window_size $WINDOW_SIZE --stride $STRIDE --n_mels $N_MELS --normalize --del_silence \
--feature_extract_by $FEATURE_EXTRACT_BY --time_mask_para $TIME_MASK_PARA --freq_mask_para $FREQ_MASK_PARA \
--time_mask_num $TIME_MASK_NUM --freq_mask_num $FREQ_MASK_NUM --save_result_every $SAVE_RESULT_EVERY \
--checkpoint_every $CHECKPOINT_EVERY --print_every $PRINT_EVERY --init_lr $INIT_LR  \
--use_multi_gpu --init_uniform --mode $MODE --dataset_path $DATASET_PATH --data_list_path $DATA_LIST_PATH \
--max_grad_norm $MAX_GRAD_NORM --rampup_period $RAMPUP_PERIOD --max_len $MAX_LEN --decay_start_improvement $DECAY_START_IMPROVEMENT \
--exp_decay_period  $EXP_DECAY_PERIOD --low_plateau_lr $LOW_PLATEAU_LR --noiseset_size $NOISESET_SIZE
