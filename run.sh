# @github{
#   title = {KoSpeech},
#   author = {Soohwan Kim, Seyoung Bae, Cheolhwang Won},
#   link = {https://github.com/sooftware/KoSpeech},
#   year = {2020}
# }

DATASET_PATH='/data3/'
DATA_LIST_PATH='../data/data_list/filter_train_list.csv'
BATCH_SIZE=32
NUM_WORKERS=4
NUM_EPOCHS=20
HIDDEN_DIM=256
DROPOUT=0.3
NUM_HEADS=4
ATTN_MECHANISM='dot'
LABEL_SMOOTHING=0.1
LISTENER_LAYER_SIZE=3
SPELLER_LAYER_SIZE=2
RNN_TYPE='lstm'
TEACHER_FORCING_RATIO=1.0
TEACHER_FORCING_STEP=0.05
MIN_TEACHER_FORCING_RATIO=0.7
VALID_RATIO=0.003
MAX_LEN=151
MAX_GRAD_NORM=400
INIT_LR=3e-04
HIGH_PLATEAU_LR=3e-04
LOW_PLATEAU_LR=3e-05
RAMPUP_PERIOD=0
DECAY_THRESHOLD=0.02
EXP_DECAY_PERIOD=120000
WINDOW_SIZE=20
SAMPLE_RATE=16000
STRIDE=10
N_MELS=80
FEATURE_EXTRACT_BY='librosa'  # You can set 'torchaudio'
EXTRACTOR='vgg'        # Support extractor : vgg, ds2 (DeepSpeech2)
ACTIVATION='hardtanh'  # Support activation : ReLU, ELU, Hardtanh, GELU, LeakyReLU
TIME_MASK_PARA=50
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

python ./main.py --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --num_epochs $NUM_EPOCHS --use_bidirectional \
--input_reverse --spec_augment --noise_augment --use_cuda --hidden_dim $HIDDEN_DIM \
--dropout $DROPOUT --num_heads $NUM_HEADS --label_smoothing $LABEL_SMOOTHING \
--listener_layer_size $LISTENER_LAYER_SIZE --speller_layer_size $SPELLER_LAYER_SIZE --rnn_type $RNN_TYPE \
--high_plateau_lr $HIGH_PLATEAU_LR --teacher_forcing_ratio $TEACHER_FORCING_RATIO --valid_ratio $VALID_RATIO \
--sample_rate $SAMPLE_RATE --window_size $WINDOW_SIZE --stride $STRIDE --n_mels $N_MELS --normalize --del_silence \
--feature_extract_by $FEATURE_EXTRACT_BY --time_mask_para $TIME_MASK_PARA --freq_mask_para $FREQ_MASK_PARA \
--time_mask_num $TIME_MASK_NUM --freq_mask_num $FREQ_MASK_NUM --save_result_every $SAVE_RESULT_EVERY \
--checkpoint_every $CHECKPOINT_EVERY --print_every $PRINT_EVERY --init_lr $INIT_LR  \
--init_uniform --mode $MODE --dataset_path $DATASET_PATH --data_list_path $DATA_LIST_PATH \
--max_grad_norm $MAX_GRAD_NORM --rampup_period $RAMPUP_PERIOD --max_len $MAX_LEN --decay_threshold $DECAY_THRESHOLD \
--exp_decay_period  $EXP_DECAY_PERIOD --low_plateau_lr $LOW_PLATEAU_LR --noiseset_size $NOISESET_SIZE \
--noise_level $NOISE_LEVEL --attn_mechanism $ATTN_MECHANISM --teacher_forcing_step $TEACHER_FORCING_STEP \
--min_teacher_forcing_ratio $MIN_TEACHER_FORCING_RATIO --extractor $EXTRACTOR --activation $ACTIVATION --mask_cnn
