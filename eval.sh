# @github{
#   title = {KoSpeech},
#   author = {Soohwan Kim, Seyoung Bae, Cheolhwang Won},
#   link = {https://github.com/sooftware/KoSpeech},
#   year = {2020}
# }

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
TRANSFORM_METHOD='mel'
FEATURE_EXTRACT_BY='torchaudio'
PRINT_EVERY=10
K=5                # if use beam search
MODE='eval'

# shellcheck disable=SC2164
cd bin

python ./eval.py --sample_rate $SAMPLE_RATE --window_size $FRAME_LENGTH=20 --frame_shift $FRAME_SHIFT --n_mels $N_MELS \
--normalize --del_silence --feature_extract_by $FEATURE_EXTRACT_BY  \
--num_workers $NUM_WORKERS --use_cuda --batch_size $BATCH_SIZE --k $K  --decode $DECODE \
--print_every $PRINT_EVERY --mode $MODE --dataset_path $DATASET_PATH --data_list_path $DATA_LIST_PATH \
--model_path $MODEL_PATH --transform_method $TRANSFORM_METHOD
