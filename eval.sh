# @github{
#   title = {KoSpeech},
#   author = {Soohwan Kim, Seyoung Bae, Cheolhwang Won},
#   link = {https://github.com/sooftware/KoSpeech},
#   year = {2020}
# }

DATASET_PATH='/data1/'
MODEL_PATH=''
DATA_LIST_PATH='../data/data_list/filter_test_list.csv'
BATCH_SIZE=32
NUM_WORKERS=4
SAMPLE_RATE=16000
WINDOW_SIZE=20
STRIDE=10
N_MELS=80
DECODE='greedy'
TRANSFORM_METHOD='spect'
FEATURE_EXTRACT_BY='librosa'
PRINT_EVERY=10
K=5
MODE='eval'

# shellcheck disable=SC2164
cd bin

python ./eval.py --sample_rate $SAMPLE_RATE --window_size $WINDOW_SIZE --stride $STRIDE --n_mels $N_MELS \
--normalize --del_silence --input_reverse --feature_extract_by $FEATURE_EXTRACT_BY  \
--num_workers $NUM_WORKERS --use_cuda --batch_size $BATCH_SIZE --k $K  --decode $DECODE \
--print_every $PRINT_EVERY --mode $MODE --dataset_path $DATASET_PATH --data_list_path $DATA_LIST_PATH \
--model_path $MODEL_PATH --transform_method $TRANSFORM_METHOD
