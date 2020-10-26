DATASET_PATH="SET_YOUR_DATASET_PATH"
VOCAB_DEST='../data/vocab/'
PREPROCESS_MODE='phonetic'         # phonetic : 칠 십 퍼센트,  spelling : 70%


python prepare-kspon.py --dataset_path $DATASET_PATH --vocab_dest $VOCAB_DEST --preprocess_mode $PREPROCESS_MODE
