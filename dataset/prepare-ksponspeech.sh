# Author
# Soohwan Kim, Seyoung Bae, Cheolhwang Won, Soyoung Cho, Jeongwon Kwak

DATASET_PATH="SET_YOUR_DATASET_PATH"
PREPROCESSED_DATASET_PATH="SET_YOUR_PREPROCESSED_DATASET_PATH"
NEW_PATH="SET_YOUR_PATH_TO_SAVE_SCRIPT"
SCRIPT_PREFIX='KsponScript_'
LABELS_DEST='SET_LABELS_DESTINATION'
GRAPHEME_SAVE_PATH='SET_YOUR_PATH_TO_SAVE_GRAPHME_TEXT'
SUBWORD_SAVE_PATH='SET_YOUR_PATH_TO_SAVE_SUBWORD_TEXT'
OUTPUT_UNIT='character'                                          # you can set character / subword / grapheme
PREPROCESS_MODE='numeric_phonetic_otherwise_spelling'            # phonetic : 칠 십 퍼센트,  spelling : 70%
VOCAB_SIZE=5000                                                  # if you use subword output unit, set vocab size

# if you want to use pretrain kober tokenizer refer https://github.com/SKTBrain/KoBERT
# And release the bottom annotation.

python main.py --dataset_path $DATASET_PATH --preprocessed_dataset_path $PREPROCESSED_DATASET_PATH --new_path $NEW_PATH \
--script_prefix $SCRIPT_PREFIX --labels_dest $LABELS_DEST --output_unit $OUTPUT_UNIT --preprocess_mode $PREPROCESS_MODE \
--vocab_size $VOCAB_SIZE --grapheme_save_path $GRAPHEME_SAVE_PATH --subword_save_path $SUBWORD_SAVE_PATH \
# --use_pretrain_kobert_tokenizer $USE_PRETRAIN_KOBERT_TOKENIZER
