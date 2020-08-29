import os
import shutil
import argparse
from tqdm import trange
from preprocess.preprocess import preprocess
from preprocess.character import (
    generate_character_labels,
    generate_character_script
)
from preprocess.subword import (
    generate_sentencepiece_input,
    train_sentencepiece,
    generate_subword_labels,
    sentence_to_subwords,
    generate_subword_script
)
from preprocess.grapheme import (
    character_to_grapheme,
    generate_grapheme_labels,
    generate_grapheme_script
)


def merge_dataset(dataset_path, new_path):
    print('merge_dataset started...')

    for folder_idx in trange(len(os.listdir(dataset_path))):
        if not dataset_path[folder_idx].startswith('KsponSpeech'):
            continue
        # folder : {KsponSpeech_01, ..., KsponSpeech_05}
        path = os.path.join(dataset_path, dataset_path[folder_idx])
        for subfolder in os.listdir(path):
            path = os.path.join(dataset_path, dataset_path[folder_idx], subfolder)
            for file in os.listdir(path):
                if file.endswith('.pcm'):
                    shutil.copy(os.path.join(path, file), os.path.join(new_path, file))


def _get_parser():
    """ Get arguments parser """
    parser = argparse.ArgumentParser(description='KsponSpeech Preprocess')
    parser.add_argument('--dataset_path', type=str,
                        default='E:/KsponSpeech/original',
                        help='path of original dataset')
    parser.add_argument('--preprocessed_dataset_path', type=str,
                        default='E:/KsponSpeech/preprocessed',
                        help='path of preprocessed dataset')
    parser.add_argument('--new_path', type=str,
                        default='E:/KsponSpeech/character',
                        help='new path to save')
    parser.add_argument('--script_prefix', type=str,
                        default='KsponScript_',
                        help='script_prefix + FILENUM.txt : KsponScript_000001.txt')
    parser.add_argument('--labels_dest', type=str,
                        default='E:/KsponSpeech',
                        help='destination to save character / subword labels file')
    parser.add_argument('--output_unit', type=str,
                        default='character',
                        help='character or subword or grapheme')
    parser.add_argument('--preprocess_mode', type=str,
                        default='numeric_phonetic_otherwise_spelling',
                        help='Ex) (70%)/(칠 십 퍼센트) 확률이라니 (뭐 뭔)/(모 몬) 소리야 진짜 (100%)/(백 프로)가 왜 안돼?'
                             'phonetic: 칠 십 퍼센트 확률이라니 모 몬 소리야 진짜 백 프로가 왜 안돼?'
                             'spelling: 70% 확률이라니 뭐 뭔 소리야 진짜 100%가 왜 안돼?'
                             'numeric_phonetic_otherwise_spelling: 칠 십 퍼센트 확률이라니 뭐 뭔 소리야 진짜 백 프로가 왜 안돼?')
    parser.add_argument('--vocab_size', type=int,
                        default=5000,
                        help='size of vocab (default: 5000)')
    parser.add_argument('--grapheme_save_path', type=str,
                        default='E:/KsponSpeech/grapheme_script',
                        help='save path of grapheme text files')
    parser.add_argument('--subword_save_path', type=str,
                        default='E:/KsponSpeech/grapheme_script',
                        help='save path of grapheme text files')
    parser.add_argument('--use_pretrain_kobert_tokenizer', '-use_pretrain_kobert_tokenizer',
                        action='store_true', default=False,
                        help='flag indication to use pretrained sentencepiece kobert tokenizer or not (default: False)')

    return parser


def log_info(opt):
    print("Dataset Path : %s" % opt.dataset_path)
    print("Preprocessed Path : %s" % opt.preprocessed_dataset_path)
    print("New Path : %s" % opt.new_path)
    print("Script Prefix : %s" % opt.script_prefix)
    print("Labels Dest : %s" % opt.labels_dest)
    print("Output-Unit : %s" % opt.output_unit)
    print("Preprocess Mode : %s" % opt.preprocess_mode)
    if opt.output_unit == 'grapheme':
        print("Grapheme Save Path : %s" % opt.grapheme_save_path)
    if opt.output_unit == 'subword':
        print("Use Pretrain Kobert Tokenizer : %s" % opt.use_pretrain_kobert_tokenizer)


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    log_info(opt)

    preprocess(opt.dataset_path, opt.preprocessed_dataset_path, opt.preprocess_mode)

    if opt.output_unit == 'character':
        generate_character_labels(opt.preprocessed_dataset_path, opt.labels_dest)
        generate_character_script(opt.preprocessed_dataset_path, opt.new_path, opt.script_prefix, opt.labels_dest)

    elif opt.output_unit == 'subword':
        if not opt.use_pretrain_kobert_tokenizer:
            generate_sentencepiece_input(opt.preprocessed_dataset_path)
            train_sentencepiece(opt.preprocessed_dataset_path, opt.vocab_size)
        sentence_to_subwords(opt.preprocessed_dataset_path, opt.subword_save_path, opt.script_prefix, opt.use_pretrain_kobert_tokenizer)
        generate_subword_labels(opt.subword_save_path, opt.labels_dest)
        generate_subword_script(opt.subword_save_path, opt.new_path, opt.script_prefix, opt.labels_dest)

    elif opt.output_unit == 'grapheme':
        character_to_grapheme(opt.preprocessed_dataset_path, opt.grapheme_save_path)
        generate_grapheme_labels(opt.grapheme_save_path, opt.labels_dest)
        generate_grapheme_script(opt.grapheme_save_path, opt.new_path, opt.script_prefix, opt.labels_dest)

    else:
        raise ValueError("Unsupported preprocess method : {0}".format(opt.output_unit))

    merge_dataset(opt.dataset_path, opt.new_path)


if __name__ == '__main__':
    main()
