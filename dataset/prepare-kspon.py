import argparse
from preprocess.preprocess import preprocess
from preprocess.character import generate_character_labels, generate_character_script


def _get_parser():
    """ Get arguments parser """
    parser = argparse.ArgumentParser(description='KsponSpeech Preprocess')
    parser.add_argument('--dataset_path', type=str,
                        default='E:/KsponSpeech/original',
                        help='path of original dataset')
    parser.add_argument('--vocab_dest', type=str,
                        default='E:/KsponSpeech',
                        help='destination to save vocab file')
    parser.add_argument('--preprocess_mode', type=str,
                        default='numeric_phonetic_otherwise_spelling',
                        help='Ex) (70%)/(칠 십 퍼센트) 확률이라니 (뭐 뭔)/(모 몬) 소리야 진짜 (100%)/(백 프로)가 왜 안돼?'
                             'phonetic: 칠 십 퍼센트 확률이라니 모 몬 소리야 진짜 백 프로가 왜 안돼?'
                             'spelling: 70% 확률이라니 뭐 뭔 소리야 진짜 100%가 왜 안돼?')

    return parser


def log_info(opt):
    print("Dataset Path : %s" % opt.dataset_path)
    print("Labels Dest : %s" % opt.labels_dest)
    print("Preprocess Mode : %s" % opt.preprocess_mode)


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    log_info(opt)

    audio_paths, transcripts = preprocess(opt.dataset_path, opt.preprocess_mode)
    generate_character_labels(transcripts, opt.vocab_dest)
    generate_character_script(audio_paths, transcripts, opt.vocab_dest)


if __name__ == '__main__':
    main()
