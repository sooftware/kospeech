import argparse
from libri.preprocess import(
    collect_transcripts,
    prepare_tokenizer,
    generate_transcript_file
)

LIBRI_SPEECH_DATASETS = [
    'train_960', 'dev-clean', 'dev-other', 'test-clean', 'test-other'
]


def _get_parser():
    """ Get arguments parser """
    parser = argparse.ArgumentParser(description='LibriSpeech Preprocess')
    parser.add_argument('--dataset_path', type=str,
                        default='your_dataset_path',
                        help='path of original dataset')
    parser.add_argument('--vocab_size', type=int,
                        default=5000,
                        help='size of vocab')

    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()

    transcripts_collection = collect_transcripts(opt.dataset_path)
    prepare_tokenizer(transcripts_collection[0], opt.vocab_size)

    for idx, dataset in enumerate(LIBRI_SPEECH_DATASETS):
        generate_transcript_file(dataset, transcripts_collection[idx])


if __name__ == '__main__':
    main()