# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

# $1 : DIR_TO_SAVE_DATA

base_url=www.openslr.org/resources/12
train_dir=train_960
vocab_size=5000

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <download_dir>>"
  exit 1
fi

download_dir=${1%/}

echo "Data Download"
for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    url=$base_url/$part.tar.gz
    if ! wget -P $download_dir $url; then
        echo "$0: wget failed for $url"
        exit 1
    fi
    if ! tar -C $download_dir -xvzf $download_dir/$part.tar.gz; then
        echo "$0: error un-tarring archive $download_dir/$part.tar.gz"
        exit 1
    fi
done

echo "Merge all train packs into one"
mkdir -p ${download_dir}/LibriSpeech/${train_dir}/
for part in train-clean-100 train-clean-360 train-other-500; do
    mv ${download_dir}/LibriSpeech/${part}/* $download_dir/LibriSpeech/${train_dir}/
done

python prepare-libri.py --dataset_path $1/LibriSpeech --vocab_size $vocab_size

for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    rm $part.tar.gz
done