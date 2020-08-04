AUDIO_PATH='../pretrain/sample_audio.pcm'
DEVICE='cuda'

# shellcheck disable=SC2164
cd bin
python run_pretrain.py --audio_path $AUDIO_PATH --device $DEVICE

# shellcheck disable=SC2103
cd ..