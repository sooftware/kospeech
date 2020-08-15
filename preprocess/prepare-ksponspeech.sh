DATASET_PATH="SET YOUR KsponSpeech corpus PATH"
NEW_PATH="SET YOUR path to store preprocessed KsponSpeech corpus"
SCRIPT_PREFIX='KsponScript_'


python prepare_ksponspeech.py --dataset_path "$DATASET_PATH" --new_path "$NEW_PATH" --script_prefix $SCRIPT_PREFIX
