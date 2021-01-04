from dataclasses import dataclass


@dataclass
class AudioConfig:
    audio_extension: str = "pcm"
    transform_method: str = "fbank"
    sample_rate: int = 16000
    frame_length: int = 20
    frame_shift: int = 10
    n_mels: int = 80
    normalize: bool = True
    del_silence: bool = True
    feature_extract_by: str = "kaldi"
    freq_mask_para: int = 18
    time_mask_num: int = 4
    freq_mask_num: int = 2
    spec_augment: bool = True
    input_reverse: bool = False


@dataclass
class TrainConfig:
    dataset: str = "kspon"
    dataset_path: str = "???"
    transcripts_path: str = "../../../data/transcripts.txt"
    output_unit: str = "character"

    num_epochs: int = 20
    batch_size: int = 32
    save_result_every: int = 1000
    checkpoint_every: int = 5000
    print_every: int = 10
    mode: str = "train"

    num_workers: int = 4
    use_cuda: bool = True

    optimizer: str = "adam"
    init_lr: float = 1e-06
    final_lr: float = 1e-06
    peak_lr: float = 1e-04
    init_lr_scale: float = 0.01
    final_lr_scale: float = 0.05
    max_grad_norm: int = 400
    warmup_steps: int = 400
    weight_decay: float = 1e-05
    reduction: str = "mean"

    seed: int = 777
    resume: bool = False


@dataclass
class ModelConfig:
    architecture: str = "???"
    teacher_forcing_ratio: float = 1.0
    teacher_forcing_step: float = 0.01
    min_teacher_forcing_ratio: float = 0.9
    dropout: float = 0.3
    bidirectional: bool = False
    joint_ctc_attention: bool = False
    max_len: int = 400


@dataclass
class DeepSpeech2Config(ModelConfig):
    architecture: str = "deepspeech2"
    use_bidirectional: bool = True
    rnn_type: str = "gru"
    hidden_dim: int = 1024
    activation: str = "hardtanh"
    num_encoder_layers: int = 3


@dataclass
class ListenAttendSpellConfig(ModelConfig):
    architecture: str = "las"
    use_bidirectional: bool = True
    dropout: float = 0.3
    num_heads: int = 4
    label_smoothing: float = 0.1
    num_encoder_layers: int = 3
    num_decoder_layers: int = 2
    rnn_type: str = "lstm"
    hidden_dim: int = 512
    teacher_forcing_ratio: float = 1.0
    attn_mechanism: str = "multi-head"
    teacher_forcing_step: float = 0.01
    min_teacher_forcing_ratio: float = 0.9
    extractor: str = "vgg"
    activation: str = "hardtanh"
    mask_conv: bool = False
    joint_ctc_attention: bool = False


@dataclass
class JointCTCAttentionLASConfig(ListenAttendSpellConfig):
    hidden_dim: int = 768
    cross_entropy_weight: float = 0.7
    ctc_weight: float = 0.3
    mask_conv: bool = True
    joint_ctc_attention: bool = True


@dataclass
class TransformerConfig(ModelConfig):
    architecture: str = "transformer"
    use_bidirectional: bool = True
    dropout: float = 0.3
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 12
    num_decoder_layers: int = 6
    ffnet_style: str = "ff"


@dataclass
class JointCTCAttentionTransformerConfig(TransformerConfig):
    cross_entropy_weight: float = 0.7
    ctc_weight: float = 0.3
    mask_conv: bool = True
    joint_ctc_attention: bool = True


@dataclass
class EvalConfig:
    dataset: str = 'kspon'
    dataset_path: str = ''
    transcript_path: str = '../../../data/eval_transcript.txt'
    model_path: str = ''
    output_unit: str = 'character'
    batch_size: int = 32
    num_workers: int = 4
    print_every: int = 20
    decode: str = 'greedy'
    k: int = 3
    use_cuda: bool = True
