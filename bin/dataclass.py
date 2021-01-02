from dataclasses import dataclass, MISSING


@dataclass
class AudioConfig:
    audio_extension: str
    transform_method: str
    sample_rate: int
    frame_length: int
    frame_shift: int
    n_mels: int
    normalize: bool
    del_silence: bool
    feature_extract_by: str
    freq_mask_para: int
    time_mask_num: int
    freq_mask_num: int
    spec_augment: bool


@dataclass
class TrainConfig:
    dataset: str
    transcript_path: str
    output_unit: str

    num_epochs: int
    batch_size: int
    save_result_every: int
    save_checkpoint_every: int
    print_every: int
    mode: str

    num_workers: int
    use_cuda: bool

    optimizer: str
    init_lr: float
    final_lr: float
    peak_lr: float
    init_lr_scale: float
    final_lr_scale: float
    max_grad_norm: int
    warmup_steps: int
    weight_decay: float
    reduction: str


@dataclass
class ModelConfig:
    architecture: str
    use_bidirectional: bool
    hidden_dim: int
    dropout: float
    num_encoder_layers: int


@dataclass
class DeepSpeech2Config(ModelConfig):
    rnn_type: str
    max_len: int
    activation: str


@dataclass
class ListenAttendSpellConfig(ModelConfig):
    num_heads: int
    label_smoothing: float
    num_decoder_layers: int
    rnn_type: str
    teacher_forcing_ratio: float
    attn_mechanism: str
    teacher_forcing_step: float
    min_teacher_forcing_ratio: float
    extractor: str
    activation: str
    mask_conv: bool
    joint_ctc_attention: bool


@dataclass
class JointCTCAttentionConfig(ModelConfig):
    num_heads: int
    label_smoothing: float
    num_decoder_layers: int
    rnn_type: str
    teacher_forcing_ratio: float
    attn_mechanism: str
    teacher_forcing_step: float
    min_teacher_forcing_ratio: float
    extractor: str
    activation: str
    cross_entropy_weight: float
    ctc_weight: float
    mask_conv: bool
    joint_ctc_attention: bool


@dataclass
class TransformerConfig(ModelConfig):
    d_model: int
    num_heads: int
    num_decoder_layers: int
    ffnet_style: str


@dataclass
class Config:
    audio: AudioConfig
    train: TrainConfig
    model: ModelConfig
