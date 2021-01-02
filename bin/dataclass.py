from dataclasses import dataclass, MISSING


@dataclass
class AudioConfig:
    audio_extension: str = MISSING
    transform_method: str = MISSING
    sample_rate: int = MISSING
    frame_length: int = MISSING
    frame_shift: int = MISSING
    n_mels: int = MISSING
    normalize: bool = MISSING
    del_silence: bool = MISSING
    feature_extract_by: str = MISSING
    freq_mask_para: int = MISSING
    time_mask_num: int = MISSING
    freq_mask_num: int = MISSING
    spec_augment: bool = MISSING


@dataclass
class TrainConfig:
    dataset: str = MISSING
    transcript_path: str = MISSING
    output_unit: str = MISSING

    num_epochs: int = MISSING
    batch_size: int = MISSING
    save_result_every: int = MISSING
    save_checkpoint_every: int = MISSING
    print_every: int = MISSING
    mode: str = MISSING

    num_workers: int = MISSING
    use_cuda: bool = MISSING

    optimizer: str = MISSING
    init_lr: float = MISSING
    final_lr: float = MISSING
    peak_lr: float = MISSING
    init_lr_scale: float = MISSING
    final_lr_scale: float = MISSING
    max_grad_norm: int = MISSING
    warmup_steps: int = MISSING
    weight_decay: float = MISSING
    reduction: str = MISSING


@dataclass
class ModelConfig:
    architecture: str = MISSING
    use_bidirectional: bool = MISSING
    hidden_dim: int = MISSING
    dropout: float = MISSING
    num_encoder_layers: int = MISSING


@dataclass
class DeepSpeech2Config(ModelConfig):
    rnn_type: str = MISSING
    max_len: int = MISSING
    activation: str = MISSING


@dataclass
class ListenAttendSpellConfig(ModelConfig):
    num_heads: int = MISSING
    label_smoothing: float = MISSING
    num_decoder_layers: int = MISSING
    rnn_type: str = MISSING
    teacher_forcing_ratio: float = MISSING
    attn_mechanism: str = MISSING
    teacher_forcing_step: float = MISSING
    min_teacher_forcing_ratio: float = MISSING
    extractor: str = MISSING
    activation: str = MISSING
    mask_conv: bool = MISSING
    joint_ctc_attention: bool = MISSING


@dataclass
class JointCTCAttentionConfig(ModelConfig):
    num_heads: int = MISSING
    label_smoothing: float = MISSING
    num_decoder_layers: int = MISSING
    rnn_type: str = MISSING
    teacher_forcing_ratio: float = MISSING
    attn_mechanism: str = MISSING
    teacher_forcing_step: float = MISSING
    min_teacher_forcing_ratio: float = MISSING
    extractor: str = MISSING
    activation: str = MISSING
    cross_entropy_weight: float = MISSING
    ctc_weight: float = MISSING
    mask_conv: bool = MISSING
    joint_ctc_attention: bool = MISSING


@dataclass
class TransformerConfig(ModelConfig):
    d_model: int = MISSING
    num_heads: int = MISSING
    num_decoder_layers: int = MISSING
    ffnet_style: str = MISSING


@dataclass
class Config:
    model: ModelConfig = MISSING
