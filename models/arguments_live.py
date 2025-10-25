from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class LiveTrainingArguments(TrainingArguments):
    live_version: str = 'live1+'
    system_prompt: str = (
        "A multimodal AI assistant is helping users with some activities."
        " Below is their conversation, interleaved with the list of video frames received by the assistant."
    )
    train_datasets: list[str] = None
    eval_datasets: list[str] = None
    stream_loss_weight: float = 1.0
    llm_pretrained: str = 'meta-llama/Meta-Llama-3-8B-Instruct'
    vision_pretrained: str = 'google/siglip-large-patch16-384'
    lora_modules: str = "model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|lm_head$"
    lora_r: int = 128
    lora_alpha: int = 256
    finetune_modules: list[str] = field(default_factory=lambda: ['connector'])
    connector_type: str = 'mlp'  # 'mlp' or 'mamba'
    frame_fps: int = 2 # for training. inference can be 10
    frame_token_cls: bool = None
    frame_token_pooled: list[int] = None
    frame_resolution: int = 384
    frame_token_interval: str  = ','
    frame_token_interval_threshold: float = 0.725
    augmentation: bool = False
    attn_implementation: str = 'flash_attention_2'
    output_dir: str = 'outputs/debug'
    vision_drop_strategy: str = None
    is_mod_weighted: bool = True
    mod_warmup_steps: int = 0
    is_return_vision_weights: bool = False
    use_conversation_eval: bool = False
    prediction_cache_dir: str = None  # 預測結果緩存目錄
    skip_inference: bool = True  # 是否跳過推理，直接從緩存讀取
    use_infcache: bool = False  # 是否使用推理緩存
    n_max: int = 2048  # 最大緩存大小
    n_min: int = 2048

@dataclass
class LiveOneTrainingArguments(LiveTrainingArguments):
    live_version: str = 'live1'
    frame_token_cls: bool = True
    frame_num_tokens: int = 1
    frame_token_interval: str  = ','
    embed_mark: str = '2fps_384_1'
    max_num_frames: int = 7200 # 1h, 2fps, 7200 frames

@dataclass
class LiveOnePlusTrainingArguments(LiveTrainingArguments):
    live_version: str = 'live1+'
    frame_token_cls: bool = True
    frame_token_pooled: list[int] = field(default_factory=lambda: [3, 3])
    frame_num_tokens: int = 5
    frame_fps: int = 2
    embed_mark: str = '2fps_384_1+3x3'
    frame_token_interval: str = ','
    max_num_frames: int = 120 # 1min, 2fps, 120 frames
    # max_num_frames: int = 600 # 5min, 2fps, 600 frames
    # max_num_frames: int = 36000 # 5hr, 2fps, 36000 frames

def get_args_class(live_version: str):
    if live_version == 'live1':
        return LiveOneTrainingArguments
    elif live_version == 'live1+':
        return LiveOnePlusTrainingArguments
    raise NotImplementedError
