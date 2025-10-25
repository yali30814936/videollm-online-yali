from transformers import PretrainedConfig
from typing import Optional, List

class LiveConfigMixin(PretrainedConfig):
    def __init__(self, *, vision_pretrained: Optional[str] = None,
        frame_resolution: Optional[int] = None, frame_token_cls: Optional[bool] = None, frame_token_pooled: Optional[List[int]] = None, frame_num_tokens: Optional[int] = None,
        v_placeholder: str = '<v>', frame_token_interval: Optional[str] = None, v_placeholder_id: Optional[int] = None, frame_token_interval_id: Optional[int] = None,
        stream_loss_weight: float = 1.0, frame_token_interval_threshold: float = 0.0,
        vision_drop_strategy: Optional[str] = None, is_mod_weighted: bool = True, mod_warmup_steps: int = 0, is_return_vision_weights: bool = True, vision_hidden_size=1024, connector_type: str = 'mlp',
        use_conversation_eval: bool = False,
        use_infcache: bool = False, n_max: int = 2048, n_min: int = 1024,
        frame_fps: int = 2, max_num_frames: int = 1200,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_pretrained = vision_pretrained
        self.frame_resolution = frame_resolution
        self.frame_token_cls = frame_token_cls
        self.frame_token_pooled = frame_token_pooled
        self.frame_num_tokens = frame_num_tokens
        self.vision_hidden_size = vision_hidden_size
        self.stream_loss_weight = stream_loss_weight
        self.connector_type = connector_type
        self.v_placeholder = v_placeholder
        self.frame_token_interval = frame_token_interval
        self.v_placeholder_id = v_placeholder_id
        self.frame_token_interval_id = frame_token_interval_id
        self.vision_drop_strategy = vision_drop_strategy
        self.is_mod_weighted = is_mod_weighted
        self.mod_warmup_steps = mod_warmup_steps
        self.is_return_vision_weights = is_return_vision_weights
        self.use_conversation_eval = use_conversation_eval
        self.frame_token_interval_threshold = frame_token_interval_threshold
        self.use_infcache = use_infcache
        self.n_max = n_max
        self.n_min = n_min
        self.frame_fps = frame_fps
        self.max_num_frames = max_num_frames
        