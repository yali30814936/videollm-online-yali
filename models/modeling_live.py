import torch, re
import numpy as np
import json
import os
from collections import deque
from typing import Optional, Any
from pathlib import Path
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, Cache, PreTrainedTokenizer
from transformers.utils import logging
from scipy.optimize import linear_sum_assignment  # type: ignore
from nltk.translate.meteor_score import meteor_score
from rouge_score.rouge_scorer import RougeScorer

from .tokenization_live import build_live_tokenizer_and_update_config
from .vision_live import build_live_vision

logger = logging.get_logger(__name__)

class LiveMixin(AutoModelForCausalLM):
    def set_vision_inside(self):
        logger.warning_once("!!! Set vision encoder in the model, only recommended for on in-the-wild inference. "
            "Please dont call this for efficient training & evaluation. Instead, do visual feature pre-extraction.")
        self.vision_encoder, self.vision_encode = build_live_vision(self.config)

    def unset_vision_inside(self):
        del self.vision_encoder
        del self.vision_encode
    
    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def _get_text_metric_config(self) -> list[str]:
        """獲取文本評估度量配置，返回標準化的度量名稱列表"""
        def _normalize_metric_name(name: str) -> str:
            n = name.strip().lower()
            if n in ('rouge', 'rougel', 'rouge-l', 'rougel-f1', 'rouge-lsum'):
                return 'rougelsum'
            if n in ('meteor',):
                return 'meteor'
            if n in ('jaccard', 'jac'):
                return 'jaccard'
            return n

        text_metrics_raw = getattr(self.config, 'eval_text_metrics', None)
        if text_metrics_raw is None:
            single_metric = getattr(self.config, 'eval_text_metric', 'rougeLsum')
            if isinstance(single_metric, str) and ',' in single_metric:
                text_metric_list = [m for m in single_metric.split(',') if m.strip()]
            else:
                text_metric_list = [single_metric]
        else:
            if isinstance(text_metrics_raw, str):
                text_metric_list = [m for m in text_metrics_raw.split(',') if m.strip()]
            else:
                text_metric_list = list(text_metrics_raw)
        text_metric_list = [_normalize_metric_name(m) for m in text_metric_list if m]
        if not text_metric_list:
            text_metric_list = ['rougelsum']
        return text_metric_list

    def _init_text_metric_scorers(self, text_metric_list: list[str]) -> tuple:
        """初始化文本度量所需的評分器
        返回: (meteor_fn, rouge_scorer)
        """
        need_meteor = any(m == 'meteor' for m in text_metric_list)
        need_rouge = any(m == 'rougelsum' for m in text_metric_list)

        meteor_fn = None
        rouge_scorer = None
        if need_meteor:
            try:
                meteor_fn = meteor_score
            except Exception:
                meteor_fn = None
        if need_rouge:
            try:
                rouge_scorer = RougeScorer(['rougeLsum'], use_stemmer=False)
            except Exception:
                rouge_scorer = None
        return meteor_fn, rouge_scorer

    def _compute_text_similarities(self, hyp: str, ref: str, text_metric_list: list[str], meteor_fn=None, rouge_scorer=None) -> list[float]:
        """計算預測文本與參考文本之間的相似度
        
        Args:
            hyp: 預測文本
            ref: 參考文本
            text_metric_list: 要使用的度量列表
            meteor_fn: METEOR 評分函數（可選）
            rouge_scorer: ROUGE 評分器（可選）
            
        Returns:
            每個度量的相似度分數列表
        """
        def _jaccard(h: str, r: str) -> float:
            a = set([t for t in re.split(r"[^\w]+", (h or '').lower()) if t])
            b = set([t for t in re.split(r"[^\w]+", (r or '').lower()) if t])
            if not a and not b:
                return 1.0
            inter = len(a & b)
            union = len(a | b)
            return inter / union if union > 0 else 0.0

        sims: list[float] = []
        for m in text_metric_list:
            if m == 'meteor':
                if meteor_fn is not None:
                    try:
                        sims.append(float(meteor_fn([ref or ''], hyp or '')))
                        continue
                    except Exception:
                        pass
                # fallback
                sims.append(_jaccard(hyp, ref))
            elif m == 'rougelsum':
                if rouge_scorer is not None:
                    try:
                        score = rouge_scorer.score(ref or '', hyp or '')
                        sims.append(float(score['rougeLsum'].fmeasure))
                        continue
                    except Exception:
                        pass
                sims.append(_jaccard(hyp, ref))
            elif m == 'jaccard':
                sims.append(_jaccard(hyp, ref))
            else:
                # 未知名稱：退回 Jaccard
                sims.append(_jaccard(hyp, ref))
        return sims

    def visual_embed(self, frames: torch.Tensor):
        if hasattr(self, 'vision_encode'):
            with torch.cuda.amp.autocast():
                frames = self.vision_encode(self.vision_encoder, frames)
            frames = frames.to(self.dtype)
        frames = self.connector(frames)
        return frames.view(-1, frames.shape[-1])

    def joint_embed(
        self,
        input_ids: torch.Tensor = None,
        frames: torch.Tensor = None,
    ):
        if frames is None:
            return self.get_input_embeddings()(input_ids)
        if input_ids is None:
            return self.visual_embed(frames)
        inputs_embeds = self.get_input_embeddings()(input_ids.clamp(max=self.vocab_size-1))
        v_mask = input_ids == self.config.v_placeholder_id
        if v_mask.any():
            inputs_embeds[v_mask] = self.visual_embed(frames)
        return inputs_embeds

    @torch.no_grad()
    def stream_evaluate(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        frames: torch.Tensor,
        ignore_token_id: int = -100,
        frame_token_interval_threshold: float = 0.0,
        **kwargs
    ):
        use_advanced_text_sim = hasattr(self.config, 'eval_text_metrics') and self.config.eval_text_metrics is not None
        llm_response_head_len = len("Assistant: ")
        
        _turn_text_sims = []
        _text_metric_list = None
        _meteor_fn = None
        _rouge_scorer = None
        
        if use_advanced_text_sim:
            _text_metric_list = self._get_text_metric_config()
            _meteor_fn, _rouge_scorer = self._init_text_metric_scorers(_text_metric_list)
        
        # 0. evaluation only supports batch_size = 1
        assert input_ids.size(0) == labels.size(0) == 1
        input_id, label = input_ids[0], labels[0]
        device = input_id.device
        zero = torch.tensor(0, dtype=torch.int, device=device)
        one = torch.tensor(1, dtype=torch.int, device=device)

        # 1. prepare multi-turn start and stop
        turn_stops = ((input_id == self.config.eos_token_id).nonzero() + 1)[:,0].tolist()
        turn_starts = [0] + turn_stops[:-1]
        num_turns = len(turn_starts)

        # 2. forward the full input_ids and labels, get tokenwise logits and losses
        outputs = self.forward(input_ids=input_ids, frames=frames, return_dict=True, use_cache=True)
        logit, past_key_values = outputs.logits[0], outputs.past_key_values

        # 3. compute metrics for each turn
        v_placeholder_id = self.config.v_placeholder_id
        use_interval = self.config.frame_token_interval_id is not None
        frame_token_interval_id = self.config.frame_token_interval_id if use_interval else self.config.eos_token_id
        frame_num_tokens = self.config.frame_token_cls
        if self.config.frame_token_pooled:
            frame_num_tokens += self.config.frame_token_pooled[0] * self.config.frame_token_pooled[1]
        past_num_frames = 0
        lm_ppls, frame_diffs, fluencies, lm_correctness = [], [], [], []
        for r, (turn_start, turn_stop) in enumerate(zip(turn_starts, turn_stops)):
            ## 3.1. we only have two losses: stream loss on frame tokens, and lm loss. prepare corresponding mask according two losses
            turn_label = label[turn_start:turn_stop]
            turn_learn_mask = turn_label != ignore_token_id
            if not turn_learn_mask.any():
                continue
            turn_logit = logit[turn_start:turn_stop]
            turn_input_id = input_id[turn_start:turn_stop]
            turn_v_mask = turn_input_id == v_placeholder_id
            turn_num_frames = turn_v_mask.sum() // frame_num_tokens
            turn_stream_mask = turn_v_mask & turn_learn_mask
            turn_lm_mask = turn_learn_mask & ~turn_stream_mask

            ## 3.2 ppl, offline metric
            if turn_lm_mask.any():
                turn_lm_masked_logit, turn_lm_masked_label = turn_logit[turn_lm_mask], turn_label[turn_lm_mask]
                lm_ppl = torch.nn.functional.cross_entropy(turn_lm_masked_logit, turn_lm_masked_label).exp()
                lm_ppls.append(lm_ppl)
                turn_lm_masked_wrong_mask = turn_lm_masked_logit.argmax(dim=-1) != turn_lm_masked_label
                if turn_lm_masked_wrong_mask.any():
                    num_lm_correct_tokens = turn_lm_masked_wrong_mask.nonzero()[0,0]
                else:
                    num_lm_correct_tokens = (~turn_lm_masked_wrong_mask).sum()
                lm_correctness.append(num_lm_correct_tokens / turn_lm_masked_label.numel())
                
                # 計算文本相似度
                if use_advanced_text_sim:
                    tokenizer = kwargs['tokenizer']
                    # 獲取預測文本
                    pred_token_ids = turn_lm_masked_logit.argmax(dim=-1)
                    pred_text = tokenizer.decode(pred_token_ids, skip_special_tokens=True)[llm_response_head_len:]
                    # 獲取真實文本
                    ref_text = tokenizer.decode(turn_lm_masked_label, skip_special_tokens=True)[llm_response_head_len:]
                    
                    # 計算文本相似度（使用函數級別的評分器）
                    text_sims = self._compute_text_similarities(
                        pred_text, ref_text, 
                        _text_metric_list,
                        _meteor_fn, 
                        _rouge_scorer
                    )
                    _turn_text_sims.append(text_sims)

            ## 3.3. frame_diff (will be casted to time_diff in compute_metrics)
            if turn_stream_mask.any():
                ## 3.3.1: reply before (at) turn_num_frames
                turn_score = turn_logit.softmax(dim=-1)
                turn_stream_masked_score = turn_score[turn_stream_mask]
                if frame_token_interval_threshold > 0:
                    lower_threshold_mask = turn_stream_masked_score[:, frame_token_interval_id] < frame_token_interval_threshold
                    turn_stream_masked_score[lower_threshold_mask] = 0
                turn_stream_masked_pred_mask = turn_stream_masked_score.argmax(dim=-1) != frame_token_interval_id
                if turn_stream_masked_pred_mask.any():
                    frame_diff = turn_stream_mask.sum() - turn_stream_masked_pred_mask.nonzero()[0,0] - 1
                else:
                    ## 3.3.2: the most complex part,reply after turn_num_frames. we assume the 'assistant: ...' not exists
                    turn_last_stream_idx = turn_stream_mask.nonzero()[-1,0]
                    past_key_values_before_assistant = self.trim_past_key_values(past_key_values, 0, turn_start + turn_last_stream_idx + 1)
                    if r == num_turns - 1: # no future frame. we assume the model should receive a signal when streaming ends (e.g. close button).
                        frame_diff = zero
                    else:
                        next_turn_num_frames = (input_id[turn_starts[r+1]:turn_stops[r+1]] == v_placeholder_id).sum() // frame_num_tokens
                        to_append_num_frames = min(next_turn_num_frames, turn_num_frames - 1) # avoid bias. current as center, two equal left/right side
                        if to_append_num_frames == 0:
                            frame_diff = zero
                        else:
                            to_append_frames = frames[past_num_frames+turn_num_frames:past_num_frames+turn_num_frames+to_append_num_frames]
                            frame_placeholder = [v_placeholder_id] * frame_num_tokens
                            if use_interval:
                                frame_placeholder = [frame_token_interval_id] + frame_placeholder
                            to_append_input_id = torch.tensor(frame_placeholder * to_append_num_frames, dtype=torch.long, device=device)
                            to_append_logit = self.forward(
                                input_ids=to_append_input_id[None],
                                past_key_values=past_key_values_before_assistant,
                                frames=to_append_frames,
                                return_dict=True, use_cache=True
                            ).logits[0]
                            # we only use the last idx of each frame
                            idxs = torch.arange(len(frame_placeholder)-1, len(to_append_input_id), len(frame_placeholder), device=device)
                            to_append_score = to_append_logit[idxs].softmax(dim=-1)
                            if frame_token_interval_threshold > 0:
                                lower_threshold_mask = to_append_score[:, frame_token_interval_id] < frame_token_interval_threshold
                                to_append_score[lower_threshold_mask] = 0
                            to_append_score_pred_mask = to_append_score.argmax(dim=-1) != frame_token_interval_id
                            if to_append_score_pred_mask.any():
                                frame_diff = -(to_append_score_pred_mask.nonzero()[0,0] + 1)
                            else:
                                frame_diff = -to_append_num_frames
                frame_diffs.append(frame_diff.abs())

            ## 2.6 fluency
            if turn_lm_mask.any() and turn_stream_mask.any():
                num_learn_v_tokens = turn_stream_mask.sum()
                num_learn_valid_tokens = turn_lm_masked_label.numel() + num_learn_v_tokens
                if frame_diff == 0:
                    fluency = (num_learn_v_tokens + num_lm_correct_tokens) / num_learn_valid_tokens
                elif frame_diff > 0:
                    fluency = (num_learn_v_tokens - frame_diff) / num_learn_valid_tokens
                else:
                    fluency = (num_learn_v_tokens - 1) / num_learn_valid_tokens
                fluencies.append(fluency)
            ## 2.7 next turn
            past_num_frames += turn_num_frames
        lm_ppl = torch.stack(lm_ppls).mean() if lm_ppls else one
        frame_diff = torch.stack(frame_diffs).float().mean() if frame_diffs else zero
        fluency = torch.stack(fluencies).float().mean() if fluencies else one
        lm_correctness = torch.stack(lm_correctness).float().mean() if lm_correctness else one
        
        # 構建返回結果，格式與 conversation_stream_evaluate 一致
        # 基礎指標：[lm_ppl, frame_diff, fluency, lm_correctness]
        result_list = [lm_ppl, frame_diff, fluency, lm_correctness]
        
        # 添加文本相似度指標（如果有計算）
        if use_advanced_text_sim and _turn_text_sims:
            # 計算每個度量的平均值
            num_metrics = len(_turn_text_sims[0])
            for i in range(num_metrics):
                avg_sim = sum(sims[i] for sims in _turn_text_sims) / len(_turn_text_sims)
                result_list.append(torch.tensor(avg_sim, dtype=torch.float, device=device))
        
        # 在模型設備上構建結果 tensor（與 _evaluate_responses 一致）
        try:
            model_device = next(self.parameters()).device
        except Exception:
            model_device = device
        
        # 將所有元素轉換為相同設備和類型
        result_metrics = []
        for metric in result_list:
            if isinstance(metric, torch.Tensor):
                result_metrics.append(metric.to(device=model_device, dtype=torch.float32))
            else:
                result_metrics.append(torch.tensor(float(metric), dtype=torch.float32, device=model_device))
        
        # 返回 [1, N] 形狀的 tensor，與 conversation_stream_evaluate 一致
        result = torch.stack(result_metrics).unsqueeze(0)
        return result

    def trim_past_key_values(self, past_key_values, start, stop):
        return [[past_keys[:,:,start:stop], past_values[:,:,start:stop]] for past_keys, past_values in past_key_values]

    class FrameCache:
        def __init__(
            self,
            frames,
            device,
            chunk_size: int = 64,
            prefetch_ratio: int = 5,
        ):
            self.chunk_size = chunk_size
            self.prefetch_ratio = prefetch_ratio
            self.write_ptr = 0
            self.read_ptr = 0
            self.num_frames = frames.size(0)
            self.device = device

            self.buffer_size = chunk_size * prefetch_ratio
            self.buffer = torch.zeros((self.buffer_size,) + frames.shape[1:], dtype=frames.dtype, device=device)
            self.cpu_frames = frames
        
        def prefill(self):
            num_to_load = min(self.chunk_size * (self.prefetch_ratio // 2), self.num_frames - self.write_ptr)
            if num_to_load > 0:
                self.buffer[:num_to_load] = self.cpu_frames[self.write_ptr:self.write_ptr+num_to_load].to(self.device, non_blocking=True)
                self.write_ptr += num_to_load

        def __iter__(self):
            self.read_ptr = 0
            return self

        def __next__(self):
            if self.read_ptr >= self.num_frames:
                raise StopIteration
            # If buffer is exhausted, prefill more frames
            buffer_idx = self.read_ptr % self.buffer_size
            if buffer_idx == 0 and self.read_ptr != 0:
                self.prefill()
            frame = self.buffer[buffer_idx]
            self.read_ptr += 1
            return frame

    @torch.no_grad()
    def conversation_stream_evaluate(
        self,
        frames: torch.Tensor,
        conversations: list,
        tokenizer: PreTrainedTokenizer,
        frame_token_interval_threshold: float = 0.725,
        max_new_tokens: int = 100,
        frame_chunk_size: int = 64,
        prefetch_ratio: int = 5,
        sample_uid: str = None,
        prediction_cache_dir: str = None,
        skip_inference: bool = False,
        **kwargs
    ):
        """
        對話流式評估方法，支持兩階段處理：
        1. 生成階段：運行模型推理並儲存預測結果
        2. 評估階段：讀取已保存的結果並計算指標
        
        Args:
            sample_uid: 測資的唯一識別碼（通常是 video_uid 或 annotation_uid）
            prediction_cache_dir: 預測結果緩存目錄
            skip_inference: 是否跳過推理階段，直接從緩存讀取
        """
        # 如果提供了緩存目錄，檢查是否已有結果
        if prediction_cache_dir and sample_uid:
            cache_path = self._get_prediction_cache_path(prediction_cache_dir, sample_uid)
            
            # 如果要求跳過推理或已有緩存，嘗試從緩存讀取
            if skip_inference or os.path.exists(cache_path):
                cached_result = self._load_prediction_from_cache(cache_path)
                if cached_result is not None:
                    logger.info(f"Loaded cached prediction for {sample_uid}")
                    return self._evaluate_responses(
                        cached_result['pd_responses'], 
                        cached_result['gt_responses']
                    )
                elif skip_inference:
                    logger.warning(f"skip_inference=True but cache not found for {sample_uid}, skipping...")
                    # 返回全零指標
                    return self._get_zero_metrics()
        
        # 執行推理生成預測
        device = self.model.device
        stream_frames = (frames.device != device)
        conversations = conversations[0]
        state = self._init_simulation_state(device, frame_token_interval_threshold, max_new_tokens, tokenizer)

        query_queue = deque()
        gt_responses = []
        pd_responses = []

        # Prefetch frames to device (non-blocking)
        if stream_frames:
            frame_buffer = self.FrameCache(
                frames=frames,
                device=device,
                chunk_size=frame_chunk_size,
                prefetch_ratio=prefetch_ratio
            )
        else:
            frame_buffer = frames

        # Prepare for response generation
        fid = 0
        for conv in conversations:
            if conv['role'] == 'system' or conv['role'] == 'user':
                query_queue.append((fid, conv))
            elif conv['role'] == 'assistant':
                gt_responses.append((fid, conv))
            elif conv['role'] == 'stream':
                fid += conv['num_frames']

        # Process frames
        last_role = None
        
        # 如果需要緩存，準備緩存路徑和即時保存機制
        cache_path = None
        if prediction_cache_dir and sample_uid:
            cache_path = self._get_prediction_cache_path(prediction_cache_dir, sample_uid)
        
        for fid, frame in enumerate(frame_buffer):
            turn_conversations = []
            while query_queue and query_queue[0][0] == fid:
                _, conv = query_queue.popleft()
                turn_conversations.append(conv)
            if turn_conversations:
                state['last_ids'] = tokenizer.apply_chat_template(
                    turn_conversations,
                    add_stream_query_prompt=(last_role == 'stream'),
                    add_generation_prompt=(turn_conversations[-1]['role'] == 'user'),
                    add_stream_prompt=(turn_conversations[-1]['role'] != 'user'),
                    add_stream_generation_prompt=False,
                    return_tensors='pt'
                ).to(device)
                if last_role == 'assistant':
                    state['last_ids'] = torch.cat([
                        torch.tensor([[tokenizer.eos_token_id]], device=device),
                        state['last_ids']], dim=1)
                last_role = turn_conversations[-1]['role']
            elif last_role == 'assistant':
                state['last_ids'] = torch.cat([state['last_ids'], state['_added_stream_prompt_ids']], dim=1)
            
            if last_role == 'user':
                output_ids = self._simulate_stream_response(state=state, device=device)
                pd_responses.append((fid, {
                    'role': 'assistant',
                    'content': tokenizer.decode(output_ids[0], skip_special_tokens=True)[1:]
                }))
                last_role = 'assistant'
                state['last_ids'] = torch.cat([state['last_ids'], state['_added_stream_prompt_ids']], dim=1)
                
            next_token = self._simulate_stream_step(state, frame)
            last_role = 'stream'
            if next_token == 933: #]\n
                state['last_ids'] = state['_added_stream_generation_ids'].to(device)
                output_ids = self._simulate_stream_response(state=state, device=device)
                pd_responses.append((fid, {
                    'role': 'assistant',
                    'content': tokenizer.decode(output_ids[0], skip_special_tokens=True)[1:]
                }))
                last_role = 'assistant'

        if cache_path:
            self._save_prediction_to_cache(
                cache_path=cache_path,
                sample_uid=sample_uid,
                pd_responses=pd_responses,
                gt_responses=gt_responses
            )

        return self._evaluate_responses(pd_responses, gt_responses)

    def _get_prediction_cache_path(self, cache_dir: str, sample_uid: str) -> str:
        """獲取預測結果的緩存路徑"""
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return os.path.join(cache_dir, f"{sample_uid}.json")
    
    def _save_prediction_to_cache(self, cache_path: str, sample_uid: str, pd_responses: list, gt_responses: list):
        """將預測結果保存到緩存檔案
        
        Args:
            cache_path: 緩存檔案路徑
            sample_uid: 測資唯一識別碼
            pd_responses: 預測的回應列表 [(frame_id, response_dict), ...]
            gt_responses: 標註的回應列表 [(frame_id, response_dict), ...]
        """
        # 獲取 fps 用於時間轉換
        fps = float(getattr(self.config, 'frame_fps', 2) or 2)
        fps = fps if fps > 0 else 1.0
        
        # 將 frame index 轉換為秒，並只保留 content（移除 role）
        def convert_to_seconds(responses):
            result = []
            for fid, resp in responses:
                timestamp = fid / fps
                # 只保留 content，移除 role 和其他不需要的欄位
                content = resp.get('content', '') if isinstance(resp, dict) else str(resp)
                result.append([timestamp, content])
            return result
        
        cache_data = {
            'sample_uid': sample_uid,
            'pd_responses': convert_to_seconds(pd_responses),
            'gt_responses': convert_to_seconds(gt_responses),
            'fps': fps
        }
        
        # 寫入 JSON 檔案（一個檔案一筆測資）
        try:
            # 確保父目錄存在
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved prediction to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save prediction to {cache_path}: {e}")
    
    def _load_prediction_from_cache(self, cache_path: str) -> Optional[dict]:
        """從緩存檔案讀取預測結果
        
        Returns:
            包含 pd_responses 和 gt_responses 的字典，時間戳記已轉回 frame index
            如果讀取失敗則返回 None
        """
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # 獲取 fps
            fps = cache_data.get('fps', 2.0)
            
            # 將秒轉換回 frame index
            # 緩存格式: [[timestamp, content], ...]
            # 返回格式: [(frame_id, content), ...] - 直接返回字符串，與 _evaluate_responses 的 extract_content 兼容
            def convert_to_frames(responses):
                result = []
                for item in responses:
                    if isinstance(item, list) and len(item) >= 2:
                        timestamp, content = item[0], item[1]
                        fid = int(round(timestamp * fps))
                        # 直接返回字符串內容，不需要包裝成字典
                        if isinstance(content, dict):
                            # 兼容舊格式（如果有）
                            result.append((fid, content.get('content', '')))
                        else:
                            result.append((fid, content))
                    else:
                        logger.warning(f"Unexpected cache format: {item}")
                return result
            
            return {
                'pd_responses': convert_to_frames(cache_data['pd_responses']),
                'gt_responses': convert_to_frames(cache_data['gt_responses'])
            }
        except Exception as e:
            logger.warning(f"Failed to load prediction from {cache_path}: {e}")
            return None
    
    def _get_zero_metrics(self):
        """返回全零的指標，用於跳過的測資"""
        text_metric_list = self._get_text_metric_config()
        num_metrics = 2 + len(text_metric_list) + 1  # time_mae, time_acc, text_metrics..., f1
        
        try:
            model_device = next(self.parameters()).device
        except Exception:
            model_device = torch.device('cpu')
        
        return torch.zeros(num_metrics, dtype=torch.float32, device=model_device)

    def _simulate_stream_step(self, state, frame):
        device = state['device']
        
        input_ids = torch.cat([
            state['last_ids'],
            torch.tensor([[self.config.v_placeholder_id] * self.config.frame_num_tokens], device=device)
        ], dim=1)
        input_embeds = torch.cat([
            self.get_input_embeddings()(state['last_ids'].to(device)),
            self.visual_embed(frame).view(1, -1, self.config.hidden_size)
        ], dim=1)
        v_mask = torch.cat([
            torch.zeros((1, state['last_ids'].size(1)), dtype=torch.bool, device=device),
            torch.ones((1, self.config.frame_num_tokens), dtype=torch.bool, device=device)
        ], dim=1)
        frame_interval_mask = torch.cat([
            state['last_ids'] == self.config.frame_token_interval_id,
            torch.zeros((1, self.config.frame_num_tokens), dtype=torch.bool, device=device)
        ], dim=1)
        
        outputs = self.forward(
            inputs_embeds = input_embeds,
            v_mask = v_mask,
            frame_interval_mask = frame_interval_mask,
            past_key_values = state.get('past_key_values', None),
            return_dict = True,
            use_cache = True,
            input_ids = input_ids
        )
        
        state['past_key_values'] = outputs.past_key_values

        next_score = outputs.logits[:,-1:].softmax(dim=-1)
        if next_score[:,:,self.config.frame_token_interval_id] < self.config.frame_token_interval_threshold:
            next_score[:,:,self.config.frame_token_interval_id].zero_()
        next_token = next_score.argmax(dim=-1)
        state['last_ids'] = next_token
        return next_token

    def _simulate_stream_response(self, state, device):
        eos_token_id = self.config.eos_token_id
        
        input_embeds = self.get_input_embeddings()(state['last_ids'].to(device))
        v_mask = torch.zeros((1, state['last_ids'].size(1)), dtype=torch.bool, device=device)
        frame_interval_mask = state['last_ids'] == self.config.frame_token_interval_id
        
        output_ids, state['past_key_values'] = fast_greedy_generate(
            model=self,
            inputs_embeds=input_embeds,
            past_key_values=state.get('past_key_values', None),
            v_mask=v_mask,
            frame_interval_mask=frame_interval_mask,
            eos_token_id=eos_token_id,
            inplace_output_ids=state['inplace_output_ids'],
            device=device,
            input_ids = state['last_ids']
        )
        
        state['last_ids'] = output_ids[:,-1:]
        return output_ids
        
    def _get_stream_token_templates(self, tokenizer):
        """預計算流式處理所需的token模板，類似 LiveInfer 的初始化"""
        if tokenizer is None:
            # 如果沒有tokenizer，返回佔位符
            return {
                '_added_stream_prompt_ids': torch.tensor([[]], dtype=torch.long),
                '_added_stream_generation_ids': torch.tensor([[]], dtype=torch.long),
            }
            
        # 計算各種模板token序列
        _added_stream_prompt_ids = tokenizer.apply_chat_template(
            [{}], 
            add_stream_prompt=True, 
            return_tensors='pt'
        )
        _added_stream_generation_ids = tokenizer.apply_chat_template(
            [{}], 
            add_stream_generation_prompt=True, 
            return_tensors='pt'
        )
        
        return {
            '_added_stream_prompt_ids': _added_stream_prompt_ids,
            '_added_stream_generation_ids': _added_stream_generation_ids,
        }

    def _init_simulation_state(self, device, frame_token_interval_threshold, max_new_tokens, tokenizer=None):
        """初始化模擬狀態，類似 LiveInfer 的初始化"""
        # 獲取流式處理的token模板
        stream_templates = self._get_stream_token_templates(tokenizer)
        
        # 將模板移動到正確的設備
        for key, value in stream_templates.items():
            if isinstance(value, torch.Tensor):
                stream_templates[key] = value.to(device)
        
        state = {
            'device': device,
            'past_key_values': None,
            'last_ids': torch.tensor([[]], device=device, dtype=torch.long),
            'inplace_output_ids': torch.zeros((1, max_new_tokens), dtype=torch.long, device=device),
            'max_new_tokens': max_new_tokens,
        }
        
        # 合併流式處理模板
        state.update(stream_templates)
        return state

    def _evaluate_responses(self, pd_responses, gt_responses):
        """
        使用匈牙利演算法（最小化 |Δframe|）將預測與標註配對，並計算：
        - time_mae: 平均時間誤差（秒），由 |Δframe| / fps 換算
        - time_acc: 時間誤差在 3 幀（= 3/fps 秒）以內的比例
        - text_sim: 文本相似度（預設 ROUGE-Lsum F1，可透過 self.config.eval_text_metric == 'meteor' 改用 METEOR），匹配對取平均
        - f1: 事件級配對的 F1（匹配數與預測/標註數量的調和平均）

        回傳: torch.float32 tensor [time_mae(sec), time_acc(<=3/fps s), text_sim, f1]
        """

        # 使用共享的配置獲取方法
        text_metric_list = self._get_text_metric_config()

        # 提取 frame index 與文本（兼容字典和字符串兩種格式）
        pd_fids = [fid for fid, _ in pd_responses]
        gt_fids = [fid for fid, _ in gt_responses]
        
        # 提取文本內容，處理兩種可能的格式
        def extract_content(response_data):
            if isinstance(response_data, dict):
                return response_data.get('content', '')
            elif isinstance(response_data, str):
                return response_data
            else:
                return str(response_data)
        
        pd_texts = [extract_content(res[1]) for res in pd_responses]
        gt_texts = [extract_content(res[1]) for res in gt_responses]

        n_pred, n_gt = len(pd_fids), len(gt_fids)
        if n_pred == 0 and n_gt == 0:
            # 無事件：時間 MAE=0, time_acc=1, 每個文字指標=1, F1=1
            out_vec = [0.0, 1.0] + [1.0] * len(text_metric_list) + [1.0]
            return torch.tensor(out_vec, dtype=torch.float32)
        if n_pred == 0 or n_gt == 0:
            # 單邊空：全部 0（除了時間 MAE=0）
            out_vec = [0.0, 0.0] + [0.0] * len(text_metric_list) + [0.0]
            return torch.tensor(out_vec, dtype=torch.float32)

        # 構建成本矩陣（L1 差值 on frames）
        cost = [[abs(pi - gi) for gi in gt_fids] for pi in pd_fids]

        # 使用匈牙利演算法進行最佳匹配
        cmat = np.array(cost, dtype=float)
        rows, cols = linear_sum_assignment(cmat)
        rows, cols = rows.tolist(), cols.tolist()

        # 使用共享的評分器初始化方法
        meteor_fn, rouge_scorer = self._init_text_metric_scorers(text_metric_list)

        # 匹配後計算指標
        abs_diffs_frames = []
        text_sims_all: list[list[float]] = []
        for pi, gi in zip(rows, cols):
            abs_diffs_frames.append(abs(pd_fids[pi] - gt_fids[gi]))
            # 使用共享的文本相似度計算方法
            text_sims_all.append(self._compute_text_similarities(
                pd_texts[pi], gt_texts[gi], 
                text_metric_list, 
                meteor_fn, 
                rouge_scorer
            ))

        matched = len(abs_diffs_frames)
        fps = float(getattr(self.config, 'frame_fps', 2) or 2)
        fps = fps if fps > 0 else 1e-6
        diffs_sec = [d / fps for d in abs_diffs_frames]
        time_mae = float(sum(diffs_sec) / matched) if matched > 0 else 0.0
        tol_sec = 3.0 / fps
        time_acc = float(sum(1 for s in diffs_sec if s <= tol_sec) / matched) if matched > 0 else 0.0
        # 逐指標平均
        if matched > 0:
            sums = [0.0] * len(text_metric_list)
            for sims in text_sims_all:
                for i, v in enumerate(sims):
                    sums[i] += float(v)
            text_avgs = [s / matched for s in sums]
        else:
            text_avgs = [0.0] * len(text_metric_list)

        precision = matched / n_pred if n_pred > 0 else 0.0
        recall = matched / n_gt if n_gt > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        out_vec = [time_mae, time_acc] + text_avgs + [f1]
        # 在模型裝置上建立結果，避免分散式蒐集報 CUDA/dense 錯誤
        try:
            model_device = next(self.parameters()).device
        except Exception:
            model_device = None
        if model_device is not None:
            return torch.tensor(out_vec, dtype=torch.float32, device=model_device)
        return torch.tensor(out_vec, dtype=torch.float32)

def fast_greedy_generate(
    *,
    model: LiveMixin,
    inputs_embeds: torch.Tensor,
    past_key_values: Cache,
    eos_token_id: int,
    inplace_output_ids: torch.Tensor,
    v_mask: Optional[torch.Tensor] = None,
    frame_interval_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    input_ids: Optional[torch.Tensor] = None,
):

    i = 0
    for i in range(inplace_output_ids.size(1)):
        outputs = model.forward(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            v_mask=v_mask,
            frame_interval_mask=frame_interval_mask,
            input_ids=input_ids,
        )
        past_key_values = outputs.past_key_values
        new_token_id = outputs.logits[:, -1:].argmax(dim=-1)
        inplace_output_ids[:, i] = new_token_id
        if new_token_id == eos_token_id:
            break
        input_ids = new_token_id
        inputs_embeds = model.get_input_embeddings()(new_token_id)
        v_mask = torch.zeros((1, new_token_id.shape[1]), device=device, dtype=torch.bool)
        frame_interval_mask = torch.zeros_like(v_mask, device=device, dtype=torch.bool)
    return inplace_output_ids[:, :i+1], past_key_values

def build_live(
    *,
    is_training: bool,
    config_class: type,
    model_class: type,
    llm_pretrained: Optional[str] = None,
    finetune_modules: Optional[list[str]] = None,
    lora_modules: Optional[str] = None,
    lora_r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    set_vision_inside: bool = False,
    resume_from_checkpoint: str = '',
    attn_implementation: str = 'flash_attention_2',
    torch_dtype: str | torch.dtype = 'auto',
    **kwargs
):
    model = model_class.from_pretrained(llm_pretrained, config=config_class.from_pretrained(llm_pretrained, **kwargs), torch_dtype=torch_dtype, attn_implementation=attn_implementation)
    tokenizer = build_live_tokenizer_and_update_config(llm_pretrained, model.config)
    if is_training:
        # Handle connector finetuning based on connector type
        connector_type = getattr(model.config, 'connector_type', 'mlp')
        
        if connector_type == 'mlp':
            # For MLP connector, use manual finetuning due to PEFT compatibility issues
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_modules,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                modules_to_save=finetune_modules,
                inference_mode=False,
            )
            model = get_peft_model(model, lora_config)
            
            # Manually enable connector training
            for param in model.base_model.model.connector.parameters():
                param.requires_grad = True
        else:
            # For other connectors (like Mamba), use modules_to_save
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_modules,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                modules_to_save=finetune_modules,
                inference_mode=False,
            )
            model = get_peft_model(model, lora_config)
        
        model.print_trainable_parameters()
    else:
        if resume_from_checkpoint:
            model = PeftModel.from_pretrained(model, resume_from_checkpoint, is_trainable=False)
        else:
            logger.warning(f'!!! Fail to load checkpoint: {resume_from_checkpoint}. Return a new initialized model.')
        if set_vision_inside:
            model.set_vision_inside()
        model.requires_grad_(False)
    return model, tokenizer
