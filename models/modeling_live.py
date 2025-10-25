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
from rouge_score.rouge_scorer import RougeScorer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers.cache_utils import DynamicCache

from .tokenization_live import build_live_tokenizer_and_update_config
from .vision_live import build_live_vision
from .live_llama.infcache_improved import CacheOverflowError
from .live_llama.infcache_improved import InfCache

logger = logging.get_logger(__name__)

# Global lazy-loaded sentence transformer model
_SENTENCE_TRANSFORMER = None

def get_sentence_transformer():
    """Lazy load sentence transformer model"""
    global _SENTENCE_TRANSFORMER
    if _SENTENCE_TRANSFORMER is None:
        _SENTENCE_TRANSFORMER = SentenceTransformer('all-MiniLM-L6-v2')
    return _SENTENCE_TRANSFORMER

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

    def _evaluate_responses(self, pd_responses, gt_responses, gt_offset_sec: float = 0.0):
        """
        使用文本相似度為主的匈牙利演算法配對預測與標註事件
        
        Args:
            pd_responses: 預測的回應列表 [(timestamp_sec, content), ...]
            gt_responses: 標註的回應列表 [(timestamp_sec, content), ...]
            gt_offset_sec: GT 時間偏移量（用於時間對齊）
        
        Returns:
            torch.float32 tensor [F1@1s, F1@2s, F1@3s, MAE(sec), BERT_Sim_Avg, ROUGE_L_Avg, Text_Sim_Avg, Composite_Score]
            - F1@1s: F1 score with 1s tolerance
            - F1@2s: F1 score with 2s tolerance
            - F1@3s: F1 score with 3s tolerance
            - MAE: Mean Absolute Error in seconds
            - BERT_Sim_Avg: Average BERT embedding cosine similarity
            - ROUGE_L_Avg: Average ROUGE-L F-measure
            - Text_Sim_Avg: Average combined text similarity (0.6*BERT + 0.4*ROUGE)
            - Composite_Score: 0.5*F1@1s + 0.5*Text_Sim_Avg
        """
        # 提取時間戳記（秒）與文本
        pd_times_sec = [t for t, _ in pd_responses]
        gt_times_sec = [t - gt_offset_sec for t, _ in gt_responses]
        
        # 提取文本內容
        def extract_content(response_data):
            if isinstance(response_data, dict):
                return response_data.get('content', '')
            elif isinstance(response_data, str):
                return response_data
            else:
                return str(response_data)
        
        pd_texts = [extract_content(res[1]) for res in pd_responses]
        gt_texts = [extract_content(res[1]) for res in gt_responses]

        n_pred, n_gt = len(pd_times_sec), len(gt_times_sec)
        
        # 邊界情況處理
        if n_pred == 0 and n_gt == 0:
            # 無事件：所有指標設為完美（8個指標）
            return torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        if n_pred == 0 or n_gt == 0:
            # 單邊空：F1=0, MAE=0, BERT=0, ROUGE=0, Text_Sim=0, Composite=0
            return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

        # 初始化評分器
        sentence_model = get_sentence_transformer()
        rouge_scorer = RougeScorer(['rougeL'], use_stemmer=False)
        
        # 計算文本 embeddings
        pd_embeddings = sentence_model.encode(pd_texts, convert_to_tensor=True, show_progress_bar=False)
        gt_embeddings = sentence_model.encode(gt_texts, convert_to_tensor=True, show_progress_bar=False)
        
        # 構建成本矩陣：只考慮時間窗口內（±3秒）的配對，並加入時間距離懲罰
        TIME_WINDOW = 3.0
        SIMILARITY_THRESHOLD = 0.3
        LENGTH_RATIO_POWER = 0.6
        TIME_PENALTY_WEIGHT = 0.3  # 時間距離懲罰權重
        
        cost_matrix = np.full((n_pred, n_gt), 1.0, dtype=float)  # 1.0 表示最大成本（最低相似度）
        
        # 保存每個配對的 BERT 和 ROUGE-L 分數（用於後續統計）
        bert_scores = {}  # {(i, j): score}
        rouge_scores = {}  # {(i, j): score}
        
        for i, (pd_time, pd_text, pd_emb) in enumerate(zip(pd_times_sec, pd_texts, pd_embeddings)):
            for j, (gt_time, gt_text, gt_emb) in enumerate(zip(gt_times_sec, gt_texts, gt_embeddings)):
                # 檢查時間窗口
                time_diff = abs(pd_time - gt_time)
                if time_diff > TIME_WINDOW:
                    continue  # 超出時間窗口，保持高成本
                
                # 計算文本相似度
                # 1. Cosine similarity (embedding)
                cosine_sim = torch.nn.functional.cosine_similarity(
                    pd_emb.unsqueeze(0), gt_emb.unsqueeze(0)
                ).item()
                
                # 2. ROUGE-L
                rouge_score = rouge_scorer.score(gt_text or '', pd_text or '')
                rouge_l = rouge_score['rougeL'].fmeasure
                
                # 保存原始分數
                bert_scores[(i, j)] = cosine_sim
                rouge_scores[(i, j)] = rouge_l
                
                # 3. 加權平均
                text_sim = 0.6 * cosine_sim + 0.4 * rouge_l
                
                # 4. 長度偏置校正
                len_pd = len(pd_text) if pd_text else 1
                len_gt = len(gt_text) if gt_text else 1
                len_penalty = min(1.0, (len_pd / len_gt) ** LENGTH_RATIO_POWER)
                adjusted_sim = text_sim * len_penalty
                
                # 5. 時間距離懲罰：線性懲罰，時間差越大，相似度越低
                adjusted_sim = adjusted_sim * (1-TIME_PENALTY_WEIGHT) + (time_diff / TIME_WINDOW) * TIME_PENALTY_WEIGHT
                
                # 成本 = 1 - 相似度
                cost_matrix[i, j] = 1.0 - adjusted_sim
        
        # 使用匈牙利演算法找最佳配對
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # 過濾低於閾值的配對
        valid_pairs = []
        for i, j in zip(row_indices, col_indices):
            similarity = 1.0 - cost_matrix[i, j]
            if similarity >= SIMILARITY_THRESHOLD:
                time_diff = abs(pd_times_sec[i] - gt_times_sec[j])
                # 獲取這個配對的 BERT 和 ROUGE-L 分數
                bert_sim = bert_scores.get((i, j), 0.0)
                rouge_l_sim = rouge_scores.get((i, j), 0.0)
                valid_pairs.append({
                    'pred_idx': i,
                    'gt_idx': j,
                    'time_diff': time_diff,
                    'similarity': similarity,
                    'bert_sim': bert_sim,
                    'rouge_l_sim': rouge_l_sim
                })
        
        # 計算指標
        n_matched = len(valid_pairs)
        
        if n_matched == 0:
            # 沒有有效配對（8個指標）
            return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        
        # 1. F1@1s、F1@2s 和 F1@3s
        n_matched_1s = sum(1 for pair in valid_pairs if pair['time_diff'] <= 1.0)
        n_matched_2s = sum(1 for pair in valid_pairs if pair['time_diff'] <= 2.0)
        n_matched_3s = sum(1 for pair in valid_pairs if pair['time_diff'] <= 3.0)
        
        precision_1s = n_matched_1s / n_pred
        recall_1s = n_matched_1s / n_gt
        f1_1s = (2 * precision_1s * recall_1s / (precision_1s + recall_1s)) if (precision_1s + recall_1s) > 0 else 0.0
        
        precision_2s = n_matched_2s / n_pred
        recall_2s = n_matched_2s / n_gt
        f1_2s = (2 * precision_2s * recall_2s / (precision_2s + recall_2s)) if (precision_2s + recall_2s) > 0 else 0.0
        
        precision_3s = n_matched_3s / n_pred
        recall_3s = n_matched_3s / n_gt
        f1_3s = (2 * precision_3s * recall_3s / (precision_3s + recall_3s)) if (precision_3s + recall_3s) > 0 else 0.0
        
        # 2. MAE（所有有效配對的平均時間誤差）
        mae = sum(pair['time_diff'] for pair in valid_pairs) / n_matched
        
        # 3. BERT Similarity Average
        bert_sim_avg = sum(pair['bert_sim'] for pair in valid_pairs) / n_matched
        
        # 4. ROUGE-L Average
        rouge_l_avg = sum(pair['rouge_l_sim'] for pair in valid_pairs) / n_matched
        
        # 5. Text Similarity Average（加權平均：0.6*BERT + 0.4*ROUGE）
        text_sim_avg = sum(pair['similarity'] for pair in valid_pairs) / n_matched
        
        # 6. Composite Score
        composite = 0.5 * f1_1s + 0.5 * text_sim_avg
        
        # 在模型設備上構建結果 tensor
        try:
            model_device = next(self.parameters()).device
        except Exception:
            model_device = torch.device('cpu')
        
        result = torch.tensor(
            [f1_1s, f1_2s, f1_3s, mae, bert_sim_avg, rouge_l_avg, text_sim_avg, composite],
            dtype=torch.float32,
            device=model_device
        )
        
        return result

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

            ## 3.4 fluency
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
            ## 3.5 next turn
            past_num_frames += turn_num_frames
        lm_ppl = torch.stack(lm_ppls).mean() if lm_ppls else one
        frame_diff = torch.stack(frame_diffs).float().mean() if frame_diffs else zero
        fluency = torch.stack(fluencies).float().mean() if fluencies else one
        lm_correctness = torch.stack(lm_correctness).float().mean() if lm_correctness else one
        
        # 返回基礎指標：[lm_ppl, frame_diff, fluency, lm_correctness]
        result_list = [lm_ppl, frame_diff, fluency, lm_correctness]
        
        # 在模型設備上構建結果 tensor
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
        
        # 返回 [1, N] 形狀的 tensor
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
        max_new_tokens: int = 32,
        frame_chunk_size: int = 64,
        prefetch_ratio: int = 5,
        sample_uid: str = None,
        prediction_cache_dir: str = None,
        skip_inference: bool = False,
        original_times: list = None,  # 原始時間戳記列表（未對齊 frame rate）
        **kwargs
    ):
        """
        對話流式評估方法，支持完全分離的兩階段處理：
        1. Inference Stage（推理階段）：運行模型推理並儲存預測結果
        2. Evaluation Stage（評估階段）：讀取已保存的結果並計算指標
        
        Args:
            sample_uid: 測資的唯一識別碼（通常是 video_uid 或 annotation_uid）
            prediction_cache_dir: 預測結果緩存目錄（必須提供以啟用兩階段模式）
            skip_inference: 跳過推理階段，直接從緩存讀取並評估
            original_times: GT 事件的原始時間戳記列表（秒），未對齊 frame rate
        
        Returns:
            - 評估指標 tensor [F1@1s, F1@2s, MAE, Text_Sim_Avg, Composite_Score]
        """
        # ====================================================================
        # Stage 1: Inference (可選，依據 skip_inference 決定)
        # ====================================================================
        if prediction_cache_dir and sample_uid:
            cache_path = self._get_prediction_cache_path(prediction_cache_dir, sample_uid)
            
            # 檢查是否需要執行推理
            need_inference = not os.path.exists(cache_path)
            
            if not skip_inference:
                if not need_inference:
                    return self._get_zero_metrics()
            else:
                # 直接從緩存讀取並評估
                max_time = self.config.max_num_frames / self.config.frame_fps
                cached_result = self._load_prediction_from_cache(cache_path, max_time)
                if cached_result is not None:
                    logger.info(f"Loaded cached prediction for {sample_uid}")
                    return self._evaluate_responses(
                        cached_result['pd_responses'], 
                        cached_result['gt_responses'],
                    )
                else:
                    logger.warning(f"Failed to load cache for {sample_uid}")
                    return self._get_zero_metrics()
                
        
        # ====================================================================
        # 執行推理階段（生成預測）
        # ====================================================================
        
        # 執行推理生成預測
        device = self.model.device
        stream_frames = (frames.device != device)
        conversations = conversations[0]
        state = self._init_simulation_state(device, frame_token_interval_threshold, max_new_tokens, tokenizer)

        query_queue = deque()
        gt_responses = []
        pd_responses = []
        
        # 獲取 fps 用於時間轉換
        fps = float(getattr(self.config, 'frame_fps', 2) or 2)
        fps = fps if fps > 0 else 1.0

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
        # 同時構建原始時間戳記的索引
        fid = 0
        gt_time_index = 0  # 用於追蹤 original_times 的索引
        for conv in conversations:
            if conv['role'] == 'system' or conv['role'] == 'user':
                query_queue.append((fid, conv))
            elif conv['role'] == 'assistant':
                # 使用原始時間戳記（如果提供）
                if original_times and gt_time_index < len(original_times):
                    original_time = original_times[gt_time_index]
                    gt_responses.append((original_time, conv))  # 直接使用原始時間（秒）
                    gt_time_index += 1
                else:
                    # 沒有原始時間時，使用 frame index 轉換的時間
                    gt_responses.append((fid / fps, conv))
            elif conv['role'] == 'stream':
                fid += conv['num_frames']

        # Process frames
        last_role = None
        
        # 準備緩存路徑
        cache_path = None
        if prediction_cache_dir and sample_uid:
            cache_path = self._get_prediction_cache_path(prediction_cache_dir, sample_uid)
        
        # 計算總幀數用於進度條
        if stream_frames:
            total_frames = frame_buffer.num_frames
        else:
            total_frames = frames.shape[0]
        
        # 創建進度條
        uid_display = (sample_uid[:20] + '...') if sample_uid and len(sample_uid) > 23 else (sample_uid or 'video')
        
        frame_pbar = tqdm(
            total=total_frames,
            desc=f"▸ {uid_display}",
            position=1,
            leave=False,
            ncols=None,
            unit='f',
        )
        
        # 記錄處理狀態
        processing_status = "success"
        last_processed_frame = 0
        
        # 記錄 peak memory usage
        peak_kv_tokens = 0  # KV cache 的最大 token 數量
        peak_vram_gb = 0.0  # 最大 VRAM 使用量（GB）
        
        try:
            for fid, frame in enumerate(frame_buffer):
                # 更新進度條
                frame_pbar.update(1)
                last_processed_frame = fid
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
                    state['last_ids'] = torch.cat([
                        torch.tensor([[tokenizer.eos_token_id]], device=device),
                        state['_added_stream_prompt_ids']], dim=1)
                
                if last_role == 'user':
                    output_ids = self._simulate_stream_response(state=state, device=device)
                    pd_responses.append((fid / fps, {
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
                    pd_responses.append((fid / fps, {
                        'role': 'assistant',
                        'content': tokenizer.decode(output_ids[0], skip_special_tokens=True)[1:]
                    }))
                    last_role = 'assistant'
                
                # 記錄 peak memory usage
                if torch.cuda.is_available():
                    # 記錄 VRAM 使用量
                    current_vram_gb = torch.cuda.memory_allocated(device) / 1e9
                    peak_vram_gb = max(peak_vram_gb, current_vram_gb)
                    
                    # 記錄 KV cache 的 token 數量
                    if state.get('past_key_values') is not None:
                        kv_cache = state['past_key_values']
                        if isinstance(kv_cache, InfCache):
                            current_kv_tokens = kv_cache.actual_cache_length
                        elif isinstance(kv_cache, DynamicCache):
                            current_kv_tokens = kv_cache.get_seq_length()
                        else:
                            # 假設 HF Cache 結構
                            current_kv_tokens = kv_cache[0][0].size(2)
                        peak_kv_tokens = max(peak_kv_tokens, current_kv_tokens)

                if self.config.use_infcache:
                    state['past_key_values'].update_memory()
        except CacheOverflowError as e:
            processing_status = "cache_overflow"
            logger.info(f"Cache overflow during inference for {sample_uid}: {e}")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                processing_status = "cuda_oom"
                logger.info(f"CUDA OOM during inference for {sample_uid}: {str(e)[:200]}")
                
                # 記錄 OOM 前的記憶體狀態
                device_id = torch.cuda.current_device()
                allocated_before = torch.cuda.memory_allocated(device_id) / 1e9
                reserved_before = torch.cuda.memory_reserved(device_id) / 1e9
                logger.info(f"Memory BEFORE cleanup: allocated={allocated_before:.2f}GB, reserved={reserved_before:.2f}GB")
                
                # ============================================================
                # 超級積極的記憶體清理策略
                # ============================================================
                import gc
                
                # 步驟 1: 診斷 - 找出最大的 tensor
                logger.info("Diagnosing large tensors...")
                large_tensors = []
                try:
                    for obj in gc.get_objects():
                        if torch.is_tensor(obj):
                            if obj.is_cuda:
                                size_mb = obj.element_size() * obj.nelement() / 1024 / 1024
                                if size_mb > 10:  # 大於 10MB 的 tensor
                                    large_tensors.append((size_mb, obj.shape, type(obj).__name__))
                    large_tensors.sort(reverse=True)
                    for size_mb, shape, typename in large_tensors[:10]:  # 顯示前 10 個最大的
                        logger.info(f"  - {size_mb:.1f}MB: {shape} ({typename})")
                except Exception as diag_e:
                    logger.warning(f"Diagnostic failed: {diag_e}")
                
                # 步驟 2: 清理 KV cache（最關鍵！）
                logger.info("Cleaning KV cache...")
                kv_freed = False
                try:
                    if state and 'past_key_values' in state and state['past_key_values'] is not None:
                        kv_cache = state['past_key_values']
                        logger.info(f"  KV cache type: {type(kv_cache).__name__}")
                        
                        # 清理 DynamicCache / HF Cache
                        if hasattr(kv_cache, 'key_cache') and hasattr(kv_cache, 'value_cache'):
                            num_layers = len(kv_cache.key_cache)
                            logger.info(f"  Clearing {num_layers} layers...")
                            for i in range(num_layers):
                                if kv_cache.key_cache[i] is not None:
                                    del kv_cache.key_cache[i]
                                if kv_cache.value_cache[i] is not None:
                                    del kv_cache.value_cache[i]
                            kv_cache.key_cache.clear()
                            kv_cache.value_cache.clear()
                            kv_freed = True
                        
                        # 刪除 cache 對象本身
                        del state['past_key_values']
                        state['past_key_values'] = None
                        logger.info(f"  KV cache deleted: {kv_freed}")
                except Exception as kv_e:
                    logger.warning(f"KV cache cleanup failed: {kv_e}")
                
                # 步驟 3: 清理所有 state 內容
                logger.info("Cleaning state dict...")
                try:
                    if state:
                        # 刪除所有已知的大對象
                        for key in ['inplace_output_ids', 'last_ids', '_added_stream_prompt_ids', 
                                    '_added_stream_generation_ids', 'inputs_embeds', 'attention_mask']:
                            if key in state:
                                del state[key]
                        
                        # 刪除所有 tensor
                        tensor_keys = [k for k, v in state.items() if isinstance(v, torch.Tensor)]
                        for key in tensor_keys:
                            del state[key]
                        
                        logger.info(f"  Cleared {len(tensor_keys)} tensors from state")
                        state.clear()
                except Exception as state_e:
                    logger.warning(f"State cleanup failed: {state_e}")
                
                # 步驟 4: 清理 frame buffer 和 frames
                logger.info("Cleaning frames...")
                try:
                    # frame_buffer 可能是 FrameCache 對象或直接是 tensor
                    if frame_buffer is not None:
                        if hasattr(frame_buffer, 'buffer'):
                            del frame_buffer.buffer
                        if hasattr(frame_buffer, 'cpu_frames'):
                            del frame_buffer.cpu_frames
                    
                    # frames 是函數參數，設為 None（不要刪除，會影響外部）
                    # 但可以嘗試釋放內部數據
                    del frame_buffer
                except Exception as frame_e:
                    logger.warning(f"Frame cleanup failed: {frame_e}")
                
                # 步驟 5: 多輪垃圾回收
                logger.info("Running garbage collection...")
                collected = [gc.collect() for _ in range(3)]
                logger.info(f"  GC collected: {sum(collected)} objects")
                
                # 步驟 6: 清空 CUDA cache
                logger.info("Emptying CUDA cache...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            else:
                processing_status = "other_error"
                raise
        finally:
            frame_pbar.close()

        # 計算處理資訊
        video_duration_sec = total_frames / fps
        processed_duration_sec = (last_processed_frame + 1) / fps
        completion_rate = processed_duration_sec / video_duration_sec if video_duration_sec > 0 else 0.0
        
        # 在處理結束後記錄最終的 KV cache token 數量
        total_kv_tokens = 0
        if state.get('past_key_values') is not None:
            kv_cache = state['past_key_values']
            if isinstance(kv_cache, DynamicCache):
                total_kv_tokens = kv_cache.get_seq_length()
            else:
                # 假設 HF Cache 結構
                total_kv_tokens = kv_cache[0][0].size(2)
        
        # 保存推理結果到緩存
        if cache_path:
            self._save_prediction_to_cache(
                cache_path=cache_path,
                sample_uid=sample_uid,
                pd_responses=pd_responses,
                gt_responses=gt_responses,
                fps=fps,
                processing_status=processing_status,
                total_frames=total_frames,
                processed_frames=last_processed_frame + 1,
                video_duration_sec=video_duration_sec,
                processed_duration_sec=processed_duration_sec,
                completion_rate=completion_rate,
                peak_kv_tokens=peak_kv_tokens,
                peak_vram_gb=peak_vram_gb,
                total_kv_tokens=total_kv_tokens
            )
        
        # ====================================================================
        # Stage 2: Evaluation（可選，依據 skip_evaluation 決定）
        # ====================================================================
        
        if not skip_inference:
            # 只執行推理，不評估（節省 VRAM）
            logger.info(f"Inference completed for {sample_uid}, skipping evaluation")
            return None
        
        # 如果發生錯誤，返回零指標
        if processing_status in ["cuda_oom", "other_error"]:
            logger.info(f"Returning zero metrics for {sample_uid} due to {processing_status}")
            # 最終清理
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            return self._get_zero_metrics()
        
        # 計算 GT 時間偏移量
        gt_time_offset = gt_responses[0][0] if gt_responses else 0.0
        
        # 執行評估（使用 Sentence-BERT）
        return self._evaluate_responses(
            pd_responses, 
            gt_responses,
            gt_offset_sec=gt_time_offset
        )
    
    def _get_prediction_cache_path(self, cache_dir: str, sample_uid: str) -> str:
        """獲取預測結果的緩存路徑"""
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return os.path.join(cache_dir, f"{sample_uid}.json")
    
    def _save_inference_to_file(self, output_dir: str, sample_uid: str, pd_responses: list, fps: float = 2.0, num_frames: int = 0):
        """將純推理結果保存到檔案
        
        Args:
            output_dir: 輸出目錄
            sample_uid: 影片唯一識別碼
            pd_responses: 生成的回應列表 [(timestamp_sec, content), ...]
            fps: 影格率
            num_frames: 影片總幀數
        """
        import json
        import os
        from pathlib import Path
        
        # 提取時間戳記和內容（不做時間平移，保留原始時間）
        def extract_data(responses):
            result = []
            for timestamp_sec, resp in responses:
                content = resp if isinstance(resp, str) else str(resp)
                result.append([timestamp_sec, content])
            return result
        
        output_data = {
            'sample_uid': sample_uid,
            'responses': extract_data(pd_responses),
            'fps': fps,
            'num_frames': num_frames,
            'duration_sec': num_frames / fps if fps > 0 else 0.0
        }
        
        # 寫入 JSON 檔案
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = os.path.join(output_dir, f"{sample_uid}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved inference result to {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save inference result: {e}")
    
    def _save_prediction_to_cache(
        self, 
        cache_path: str, 
        sample_uid: str, 
        pd_responses: list, 
        gt_responses: list, 
        fps: float = 2.0,
        processing_status: str = "success",
        total_frames: int = 0,
        processed_frames: int = 0,
        video_duration_sec: float = 0.0,
        processed_duration_sec: float = 0.0,
        completion_rate: float = 1.0,
        peak_kv_tokens: int = 0,
        peak_vram_gb: float = 0.0,
        total_kv_tokens: int = 0
    ):
        """將預測結果保存到緩存檔案
        
        Args:
            cache_path: 緩存檔案路徑
            sample_uid: 測資唯一識別碼
            pd_responses: 預測的回應列表 [(timestamp_sec, response_dict), ...]
            gt_responses: 標註的回應列表 [(timestamp_sec, response_dict), ...]
            fps: 影格率（用於記錄）
            processing_status: 處理狀態 ("success", "cache_overflow", "cuda_oom", "other_error")
            total_frames: 影片總幀數
            processed_frames: 實際處理的幀數
            video_duration_sec: 影片總時長（秒）
            processed_duration_sec: 實際處理的時長（秒）
            completion_rate: 完成率 (0.0-1.0)
            peak_kv_tokens: KV cache 的峰值 token 數量
            peak_vram_gb: 峰值 VRAM 使用量（GB）
            total_kv_tokens: 影片處理結束後的總 token 數量
        
        注意：
        - gt_responses 的時間戳記會被平移，使第一個事件的時間為 0
        - pd_responses 保留原始時間戳記（相對於影片幀位置）
        """
        # 提取時間戳記和內容，並對 GT 時間進行平移
        def extract_gt_data(responses):
            result = []
            # 找到第一個事件的時間作為偏移量
            time_offset = responses[0][0] if responses else 0.0
            
            for timestamp_sec, resp in responses:
                # 只保留 content，移除 role 和其他不需要的欄位
                content = resp.get('content', '') if isinstance(resp, dict) else str(resp)
                # 平移時間，使第一個事件為 0
                shifted_time = timestamp_sec - time_offset
                result.append([shifted_time, content])
            return result, time_offset
        
        def extract_pd_data(responses):
            result = []
            for timestamp_sec, resp in responses:
                # 只保留 content，移除 role 和其他不需要的欄位
                content = resp.get('content', '') if isinstance(resp, dict) else str(resp)
                result.append([timestamp_sec, content])
            return result
        
        gt_data, time_offset = extract_gt_data(gt_responses)
        pd_data = extract_pd_data(pd_responses)
        
        cache_data = {
            'sample_uid': sample_uid,
            'pd_responses': pd_data,
            'gt_responses': gt_data,
            'fps': fps,
            'gt_time_offset': time_offset,  # 記錄 GT 的時間偏移量，以便必要時還原
            # 處理狀態資訊
            'processing_status': processing_status,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'video_duration_sec': video_duration_sec,
            'processed_duration_sec': processed_duration_sec,
            'completion_rate': completion_rate,
            # 峰值記憶體使用量
            'peak_kv_tokens': peak_kv_tokens,
            'peak_vram_gb': peak_vram_gb,
            'total_kv_tokens': total_kv_tokens
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
    
    def _load_prediction_from_cache(self, cache_path: str, max_time: float = 0.0) -> Optional[dict]:
        """從緩存檔案讀取預測結果
        
        Returns:
            包含 pd_responses, gt_responses, gt_time_offset 的字典
            如果讀取失敗則返回 None
        """
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # 緩存格式: [[timestamp_sec, content], ...]
            # 返回格式: [(timestamp_sec, content), ...] - 直接返回字符串
            def parse_responses(responses):
                result = []
                for item in responses:
                    if isinstance(item, list) and len(item) >= 2:
                        timestamp_sec, content = item[0], item[1]
                        if max_time > 0.0 and timestamp_sec > max_time:
                            break
                        # 直接返回字符串內容
                        if isinstance(content, dict):
                            # 兼容舊格式（如果有）
                            result.append((timestamp_sec, content.get('content', '')))
                        else:
                            result.append((timestamp_sec, content))
                    else:
                        logger.warning(f"Unexpected cache format: {item}")
                return result
            
            return {
                'pd_responses': parse_responses(cache_data['pd_responses']),
                'gt_responses': parse_responses(cache_data['gt_responses']),
                'gt_time_offset': cache_data.get('gt_time_offset', 0.0)  # 讀取 GT 時間偏移量
            }
        except Exception as e:
            logger.warning(f"Failed to load prediction from {cache_path}: {e}")
            return None
    
    def _get_zero_metrics(self):
        """返回全零的指標，用於跳過的測資"""
        # 指標格式：[F1@1s, F1@2s, F1@3s, MAE, BERT_Sim_Avg, ROUGE_L_Avg, Text_Sim_Avg, Composite_Score]
        num_metrics = 8
        
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
