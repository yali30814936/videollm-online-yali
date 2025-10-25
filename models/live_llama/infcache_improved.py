"""
Improved InfCache implementation with better separation of concerns.
改進的 InfCache 實作，更好的關注點分離。
"""

import torch
from typing import Dict, Any, Optional, Tuple, List, Union
from transformers.cache_utils import DynamicCache
from enum import Enum
from collections import deque


class CacheOverflowError(Exception):
    """
    當 cache 超過 n_max 且使用 crash 模式時拋出的異常
    
    用於模擬記憶體不足的情況，作為其他 eviction 策略的 baseline 比較。
    
    Attributes:
        cache_size: 當前 cache 大小
        n_max: 最大允許的 cache 大小
        message: 錯誤訊息
    """
    def __init__(self, cache_size: int, n_max: int, message: str = "Cache overflow"):
        self.cache_size = cache_size
        self.n_max = n_max
        self.message = message
        super().__init__(f"{message}: cache_size={cache_size}, n_max={n_max}")


class TurnAttendTracker:
    """
    追蹤最近 k 步的 turn attention quality
    
    核心概念：
    - 只保留最近 k 次 forward 的歷史記錄
    - 每次 forward 計算所有 turns 的 attend quality
    - Eviction 時選擇最近 k 步平均 quality 最低的 turn
    
    計算方法：
    1. Token attend score = 跨層加權平均（deeper layers 權重更高）
    2. Turn quality = sum(token_scores) / sqrt(turn_length)
    3. Turn eviction score = mean(recent_k_qualities)
    """
    
    def __init__(
        self, 
        history_k: int = 50,
        protect_newest_turns: int = 1,
        protect_oldest_turns: int = 1
    ):
        """
        Args:
            history_k: 保留最近 k 次 forward 的歷史
            protect_newest_turns: 保護最新的 N 個 turns
            protect_oldest_turns: 保護最早的 N 個 turns
        """
        self.history_k = history_k
        self.protect_newest_turns = protect_newest_turns
        self.protect_oldest_turns = protect_oldest_turns
        
        # 儲存每個 turn 的最近 k 步 quality
        # turn_id -> deque([q1, q2, ...], maxlen=history_k)
        self.turn_history: Dict[int, deque] = {}
        
        # Layer weights (deeper layers 權重更高)
        # 延遲初始化，等第一次看到 attention 時才計算
        self.layer_weights: Optional[torch.Tensor] = None
        self.num_layers: int = 0
    
    def _initialize_layer_weights(self, num_layers: int, device: torch.device):
        """
        初始化層權重（線性遞增 + 歸一化）
        
        Args:
            num_layers: 模型層數
            device: 計算設備
        """
        if self.layer_weights is not None and self.num_layers == num_layers:
            return
        
        self.num_layers = num_layers
        # 線性權重: [1, 2, 3, ..., num_layers]
        weights = torch.arange(1, num_layers + 1, dtype=torch.float32, device=device)
        # 歸一化（總和 = 1）
        self.layer_weights = weights / weights.sum()
    
    def compute_token_attend_scores(
        self, 
        all_self_attns: tuple,
        new_token_range: Tuple[int, int]
    ) -> Optional[torch.Tensor]:
        """
        計算每個 key token 的 attend score（跨層加權平均）
        
        工作流程：
        1. 對於每一層：
           - 提取新增 queries 對所有 keys 的 attention
           - 對 heads 取平均
           - 對 queries 取平均
        2. 跨層加權平均（deeper layers 權重更高）
        
        Args:
            all_self_attns: Tuple of attention weights from all layers
                每層: (batch_size, num_heads, query_len, key_len)
            new_token_range: (start_idx, end_idx) 新增的 query tokens 範圍
        
        Returns:
            token_attend_scores: (key_len,) 每個 key token 的 attend score
                如果無法計算則返回 None
        """
        if not all_self_attns or len(all_self_attns) == 0:
            return None
        
        # 只處理 batch_size=1 的情況
        first_attn = all_self_attns[0]
        if first_attn.size(0) != 1:
            return None
        
        num_layers = len(all_self_attns)
        key_len = first_attn.shape[-1]
        device = first_attn.device
        
        # 初始化層權重
        self._initialize_layer_weights(num_layers, device)
        
        # 提取新增的 query 範圍
        # new_token_range 是絕對位置，但 attention 的 query 維度是相對的
        # 例如：new_token_range = (90, 100)，attention shape = (1, 8, 10, 100)
        # 這意味著新增了 10 個 queries，它們在 attention 中的索引是 [0:10]
        new_start, new_end = new_token_range
        num_new_queries = new_end - new_start
        query_len = first_attn.shape[-2]
        
        # 檢查：新增的 queries 應該是 attention 的最後幾個
        if num_new_queries > query_len or num_new_queries <= 0:
            return None
        
        # 在 attention 中，新增的 queries 位於最後 num_new_queries 個位置
        query_start_in_attn = query_len - num_new_queries
        query_end_in_attn = query_len
        
        # 收集每層的 per-token importance
        layer_importances = []
        
        for attn_weights in all_self_attns:
            # attn_weights: (1, num_heads, query_len, key_len)
            attn_weights = attn_weights.squeeze(0)  # (num_heads, query_len, key_len)
            
            # 只取新增的 queries
            new_queries_attn = attn_weights[:, query_start_in_attn:query_end_in_attn, :]  # (num_heads, num_new_queries, key_len)
            
            # 對 heads 取平均
            attn_avg_heads = new_queries_attn.mean(dim=0)  # (num_new_queries, key_len)
            
            # 對 queries 取平均（每個 key 被新 queries 關注的平均程度）
            per_token_importance = attn_avg_heads.mean(dim=0)  # (key_len,)
            
            layer_importances.append(per_token_importance)
        
        # Stack 成 (num_layers, key_len)
        layer_importances = torch.stack(layer_importances, dim=0)
        
        # 跨層加權平均（deeper layers 權重更高）
        # layer_weights: (num_layers,) -> (num_layers, 1)
        weighted_importances = layer_importances * self.layer_weights.unsqueeze(1)
        final_importances = weighted_importances.sum(dim=0)  # (key_len,)
        
        return final_importances
    
    def compute_turn_quality(
        self, 
        token_scores: torch.Tensor, 
        turn_token_range: Tuple[int, int]
    ) -> float:
        """
        計算 turn 的 attend quality（有長度校正）
        
        公式：quality = sum(token_scores) / sqrt(turn_length)
        
        Args:
            token_scores: (key_len,) 每個 token 的 attend score
            turn_token_range: (start_idx, end_idx) turn 的 token 範圍（end_idx 包含）
        
        Returns:
            turn_quality: 該 turn 的品質分數
        """
        start_idx, end_idx = turn_token_range
        
        # 檢查範圍是否有效
        if start_idx < 0 or end_idx < 0 or start_idx > end_idx:
            return 0.0
        
        if end_idx + 1 > len(token_scores):
            return 0.0
        
        # 提取 turn 內的 token scores
        turn_scores = token_scores[start_idx:end_idx + 1]
        
        # 計算長度
        turn_length = len(turn_scores)
        if turn_length == 0:
            return 0.0
        
        # 長度校正：sum / sqrt(N)
        quality = turn_scores.sum().item() / (turn_length ** 0.5)
        
        return quality
    
    def update(
        self, 
        all_self_attns: tuple, 
        turns: List,  # List[Turn]
        new_token_range: Tuple[int, int]
    ):
        """
        更新所有 turns 的 attend quality（添加一步記錄）
        
        Args:
            all_self_attns: 所有層的 attention weights
            turns: 所有 Turn 對象的列表
            new_token_range: 本次 forward 新增的 token 範圍
        """
        # 計算所有 tokens 的 attend scores
        token_scores = self.compute_token_attend_scores(all_self_attns, new_token_range)
        
        if token_scores is None:
            return
        
        # 為每個 finalized turn 計算 quality
        for turn in turns:
            if not turn.is_finalized:
                continue
            
            # 初始化該 turn 的歷史記錄（如果還沒有）
            if turn.turn_idx not in self.turn_history:
                self.turn_history[turn.turn_idx] = deque(maxlen=self.history_k)
            
            # 計算 turn quality
            turn_quality = self.compute_turn_quality(
                token_scores, 
                turn.get_token_range()
            )
            
            # 添加到歷史記錄
            self.turn_history[turn.turn_idx].append(turn_quality)
    
    def get_worst_turn(self, turns: List) -> Optional[Any]:  # Optional[Turn]
        """
        獲取最近 k 步中 attend 最小的 turn
        
        策略：
        1. 累積 k 步後才開始刪除
        2. 計算每個 turn 最近 k 步的平均 quality
        3. 選擇平均值最低的 turn
        4. 豁免最新和最早的 turns（基於相對位置，不是 turn_idx）
        
        Args:
            turns: 所有 Turn 對象的列表
        
        Returns:
            最該被刪除的 turn（或 None）
        """
        # 先篩選出所有 finalized turns（用於計算相對位置）
        finalized_turns = [t for t in turns if t.is_finalized]
        
        if not finalized_turns:
            return None
        
        # 收集候選 turns（帶有位置信息）
        candidates = []
        
        for position, turn in enumerate(finalized_turns):
            # 檢查是否在歷史記錄中
            if turn.turn_idx not in self.turn_history:
                continue
            
            history = self.turn_history[turn.turn_idx]
            
            # 檢查是否累積了足夠的歷史（至少 k 步）
            if len(history) < self.history_k:
                continue
            
            # 計算平均 quality
            avg_quality = sum(history) / len(history)
            
            candidates.append((turn, avg_quality, position))
        
        if not candidates:
            return None
        
        # 排序：quality 最低的優先
        candidates.sort(key=lambda x: x[1])
        
        # 應用豁免規則（基於相對位置）
        num_finalized = len(finalized_turns)
        
        # 從候選中找到第一個未被保護的 turn
        for turn, quality, position in candidates:
            # 保護最早的 N 個 turns
            if position < self.protect_oldest_turns:
                continue
            
            # 保護最新的 N 個 turns
            if position >= num_finalized - self.protect_newest_turns:
                continue
            
            # 這個 turn 沒有被保護，可以刪除
            return turn
        
        # 所有候選都被保護了
        return None
    
    def remove_turn_history(self, turn_idx: int):
        """
        刪除指定 turn 的歷史記錄
        
        Args:
            turn_idx: 要刪除的 turn 索引
        """
        if turn_idx in self.turn_history:
            del self.turn_history[turn_idx]
    
    def get_statistics_summary(self) -> str:
        """獲取統計摘要（用於調試）"""
        lines = [
            f"TurnAttendTracker:",
            f"  History K: {self.history_k}",
            f"  Tracked Turns: {len(self.turn_history)}",
            f"  Layer Weights: {self.layer_weights.tolist() if self.layer_weights is not None else 'Not initialized'}",
            ""
        ]
        
        if self.turn_history:
            lines.append("  Turn Histories:")
            for turn_idx, history in sorted(self.turn_history.items()):
                avg_quality = sum(history) / len(history) if history else 0.0
                lines.append(
                    f"    Turn {turn_idx}: {len(history)} steps, "
                    f"avg_quality={avg_quality:.4f}"
                )
        
        return "\n".join(lines)


class InfCacheState(Enum):
    """對話狀態追蹤"""
    INIT                  = 0
    START                 = 1  # <BOS>{system prompt}\n\n[<v>
    ASSISTANT_HEAD        = 2  # ]\nAssistant:
    USER_AND_ASSISTANT    = 3  # ]\n{user query}\nAssistant:
    RESPONSE_A_TOKEN      = 4  # {single assistant response token}
    STREAM_AFTER_RESPONSE = 5  # <EOS>\n[<v>
    NEXT_FRAME            = 6  # ,<v>
    EOS_AND_QUERY         = 7  # <EOS>\n{user query}\nAssistant:


class Sentence:
    """對話片段基類"""
    def __init__(self, start_idx: int, end_idx: int = -1):
        self.start_idx = start_idx
        self.end_idx = end_idx
        # Reference-based eviction 相關統計
        self.reference_count: int = 0
        self.importance_score: float = 0.0
        # Recency tracking (類似 LRU)
        self.last_access_step: int = 0  # 最後一次 reference_count 被增加的 forward step

    def __repr__(self):
        return f"{self.__class__.__name__}(start={self.start_idx}, end={self.end_idx}, ref={self.reference_count}, score={self.importance_score:.2f}, last_access={self.last_access_step})"


class SystemPrompt(Sentence):
    pass


class UserQuery(Sentence):
    pass


class AssistantResponse(Sentence):
    def end_response(self, end_idx: int):
        self.end_idx = end_idx


class Turn:
    """
    一個完整的對話輪次 (Turn)
    
    定義：從上一個 AssistantResponse 結束到當前 AssistantResponse 結束的所有內容
    
    典型結構:
        Turn 0: SystemPrompt + StreamFrame
        Turn 1: UserQuery(opt) + StreamFrame + AssistantResponse
        Turn 2: StreamFrame + AssistantResponse
        Turn 3: StreamFrame + AssistantResponse
        ...
    
    Reference 計算邏輯:
        - 只要 turn 內任何一個 token 被關注 (importance > threshold)
        - 整個 turn 的 reference_count += 1
    
    Eviction 邏輯:
        - 刪除整個 turn (從 start_idx 到 end_idx)
        - 優先刪除 reference_count 最低的 turns
    """
    
    def __init__(self, turn_idx: int):
        self.turn_idx = turn_idx
        # 包含的 events (StreamFrame, UserQuery, AssistantResponse)
        self.event_start_idx: int = -1  # 在 conversations list 中的起始索引
        self.event_end_idx: int = -1    # 在 conversations list 中的結束索引（包含）
        
        # Token 範圍
        self.start_idx: int = -1  # 第一個 event 的 start_idx
        self.end_idx: int = -1    # 最後一個 AssistantResponse 的 end_idx
        
        # Reference tracking
        self.reference_count: int = 0
        self.importance_score: float = 0.0  # Turn 內最大的 token importance
        # Recency tracking (類似 LRU)
        self.last_access_step: int = 0  # 最後一次 reference_count 被增加的 forward step
        
        self.is_finalized: bool = False
    
    def set_events(self, start_event_idx: int, end_event_idx: int):
        """設置包含的 events 範圍"""
        self.event_start_idx = start_event_idx
        self.event_end_idx = end_event_idx
    
    def set_token_range(self, start_idx: int, end_idx: int):
        """設置 token 範圍"""
        self.start_idx = start_idx
        self.end_idx = end_idx
    
    def finalize(self):
        """標記為已完成（當 AssistantResponse 結束時）"""
        self.is_finalized = True
    
    def get_token_range(self) -> Tuple[int, int]:
        """返回整個 turn 的 token 範圍"""
        return (self.start_idx, self.end_idx)
    
    def __repr__(self):
        return (f"Turn(idx={self.turn_idx}, "
                f"events=[{self.event_start_idx}:{self.event_end_idx+1}], "
                f"tokens=[{self.start_idx}:{self.end_idx+1}], "
                f"ref={self.reference_count}, "
                f"score={self.importance_score:.2f}, "
                f"finalized={self.is_finalized})")


class StreamFrame(Sentence):
    def __init__(self, num_frame_tokens: int, start_idx: int):
        super().__init__(start_idx)
        self.num_frame_tokens = num_frame_tokens
        # 改用相對索引：存儲 frame 的序號 (0, 1, 2, ...)
        self.frame_indices: deque[int] = deque()
        # 每個 frame 的統計 (frame_idx -> {"reference_count": int, "importance_score": float})
        self.frame_stats: Dict[int, Dict[str, float]] = {}
        # 追蹤下一個 frame 的索引
        self.next_frame_idx = 0
        # 添加第一個 frame
        self.append_frame()

    def append_frame(self):
        """添加新的 frame（使用自動遞增的索引）"""
        frame_idx = self.next_frame_idx
        self.frame_indices.append(frame_idx)
        # 初始化該 frame 的統計
        self.frame_stats[frame_idx] = {
            "reference_count": 0,
            "importance_score": 0.0,
            "last_access_step": 0,  # Recency tracking
        }
        self.next_frame_idx += 1

    def end_stream(self, end_idx: int):
        self.end_idx = end_idx

    def pop_frame(self) -> Optional[int]:
        """
        刪除最早的 frame (FIFO)，返回該 frame 在 cache 中的起始位置
        
        注意：此方法只返回要刪除的位置，不更新座標
        座標更新由 remove_range() 統一處理
        """
        if len(self.frame_indices) > 1:
            frame_idx = self.frame_indices.popleft()
            # 清理該 frame 的統計
            if frame_idx in self.frame_stats:
                del self.frame_stats[frame_idx]
            
            # 計算被刪除的 frame 在 cache 中的位置
            # 第一個 frame 在 start_idx + 1 (跳過 '[')
            token_start = self.start_idx + 1
            
            # ⚠️ 不在這裡更新 end_idx，由 remove_range() 統一處理
            
            return token_start
        return None
    
    def remove_specific_frame(self, frame_idx: int) -> Optional[int]:
        """
        移除指定的 frame（用於 reference-based eviction）
        返回該 frame 在序列中的 token 起始位置
        
        Args:
            frame_idx: frame 的相對索引
        
        Returns:
            該 frame 在 cache 中的起始位置
        
        注意：此方法只返回要刪除的位置，不更新座標
        座標更新由 remove_range() 統一處理
        """
        if frame_idx not in self.frame_indices:
            return None
        
        # 計算該 frame 在當前 frame 列表中的位置
        frame_position = list(self.frame_indices).index(frame_idx)
        
        # 計算該 frame 在 cache 中的 token 起始位置
        # start_idx 是 '[' 的位置
        # 第一個 frame: start_idx + 1 (沒有前導逗號)
        # 後續 frames: start_idx + 1 + position * (num_frame_tokens + 1)
        if frame_position == 0:
            token_start = self.start_idx + 1  # '[<v>...'
        else:
            token_start = self.start_idx + 1 + frame_position * (self.num_frame_tokens + 1)
        
        # 移除 frame
        self.frame_indices.remove(frame_idx)
        if frame_idx in self.frame_stats:
            del self.frame_stats[frame_idx]
        
        # ⚠️ 不在這裡更新 end_idx，由 remove_range() 統一處理
        
        return token_start
    
    def get_token_range_for_frame(self, frame_idx: int) -> Optional[Tuple[int, int]]:
        """
        獲取指定 frame 在 cache 中的 token 範圍
        
        Args:
            frame_idx: frame 的相對索引
        
        Returns:
            (token_start, token_end) 或 None
            token_end 是不包含的 (即 [token_start, token_end) 區間)
        """
        if frame_idx not in self.frame_indices:
            return None
        
        frame_position = list(self.frame_indices).index(frame_idx)
        
        if frame_position == 0:
            token_start = self.start_idx + 1
        else:
            token_start = self.start_idx + 1 + frame_position * (self.num_frame_tokens + 1)
        
        token_end = token_start + self.num_frame_tokens
        
        return (token_start, token_end)
    
    def get_least_important_frame(self, exclude_latest: bool = True) -> Optional[int]:
        """
        獲取最不重要的 frame
        
        Args:
            exclude_latest: 是否排除最新的 frame（豁免權）
        
        Returns:
            frame_idx or None
        """
        if len(self.frame_indices) == 0:
            return None
        
        candidates = list(self.frame_indices)
        
        # 最新的 frame 有豁免權
        if exclude_latest and len(candidates) > 1:
            candidates = candidates[:-1]
        
        if not candidates:
            return None
        
        # 找到引用次數最低的 frame
        min_frame = min(
            candidates,
            key=lambda f: (
                self.frame_stats[f]["reference_count"],
                self.frame_stats[f]["importance_score"]
            )
        )
        
        return min_frame

    def __repr__(self):
        return f"StreamFrame(start={self.start_idx}, end={self.end_idx}, frames={len(self.frame_indices)}, ref={self.reference_count}, score={self.importance_score:.2f})"


class ConversationTracker:
    """
    獨立的對話結構追蹤器
    Separate conversation structure tracker
    """
    def __init__(
        self,
        num_frame_tokens: int = 1,
        bos_token_id: int = 128000,
        eos_token_id: int = 128009,
        vision_token_id: int = 128256,
        assistant_tokens: Tuple[int, ...] = (933, 72803, 25),  # "]\nAssistant:"
    ):
        self.num_frame_tokens = num_frame_tokens
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.vision_token_id = vision_token_id
        self.assistant_tokens = torch.tensor(assistant_tokens)
        
        self.conversations: List[Sentence] = []
        self.current_position = 0
        self.state = InfCacheState.INIT
        
        # Turn-based tracking
        self.turns: List[Turn] = []
        self.current_turn: Optional[Turn] = None
        self.current_turn_start_event_idx: int = 0
        self.next_turn_idx: int = 0  # 全局單調遞增計數器，避免 turn_idx 重複

    def update(self, input_ids: torch.Tensor):
        """更新對話結構追蹤"""
        input_ids = input_ids.squeeze(0)  # (seq_len,)
        
        if self._is_start_sequence(input_ids):
            self._handle_start(input_ids)
        elif self._is_assistant_head(input_ids):
            self._handle_assistant_head()
        elif self._is_user_and_assistant(input_ids):
            self._handle_user_and_assistant(input_ids)
        elif self._is_single_token(input_ids):
            self._handle_response_token()
        elif self._is_stream_after_response(input_ids):
            self._handle_stream_after_response()
        elif self._is_next_frame(input_ids):
            self._handle_next_frame()
        elif self._is_eos_and_query(input_ids):
            self._handle_eos_and_query(input_ids)
        else:
            raise ValueError(f"Unrecognized input_ids pattern: {input_ids}")

    def _is_start_sequence(self, input_ids: torch.Tensor) -> bool:
        return input_ids[0].item() == self.bos_token_id

    def _is_assistant_head(self, input_ids: torch.Tensor) -> bool:
        return (input_ids.size(0) == 3 and 
                torch.equal(input_ids, self.assistant_tokens.to(input_ids.device)))

    def _is_user_and_assistant(self, input_ids: torch.Tensor) -> bool:
        return (input_ids.size(0) > 3 and 
                input_ids[0].item() != self.eos_token_id and
                torch.equal(input_ids[-2:], self.assistant_tokens[-2:].to(input_ids.device)))

    def _is_single_token(self, input_ids: torch.Tensor) -> bool:
        return input_ids.size(0) == 1

    def _is_stream_after_response(self, input_ids: torch.Tensor) -> bool:
        return input_ids[0].item() == self.eos_token_id and \
               input_ids.size(0) == 1 + 1 + 1 + self.num_frame_tokens  # <EOS>\n[<v>

    def _is_next_frame(self, input_ids: torch.Tensor) -> bool:
        return input_ids[-1].item() == self.vision_token_id

    def _is_eos_and_query(self, input_ids: torch.Tensor) -> bool:
        return input_ids[0].item() == self.eos_token_id
    
    def _handle_start(self, input_ids: torch.Tensor):
        if self.state != InfCacheState.INIT:
            raise ValueError("System prompt should be the first input.")
        
        system_ed = input_ids.size(0) - (1 + self.num_frame_tokens) - 1
        self.conversations.append(SystemPrompt(0, system_ed))
        self.current_position = system_ed + 1
        
        self.conversations.append(StreamFrame(self.num_frame_tokens, self.current_position))
        self.current_position = input_ids.size(0)
        
        self.state = InfCacheState.START

    def _handle_assistant_head(self):
        if self.state not in (InfCacheState.START, 
                              InfCacheState.STREAM_AFTER_RESPONSE, 
                              InfCacheState.NEXT_FRAME):
            raise ValueError(f"assistant_head invalid from state {self.state}")
        
        # End stream
        if isinstance(self.conversations[-1], StreamFrame):
            self.conversations[-1].end_stream(self.current_position)
        self.current_position += 1  # ']\n'
        
        # Start assistant response
        self.conversations.append(AssistantResponse(self.current_position))
        self.current_position += 2  # "Assistant:"
        
        self.state = InfCacheState.ASSISTANT_HEAD

    def _handle_user_and_assistant(self, input_ids: torch.Tensor):
        if self.state not in (InfCacheState.START, 
                              InfCacheState.STREAM_AFTER_RESPONSE, 
                              InfCacheState.NEXT_FRAME):
            raise ValueError(f"user_and_assistant invalid state {self.state}")
        
        # End stream
        if isinstance(self.conversations[-1], StreamFrame):
            self.conversations[-1].end_stream(self.current_position)
        self.current_position += 1  # ']\n'
        
        # Add user query
        query_ed = self.current_position + input_ids.size(0) - 1 - 2 - 1
        self.conversations.append(UserQuery(self.current_position, query_ed))
        self.current_position = query_ed + 1
        
        # Start assistant response
        self.conversations.append(AssistantResponse(self.current_position))
        self.current_position += 2  # "Assistant:"
        
        self.state = InfCacheState.USER_AND_ASSISTANT

    def _handle_response_token(self):
        if self.state not in (InfCacheState.ASSISTANT_HEAD,
                              InfCacheState.USER_AND_ASSISTANT,
                              InfCacheState.EOS_AND_QUERY,
                              InfCacheState.RESPONSE_A_TOKEN):
            raise ValueError(f"response_token invalid from state {self.state}")
        
        self.current_position += 1
        self.state = InfCacheState.RESPONSE_A_TOKEN

    def _handle_stream_after_response(self):
        if self.state not in (InfCacheState.RESPONSE_A_TOKEN,
                              InfCacheState.ASSISTANT_HEAD):
            raise ValueError(f"stream_after_response invalid from state {self.state}")
        
        # End assistant response
        if isinstance(self.conversations[-1], AssistantResponse):
            self.conversations[-1].end_response(self.current_position + 2 - 1)
            # ⭐ Finalize current turn (response 結束 = turn boundary)
            self._finalize_current_turn()
        
        self.current_position += 2  # "<EOS>\n"
        
        # Start stream
        self.conversations.append(StreamFrame(self.num_frame_tokens, self.current_position))
        self.current_position += 1 + self.num_frame_tokens  # [<v>
        
        self.state = InfCacheState.STREAM_AFTER_RESPONSE

    def _handle_next_frame(self):
        if self.state not in (InfCacheState.STREAM_AFTER_RESPONSE,
                              InfCacheState.NEXT_FRAME):
            raise ValueError(f"next_frame invalid from state {self.state}")
        
        # Append stream frame
        if isinstance(self.conversations[-1], StreamFrame):
            self.conversations[-1].append_frame()
        self.current_position += 1 + self.num_frame_tokens
        
        self.state = InfCacheState.NEXT_FRAME

    def _handle_eos_and_query(self, input_ids: torch.Tensor):
        if self.state != InfCacheState.RESPONSE_A_TOKEN:
            raise ValueError("eos_and_query should follow response_a_token.")
        # End assistant response
        if isinstance(self.conversations[-1], AssistantResponse):
            self.conversations[-1].end_response(self.current_position + 2 - 1)
            # ⭐ Finalize current turn
            self._finalize_current_turn()
        
        self.current_position += 2  # <EOS>\n
        # Add user query
        query_ed = self.current_position + input_ids.size(0) - 2 - 2 - 1 # -2 for <EOS>\n, -2 for Assistant:
        self.conversations.append(UserQuery(self.current_position, query_ed))
        self.current_position = query_ed + 1
        # Start assistant response
        self.conversations.append(AssistantResponse(self.current_position))
        self.current_position += 2  # "Assistant:"
    
    def _finalize_current_turn(self):
        """
        完成當前 turn（當 AssistantResponse 結束時調用）
        
        創建一個 Turn 對象，記錄從 current_turn_start_event_idx 到當前最後一個 event
        """
        if self.current_turn is None:
            # 創建新 turn，使用單調遞增的計數器確保 turn_idx 唯一
            turn_idx = self.next_turn_idx
            self.next_turn_idx += 1
            self.current_turn = Turn(turn_idx)
        
        # 設置 event 範圍
        end_event_idx = len(self.conversations) - 1
        self.current_turn.set_events(self.current_turn_start_event_idx, end_event_idx)
        
        # 設置 token 範圍
        start_event = self.conversations[self.current_turn_start_event_idx]
        end_event = self.conversations[end_event_idx]
        self.current_turn.set_token_range(start_event.start_idx, end_event.end_idx)
        
        # Finalize
        self.current_turn.finalize()
        self.turns.append(self.current_turn)
        
        # 準備下一個 turn
        self.current_turn = None
        self.current_turn_start_event_idx = len(self.conversations)
        
    def get_conversation_summary(self) -> str:
        """獲取對話摘要（用於調試）"""
        lines = [f"State: {self.state.name}", f"Position: {self.current_position}", ""]
        for i, conv in enumerate(self.conversations):
            lines.append(f"{i}: {conv}")
        return "\n".join(lines)


class InfCache(DynamicCache):
    """
    Inference cache with conversation tracking for video streaming scenarios.
    支持視訊串流場景的推理快取，包含對話追蹤功能。
    
    繼承 DynamicCache 處理 KV cache，額外追蹤對話結構。
    Inherits DynamicCache for KV cache management, with additional conversation tracking.
    """

    def __init__(
        self,
        n_max: int = 2048,
        n_min: int = 1536,  # 刪減目標：當 cache 超過 n_max 時，刪減到 n_min
        num_frame_tokens: int = 1,
        bos_token_id: int = 128000,
        eos_token_id: int = 128009,
        vision_token_id: int = 128256,
        enable_position_remapping: bool = False,
        eviction_strategy: str = "fifo",  # "fifo", "reference_based", "turn_based", or "crash"
        turn_eviction_policy: str = "reference_based",  # "fifo" or "reference_based" (僅用於 turn_based 策略)
        top_k_references: int = 5,  # 每次統計時給 top-k 個對象增加引用計數（reference_based 使用）
        enable_recency_tracking: bool = False,  # 是否啟用 recency tracking (類似 LRU)
        recency_weight: float = 0.5,  # Recency 的權重（0.0 = 只看 ref_count，1.0 = 只看 recency）
        enable_rope_repositioning: bool = False,  # 啟用 RoPE 重新定位
        rope_theta: float = 10000.0,  # RoPE 的 theta 參數
        rope_scaling: Optional[Dict] = None,  # RoPE scaling 配置
        head_dim: int = 128,  # Head dimension for RoPE calculation
        # 新增：Recent attend 策略相關參數
        use_recent_attend: bool = False,  # 是否啟用基於最近 k 步的 attend quality 策略
        recent_attend_history_k: int = 50,  # 保留最近 k 次 forward 的歷史
        protect_newest_turns: int = 1,  # 保護最新的 N 個 turns
        protect_oldest_turns: int = 1,  # 保護最早的 N 個 turns
        **kwargs
    ):
        super().__init__()
        self.n_max = n_max
        self.n_min = n_min if n_min <= n_max else int(n_max * 0.75)  # 確保 n_min <= n_max
        self.num_frame_tokens = num_frame_tokens
        self.enable_position_remapping = enable_position_remapping
        self.eviction_strategy = eviction_strategy
        self.turn_eviction_policy = turn_eviction_policy  # 新增：turn-based 的刪除策略
        self.top_k_references = top_k_references
        self.enable_recency_tracking = enable_recency_tracking  # 新增：recency tracking 開關
        self.recency_weight = recency_weight if 0.0 <= recency_weight <= 1.0 else 0.5  # 確保在 [0, 1] 範圍
        
        # Forward step counter (用於 recency tracking)
        self.forward_step: int = 0
        
        # RoPE repositioning 相關
        self.enable_rope_repositioning = enable_rope_repositioning
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.head_dim = head_dim
        
        # RoPE rerotation cache (類似 SinkCache)
        self._cos_cache = None
        self._sin_cache = None
        self._max_position_computed = 0
        
        # 獨立的對話追蹤器
        self.conversation_tracker = ConversationTracker(
            num_frame_tokens=num_frame_tokens,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            vision_token_id=vision_token_id,
        )
        
        # Attention tracking (optional)
        self.enable_attention_tracking = eviction_strategy == "reference_based" and turn_eviction_policy != "fifo"
        
        # Recent attend tracking (新增)
        self.use_recent_attend = use_recent_attend
        self.recent_attend_tracker: Optional[TurnAttendTracker] = None
        if use_recent_attend:
            self.recent_attend_tracker = TurnAttendTracker(
                history_k=recent_attend_history_k,
                protect_newest_turns=protect_newest_turns,
                protect_oldest_turns=protect_oldest_turns
            )
        
        # Position remapping 相關
        # original_position -> actual_cache_position 的映射
        self.position_map: Dict[int, int] = {}
        # 記錄被刪除的 positions (original positions)
        self.dropped_positions: List[int] = []
        # 當前實際的 cache 長度（扣除被刪除的 tokens）
        self.actual_cache_length: int = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 使用父類的標準實作
        key_cache, value_cache = super().update(
            key_states, value_states, layer_idx, cache_kwargs
        )
        
        return key_cache, value_cache

    def update_conversation(self, input_ids: torch.Tensor):
        """更新對話結構追蹤（向後相容的介面）"""
        self.conversation_tracker.update(input_ids)
    
    def update_reference_counts(self, all_self_attns: tuple):
        """
        根據 attention statistics 更新對話對象的引用計數
        
        工作流程：
        1. 從 all_self_attns 計算每個 token 的跨層聚合 importance
        2. 為每個 Sentence/Frame 計算重要性分數（區間內最大值）
        3. 全局排序，給 top-k 個對象的引用計數 +1
        
        Args:
            all_self_attns: Tuple of attention weights from all layers
                每層: (batch_size, num_heads, query_len, key_len)
        """
        if not self.enable_attention_tracking:
            return
        
        if not all_self_attns or len(all_self_attns) == 0:
            return
        
        # 批量計算所有 token 的 importance（高效向量化實現）
        token_importances = self._compute_token_importances_batch(all_self_attns)
        
        if token_importances is None:
            return
        
        # 收集所有對象及其重要性分數
        # 格式: (object_type, object_ref, importance_score)
        objects_with_scores = []
        
        for event in self.conversation_tracker.conversations:
            if isinstance(event, StreamFrame):
                # StreamFrame: 每個 frame 單獨計算
                for frame_idx in event.frame_indices:
                    # 獲取該 frame 的 token 範圍
                    token_range = event.get_token_range_for_frame(frame_idx)
                    if token_range is None:
                        continue
                    
                    token_start, token_end = token_range
                    
                    # 檢查範圍是否有效
                    if token_end > len(token_importances):
                        continue
                    
                    # 該 frame 的重要性 = 區間內最大 importance
                    frame_importance = token_importances[token_start:token_end].max().item()
                    
                    # 記錄重要性分數
                    event.frame_stats[frame_idx]["importance_score"] = frame_importance
                    
                    objects_with_scores.append(("frame", (event, frame_idx), frame_importance))
            else:
                # SystemPrompt, UserQuery, AssistantResponse
                if event.end_idx == -1:
                    continue  # 還沒結束，跳過
                
                # 檢查範圍是否有效
                if event.start_idx < 0 or event.end_idx < 0:
                    continue  # 無效座標
                
                if event.end_idx + 1 > len(token_importances):
                    continue
                
                if event.start_idx >= event.end_idx + 1:
                    continue  # 空區間
                
                # 區間內最大 importance
                segment_importance = token_importances[event.start_idx:event.end_idx + 1].max().item()
                
                # 記錄重要性分數
                event.importance_score = segment_importance
                
                objects_with_scores.append(("sentence", event, segment_importance))
        
        # 全局排序，取 top-k
        objects_with_scores.sort(key=lambda x: x[2], reverse=True)
        top_k_objects = objects_with_scores[:self.top_k_references]
        
        # 更新引用計數
        for obj_type, obj_ref, score in top_k_objects:
            if obj_type == "frame":
                event, frame_idx = obj_ref
                event.frame_stats[frame_idx]["reference_count"] += 1
                # 更新 recency tracking
                if self.enable_recency_tracking:
                    event.frame_stats[frame_idx]["last_access_step"] = self.forward_step
            else:  # sentence
                obj_ref.reference_count += 1
                # 更新 recency tracking
                if self.enable_recency_tracking:
                    obj_ref.last_access_step = self.forward_step
        
        # 遞增 forward step counter
        if self.enable_recency_tracking:
            self.forward_step += 1
    
    def update_reference_counts_turn_based(self, all_self_attns: tuple):
        """
        Turn-based reference counting 策略
        
        核心思想：
        1. 以 AssistantResponse 結束為邊界，劃分 turns
        2. 只要 turn 內任何 token 的 importance 超過閾值，整個 turn 的 ref_count +1
        3. 刪除時以整個 turn 為單位
        
        工作流程：
        1. 計算所有 token 的 importance
        2. 對於每個 turn，檢查是否有 token 被顯著關注
        3. 如果有，turn 的 reference_count +1
        
        Args:
            all_self_attns: Tuple of attention weights from all layers
        """
        if not self.enable_attention_tracking:
            return
        
        if not all_self_attns or len(all_self_attns) == 0:
            return
        
        # 批量計算所有 token 的 importance
        token_importances = self._compute_token_importances_batch(all_self_attns)
        
        if token_importances is None:
            return
        
        # 計算全局平均 importance 作為閾值
        avg_importance = token_importances.mean().item()
        threshold = avg_importance * 0.5  # 超過平均值的 50% 才算被關注
        
        # 對於每個 finalized turn，檢查是否有 token 被關注
        for turn in self.conversation_tracker.turns:
            if not turn.is_finalized:
                continue
            
            start_idx, end_idx = turn.get_token_range()
            
            # 檢查範圍是否有效
            if start_idx < 0 or end_idx < 0:
                continue
            
            if end_idx + 1 > len(token_importances):
                continue
            
            if start_idx >= end_idx + 1:
                continue
            
            # 提取 turn 內所有 tokens 的 importance
            turn_importances = token_importances[start_idx:end_idx + 1]
            
            # Turn 的重要性分數 = 區間內最大值
            turn.importance_score = turn_importances.max().item()
            
            # 檢查是否有任何 token 超過閾值
            if (turn_importances > threshold).any():
                turn.reference_count += 1
                # 更新 recency tracking
                if self.enable_recency_tracking:
                    turn.last_access_step = self.forward_step
        
        # 遞增 forward step counter
        if self.enable_recency_tracking:
            self.forward_step += 1
    
    def update_recent_attend_turn_based(
        self, 
        all_self_attns: tuple,
        new_token_range: Tuple[int, int]
    ):
        """
        基於最近 k 步的 attend quality 更新策略（新方法）
        
        與 update_reference_counts_turn_based 的區別：
        - 舊方法：累積式引用計數（時間越長累積越多）
        - 新方法：只看最近 k 步的 attend quality（即時重要性）
        
        工作流程：
        1. 計算所有 tokens 的 attend scores（跨層加權平均）
        2. 為每個 turn 計算 quality（有長度校正）
        3. 記錄到歷史中（最近 k 步）
        
        Args:
            all_self_attns: Tuple of attention weights from all layers
            new_token_range: (start_idx, end_idx) 本次 forward 新增的 token 範圍
        """
        if not self.use_recent_attend or self.recent_attend_tracker is None:
            return
        
        if not all_self_attns or len(all_self_attns) == 0:
            return
        
        # 更新 tracker
        self.recent_attend_tracker.update(
            all_self_attns,
            self.conversation_tracker.turns,
            new_token_range
        )
    
    def _compute_weighted_eviction_score(
        self, 
        ref_count: int, 
        last_access_step: int,
        importance_score: float = 0.0
    ) -> Tuple[float, ...]:
        """
        計算加權刪除分數（數值越小越優先刪除）
        
        公式：
            score = (1 - w) * normalized_ref_count - w * normalized_recency
        
        其中：
            - normalized_ref_count: ref_count 標準化到 [0, 1]（越高越不該刪）
            - normalized_recency: (current_step - last_access) 標準化到 [0, 1]（越近越不該刪）
            - w: recency_weight
        
        範例（recency_weight = 0.5）：
            - ref_count=10, last_access=100, current=100 → score 低（最近訪問，不刪）
            - ref_count=10, last_access=0, current=100 → score 高（很久沒訪問，可刪）
            - ref_count=1, last_access=100 → score 高（少訪問，可刪）
        
        Args:
            ref_count: 引用次數
            last_access_step: 最後訪問的 step
            importance_score: 重要性分數（用於 tie-breaking）
        
        Returns:
            加權分數（越小越優先刪除）
        """
        if not self.enable_recency_tracking:
            # 如果沒啟用 recency tracking，只考慮 ref_count 和 importance
            return (ref_count, importance_score)
        
        # 計算 recency（距離當前 step 的時間差）
        time_since_access = self.forward_step - last_access_step
        
        # 簡單的標準化策略：
        # - ref_count 標準化：除以一個合理的上限（例如 100）
        # - recency 標準化：除以一個時間窗口（例如當前 forward_step）
        
        # 避免除以零
        max_ref_count = 100.0  # 假設 ref_count 不會超過 100
        max_time_window = max(self.forward_step, 1)  # 避免除以零
        
        normalized_ref = min(ref_count / max_ref_count, 1.0)  # 限制在 [0, 1]
        normalized_recency = min(time_since_access / max_time_window, 1.0)  # 限制在 [0, 1]
        
        # 加權組合（注意：我們要的是「應該被刪除的分數」，所以）
        # - ref_count 越高 → 越不該刪 → 貢獻負分
        # - recency 越大（越舊）→ 越該刪 → 貢獻正分
        w = self.recency_weight
        weighted_score = -(1 - w) * normalized_ref + w * normalized_recency
        
        # 返回 tuple: (weighted_score, importance_score) 用於 tie-breaking
        return (weighted_score, importance_score)
    
    def _compute_token_importances_batch(self, all_self_attns: tuple) -> Optional[torch.Tensor]:
        """
        批量計算所有 token 的重要性分數（高效向量化實現）
        
        策略：跨層平均池化 (Average Pooling across Layers)
        
        對於每個 token position k：
            importance[k] = average over all layers of (
                sum over all queries of (
                    average over all heads of attention[h, q, k]
                )
            )
        
        這個策略：
        - ✅ 完全向量化，單次 GPU 操作
        - ✅ 捕捉「sustained importance」- 跨層一致重要的 tokens
        - ✅ 數值穩定，不受單層異常值影響
        
        Args:
            all_self_attns: Tuple of attention weights from all layers
                每層: (batch_size, num_heads, query_len, key_len)
        
        Returns:
            token_importances: (key_len,) tensor，每個 token 的重要性分數
                如果無法計算則返回 None
        """
        if not all_self_attns or len(all_self_attns) == 0:
            return None
        
        # 只處理 batch_size=1 的情況（生成模式）
        first_attn = all_self_attns[0]
        if first_attn.size(0) != 1:
            return None
        
        num_layers = len(all_self_attns)
        key_len = first_attn.shape[-1]
        
        # 收集所有層的 per-token importance
        # 每層計算: mean over heads -> sum over queries -> per-token importance
        layer_importances = []
        
        for attn_weights in all_self_attns:
            # attn_weights: (1, num_heads, query_len, key_len)
            attn_weights = attn_weights.squeeze(0)  # (num_heads, query_len, key_len)
            
            # 先對 heads 取平均（減少噪音）
            attn_avg_heads = attn_weights.mean(dim=0)  # (query_len, key_len)
            
            # 對 queries 求和（每個 key 被所有新 queries 關注的總量）
            per_token_importance = attn_avg_heads.sum(dim=0)  # (key_len,)
            
            layer_importances.append(per_token_importance)
        
        # Stack 成 (num_layers, key_len)
        layer_importances = torch.stack(layer_importances, dim=0)
        
        # 跨層平均（捕捉持續重要性）
        final_importances = layer_importances.mean(dim=0)  # (key_len,)
        
        return final_importances

    def remove_range(self, start: int, end: int):
        """
        從 KV cache 中刪除指定範圍的 tokens
        
        Args:
            start: 起始位置（包含）
            end: 結束位置（不包含，即刪除 [start, end) 區間）
        
        例如：remove_range(10, 12) 會刪除位置 10 和 11 的 tokens
        """
        if not self.key_cache or len(self.key_cache) == 0:
            return
        
        # 更新每一層的 KV cache
        for layer_idx in range(len(self.key_cache)):
            # 刪除指定範圍：保留 [0, start) 和 [end, ...)
            self.key_cache[layer_idx] = torch.cat([
                self.key_cache[layer_idx][..., :start, :],
                self.key_cache[layer_idx][..., end:, :]
            ], dim=-2)
            
            self.value_cache[layer_idx] = torch.cat([
                self.value_cache[layer_idx][..., :start, :],
                self.value_cache[layer_idx][..., end:, :]
            ], dim=-2)
        
        # 更新 actual_cache_length
        removed_count = end - start
        self.actual_cache_length = self.key_cache[0].shape[-2] if self.key_cache else 0
        
        # 如果啟用了 position remapping，記錄被刪除的位置
        if self.enable_position_remapping:
            for pos in range(start, end):
                if pos not in self.dropped_positions:
                    self.dropped_positions.append(pos)
        
        # ⭐ 關鍵：更新所有受影響的對話對象的座標
        self._update_conversation_positions_after_deletion(start, end)
        
        # ⭐ RoPE Repositioning：如果啟用，重新計算 RoPE
        if self.enable_rope_repositioning:
            self._reposition_rope_after_deletion(start, end)
    
    def _update_conversation_positions_after_deletion(self, deleted_start: int, deleted_end: int):
        """
        在刪除 tokens 後更新所有對話對象的座標
        
        Args:
            deleted_start: 被刪除區間的起始位置（包含）
            deleted_end: 被刪除區間的結束位置（不包含）
        
        邏輯：
        - 如果對象完全在刪除區間之前：座標不變
        - 如果對象完全在刪除區間之後：座標需要向前移動 (deleted_end - deleted_start) 個位置
        - 如果對象跨越刪除區間：需要特殊處理（通常這種情況不應該發生）
        """
        deleted_count = deleted_end - deleted_start
        
        # 更新 ConversationTracker 的 current_position
        if self.conversation_tracker.current_position >= deleted_end:
            self.conversation_tracker.current_position -= deleted_count
        
        # 更新所有 conversation 對象的座標
        for event in self.conversation_tracker.conversations:
            # 處理 start_idx
            if event.start_idx >= deleted_end:
                # 完全在刪除區間之後，向前移動
                event.start_idx -= deleted_count
            elif event.start_idx >= deleted_start:
                # 在刪除區間內，這通常不應該發生
                # 但為了安全起見，我們將其設為刪除點
                event.start_idx = deleted_start
            
            # 處理 end_idx
            if event.end_idx >= deleted_end:
                # 完全在刪除區間之後，向前移動
                event.end_idx -= deleted_count
            elif event.end_idx >= deleted_start and event.end_idx != -1:
                # 在刪除區間內或跨越刪除區間
                # 將 end_idx 設為刪除點之前
                event.end_idx = deleted_start - 1
        
        # ⭐ 更新所有 turns 的 token 範圍
        for turn in self.conversation_tracker.turns:
            # 處理 start_idx
            if turn.start_idx >= deleted_end:
                # 完全在刪除區間之後，向前移動
                turn.start_idx -= deleted_count
            elif turn.start_idx >= deleted_start:
                # 在刪除區間內（理論上不應該發生）
                turn.start_idx = deleted_start
            
            # 處理 end_idx
            if turn.end_idx >= deleted_end:
                # 完全在刪除區間之後，向前移動
                turn.end_idx -= deleted_count
            elif turn.end_idx >= deleted_start and turn.end_idx != -1:
                # 在刪除區間內或跨越刪除區間
                turn.end_idx = deleted_start - 1
    
    def _update_turn_event_indices_after_event_deletion(self, deleted_event_idx: int):
        """
        在刪除 conversation event 後更新所有 turns 的 event indices
        
        Args:
            deleted_event_idx: 被刪除的 event 在 conversations 列表中的索引（刪除前）
        
        邏輯：
        - 所有 event_start_idx 或 event_end_idx 大於 deleted_event_idx 的，都要減 1
        """
        for turn in self.conversation_tracker.turns:
            # 更新 event_start_idx
            if turn.event_start_idx > deleted_event_idx:
                turn.event_start_idx -= 1
            
            # 更新 event_end_idx
            if turn.event_end_idx > deleted_event_idx:
                turn.event_end_idx -= 1
            
            # 特殊情況：如果被刪除的 event 正好在 turn 的範圍內
            # 需要調整 turn 的範圍
            if turn.event_start_idx == deleted_event_idx and turn.event_end_idx == deleted_event_idx:
                # Turn 只包含這一個 event，且被刪除了
                # 這個 turn 應該被標記為無效或移除
                # 但為了安全，我們暫時不做處理，留待上層邏輯處理
                pass
    
    def _reposition_rope_after_deletion(self, deleted_start: int, deleted_end: int):
        """
        在刪除 tokens 後重新旋轉受影響的 keys
        
        使用 SinkCache 的方法：通過三角恒等式計算差分旋轉，
        不需要存儲原始的 unrotated keys。
        
        數學原理：
        - RoPE 是可疊加的旋轉操作
        - 從 position A 旋轉到 position B 可以通過應用差分旋轉 (B - A)
        - 使用三角恒等式：cos(A-B) = cos(A)cos(B) + sin(A)sin(B)
        
        Args:
            deleted_start: 被刪除區間的起始位置（包含）
            deleted_end: 被刪除區間的結束位置（不包含）
        """
        if not self.key_cache or len(self.key_cache) == 0:
            return
        
        deleted_count = deleted_end - deleted_start
        batch_size, num_heads, current_seq_len, head_dim = self.key_cache[0].shape
        
        if deleted_start >= current_seq_len:
            return
        
        # 計算受影響的範圍
        affected_len = current_seq_len - deleted_start
        
        # 舊位置：deleted_end, deleted_end+1, ..., deleted_end+affected_len-1
        old_positions = torch.arange(
            deleted_end, 
            deleted_end + affected_len, 
            device=self.key_cache[0].device
        )
        
        # 新位置：deleted_start, deleted_start+1, ..., deleted_start+affected_len-1
        new_positions = torch.arange(
            deleted_start,
            deleted_start + affected_len,
            device=self.key_cache[0].device
        )
        
        # 計算差分旋轉的 cos/sin
        rerotation_cos, rerotation_sin = self._get_rerotation_cos_sin(
            old_positions,
            new_positions,
            self.key_cache[0].device,
            self.key_cache[0].dtype
        )
        
        # 對每一層的 keys 應用重新旋轉
        for layer_idx in range(len(self.key_cache)):
            # 提取受影響的 keys
            keys_to_rerotate = self.key_cache[layer_idx][:, :, deleted_start:, :]
            
            # 應用差分旋轉
            rerotated_keys = self._apply_key_rotary_pos_emb(
                keys_to_rerotate,
                rerotation_cos,
                rerotation_sin
            )
            
            # 更新 cache
            self.key_cache[layer_idx][:, :, deleted_start:, :] = rerotated_keys
        
        # print(f"✅ RoPE rerotation completed: positions [{deleted_start}:{current_seq_len}) "
        #       f"shifted by -{deleted_count}")
    
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        將向量的前半和後半交換並取負（RoPE 的基本操作）
        
        這是 RoPE 的核心變換：將 [a, b, c, d] 變成 [-c, -d, a, b]
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def _apply_key_rotary_pos_emb(
        self, 
        key_states: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor
    ) -> torch.Tensor:
        """
        應用 RoPE 到 key states
        
        Args:
            key_states: [batch_size, num_heads, seq_len, head_dim]
            cos: [1, 1, seq_len, head_dim]
            sin: [1, 1, seq_len, head_dim]
        
        Returns:
            rotated keys
        """
        rotated_key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)
        return rotated_key_states
    
    def _compute_cos_sin_cache(self, max_position: int, device: torch.device):
        """
        計算並快取 cos/sin 值以避免重複計算
        
        Args:
            max_position: 需要計算到的最大位置
            device: 計算設備
        """
        if self._cos_cache is not None and max_position <= self._max_position_computed:
            return
        
        # 計算 inverse frequency
        inv_freq = 1.0 / (
            self.rope_theta ** (
                torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim
            )
        )
        
        # 處理 RoPE scaling（如果有的話）
        if self.rope_scaling is not None:
            scaling_factor = self.rope_scaling.get("factor", 1.0)
            inv_freq = inv_freq / scaling_factor
        
        # 計算所有位置的 embeddings
        position_ids = torch.arange(max_position, device=device, dtype=torch.float32).unsqueeze(1)
        freqs = position_ids * inv_freq.unsqueeze(0)  # [max_position, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_position, head_dim]
        
        # 儲存 cos/sin，使用 float32 以保持精度
        self._cos_cache = emb.cos()
        self._sin_cache = emb.sin()
        self._max_position_computed = max_position
    
    def _get_rerotation_cos_sin(
        self,
        old_positions: torch.Tensor,
        new_positions: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        計算從 old_positions 到 new_positions 的差分旋轉
        
        使用三角恒等式：
        - cos(A - B) = cos(A)cos(B) + sin(A)sin(B)
        - sin(A - B) = sin(A)cos(B) - cos(A)sin(B)
        
        其中 A = old_position, B = new_position
        
        Args:
            old_positions: 原始位置 [affected_len]
            new_positions: 新位置 [affected_len]
            device: 設備
            dtype: 輸出資料類型
        
        Returns:
            (rerotation_cos, rerotation_sin): 差分旋轉的 cos/sin
                shape: [1, 1, affected_len, head_dim]
        """
        # 確保 cos/sin cache 已計算
        max_pos = max(old_positions.max().item(), new_positions.max().item()) + 1
        self._compute_cos_sin_cache(max_pos, device)
        
        # 提取對應位置的 cos/sin
        # shape: [affected_len, head_dim]
        original_cos = self._cos_cache[old_positions]
        shifted_cos = self._cos_cache[new_positions]
        original_sin = self._sin_cache[old_positions]
        shifted_sin = self._sin_cache[new_positions]
        
        # 應用三角恒等式計算差分旋轉
        # 注意：這裡計算的是從 old 到 new 的旋轉，即 new - old
        # 因為我們要將 RoPE(key, old) 轉換成 RoPE(key, new)
        # 等價於在已旋轉的 key 上應用 (new - old) 的旋轉
        rerotation_cos = shifted_cos * original_cos + shifted_sin * original_sin
        rerotation_sin = shifted_sin * original_cos - shifted_cos * original_sin
        
        # 調整形狀以匹配 key_states: [1, 1, affected_len, head_dim]
        return (
            rerotation_cos.to(dtype).unsqueeze(0).unsqueeze(0),
            rerotation_sin.to(dtype).unsqueeze(0).unsqueeze(0)
        )
        

    def update_memory(self):
        """
        當 cache 超過最大長度時進行修剪
        
        策略：
        - 當 cache 超過 n_max 時觸發
        - 循環刪除直到 cache 降到 n_min 以下
        - 減少記憶管理的頻率
        
        支持四種策略：
        - "fifo": FIFO 策略
        - "reference_based": 基於引用計數的策略
        - "turn_based": 基於對話輪次的策略
        - "crash": 拋出 CacheOverflowError（用於 baseline 比較）
        """
        self.actual_cache_length = current_length = self.key_cache[0].shape[-2]
        
        if current_length <= self.n_max:
            return
        
        # Crash 模式：直接拋出異常
        if self.eviction_strategy == "crash":
            raise CacheOverflowError(
                cache_size=current_length,
                n_max=self.n_max,
                message=f"Cache overflow in crash mode"
            )
        
        # 循環刪除直到達到 n_min
        eviction_count = 0
        while current_length > self.n_min:
            # 記錄刪除前的長度
            length_before = current_length
            
            # 執行一次刪除
            if self.eviction_strategy == "fifo":
                self._update_memory_fifo()
            elif self.eviction_strategy == "reference_based":
                self._update_memory_reference_based()
            elif self.eviction_strategy == "turn_based":
                self._update_memory_turn_based()
            else:
                raise ValueError(f"Unknown eviction strategy: {self.eviction_strategy}")
            
            # 更新當前長度
            current_length = self.key_cache[0].shape[-2]
            eviction_count += 1
            
            # 安全檢查：如果刪除沒有效果，避免無限循環
            if current_length >= length_before:
                # print(f"⚠️  Eviction had no effect, stopping. Current length: {current_length}")
                break
            
            # 安全檢查：避免刪除過多次（例如超過 10 次）
            if eviction_count >= 10:
                # print(f"⚠️  Too many evictions ({eviction_count}), stopping. Current length: {current_length}")
                break
        
        # if eviction_count > 0:
        #     print(f"✅ Evicted {eviction_count} times, reduced from {self.n_max}+ to {current_length}")
    
    def _update_memory_fifo(self):
        """原有的 FIFO 策略"""
        is_first_stream_frame_found = True
        for idx, event in enumerate(self.conversation_tracker.conversations):
            if isinstance(event, StreamFrame):
                # 跳過第一個 StreamFrame 物件
                if is_first_stream_frame_found:
                    is_first_stream_frame_found = False
                    continue
                
                # 檢查是否為最後一個 event
                is_last_event = (idx == len(self.conversation_tracker.conversations) - 1)
                
                if len(event.frame_indices) > 1:
                    # 有多個 frames，可以刪除一個
                    rm_st = event.pop_frame()
                    if rm_st is not None:
                        rm_ed = rm_st + event.num_frame_tokens + 1  # +1 for ','
                        self.remove_range(rm_st, rm_ed)
                elif not is_last_event:
                    # 只剩一個 frame，但不是最後一個 event，可以整個刪除
                    rm_st = event.start_idx
                    rm_ed = event.end_idx + 1  # ✅ +1 因為 remove_range 是 [start, end) 區間
                    
                    # ⭐ 在從列表移除前，先記錄 idx
                    event_idx_to_delete = idx
                    
                    self.conversation_tracker.conversations.remove(event)
                    self.remove_range(rm_st, rm_ed)
                    
                    # ⭐ 更新所有 turns 的 event indices（新增）
                    self._update_turn_event_indices_after_event_deletion(event_idx_to_delete)
                    
                    # 更新 current_turn_start_event_idx：刪除的 event 如果在當前 turn 起始位置之前，需要調整
                    if idx < self.conversation_tracker.current_turn_start_event_idx:
                        self.conversation_tracker.current_turn_start_event_idx -= 1
                # else: 是最後一個 event 且只剩一個 frame，不做刪除
                break
    
    def _update_memory_reference_based(self):
        """基於引用計數的刪除策略"""
        # 收集所有可刪除的對象及其優先級
        # 格式: (priority, event_idx, event, frame_idx_or_none)
        candidates = []
        
        for idx, event in enumerate(self.conversation_tracker.conversations):
            is_last_event = (idx == len(self.conversation_tracker.conversations) - 1)
            
            if isinstance(event, StreamFrame):
                # 對於 StreamFrame，檢查每個 frame
                for frame_idx in event.frame_indices:
                    # 最新的 frame 有豁免權
                    is_latest_frame = (frame_idx == event.frame_indices[-1]) and is_last_event
                    
                    if is_latest_frame:
                        continue
                    
                    # 計算加權刪除分數
                    priority = self._compute_weighted_eviction_score(
                        ref_count=event.frame_stats[frame_idx]["reference_count"],
                        last_access_step=event.frame_stats[frame_idx]["last_access_step"],
                        importance_score=event.frame_stats[frame_idx]["importance_score"]
                    )
                    candidates.append((priority, idx, event, frame_idx))
            
            elif not is_last_event:
                # SystemPrompt, UserQuery, AssistantResponse
                # 最後一個事件不刪除
                if event.end_idx != -1:  # 已結束的事件
                    # 計算加權刪除分數
                    priority = self._compute_weighted_eviction_score(
                        ref_count=event.reference_count,
                        last_access_step=event.last_access_step,
                        importance_score=event.importance_score
                    )
                    candidates.append((priority, idx, event, None))
        
        if not candidates:
            # 沒有可刪除的候選，降級使用 FIFO
            self._update_memory_fifo()
            return
        
        # 排序：引用次數最低的優先
        candidates.sort(key=lambda x: x[0])
        
        # 刪除優先級最高（最不重要）的對象
        priority, event_idx, event, frame_idx = candidates[0]
        
        if frame_idx is not None:
            # 刪除 StreamFrame 中的特定 frame
            token_start = event.remove_specific_frame(frame_idx)
            if token_start is not None:
                token_end = token_start + event.num_frame_tokens + 1  # +1 for ','
                self.remove_range(token_start, token_end)
            
            # 如果 StreamFrame 被清空且不是最後一個 event，刪除整個結構
            if len(event.frame_indices) == 0 and event_idx < len(self.conversation_tracker.conversations) - 1:
                rm_st = event.start_idx
                rm_ed = event.end_idx + 1  # ✅ +1 因為 remove_range 是 [start, end) 區間
                self.conversation_tracker.conversations.remove(event)
                self.remove_range(rm_st, rm_ed)
                
                # ⭐ 更新所有 turns 的 event indices（新增）
                self._update_turn_event_indices_after_event_deletion(event_idx)
                
                # 更新 current_turn_start_event_idx：刪除的 event 如果在當前 turn 起始位置之前，需要調整
                if event_idx < self.conversation_tracker.current_turn_start_event_idx:
                    self.conversation_tracker.current_turn_start_event_idx -= 1
        else:
            # 刪除整個句子
            rm_st = event.start_idx
            rm_ed = event.end_idx + 1  # ✅ +1 因為 remove_range 是 [start, end) 區間
            self.conversation_tracker.conversations.remove(event)
            self.remove_range(rm_st, rm_ed)
            
            # ⭐ 更新所有 turns 的 event indices（新增）
            self._update_turn_event_indices_after_event_deletion(event_idx)
            
            # 更新 current_turn_start_event_idx：刪除的 event 如果在當前 turn 起始位置之前，需要調整
            if event_idx < self.conversation_tracker.current_turn_start_event_idx:
                self.conversation_tracker.current_turn_start_event_idx -= 1
    
    def _update_memory_turn_based(self):
        """
        Turn-based 刪除策略
        
        核心邏輯：
        1. 以整個 Turn 為單位進行刪除
        2. Turn 的範圍：從上一個 AssistantResponse 結束到當前 AssistantResponse 結束
        3. 刪除優先級：根據 turn_eviction_policy 和 use_recent_attend 決定
           - use_recent_attend=True: 使用最近 k 步的 attend quality
           - use_recent_attend=False + "reference_based": 使用累積 reference_count
           - use_recent_attend=False + "fifo": 刪除最早的 turn
        4. 豁免權：
           - 第一個 turn（turn_idx == 0）不刪除
           - 最新的 turn（current_turn 未 finalize 的部分）不刪除
        
        刪除範圍：
        - 包含 Turn 內的所有 events: StreamFrame, UserQuery, AssistantResponse
        - 從 turn.start_idx 到 turn.end_idx
        """
        # 根據策略選擇要刪除的 turn
        if self.use_recent_attend and self.recent_attend_tracker is not None:
            # 新策略：使用最近 k 步的 attend quality
            turn_to_delete = self.recent_attend_tracker.get_worst_turn(
                self.conversation_tracker.turns
            )
        else:
            # 舊策略：使用累積 reference_count 或 FIFO
            turn_to_delete = self._get_worst_turn_by_reference_count()
        
        if turn_to_delete is None:
            # 沒有可刪除的 turn，降級使用 FIFO
            # print("⚠️  No turns available for eviction, falling back to FIFO")
            self._update_memory_fifo()
            return
        
        # print(f"🗑️  Evicting {turn_to_delete} (strategy: {'recent_attend' if self.use_recent_attend else self.turn_eviction_policy})")
        
        # 刪除 Turn 對應的所有 events
        events_to_remove = self.conversation_tracker.conversations[
            turn_to_delete.event_start_idx:turn_to_delete.event_end_idx + 1
        ]
        
        num_events_removed = len(events_to_remove)
        
        for event in events_to_remove:
            self.conversation_tracker.conversations.remove(event)
        
        # 更新 current_turn_start_event_idx：刪除的 events 如果在當前 turn 起始位置之前，需要調整
        if turn_to_delete.event_end_idx < self.conversation_tracker.current_turn_start_event_idx:
            # 整個被刪除的 turn 都在當前 turn 之前
            self.conversation_tracker.current_turn_start_event_idx -= num_events_removed
        elif turn_to_delete.event_start_idx < self.conversation_tracker.current_turn_start_event_idx:
            # 被刪除的 turn 與當前 turn 起始位置有重疊（理論上不應該發生，但為了安全處理）
            self.conversation_tracker.current_turn_start_event_idx = turn_to_delete.event_start_idx
        
        # 刪除 Turn 的 tokens
        rm_st = turn_to_delete.start_idx
        rm_ed = turn_to_delete.end_idx + 1  # +1 因為 remove_range 是 [start, end) 區間
        self.remove_range(rm_st, rm_ed)
        
        # 從 turns 列表中移除
        self.conversation_tracker.turns.remove(turn_to_delete)
        
        # 如果使用 recent_attend，也要從 tracker 中移除歷史記錄
        if self.use_recent_attend and self.recent_attend_tracker is not None:
            self.recent_attend_tracker.remove_turn_history(turn_to_delete.turn_idx)
        
        # 更新所有後續 turns 的 event indices（因為前面的 events 被刪除了）
        for turn in self.conversation_tracker.turns:
            if turn.event_start_idx > turn_to_delete.event_start_idx:
                turn.event_start_idx -= num_events_removed
                turn.event_end_idx -= num_events_removed
    
    def _get_worst_turn_by_reference_count(self) -> Optional[Any]:  # Optional[Turn]
        """
        使用累積 reference_count 或 FIFO 策略選擇要刪除的 turn
        
        豁免規則基於相對位置，不是絕對 turn_idx
        
        Returns:
            要刪除的 turn 或 None
        """
        # 先篩選出所有 finalized turns（用於計算相對位置）
        finalized_turns = [t for t in self.conversation_tracker.turns if t.is_finalized]
        
        if not finalized_turns:
            return None
        
        # 計算豁免的位置範圍
        num_finalized = len(finalized_turns)
        
        # 使用 recent_attend_tracker 的參數（如果有），否則使用默認值
        if self.recent_attend_tracker is not None:
            protect_oldest = self.recent_attend_tracker.protect_oldest_turns
            protect_newest = self.recent_attend_tracker.protect_newest_turns
        else:
            protect_oldest = 1  # 默認保護最早 1 個
            protect_newest = 1  # 默認保護最新 1 個
        
        # 收集候選 turns（帶有位置信息）
        candidates = []
        
        for position, turn in enumerate(finalized_turns):
            # 豁免：保護最早的 N 個 turns（基於相對位置）
            if position < protect_oldest:
                continue
            
            # 豁免：保護最新的 N 個 turns（基於相對位置）
            if position >= num_finalized - protect_newest:
                continue
            
            candidates.append((turn, position))
        
        if not candidates:
            return None
        
        # 根據 turn_eviction_policy 選擇要刪除的 turn
        if self.turn_eviction_policy == "fifo":
            # FIFO：刪除最早的候選 turn（position 最小，即 turn_idx 最小）
            turn_to_delete = min(candidates, key=lambda x: x[0].turn_idx)[0]
        else:  # "reference_based"
            # Reference-based with weighted score：使用加權分數決定刪除優先級
            turn_to_delete = min(
                candidates,
                key=lambda x: self._compute_weighted_eviction_score(
                    ref_count=x[0].reference_count,
                    last_access_step=x[0].last_access_step,
                    importance_score=x[0].importance_score
                ) + (x[0].turn_idx,)  # 加上 turn_idx 用於 tie-breaking
            )[0]
        
        return turn_to_delete
                    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        獲取序列長度
        
        行為取決於 enable_rope_repositioning：
        - False (預設): 返回 seen_tokens（實際看過的 token 數量，包含已刪除的）
        - True: 返回 actual_cache_length（當前 KV cache 中實際存儲的 token 數量）
        
        當啟用 RoPE repositioning 時，get_seq_length 應該返回實際的 cache 大小，
        因為此時 positional embedding 與 cache 位置是同步的。
        """
        if self.enable_rope_repositioning:
            # RoPE repositioning 模式：返回實際的 cache 大小
            if self.key_cache and len(self.key_cache) > 0:
                return self.key_cache[0].shape[-2]
            return 0
        else:
            # 標準模式：返回看過的 token 總數
            return self.seen_tokens if self.seen_tokens is not None else 0

    def to_legacy_cache(self) -> "InfCache":  # type: ignore[override]
        """
        覆寫父類方法，避免被轉換成 tuple。
        返回 self 以保持完整功能（包括對話追蹤狀態）。
        """
        return self

    def get_conversation_summary(self) -> str:
        """獲取對話結構摘要（用於調試和分析）"""
        return self.conversation_tracker.get_conversation_summary()
    
    def get_reference_statistics(self) -> str:
        """獲取引用計數統計信息"""
        if self.eviction_strategy != "reference_based":
            return f"Eviction strategy: {self.eviction_strategy} (reference counting disabled)"
        
        lines = [
            f"Eviction Strategy: {self.eviction_strategy}",
            f"Top-K References: {self.top_k_references}",
            "",
            "Object Reference Counts:",
            ""
        ]
        
        for idx, event in enumerate(self.conversation_tracker.conversations):
            if isinstance(event, StreamFrame):
                lines.append(f"  {idx}. StreamFrame (pos {event.start_idx}-{event.end_idx}):")
                for frame_idx in event.frame_indices:
                    stats = event.frame_stats.get(frame_idx, {})
                    ref_count = stats.get("reference_count", 0)
                    importance = stats.get("importance_score", 0.0)
                    lines.append(f"      Frame {frame_idx}: ref={ref_count}, score={importance:.2f}")
            else:
                event_type = type(event).__name__
                lines.append(
                    f"  {idx}. {event_type} (pos {event.start_idx}-{event.end_idx}): "
                    f"ref={event.reference_count}, score={event.importance_score:.2f}"
                )
        
        return "\n".join(lines)
    
    def get_position_mapping_summary(self) -> str:
        """獲取 position mapping 的摘要資訊"""
        lines = [
            f"Position Remapping: {'Enabled' if self.enable_position_remapping else 'Disabled'}",
            f"Actual Cache Length: {self.actual_cache_length}",
            f"Logical Position: {self.conversation_tracker.current_position}",
            f"Dropped Positions: {len(self.dropped_positions)} tokens",
            ""
        ]
        
        if self.dropped_positions and len(self.dropped_positions) <= 20:
            lines.append(f"Dropped: {self.dropped_positions}")
        elif self.dropped_positions:
            lines.append(f"Dropped (first 10): {self.dropped_positions[:10]}...")
        
        if self.position_map and len(self.position_map) <= 10:
            lines.append(f"Position Map: {self.position_map}")
        elif self.position_map:
            sample_items = list(self.position_map.items())[:5]
            lines.append(f"Position Map (sample): {dict(sample_items)}...")
        
        return "\n".join(lines)
    
    def reset_position_mapping(self):
        """重置 position mapping（用於新的生成會話）"""
        self.position_map.clear()
        self.dropped_positions.clear()
        self.actual_cache_length = 0

    def __repr__(self):
        dropped_info = f", dropped={len(self.dropped_positions)}" if self.enable_position_remapping else ""
        strategy_info = f", evict={self.eviction_strategy}"
        # 如果是 turn_based 策略，顯示 turn_eviction_policy
        if self.eviction_strategy == "turn_based":
            strategy_info += f"({self.turn_eviction_policy})"
        rope_info = f", rope_repos={self.enable_rope_repositioning}" if self.enable_rope_repositioning else ""
        cache_limit_info = f", n_max={self.n_max}, n_min={self.n_min}"
        return (
            f"InfCache(layers={len(self.key_cache)}, "
            f"seq_len={self.get_seq_length()}, "
            f"actual_len={self.actual_cache_length}, "
            f"conversations={len(self.conversation_tracker.conversations)}, "
            f"state={self.conversation_tracker.state.name}"
            f"{dropped_info}{strategy_info}{rope_info}{cache_limit_info})"
        )
