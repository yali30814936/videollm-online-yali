import torch
from transformers.cache_utils import DynamicCache
from enum import Enum
from typing import Union, Tuple

class InfCacheState(Enum):
    INIT                  = 0
    START                 = 1 # <BOS>{system prompt}\n\n[<v>
    ASSISTANT_HEAD        = 2 # ]\nAssistant:
    USER_AND_ASSISTANT   = 3 # ]\n{user query}.\nAssistant:
    RESPONSE_A_TOKEN      = 4 # {single assistant response token}
    STREAM_AFTER_RESPONSE = 5 # <EOS>\n[<v>
    NEXT_FRAME            = 6 # ,<v>

class Sentence:
    def __init__(
            self,
            start_idx: int,
            end_idx: int = -1,
        ):
        self.start_idx = start_idx
        self.end_idx = end_idx

class SystemPrompt(Sentence):
    pass
class UserQuery(Sentence):
    pass
class AssistantResponse(Sentence):
    def end_response(self, end_idx: int):
        self.end_idx = end_idx

class StreamFrame(Sentence):
    def __init__(self, num_frame_tokens: int, start_idx: int):
        super().__init__(start_idx)
        self.num_frame_tokens = num_frame_tokens
        self.frame_indices = []
        self.append_frame(start_idx+1)

    def append_frame(self, frame_idx: int):
        self.frame_indices.append(frame_idx)
    
    def end_stream(self, end_idx: int):
        self.end_idx = end_idx

class InfCache(DynamicCache):
    """
    Inference cache that supports skipping certain tokens (e.g., visual tokens).
    Extends DynamicCache to handle non-contiguous cache positions.
    """

    def __init__(self,
                 max_n: int = 2048,
                 num_frame_tokens: int = 1,
                 ):
        super().__init__()
        self.max_n = max_n
        self.num_frame_tokens = num_frame_tokens
        self.conversations: list[Sentence|StreamFrame|AssistantResponse] = []
        self.past_seen_tokens = 0
        self.last_state: InfCacheState = InfCacheState.INIT

    def to_legacy_cache(self) -> "InfCache":  # type: ignore[override]
        """
        覆寫父類方法，避免被轉換成 tuple。
        返回 self 以保持 InfCache 的完整功能。
        """
        return self

    def update_conversation(self, input_ids: torch.Tensor):
        input_ids = input_ids.squeeze(0) # (seq_len,)
        if input_ids[0] == 128000: # <BOS>{system prompt}\n\n[<v>
            if self.last_state != InfCacheState.INIT:
                raise ValueError("System prompt should be the first input.")
            # append system prompt
            system_ed = input_ids.size(0) - (1 + self.num_frame_tokens) - 1 # exclude "[<v>"
            self.conversations.append(SystemPrompt(0, system_ed))
            self.past_seen_tokens = system_ed + 1
            # initiate stream
            self.conversations.append(StreamFrame(self.num_frame_tokens, self.past_seen_tokens))
            self.past_seen_tokens = input_ids.size(0)
            
            self.last_state = InfCacheState.START
        elif input_ids.size(0) == 3 and input_ids == torch.tensor([933, 72803, 25]): # ]\nAssistant:
            if  self.last_state != InfCacheState.START or \
                self.last_state != InfCacheState.STREAM_AFTER_RESPONSE or \
                self.last_state != InfCacheState.NEXT_FRAME:
                raise ValueError("assistant_head should follow start, stream_after_response, or next_frame.")
            # end stream
            assert isinstance(self.conversations[-1], StreamFrame)
            self.conversations[-1].end_stream(self.past_seen_tokens) # include ']\n'
            self.past_seen_tokens += 1 # ']\n'
            # initiate assistant response
            self.conversations.append(AssistantResponse(self.past_seen_tokens))
            self.past_seen_tokens += 2 # "Assistant:"
            
            self.last_state = InfCacheState.ASSISTANT_HEAD
        elif input_ids.size(0) > 3 and input_ids[-2:] == torch.tensor([72803, 25]): # ]\n{user query}.\nAssistant:
            if self.last_state != InfCacheState.START:
                raise ValueError("user_and_assistant should follow start.")
            # end stream
            assert isinstance(self.conversations[-1], StreamFrame)
            self.conversations[-1].end_stream(self.past_seen_tokens) # include ']\n'
            self.past_seen_tokens += 1 # ']\n'
            # append user query
            query_ed = self.past_seen_tokens + input_ids.size(0) -1 -2 -1 # exclude ']\n' and "Assistant:"
            self.conversations.append(UserQuery(
                self.past_seen_tokens,
                query_ed
            ))
            self.past_seen_tokens = query_ed + 1
            # initiate assistant response
            self.conversations.append(AssistantResponse(self.past_seen_tokens))
            self.past_seen_tokens += 2 # "Assistant:"
            
            self.last_state = InfCacheState.USER_AND_ASSISTANT
        elif input_ids.size(0) == 1: # {single assistant response token}
            if  self.last_state != InfCacheState.ASSISTANT_HEAD or \
                self.last_state != InfCacheState.USER_AND_ASSISTANT or \
                self.last_state != InfCacheState.RESPONSE_A_TOKEN:
                raise ValueError("response_a_token should follow assistant_head, user_and_assistant, or response_a_token.")
            # append response token (noop)
            self.past_seen_tokens += 1
            self.last_state = InfCacheState.RESPONSE_A_TOKEN
        elif input_ids[0] == 128009: # <EOS>\n[<v>
            if self.last_state != InfCacheState.RESPONSE_A_TOKEN:
                raise ValueError("stream_after_response should follow response_a_token.")
            # end assistant response
            assert isinstance(self.conversations[-1], AssistantResponse)
            self.conversations[-1].end_response(self.past_seen_tokens +2 -1)
            self.past_seen_tokens += 2 # "<EOS>\n"
            # initiate stream
            self.conversations.append(StreamFrame(self.num_frame_tokens, self.past_seen_tokens))
            self.past_seen_tokens += 1 + self.num_frame_tokens # [<v>
            
            self.last_state = InfCacheState.STREAM_AFTER_RESPONSE
        elif input_ids[-1] == 128256: # ,<v>
            if  self.last_state != InfCacheState.STREAM_AFTER_RESPONSE or \
                self.last_state != InfCacheState.NEXT_FRAME:
                raise ValueError("next frame should follow after stream_and_resopnse or next_frame")
            # append stream frame
            assert isinstance(self.conversations[-1], StreamFrame)
            self.conversations[-1].append_frame(self.past_seen_tokens +1)
            self.past_seen_tokens += 1 + self.num_frame_tokens
        else:
            raise ValueError("Unrecognized input_ids pattern.")
