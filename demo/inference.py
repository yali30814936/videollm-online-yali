import torch, torchvision, transformers, collections
torchvision.set_video_backend('video_reader')
from dataclasses import asdict
from torchvision.io import read_video

from models import build_model_and_tokenizer, parse_args, fast_greedy_generate

logger = transformers.logging.get_logger('liveinfer')

# python -m demo.cli --resume_from_checkpoint ... 

class LiveInfer:
    def __init__(self, ) -> None:
        args = parse_args()
        self.model, self.tokenizer = build_model_and_tokenizer(is_training=False, set_vision_inside=True, **asdict(args))
        self.model.to('cuda')

        # visual
        self.hidden_size = self.model.config.hidden_size
        self.frame_fps = args.frame_fps
        self.frame_interval = 1 / self.frame_fps
        self.frame_resolution = self.model.config.frame_resolution
        self.frame_num_tokens = self.model.config.frame_num_tokens
        self.frame_v_placeholder = self.model.config.v_placeholder * self.frame_num_tokens
        self.frame_token_interval_id = self.model.config.frame_token_interval_id
        self.frame_placeholder_ids = torch.tensor(self.model.config.v_placeholder_id).repeat(self.model.config.frame_num_tokens).reshape(1, -1)

        # generation
        self.system_prompt = args.system_prompt
        self.inplace_output_ids = torch.zeros(1, 100, device='cuda', dtype=torch.long)
        self.frame_token_interval_threshold = 0.725
        self.eos_token_id = self.model.config.eos_token_id
        self._start_ids = self.tokenizer.apply_chat_template([{'role': 'system', 'content': self.system_prompt}], add_stream_prompt=True, return_tensors='pt').to('cuda')
        self._added_stream_prompt_ids = self.tokenizer.apply_chat_template([{}], add_stream_prompt=True, return_tensors='pt').to('cuda')
        self._added_stream_generation_ids = self.tokenizer.apply_chat_template([{}], add_stream_generation_prompt=True, return_tensors='pt').to('cuda')

        # memory trim helpers
        self._open_prompt_len = int(self._added_stream_prompt_ids.shape[1])  # "\n["
        self._close_tokens_len = int(self.tokenizer("]\n", add_special_tokens=False, return_tensors='pt').input_ids.shape[1])
        self._assistant_prompt_simple_len = int(self.tokenizer("\nAssistant:", add_special_tokens=False, return_tensors='pt').input_ids.shape[1])
        # For pattern "]\nAssistant:", compute assistant-only part
        self._stream_close_plus_assistant_len = int(self._added_stream_generation_ids.shape[1])
        self._stream_assistant_prompt_len = self._stream_close_plus_assistant_len - self._close_tokens_len

        # memory management settings
        default_ctx = int(getattr(self.model.config, 'max_position_embeddings', 4096))
        self.memory_trim_trigger = int(max(1024, default_ctx - 512))
        self.enable_memory_trim = True
        # modes: 'hybrid' (user/assistant/frame) or 'frames_only' (only frame sessions)
        # self.memory_trim_mode = 'hybrid'
        self.memory_trim_mode = 'frames_only'

        # runtime memory accounting
        self._segments = collections.deque()
        self._curr_frame_session = None
        self._ctx_len = 0

        # debug conversation tracking
        self._debug_chat = [{'role': 'system', 'content': self.system_prompt}]
        self._debug_pending_stream_frames = 0

        # app
        self.reset()

    def _safe_decode_ids(self, ids_tensor):
        """Safely decode token ids to text, tolerating tokenizer variations.
        Tries decode, then batch_decode, with/without special-tokens skip.
        """
        tok = getattr(self, 'tokenizer', None)
        if tok is None:
            return ''
        ids = ids_tensor
        try:
            return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except Exception:
            pass
        try:
            return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except Exception:
            pass
        try:
            return tok.batch_decode([ids], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        except Exception:
            pass
        try:
            return tok.batch_decode([ids], skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        except Exception:
            return ''

    def _call_for_response(self, video_time, query):
        if query is not None:
            self.last_ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': query}], add_stream_query_prompt=True, add_generation_prompt=True, return_tensors='pt').to('cuda')
        else:
            assert self.last_ids == 933, f'{self.last_ids} != 933' # HACK, 933 = ]\n
            self.last_ids = self._added_stream_generation_ids
        inputs_embeds = self.model.get_input_embeddings()(self.last_ids)
        v_mask = torch.zeros((1, self.last_ids.shape[1]), device='cuda', dtype=torch.bool)
        frame_interval_mask = self.last_ids == self.frame_token_interval_id
        # memory accounting for prefixes: close stream + optional user + assistant prompt
        self._account_response_prefix(query_present=(query is not None))
        # DEBUG HOOK: finalize pending stream block and record user query
        if hasattr(self, '_debug_pending_stream_frames') and self._debug_pending_stream_frames > 0:
            self._debug_chat.append({'role': 'stream', 'num_frames': int(self._debug_pending_stream_frames)})
            self._debug_pending_stream_frames = 0
        if query is not None:
            self._debug_chat.append({'role': 'user', 'content': query})
        output_ids, self.past_key_values = fast_greedy_generate(
            model=self.model,
            inputs_embeds=inputs_embeds,
            past_key_values=self.past_key_values,
            eos_token_id=self.eos_token_id,
            inplace_output_ids=self.inplace_output_ids,
            v_mask=v_mask,
            frame_interval_mask=frame_interval_mask
        )
        self.last_ids = output_ids[:, -1:]
        # account assistant generated tokens and maybe trim
        self._account_assistant_generation(gen_len=int(output_ids.shape[1]))
        self._maybe_trim_memory()
        # DEBUG HOOK: record assistant response text with robust fallback
        resp_text = self._safe_decode_ids(output_ids[0])
        self._debug_chat.append({'role': 'assistant', 'content': resp_text})
        if query:
            query = f'(Video Time = {video_time}s) User: {query}'
        response = f'(Video Time = {video_time}s) Assistant:{self._safe_decode_ids(output_ids[0])}'
        return query, response
    
    def _call_for_streaming(self, ):
        while self.frame_embeds_queue:
            # 1. if query is before next frame, response
            if self.query_queue and self.frame_embeds_queue[0][0] > self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                return video_time, query
            video_time, frame_embeds = self.frame_embeds_queue.popleft()
            if not self.past_key_values:
                self.last_ids = self._start_ids
            elif self.last_ids == self.eos_token_id:
                self.last_ids = torch.cat([self.last_ids, self._added_stream_prompt_ids], dim=1)
            inputs_embeds = torch.cat([
                self.model.get_input_embeddings()(self.last_ids).view(1, -1, self.hidden_size),
                frame_embeds.view(1, -1, self.hidden_size),
            ], dim=1)
            v_mask = torch.cat([
                torch.zeros((1, self.last_ids.shape[1]), device='cuda', dtype=torch.bool),
                torch.ones((1, frame_embeds.shape[0]), device='cuda', dtype=torch.bool)
            ], dim=1)
            frame_interval_mask = torch.cat([
                self.last_ids == self.frame_token_interval_id,
                torch.zeros((1, frame_embeds.shape[0]), device='cuda', dtype=torch.bool)
            ], dim=1)
            outputs = self.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=self.past_key_values, v_mask=v_mask, frame_interval_mask=frame_interval_mask)
            self.past_key_values = outputs.past_key_values
            # memory accounting for this streaming append
            try:
                prefix_len = int(self.last_ids.shape[1])
            except Exception:
                prefix_len = 0
            frame_len = int(frame_embeds.shape[0])
            self._account_stream_append(prefix_len=prefix_len, frame_len=frame_len)
            self._maybe_trim_memory()
            # DEBUG HOOK: count streamed frame for pending block
            if hasattr(self, '_debug_pending_stream_frames'):
                self._debug_pending_stream_frames += 1
            # 2. if the same time, response after frame at that time
            if self.query_queue and video_time >= self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                return video_time, query
            # 3. if the next is frame but next is not interval, then response
            next_score = outputs.logits[:,-1:].softmax(dim=-1)
            if next_score[:,:,self.frame_token_interval_id] < self.frame_token_interval_threshold:
                next_score[:,:,self.frame_token_interval_id].zero_()
            self.last_ids = next_score.argmax(dim=-1)
            if self.last_ids != self.frame_token_interval_id: 
                return video_time, None
        return None, None
    
    def reset(self, ):
        self.query_queue = collections.deque()
        self.frame_embeds_queue = collections.deque()
        self.video_time = 0
        self.last_frame_idx = -1
        self.video_tensor = None
        self.last_ids = torch.tensor([[]], device='cuda', dtype=torch.long)
        self.past_key_values = None
        # memory tracking reset
        if hasattr(self, '_segments'):
            self._segments.clear()
            self._curr_frame_session = None
            self._ctx_len = 0
            self._pending_assistant_prompt_len = 0
        # DEBUG HOOK: debug conversation reset
        if hasattr(self, '_debug_chat'):
            self._debug_chat = [{'role': 'system', 'content': self.system_prompt}]
            self._debug_pending_stream_frames = 0

    def input_query_stream(self, query, history=None, video_time=None):
        if video_time is None:
            self.query_queue.append((self.video_time, query))
        else:
            self.query_queue.append((video_time, query))
        if not self.past_key_values:
            return f'(NOTE: No video stream here. Please select or upload a video. Then the assistant will answer "{query} (at {self.video_time}s)" in the video stream)'
        return f'(NOTE: Received "{query}" (at {self.video_time}s). Please wait until previous frames have been processed)'
    
    def input_video_stream(self, video_time):
        frame_idx = int(video_time * self.frame_fps)
        if frame_idx > self.last_frame_idx:
            ranger = range(self.last_frame_idx + 1, frame_idx + 1)
            # Move only the needed frames to GPU on-the-fly to reduce initial VRAM usage
            if self.video_tensor is None:
                return
            frames_chunk = self.video_tensor[ranger].to('cuda', non_blocking=True)
            frames_embeds = self.model.visual_embed(frames_chunk).split(self.frame_num_tokens)
            self.frame_embeds_queue.extend([(r / self.frame_fps, frame_embeds) for r, frame_embeds in zip(ranger, frames_embeds)])
        self.last_frame_idx = frame_idx
        self.video_time = video_time
    
    def load_video(self, video_path):
        # Keep decoded frames on CPU; transfer per-frame to GPU during streaming to minimize initial VRAM
        self.video_tensor = read_video(video_path, pts_unit='sec', output_format='TCHW')[0]
        try:
            self.video_tensor = self.video_tensor.pin_memory()
        except Exception:
            pass
        self.num_video_frames = self.video_tensor.size(0)
        self.video_duration = self.video_tensor.size(0) / self.frame_fps
        logger.warning(f'{video_path} -> {self.video_tensor.shape}, {self.frame_fps} FPS')

    def __call__(self, ):
        while not self.frame_embeds_queue:
            continue
        video_time, query = self._call_for_streaming()
        response = None
        if video_time is not None:
            query, response = self._call_for_response(video_time, query)
        return query, response

    # =====================
    # Memory management impl
    # =====================
    def _ensure_system_segment(self):
        if self._segments and self._segments[0]['type'] == 'system':
            return
        sys_len = int(self._start_ids.shape[1]) - self._open_prompt_len
        if sys_len > 0:
            self._segments.append({'type': 'system', 'length': sys_len})
            self._ctx_len += sys_len

    def _start_new_frame_session(self):
        session = {'open_len': self._open_prompt_len, 'close_len': 0, 'frames': []}
        self._segments.append({'type': 'frame_session', 'length': session['open_len'], 'session': session})
        self._curr_frame_session = session
        self._ctx_len += session['open_len']

    def _account_stream_append(self, *, prefix_len: int, frame_len: int):
        if self.past_key_values is None:
            return
        if not self._segments:
            self._ensure_system_segment()
            if self._open_prompt_len > 0:
                self._start_new_frame_session()
            if frame_len > 0:
                if self._curr_frame_session is None:
                    self._start_new_frame_session()
                self._curr_frame_session['frames'].append({'frame_len': frame_len, 'sep_len': 0})
                self._segments[-1]['length'] += frame_len
                self._ctx_len += frame_len
            return
        if self._curr_frame_session is None:
            if self._open_prompt_len > 0:
                self._start_new_frame_session()
            extra = max(0, prefix_len - self._open_prompt_len)
            if extra > 0:
                self._segments[-1]['length'] += extra
                self._ctx_len += extra
            if frame_len > 0:
                if self._curr_frame_session is None:
                    self._start_new_frame_session()
                self._curr_frame_session['frames'].append({'frame_len': frame_len, 'sep_len': 0})
                self._segments[-1]['length'] += frame_len
                self._ctx_len += frame_len
            return
        if self._curr_frame_session['frames']:
            self._curr_frame_session['frames'][-1]['sep_len'] += prefix_len
            self._segments[-1]['length'] += prefix_len
            self._ctx_len += prefix_len
        if frame_len > 0:
            self._curr_frame_session['frames'].append({'frame_len': frame_len, 'sep_len': 0})
            self._segments[-1]['length'] += frame_len
            self._ctx_len += frame_len

    def _account_response_prefix(self, *, query_present: bool):
        ids_len = int(self.last_ids.shape[1])
        if query_present:
            close_len = self._close_tokens_len if (self._curr_frame_session is not None) else 0
            if close_len and self._curr_frame_session is not None:
                self._curr_frame_session['close_len'] += close_len
                self._segments[-1]['length'] += close_len
                self._ctx_len += close_len
                self._curr_frame_session = None
            user_len = ids_len - self._assistant_prompt_simple_len - close_len
            if user_len > 0:
                self._segments.append({'type': 'user', 'length': user_len})
                self._ctx_len += user_len
            self._pending_assistant_prompt_len = self._assistant_prompt_simple_len
            self._ctx_len += self._pending_assistant_prompt_len
        else:
            if self._curr_frame_session is not None:
                self._curr_frame_session['close_len'] += self._close_tokens_len
                self._segments[-1]['length'] += self._close_tokens_len
                self._ctx_len += self._close_tokens_len
                self._curr_frame_session = None
            self._pending_assistant_prompt_len = self._stream_assistant_prompt_len
            self._ctx_len += self._pending_assistant_prompt_len

    def _account_assistant_generation(self, *, gen_len: int):
        prompt_len = int(getattr(self, '_pending_assistant_prompt_len', 0))
        total = prompt_len + gen_len
        if total > 0:
            self._segments.append({'type': 'assistant', 'length': total})
            self._ctx_len += gen_len
        self._pending_assistant_prompt_len = 0

    def _maybe_trim_memory(self):
        if not self.enable_memory_trim or self.past_key_values is None:
            return
        if self._ctx_len <= self.memory_trim_trigger:
            return
        guard = 0
        while self._ctx_len > self.memory_trim_trigger and guard < 10000:
            guard += 1
            if not self._segments:
                break
            # decide which segment to trim
            if self.memory_trim_mode == 'frames_only':
                start = 1 if (self._segments and self._segments[0]['type'] == 'system') else 0
                idx = -1
                for j in range(start, len(self._segments)):
                    if self._segments[j]['type'] == 'frame_session':
                        idx = j
                        break
                if idx == -1:
                    # no frame sessions to drop; preserve dialogue history
                    break
                seg = self._segments[idx]
            else:
                idx = 0
                if self._segments[0]['type'] == 'system':
                    if len(self._segments) == 1:
                        break
                    idx = 1
                seg = self._segments[idx]
            drop = 0
            if seg['type'] in ('user', 'assistant'):
                drop = seg['length']
                # in frames_only mode we never delete dialogue
                if self.memory_trim_mode == 'frames_only':
                    break
                else:
                    self._segments.remove(seg)
            elif seg['type'] == 'frame_session':
                session = seg['session']
                if session['frames']:
                    if len(session['frames']) > 1:
                        f0 = session['frames'].pop(0)
                        drop = f0['frame_len'] + f0['sep_len']
                        seg['length'] -= drop
                    else:
                        f0 = session['frames'].pop(0)
                        drop = session['open_len'] + f0['frame_len'] + session.get('close_len', 0)
                        self._segments.remove(seg)
                else:
                    drop = session.get('open_len', 0) + session.get('close_len', 0)
                    self._segments.remove(seg)
            else:
                drop = seg.get('length', 0)
                if self.memory_trim_mode == 'frames_only':
                    break
                else:
                    self._segments.remove(seg)
            if drop <= 0:
                break
            self._apply_trim_left(drop)

    def set_memory_trim_mode(self, mode: str):
        assert mode in ('hybrid', 'frames_only')
        self.memory_trim_mode = mode

    def _apply_trim_left(self, n_tokens: int):
        if n_tokens <= 0 or self.past_key_values is None:
            return
        kv_len = int(self.past_key_values[0][0].shape[2])
        keep_start = min(n_tokens, kv_len)
        self.past_key_values = self.model.trim_past_key_values(self.past_key_values, keep_start, kv_len)
        self._ctx_len -= keep_start

    # =====================
    # Debug helpers
    # =====================
    def debug_get_full_chat(self):
        """Return a list of messages describing the current chat, including pending frames."""
        chat = list(getattr(self, '_debug_chat', []))
        pending = int(getattr(self, '_debug_pending_stream_frames', 0))
        if pending > 0:
            chat.append({'role': 'stream', 'num_frames': pending})
        return chat

    def debug_get_full_chat_text(self):
        """Return a readable transcript of the current chat with compact frame markers."""
        chat = self.debug_get_full_chat()
        parts = []
        for m in chat:
            r = m.get('role')
            if r == 'system':
                parts.append(f"System: {m.get('content','')}")
            elif r == 'user':
                parts.append(f"\nUser: {m.get('content','')}")
            elif r == 'assistant':
                parts.append(f"\nAssistant: {m.get('content','')}")
            elif r == 'stream':
                n = int(m.get('num_frames', 0))
                if n > 0:
                    parts.append(f"\n[frames x {n}]")
        return ''.join(parts)

    def debug_print_full_chat(self):
        """Log/print the current chat transcript for debugging."""
        text = self.debug_get_full_chat_text()
        try:
            logger.warning('[DEBUG CHAT]\n' + text)
        except Exception:
            print('[DEBUG CHAT]\n' + text)

    def debug_get_cropped_chat(self):
        """Return messages that reflect the CURRENT, cropped chat (based on self._segments).

        Notes:
        - System message is always included from self.system_prompt.
        - User/Assistant turns are matched in order from the historical _debug_chat
          but only as many as there are surviving segments, so trimmed turns are skipped.
        - Frame sessions are summarized as {'role': 'stream', 'num_frames': <remaining_frames>}.
        """
        result = [{'role': 'system', 'content': getattr(self, 'system_prompt', '')}]
        # Fast path: if no segments, return system only
        if not hasattr(self, '_segments') or not self._segments:
            return result

        # Prepare tail-aligned per-role lists
        seg_roles = [s.get('type') for s in self._segments if s.get('type') in ('user','assistant')]
        remain_user = sum(1 for r in seg_roles if r == 'user')
        remain_assistant = sum(1 for r in seg_roles if r == 'assistant')
        hist_user = [m for m in list(getattr(self, '_debug_chat', [])) if m.get('role') == 'user']
        hist_assistant = [m for m in list(getattr(self, '_debug_chat', [])) if m.get('role') == 'assistant']
        tail_user = hist_user[-remain_user:] if remain_user > 0 else []
        tail_assistant = hist_assistant[-remain_assistant:] if remain_assistant > 0 else []
        u_i, a_i = 0, 0

        for seg in self._segments:
            st = seg.get('type')
            if st == 'frame_session':
                session = seg.get('session', {})
                frames = session.get('frames', []) or []
                n = len(frames)
                if n > 0:
                    result.append({'role': 'stream', 'num_frames': int(n)})
            elif st == 'user':
                m = tail_user[u_i] if u_i < len(tail_user) else {'role':'user','content':''}
                u_i += 1
                result.append({'role': 'user', 'content': m.get('content','')})
            elif st == 'assistant':
                m = tail_assistant[a_i] if a_i < len(tail_assistant) else {'role':'assistant','content':''}
                a_i += 1
                result.append({'role': 'assistant', 'content': m.get('content','')})
            elif st == 'system':
                # Already included at the beginning; skip duplicates
                continue
            else:
                # Unknown segment type; skip
                continue
        return result

    def debug_get_cropped_chat_text(self):
        """Return a readable transcript of the CURRENT, cropped chat."""
        chat = self.debug_get_cropped_chat()
        parts = []
        for m in chat:
            r = m.get('role')
            if r == 'system':
                parts.append(f"System: {m.get('content','')}")
            elif r == 'user':
                parts.append(f"\nUser: {m.get('content','')}")
            elif r == 'assistant':
                parts.append(f"\nAssistant: {m.get('content','')}")
            elif r == 'stream':
                n = int(m.get('num_frames', 0))
                if n > 0:
                    parts.append(f"\n[frames x {n}]")
        return ''.join(parts)

    def debug_print_cropped_chat(self):
        """Log/print the CURRENT, cropped chat transcript for debugging."""
        text = self.debug_get_cropped_chat_text()
        try:
            logger.warning('[DEBUG CROPPED CHAT]\n' + text)
        except Exception:
            print('[DEBUG CROPPED CHAT]\n' + text)