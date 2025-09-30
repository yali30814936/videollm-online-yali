import os, torch, json, tqdm, collections, random
from transformers import EvalPrediction

from .ego4d import Ego4D
from ..stream import StreamMixIn
from ..utils import ceil_time_by_fps, DictWithTo

class Ego4DNarrationStream(Ego4D, StreamMixIn):
    benchmarks_with_keys = {
        'goalstep': 'videos', 'fho_lta': 'clips', 'nlq': 'videos', 'moments': 'videos',
        'av': 'videos', 'fho_oscc-pnr': 'clips', 'fho_sta': 'annotations', 'vq': 'videos'
    }
    instructions = [{"role": "user", "content": "Please concisely narrate the video in real time. Use the tag 'C' to denote the camera wearer, and other letter tags, such as 'X', to denote other individuals in the scene."}]
    evaluation_kwargs = DictWithTo(evaluator='stream_evaluate')

    def get_annos(self, split: str) -> dict[str, dict[str, list]]:
        annos = json.load(open(os.path.join(Ego4D.anno_root, 'all_narrations_redacted.json')))['videos']
        assert split in ['train', 'val', 'test']
        anno_path = os.path.join(Ego4D.anno_root, f'narration_stream_video_uids_{split}.json')
        if os.path.exists(anno_path):
            split_video_uids = json.load(open(anno_path))
        else:
            all_video_uids = set(annos.keys())
            val_video_uids, test_video_uids = [], []
            for benchmark, key in tqdm.tqdm(Ego4DNarrationStream.benchmarks_with_keys.items(), desc=f'prepare {anno_path}'):
                val_video_uids.extend([anno['video_uid'] for anno in json.load(open(os.path.join(Ego4D.root, 'annotations', f'{benchmark}_val.json')))[key]])
                test_video_uids.extend([anno['video_uid'] for anno in json.load(open(os.path.join(Ego4D.root, 'annotations', f'{benchmark}_test_unannotated.json')))[key]])
            val_video_uids = set(val_video_uids)
            test_video_uids = set(test_video_uids) - val_video_uids
            if split == 'train':
                split_video_uids = list(all_video_uids - val_video_uids - test_video_uids)
            elif split == 'val':
                split_video_uids = list(all_video_uids.intersection(val_video_uids))
            else:
                split_video_uids = list(all_video_uids.intersection(test_video_uids))
            json.dump(split_video_uids, open(anno_path, 'w'), indent=4)
        anno_path = os.path.join(Ego4D.anno_root, f'narration_stream_{split}.json')
        narration_streams = {}
        if os.path.exists(anno_path):
            narration_streams = json.load(open(anno_path))
        else:
            for video_uid in tqdm.tqdm(split_video_uids, desc=f'prepare {anno_path}...'):
                if video_uid not in split_video_uids:
                    continue
                anno = annos[video_uid]
                # 1. sort & clean narration text
                narrations = []
                for ns in anno['narrations']:
                    text = Ego4DNarrationStream._clean_text(ns['text'])
                    if len(text.split(' ')) >= 2: # at least, C verb.
                        narrations.append({
                            'time': ns['time'],
                            'text': text,
                            '_annotation_uid': ns['_annotation_uid']
                        })
                narrations = sorted(narrations, key=lambda x:x['time'])
                # 2. match narration with summary
                _annotation_uid_narrations = collections.defaultdict(list)
                for narration in narrations:
                    _annotation_uid_narrations[narration.pop('_annotation_uid')].append(narration)
                narration_streams[video_uid] = _annotation_uid_narrations
            json.dump(narration_streams, open(anno_path, 'w'), indent=4)
        return narration_streams

    def __init__(self, *, split: str, frame_fps: int, is_training: bool, augmentation: bool, use_conversation_eval: bool = False, top_n_samples: int | None = None, **kwargs):
        """
        Args:
            use_conversation_eval: 如果為 True，使用新的 conversation_stream_evaluate 方法
            top_n_samples: 僅保留依片段時長由長到短的前 N 筆（為 None 或 <=0 時不啟用）。
        """
        super().__init__(split=split, frame_fps=frame_fps, augmentation=augmentation, is_training=is_training, use_conversation_eval=use_conversation_eval, **kwargs)
        self.is_training = is_training
        self.frame_fps = frame_fps
        self.use_conversation_eval = use_conversation_eval
        # 允許用環境變數覆蓋，以便快速小規模測試（例如：EGO4D_NARR_TOPN=50）
        if top_n_samples is None:
            try:
                env_topn = int(os.environ.get('EGO4D_NARR_TOPN', '0') or '0')
            except ValueError:
                env_topn = 0
            top_n_samples = env_topn if env_topn > 0 else None
        self.top_n_samples = top_n_samples
        self.min_selected_length_sec = None  # 被截斷後的最短片段長度（秒）
        
        # 根據評估方式設置 evaluation_kwargs
        if use_conversation_eval:
            self.evaluation_kwargs = DictWithTo(evaluator='conversation_stream_evaluate')

        annos = self.get_annos(split)
        self.annos = []
        for video_uid, _annotation_uid_narrations in tqdm.tqdm(annos.items(), desc=f'narration_stream_{split}...'):
            duration = self.metadata[video_uid]['duration']
            for narrations in _annotation_uid_narrations.values():
                if not narrations:
                    continue
                start_time = ceil_time_by_fps(narrations[0]['time'], frame_fps, min_time=0, max_time=duration)
                conversation = []
                last_time = start_time - 1 / frame_fps
                last_text = None
                for narration in narrations:
                    if last_time >= duration:
                        break
                    text = narration['text']
                    if text == last_text:
                        continue
                    time = ceil_time_by_fps(narration['time'], frame_fps, min_time=0, max_time=duration)
                    if time == last_time: # since we have sorted and ceiled, so directly replace, this time is more close
                        conversation[-1]['content'] = text
                    else: # time > last_time
                        num_frames = int((time - last_time) * frame_fps)
                        conversation.extend([
                            {"role": "stream", 'num_frames': num_frames, 'learn': True},
                            {"role": "assistant", "content": text, 'learn': True},
                        ])
                    last_time = time
                    last_text = text
                if not conversation:
                    continue
                # 計算實際載入的影格範圍與片段長度（秒）
                start_idx = int(start_time * frame_fps)
                end_idx = int(last_time * frame_fps)
                num_loaded_frames = max(0, end_idx - start_idx + 1)
                length_sec = num_loaded_frames / frame_fps if frame_fps > 0 else 0.0
                self.annos.append({
                    'conversation': conversation,
                    'load_ranges': {self.metadata[video_uid]['path']: range(start_idx, end_idx + 1)},
                    'length_sec': length_sec,
                    'video_uid': video_uid,
                })

        # 依片段時長由長到短排序，並根據需求只保留前 N 筆（僅當 top_n_samples>0 時啟用）
        if self.annos and (self.top_n_samples is not None and self.top_n_samples > 0):
            total = len(self.annos)
            self.annos.sort(key=lambda a: a.get('length_sec', 0.0), reverse=True)
            if total > self.top_n_samples:
                self.annos = self.annos[:self.top_n_samples]
            # 記錄並回報被截斷清單中的最短長度（秒）
            self.min_selected_length_sec = min(a.get('length_sec', 0.0) for a in self.annos) if self.annos else None
            try:
                shortest = f"{self.min_selected_length_sec:.2f}s" if self.min_selected_length_sec is not None else "N/A"
                longest = f"{self.annos[0].get('length_sec', 0.0):.2f}s" if self.annos else "N/A"
                kept = len(self.annos)
                print(f"[Ego4DNarrationStream] Sorted by length desc; kept {kept}/{total} samples. Shortest among kept: {shortest}, longest: {longest}.")
            except Exception:
                pass

    def preprocess_conversation(self, conversation):
        assert conversation[0]['role'] == 'stream' and conversation[0]['num_frames'] == 1
        conversation[0]['learn'] = False
        return conversation[:1] + [random.choice(self.instructions)] + conversation[1:] # first is stream

    def __getitem__(self, index):
        anno = self.annos[index]
        return *super().__getitem__(
            conversation=self.preprocess_conversation(anno['conversation']),
            load_ranges=anno['load_ranges'],
        ), index, self.evaluation_kwargs

    @staticmethod
    def _clean_text(src: str):
        # 1. remove #
        dst = src.replace('#C', '').replace('#c', '').replace('@c', '')
        dst = dst.replace('#O', '').replace('#o', '')
        dst = dst.replace('#Unsure', '').replace('#unsure', '')
        dst = dst.replace('#', '')
        # 2. remove start&end extra space and ,.
        dst = dst.strip('.,\n ') + '.'
        # 3. make the first word capitalize and remove extra space within the sentence
        words = dst.split()
        words[0] = words[0].capitalize()
        dst = ' '.join(words)
        return dst

    def compute_metrics(self, eval_predictions: EvalPrediction, *args, **kwargs):
        # 若使用 conversation_stream_evaluate，predictions 為:
        # [time_mae, time_acc, <text metrics...>, f1]
        if getattr(self, 'use_conversation_eval', False):
            vec = torch.from_numpy(eval_predictions.predictions).mean(dim=0).tolist()

            # 解析文字指標設定
            def _normalize_metric_name(name: str) -> str:
                n = str(name).strip().lower()
                if n in ('rouge', 'rougel', 'rouge-l', 'rougel-f1', 'rouge-lsum'):
                    return 'rougelsum'
                if n in ('meteor',):
                    return 'meteor'
                if n in ('jaccard', 'jac'):
                    return 'jaccard'
                return n

            text_metrics_raw = kwargs.get('eval_text_metrics', None)
            if text_metrics_raw is None:
                single_metric = kwargs.get('eval_text_metric', 'rougeLsum')
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

            # 映射輸出向量
            time_mae = float(vec[0]) if len(vec) > 0 else 0.0
            time_acc = float(vec[1]) if len(vec) > 1 else 0.0
            k = len(text_metric_list)
            text_vals = vec[2:2+k] if len(vec) >= 2+k else [0.0]*k
            match_f1 = float(vec[2+k]) if len(vec) > 2+k else 0.0

            metrics = {
                'time_mae': time_mae,
                'time_acc_3f': time_acc,
                'match_f1': match_f1,
            }
            for name, val in zip(text_metric_list, text_vals):
                metrics[name] = float(val)
            return metrics

        # 舊的 stream_evaluate 路徑
        lm_ppl, frame_diff, fluency, lm_correctness = torch.from_numpy(eval_predictions.predictions).mean(dim=0).tolist()
        return {
            'lm_ppl': lm_ppl,
            'time_diff': frame_diff / self.frame_fps,
            'fluency': fluency,
            'lm_correctness': lm_correctness
        }

def build_ego4d_narration_stream_train(**kwargs):
    return Ego4DNarrationStream(split='train', **kwargs)

def build_ego4d_narration_stream_val(**kwargs):
    return Ego4DNarrationStream(split='val', **kwargs)

class Ego4DRefinedNarrationStream(Ego4DNarrationStream):
    instructions = [
        {"role": "user", "content": "Please concisely narrate the video in real time."},
        {"role": "user", "content": "Help me to illustrate my view in short."},
        {"role": "user", "content": "Please simply describe what do you see."},
        {"role": "user", "content": "Continuously answer what you observed with simple text."},
        {"role": "user", "content": "Do concise real-time narration."},
        {"role": "user", "content": "Hey assistant, do you know the current video content? Reply me concisely."},
        {"role": "user", "content": "Simply interpret the scene for me."},
        {"role": "user", "content": "What can you tell me about? Be concise."},
        {"role": "user", "content": "Use simple text to explain what is shown in front of me."},
        {"role": "user", "content": "What is the action now? Please response in short."},
    ]

    def get_annos(self, split: str) -> dict:
        anno_path = os.path.join(Ego4D.anno_root, f'refined_narration_stream_{split}.json')
        assert os.path.exists(anno_path)
        narration_streams = json.load(open(anno_path))
        return narration_streams

def build_ego4d_refined_narration_stream_train(**kwargs):
    return Ego4DRefinedNarrationStream(split='train', **kwargs)

def build_ego4d_refined_narration_stream_val(**kwargs):
    return Ego4DRefinedNarrationStream(split='val', **kwargs)

if __name__ == '__main__':
    build_ego4d_refined_narration_stream_train(
        frame_fps=2, is_training=True, augmentation=True,
        system_prompt='', tokenizer=None,
        vision_pretrained='google/siglip-large-patch16-384',
        embed_mark='2fps_384_1+3x3'
    )
    build_ego4d_refined_narration_stream_val(
        frame_fps=2, is_training=True, augmentation=True,
        system_prompt='', tokenizer=None,
        vision_pretrained='google/siglip-large-patch16-384',
        embed_mark='2fps_384_1+3x3'
    )
