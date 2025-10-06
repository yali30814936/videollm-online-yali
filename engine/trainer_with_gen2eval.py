import torch
from transformers import Trainer

class TrainerWithGenToEval(Trainer):
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        with torch.no_grad(), self.compute_loss_context_manager():
            # frames = inputs.pop('frames')
            inputs = self._prepare_inputs(inputs)
            # if frames is not None:
            #     inputs['frames'] = frames
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs, return_outputs=False)
                return (loss, None, None)
            sample_idxs = inputs.pop('sample_idxs')
            evaluation_kwargs = inputs.pop('evaluation_kwargs')
            evaluator = evaluation_kwargs.pop('evaluator')
            
            # 處理 conversation_stream_evaluate 的特殊情況
            if evaluator == 'conversation_stream_evaluate':
                # 從數據集的 cache 中獲取原始對話數據
                dataset = self.eval_dataset if hasattr(self, 'eval_dataset') else None
                if dataset and hasattr(dataset, '_conversation_cache'):  # type: ignore
                    sample_idx = sample_idxs[0].item() if hasattr(sample_idxs[0], 'item') else sample_idxs[0]
                    cached_data = getattr(dataset, '_conversation_cache', {}).get(sample_idx, {})  # type: ignore
                    if isinstance(evaluation_kwargs, dict):
                        evaluation_kwargs.update(cached_data)  # type: ignore
            
            # 確保tokenizer存在
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', None) if self.tokenizer else None
            eos_token_id = getattr(self.tokenizer, 'eos_token_id', None) if self.tokenizer else None
            
            # 準備額外參數
            extra_kwargs = {
                'pad_token_id': pad_token_id,
                'eos_token_id': eos_token_id
            }
            
            # 添加 tokenizer（如果存在）
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                extra_kwargs['tokenizer'] = self.tokenizer
            
            # 合併所有參數
            all_kwargs = {**inputs, **evaluation_kwargs, **extra_kwargs}  # type: ignore
            output_ids = getattr(model, evaluator)(**all_kwargs)
            
            # 處理不同類型的evaluator返回值
            if evaluator in ['causal_stream_evaluate', 'stream_evaluate', 'conversation_stream_evaluate']:
                # 這些evaluator返回評估分數，不需要reshape
                # 確保output_ids是二維的以符合trainer期望
                if isinstance(output_ids, (list, tuple)):
                    output_ids = torch.tensor(output_ids)
                if hasattr(output_ids, 'dim') and output_ids.dim() == 1:
                    output_ids = output_ids.unsqueeze(0)  # [N] -> [1, N]
                # 移到與模型相同的裝置，避免 accelerate all_gather 報 "Tensors must be CUDA and dense"
                try:
                    device = getattr(model, 'device', None)
                    if device is None and hasattr(self.args, 'device'):
                        device = self.args.device  # type: ignore
                    if hasattr(output_ids, 'to') and device is not None:
                        output_ids = output_ids.to(device)
                    if hasattr(output_ids, 'contiguous'):
                        output_ids = output_ids.contiguous()
                except Exception:
                    pass
                return (None, output_ids, sample_idxs)
            else:
                # 其他evaluator（如generate）返回token序列，需要reshape
                return (None, output_ids.reshape(1, -1), sample_idxs)