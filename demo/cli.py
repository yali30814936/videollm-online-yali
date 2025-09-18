import os, torchvision, transformers, tqdm, time, json
import torch.multiprocessing as mp
torchvision.set_video_backend('pyav')

from data.utils import ffmpeg_once

from .inference import LiveInfer
logger = transformers.logging.get_logger('liveinfer')

# python -m demo.cli --resume_from_checkpoint ... 

def main(liveinfer):
    # Optional: configure memory trim via env for testing
    import os
    mode = os.getenv('LIVE_TRIM_MODE')
    if mode in ('hybrid', 'frames_only'):
        try:
            liveinfer.set_memory_trim_mode(mode)
            logger.warning(f'[CLI] Memory trim mode set to: {mode}')
        except Exception:
            pass
    trig = os.getenv('LIVE_TRIM_TRIGGER')
    if trig and trig.isdigit():
        try:
            liveinfer.memory_trim_trigger = int(trig)
            logger.warning(f'[CLI] Memory trim trigger set to: {liveinfer.memory_trim_trigger}')
        except Exception:
            pass
    # src_video_path = 'datasets/ego4d/v2/full_scale/1cfcca5d-7c45-46e3-8f2e-90b6d9bcf04c.mp4' # drum
    # src_video_path = 'datasets/ego4d/v2/full_scale/ec4b530e-f01c-420a-915d-4a11bc26c3ae.mp4' # long video (val)
    src_video_path = 'datasets/ego4d/v2/full_scale/de493aa9-e00c-4c54-bd8b-304e94291ba2.mp4' # long video (?)
    # src_video_path = 'datasets/ego4d/v2/full_scale/0c190d90-8230-42c8-b574-5ff9d513ecaf.mp4' # crossing street
    # src_video_path = 'datasets/ego4d/v2/full_scale/7b8c29ef-fcb6-4a9d-9b99-0e6bb64eadb9.mp4' # dinning room
    # src_video_path = 'datasets/ego4d/v2/full_scale/9ff8c35c-bd28-436f-b35b-ee460f983a67.mp4' # driving
    # src_video_path = 'datasets/ego4d/v2/full_scale/73f567d0-7f65-4f33-9331-59936ef97f7a.mp4' # guitar
    # src_video_path = 'datasets/ego4d/v2/full_scale/194612d8-4baa-4e08-a382-158974395e45.mp4' # mall
    # src_video_path = 'datasets/ego4d/v2/full_scale/9439167f-026f-4188-a671-96f068000fd3.mp4' # construction site
    
    # src_video_path = 'demo/assets/bicycle.mp4'
    # src_video_path = 'demo/assets/cooking.mp4'
    name, ext = os.path.splitext(src_video_path)
    ffmpeg_video_path = os.path.join('demo/assets/cache', name + f'_{liveinfer.frame_fps}fps_{liveinfer.frame_resolution}' + ext)
    save_history_path = src_video_path.replace('.mp4', '.json')
    if not os.path.exists(ffmpeg_video_path):
        os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
        ffmpeg_once(src_video_path, ffmpeg_video_path, fps=liveinfer.frame_fps, resolution=liveinfer.frame_resolution)
        logger.warning(f'{src_video_path} -> {ffmpeg_video_path}, {liveinfer.frame_fps} FPS, {liveinfer.frame_resolution} Resolution')
    
    liveinfer.load_video(ffmpeg_video_path)
    liveinfer.input_query_stream('Please narrate the video in real time.', video_time=0.0)
    # liveinfer.input_query_stream('Hi, who are you?', video_time=1.0)
    # liveinfer.input_query_stream('Yes, I want to check its safety.', video_time=3.0)
    # liveinfer.input_query_stream('No, I am going to install something to alert pedestrians to move aside. Could you guess what it is?', video_time=12.5)

    timecosts = []
    pbar = tqdm.tqdm(total=liveinfer.num_video_frames, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}]")
    history = {'video_path': src_video_path, 'frame_fps': liveinfer.frame_fps, 'conversation': []} 
    for i in range(min(liveinfer.num_video_frames, 10000)):
        # liveinfer.frame_token_interval_threshold -= 0.00175 # decay
        start_time = time.time()
        liveinfer.input_video_stream(i / liveinfer.frame_fps)
        query, response = liveinfer()
        end_time = time.time()
        timecosts.append(end_time - start_time)
        fps = (i + 1) / sum(timecosts)
        pbar.set_postfix_str(f"Average Processing FPS: {fps:.1f}")
        pbar.update(1)
        if query:
            history['conversation'].append({'role': 'user', 'content': query, 'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
            print(query)
        if response:
            history['conversation'].append({'role': 'assistant', 'content': response, 'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
            print(response)
        # Optional: print full/cropped chat snapshot for debugging when enabled
        if os.getenv('LIVE_DEBUG_CHAT') == '1':
            try:
                liveinfer.debug_print_full_chat()
            except Exception:
                pass
        if os.getenv('LIVE_DEBUG_CROPPED_CHAT') == '1':
            try:
                liveinfer.debug_print_cropped_chat()
            except Exception:
                pass
        if not query and not response:
            history['conversation'].append({'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
    json.dump(history, open(save_history_path, 'w'), indent=4)
    print(f'The conversation history has been saved to {save_history_path}.')

if __name__ == '__main__':
    liveinfer = LiveInfer()
    main(liveinfer)