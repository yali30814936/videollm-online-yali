import os, transformers, tqdm, time, json, torch

from .inference_vanilla import LiveInfer
logger = transformers.logging.get_logger('liveinfer')

# python -m demo.cli --resume_from_checkpoint ... 

def main(liveinfer: LiveInfer, video_uid, save_history_path):
    # src_video_path = 'demo/assets/cooking.mp4'
    # name, ext = os.path.splitext(src_video_path)
    # ffmpeg_video_path = os.path.join('demo/assets/cache', name + f'_{liveinfer.frame_fps}fps_{liveinfer.frame_resolution}' + ext)
    # save_history_path = src_video_path.replace('.mp4', '.json')
    # if not os.path.exists(ffmpeg_video_path):
    #     os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
    #     ffmpeg_once(src_video_path, ffmpeg_video_path, fps=liveinfer.frame_fps, resolution=liveinfer.frame_resolution)
    #     logger.warning(f'{src_video_path} -> {ffmpeg_video_path}, {liveinfer.frame_fps} FPS, {liveinfer.frame_resolution} Resolution')
    
    video_path = f"datasets/ego4d/v2/full_scale_2fps_384/{video_uid}.mp4"
    if not os.path.exists(save_history_path):
        os.makedirs(save_history_path)
    save_history_path = os.path.join(save_history_path, f"{video_uid}.json")
    
    # Initialize variables outside try block to ensure they're accessible in except/finally
    oom_occurred = False
    history = {'video_uid': video_uid, 'conversation': []}
    pbar = None
    i = -1  # Track current frame index
    
    try:
        liveinfer.load_video(video_path)
        liveinfer.input_query_stream('Please concisely narrate the video in real time.', video_time=0.0)

        timecosts = []
        pbar = tqdm.tqdm(total=liveinfer.num_video_frames, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}]")
        history['frame_fps'] = liveinfer.frame_fps
        history['total_video_frames'] = liveinfer.num_video_frames
        history['total_video_duration'] = liveinfer.num_video_frames / liveinfer.frame_fps
        
        for i in range(liveinfer.num_video_frames): # max 1h
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
            if not query and not response:
                history['conversation'].append({'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
                
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            oom_occurred = True
            processed_duration = (i + 1) / liveinfer.frame_fps if i >= 0 else 0
            logger.error(f"CUDA Out of Memory error occurred for video {video_uid} at frame {i}/{liveinfer.num_video_frames}")
            logger.error(f"Processed {processed_duration:.2f}s out of {history.get('total_video_duration', 0):.2f}s")
            logger.error(f"Error details: {str(e)}")
            history['oom_error'] = {
                'frame': i,
                'total_frames': liveinfer.num_video_frames,
                'error_message': str(e),
                'video_time': liveinfer.video_time,
                'processed_duration': processed_duration
            }
            history['processed_duration'] = processed_duration
        else:
            raise  # Re-raise if it's not an OOM error
    except Exception as e:
        logger.error(f"Unexpected error occurred for video {video_uid}: {str(e)}")
        processed_duration = (i + 1) / liveinfer.frame_fps if i >= 0 else 0
        history['error'] = {
            'type': type(e).__name__,
            'message': str(e)
        }
        history['processed_duration'] = processed_duration
        raise  # Re-raise unexpected errors
    finally:
        # Calculate and record the processed duration if not already set
        if 'processed_duration' not in history:
            history['processed_duration'] = (i + 1) / history['frame_fps'] if i >= 0 and 'frame_fps' in history else 0
        
        # Always save the history, even if OOM occurred
        if pbar is not None:
            pbar.close()
        json.dump(history, open(save_history_path, 'w'), indent=4)
        if oom_occurred:
            print(f'⚠️  OOM occurred! Partial results saved to {save_history_path}.')
        else:
            print(f'The conversation history has been saved to {save_history_path}.')
        
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"CUDA cache cleared after processing {video_uid}")

if __name__ == '__main__':
    import datetime
    
        # "0e0d6704-1c6c-4a62-bc97-cc55658cf8ac"
    video_list = [ # val
        "0a3097fc-baed-4d11-a4c9-30f07eb91af6"
        # "d9db7166-e5fe-4ce5-85eb-0c437c4ddf32", # grinding, black screen
        # "8c730e66-aa1b-4a0b-a4ee-db42b1b41f96", # packaging, black screen
        # "155f8d74-4c5c-4821-a18b-fceaa9c6199c", # sweeping around the house, black screen
        # "3e4ffe07-a54f-42b6-9a0b-b093adad26e8",
        # "1e5bd816-e1dd-43d3-8709-42c83114dc7c",
        # "2d560d56-dc47-4c76-8d41-889c8aa55d66",
        # "16d55886-6e1e-4195-9918-12dc4568320e",
        # "c3065f43-abc1-4868-84fd-3f937010fd17",
        # "d36bb05b-2b9a-468c-93c1-7469aa559baf",
        # "fd06094f-8e2f-414c-9f66-3c99cd14a151"
    ]
    # video_list = [ # train
    #     "878666b9-5f3d-476a-960f-3a5d02da7619",
    #     "fe1f9d74-2a66-418f-88e8-80a3f6dc6e9d",
    #     "d3f76c88-5996-4da0-8b8f-6c1b8f85a4d8",
    #     "91155366-856d-445b-8324-f7db85a51628",
    #     "3e057cdc-bf48-4090-81d6-b66bb18542e7",
    #     "ada73c3a-765f-46de-afb5-d152d058d7e2",
    #     "3efc152d-ea0e-4372-b552-7d5e1cf07259",
    #     "7376150d-53e2-43be-b97a-1c31723556ab",
    #     "0b530687-26d8-4c9d-9771-c758ecd2ecbf"
    # ]
    save_history_path = "duration_test_results/nomod_test_joya"
    
    # Record start time
    execution_start_time = time.time()
    execution_start_datetime = datetime.datetime.now()
    
    liveinfer = LiveInfer()
    
    # Track processing results
    results = {
        'completed': [], 
        'oom_errors': [], 
        'other_errors': [],
        'processing_details': {},  # Store detailed info for each video
        'video_timings': {}  # Track processing time for each video
    }
    
    for vid in video_list:
        print(f'\n{"="*60}')
        print(f'Processing video {vid} ...')
        print(f'{"="*60}\n')
        
        video_start_time = time.time()
        
        try:
            main(liveinfer, vid, save_history_path)
            liveinfer.reset()
            
            video_end_time = time.time()
            video_processing_time = video_end_time - video_start_time
            results['video_timings'][vid] = video_processing_time
            
            # Check if OOM occurred by reading the saved file
            result_path = os.path.join(save_history_path, f"{vid}.json")
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    result_data = json.load(f)
                    processed_duration = result_data.get('processed_duration', 0)
                    total_duration = result_data.get('total_video_duration', 0)
                    
                    # Store processing details
                    results['processing_details'][vid] = {
                        'processed_duration': processed_duration,
                        'total_duration': total_duration,
                        'completion_rate': (processed_duration / total_duration * 100) if total_duration > 0 else 0
                    }
                    
                    if 'oom_error' in result_data:
                        results['oom_errors'].append(vid)
                    else:
                        results['completed'].append(vid)
            else:
                results['completed'].append(vid)
                
        except Exception as e:
            video_end_time = time.time()
            video_processing_time = video_end_time - video_start_time
            results['video_timings'][vid] = video_processing_time
            
            logger.error(f"Failed to process video {vid}: {str(e)}")
            results['other_errors'].append({'video_uid': vid, 'error': str(e)})
            liveinfer.reset()  # Reset even on error
            continue
    
    # Calculate total execution time
    execution_end_time = time.time()
    execution_end_datetime = datetime.datetime.now()
    total_execution_time = execution_end_time - execution_start_time
    
    # Print summary
    print(f'\n\n{"="*60}')
    print("PROCESSING SUMMARY")
    print(f'{"="*60}')
    print(f"✅ Completed successfully: {len(results['completed'])}/{len(video_list)}")
    for vid in results['completed']:
        details = results['processing_details'].get(vid, {})
        duration = details.get('processed_duration', 0)
        processing_time = results['video_timings'].get(vid, 0)
        print(f"   - {vid}: {duration:.2f}s (took {processing_time:.2f}s to process)")
    
    print(f"\n⚠️  OOM errors (partial results saved): {len(results['oom_errors'])}/{len(video_list)}")
    for vid in results['oom_errors']:
        details = results['processing_details'].get(vid, {})
        processed = details.get('processed_duration', 0)
        total = details.get('total_duration', 0)
        completion = details.get('completion_rate', 0)
        processing_time = results['video_timings'].get(vid, 0)
        print(f"   - {vid}: {processed:.2f}s / {total:.2f}s ({completion:.1f}%) (took {processing_time:.2f}s)")
    
    print(f"\n❌ Other errors: {len(results['other_errors'])}/{len(video_list)}")
    for error_info in results['other_errors']:
        processing_time = results['video_timings'].get(error_info['video_uid'], 0)
        print(f"   - {error_info['video_uid']}: {error_info['error']} (took {processing_time:.2f}s)")
    
    # Calculate statistics
    all_processed_durations = [details['processed_duration'] for details in results['processing_details'].values()]
    all_total_durations = [details['total_duration'] for details in results['processing_details'].values()]
    
    if all_processed_durations:
        avg_processed = sum(all_processed_durations) / len(all_processed_durations)
        avg_total = sum(all_total_durations) / len(all_total_durations)
        avg_completion = (avg_processed / avg_total * 100) if avg_total > 0 else 0
        
        # Calculate average processing time
        all_processing_times = list(results['video_timings'].values())
        avg_processing_time = sum(all_processing_times) / len(all_processing_times) if all_processing_times else 0
        
        print(f'\n{"="*60}')
        print("STATISTICS")
        print(f'{"="*60}')
        print(f"Average processed duration: {avg_processed:.2f}s")
        print(f"Average total duration: {avg_total:.2f}s")
        print(f"Average completion rate: {avg_completion:.1f}%")
        print(f"Average processing time per video: {avg_processing_time:.2f}s")
        print(f"Total videos processed: {len(results['processing_details'])}")
        print(f"Total execution time: {total_execution_time:.2f}s ({total_execution_time/60:.2f} minutes)")
        
        # Add statistics to results
        results['statistics'] = {
            'average_processed_duration': avg_processed,
            'average_total_duration': avg_total,
            'average_completion_rate': avg_completion,
            'average_processing_time_per_video': avg_processing_time,
            'total_videos_processed': len(results['processing_details']),
            'total_execution_time': total_execution_time,
            'total_execution_time_minutes': total_execution_time / 60
        }
    
    print(f'{"="*60}\n')
    
    # Save summary to file
    summary_path = os.path.join(save_history_path, "processing_summary.json")
    
    # Enrich the summary with detailed information
    detailed_summary = {
        'execution_info': {
            'start_time': execution_start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': execution_end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'total_execution_time_seconds': total_execution_time,
            'total_execution_time_minutes': total_execution_time / 60,
            'total_videos': len(video_list),
            'completed': len(results['completed']),
            'oom_errors': len(results['oom_errors']),
            'other_errors': len(results['other_errors']),
            'video_list': video_list,
            'save_path': save_history_path
        },
        'statistics': results.get('statistics', {}),
        'video_details': []
    }
    
    # Add detailed info for each video
    for vid in video_list:
        video_info = {
            'video_uid': vid,
            'status': 'completed' if vid in results['completed'] else ('oom' if vid in results['oom_errors'] else 'error'),
            'processing_time_seconds': results['video_timings'].get(vid, 0)
        }
        
        if vid in results['processing_details']:
            details = results['processing_details'][vid]
            video_info.update({
                'processed_duration': details['processed_duration'],
                'total_duration': details['total_duration'],
                'completion_rate': details['completion_rate']
            })
        
        # Check if there's error info
        for error_info in results['other_errors']:
            if error_info['video_uid'] == vid:
                video_info['error'] = error_info['error']
                break
        
        detailed_summary['video_details'].append(video_info)
    
    # Add lists for quick reference
    detailed_summary['completed_videos'] = results['completed']
    detailed_summary['oom_videos'] = results['oom_errors']
    detailed_summary['error_videos'] = [e['video_uid'] for e in results['other_errors']]
    
    # Add timing summary
    detailed_summary['timing_summary'] = {
        'per_video_processing_time': results['video_timings'],
        'total_processing_time': sum(results['video_timings'].values()),
        'average_processing_time': sum(results['video_timings'].values()) / len(results['video_timings']) if results['video_timings'] else 0
    }
    
    json.dump(detailed_summary, open(summary_path, 'w'), indent=4)
    print(f"Summary saved to {summary_path}")