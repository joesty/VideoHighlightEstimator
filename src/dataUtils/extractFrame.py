import os
import argparse
from tqdm import tqdm
import subprocess
import re
import os
import argparse
from tqdm import tqdm
import subprocess
import re

def extract_frames_ffmpeg(video_path, output_path, frames_per_second=1):
    #shutil.rmtree(output_pather)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    #frame_interval = target_fps / frames_per_second
    
    if frames_per_second:
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'fps={str(frames_per_second)}',
            '-q:v', '2',
            os.path.join(output_path, 'frame_%05d.jpg')
        ]
    else:
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vsync', 'vfr',
            '-q:v', '2',
            os.path.join(output_path, 'frame_%05d.jpg')
        ]
    subprocess.run(command, check=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract audio from video files')
    parser.add_argument('--input_path', type=str, default='../datasets/SumMe/videos', help='Path to the video files')
    parser.add_argument('--output_path', type=str, default='../datasets/SumMe/frames', help='Path to save the audio files')
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    videos = os.listdir(input_path)
    for video in tqdm(videos):
        video_file = os.path.join(input_path, video)
        dst_video_dir = os.path.join(output_path, video.split('.mp4')[0])
        os.makedirs(dst_video_dir, exist_ok=True)
        try:
            extract_frames_ffmpeg(video_file, dst_video_dir, frames_per_second=1)
        except:
            print(f'Error: {video_file}')