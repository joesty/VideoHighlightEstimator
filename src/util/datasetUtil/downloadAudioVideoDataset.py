from pytube import YouTube
import os
import argparse 
from tqdm import tqdm
from pydub import AudioSegment
from moviepy.editor import AudioFileClip



os.makedirs('../../../data/video', exist_ok=True)
os.makedirs('../../../data/audio', exist_ok=True)


def openFile(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip()

def downloadYoutubeVideo(video_id):
    yt = YouTube(f'https://www.youtube.com/watch?v={video_id}')
    stream = yt.streams.get_highest_resolution()
    stream.download(output_path='../../../data/video', filename=f'{video_id}.mp4')


def downloadYoutubeAudio(video_id):
    yt = YouTube(f'https://www.youtube.com/watch?v={video_id}')
    stream = yt.streams.get_audio_only()
    webm_file = stream.download(output_path='../../../data/audio', filename=video_id)
    wav_file = f'../../../data/audio/{video_id}.wav'
    audio = AudioFileClip(webm_file)
    audio.write_audiofile(wav_file, codec='pcm_s16le')  # codec for wav format
    os.remove(webm_file)  # remove the original webm file



if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--video_id')
    argparse.add_argument('--video_file')
    argparse.add_argument('--download_type', default='video', choices=['video', 'audio'])
    args = argparse.parse_args()
    try:
        if args.video_id:
            if args.download_type == 'audio':
                downloadYoutubeAudio(args.video_id)
            else:
                downloadYoutubeVideo(args.video_id)
        elif args.video_file:
            open_file = openFile(args.video_file)
            for video_id in tqdm(open_file):
                if args.download_type == 'audio':
                    downloadYoutubeAudio(video_id)
                else:
                    downloadYoutubeVideo(video_id)
        else:
            print("Please provide either video_id or video_file path.")
    except:
        print("Error in downloading the video/audio. Please check the video_id or video_file path.")
