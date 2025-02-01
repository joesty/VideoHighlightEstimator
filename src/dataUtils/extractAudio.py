import os
import numpy as np
import argparse 
import ffmpeg
def extract_audio(input_path, file_name, output_path, audio_extractor='python'):
    #file_name = file_name
    input_f = input_path + '/' + file_name
    file_name = file_name.replace(' ', '_').replace('.mp4', '')
    try:
        output_f_1 = output_path + '/' + file_name + '_intermediate.wav'
        output_f_2 = output_path + '/' + file_name + '.wav'
        if audio_extractor == 'python':
            ffmpeg.input(input_f).output(output_f_2, ac=1, ar=16000).run() 
        else:
            print(input_f)
            
            os.system('ffmpeg -i {:s} -vn -ar 16000 ac 1 {:s}'.format(input_f, output_f_1)) # save an intermediate file
            os.system('sox {:s} {:s} remix 1'.format(output_f_1, output_f_2))
            os.remove(output_f_1)
    except ValueError as e:
        print('Error: ', e)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract audio from video files')
    parser.add_argument('--input_path', type=str, default='../datasets/SumMe/videos', help='Path to the video files')
    parser.add_argument('--output_path', type=str, default='../datasets/SumMe/audios', help='Path to save the audio files')
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    for file_name in os.listdir(args.input_path):
        if file_name.endswith('.mp4'):
            extract_audio(args.input_path, file_name, args.output_path)
    print('Done!')
    