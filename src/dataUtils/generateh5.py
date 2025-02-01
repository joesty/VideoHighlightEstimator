import h5py
import json
import pandas as pd
import os
import numpy as np
import random
from tqdm import tqdm
hisum_completo = h5py.File('../datasets/HiSum/mr_hisum.h5', 'r')
hisum_completo.keys()
#metadata = pd.read_csv('../datasets/HiSum/metadata.csv')  
feats = os.listdir('../datasets/HiSum/features')
with open(f'../paExtraction/src/HiSumPlus_pas.json', 'r') as f:
    hisumpas = json.load(f)
hisumpas
split = ""
ff = []
for feat in feats:
    with open(f'../datasets/HiSum/features/{feat}', 'r') as f:
        data = json.load(f)
    frame_size = (len(data[feat.replace('.json', '')]))
    if frame_size != len(hisum_completo[feat.replace('.json', '')]['gtscore'][()]):
        data[feat.replace('.json', '')].append(data[feat.replace('.json', '')][-1])
        with open(f'../datasets/HiSum/features/{feat}', 'w') as f:
            json.dump(data, f)
        with open(f'../datasets/HiSum/features/{feat}', 'r') as f:
            data = json.load(f)
        frame_size = (len(data[feat.replace('.json', '')]))
        print("Diferente")
    ff.append({
        'video_id': feat.replace('.json', ''),
        'frame_size': frame_size
    })
ff = pd.DataFrame(ff)
ff
import os
videos = os.listdir('/mnt/j/datasets/videos')
invalid_videos = []
for video in videos:
    invalid_videos.append(video.replace('r_', '').replace('r', '').replace('.mp4', ''))
invalid_videos
feats = list(ff[(~ff['video_id'].isin(invalid_videos))]['video_id'].values)
feats
print(feats)
random.seed(2048)
dataset_size = 350
feats = feats[:dataset_size]
for i in range(5):
    random.shuffle(feats)
    train = []
    test = []
    split = split+f"HiSum/split{i}/"
    for idx, v in enumerate(feats):
        if idx < int(0.8*dataset_size):
            video_id = v.split('.json')[0]
            train.append(video_id)
            split = split+f"{video_id},"
        elif idx >= int(0.8*dataset_size) and idx < dataset_size:
            if idx == dataset_size:
                split = split[:-1]+'/'
            video_id = v.split('.json')[0]
            test.append(video_id)
            split = split+f"{video_id},"
    split = split[:-1]+"\n"
with open(f'../CSTA/splits/HiSum350_splits.txt', 'w') as f:
    f.write(split[:-1])
len(feats)
hisum_350 = h5py.File('../CSTA/data/eccv16_dataset_hisum350_google_pool5.h5', 'w')
def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

for feat in tqdm(feats):
    video_id = feat.split('.json')[0]
    with open(f'../datasets/HiSum/features/{feat}.json', 'r') as f:
        data = json.load(f)
        
    n_frames = 30*len(hisum_completo[video_id]['gtscore'][()])
    picks = [i for i in range(0, n_frames, 30)]
    #youtube_id = metadata[metadata['video_id'] == video_id]['youtube_id'].values[0]
    hisum_350.create_group(video_id)
    hisum_350[video_id].create_dataset('features', data=np.array(data[video_id]))
    hisum_350[video_id].create_dataset('gtscore', data=np.array(hisum_completo[video_id]['gtscore'][()]))
    hisum_350[video_id].create_dataset('change_points', data=np.array(hisum_completo[video_id]['change_points'][()]))
    hisum_350[video_id].create_dataset('gtsummary', data=np.array(hisum_completo[video_id]['gt_summary'][()]))
    hisum_350[video_id].create_dataset('n_frames', data=np.array(n_frames))
    hisum_350[video_id].create_dataset('picks', data=np.array(picks))
    #hisum_350[video_id].create_dataset('video_name', data=youtube_id)
    hisum_350[video_id].create_dataset('pas', data=np.array(hisumpas[video_id]))
    hisum_350[video_id].create_dataset('pascore', data=min_max_normalize(np.array(hisumpas[video_id])))
hisum_350.close()
        