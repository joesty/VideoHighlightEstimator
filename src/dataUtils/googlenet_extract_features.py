import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from PIL import Image
import cv2
import json
import os
from tqdm import tqdm
import numpy as np
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Extract features from pool5
weights = models.GoogLeNet_Weights.DEFAULT
model = models.googlenet(weights=weights)
model = model.to(device)
model.eval()
feature_extractor = torch.nn.Sequential(*list(model.children())[:-2]).to(device)



def preprocess_image(image_path):
    # Open image using PIL and apply transformations
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match GoogLeNet input size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

def extract_features_from_image(image_path):
    # Preprocess image
    input_tensor = preprocess_image(image_path)

    # Extract features using the feature extractor
    with torch.no_grad():
        feature = feature_extractor(input_tensor)
    
    return feature.squeeze().cpu().numpy()  # Return the feature as a NumPy array

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_path', type=str, default='/mnt/j/datasets/HiSum/frames/')
    args = argparser.parse_args()
    frames_path = args.input_path
    videos = os.listdir(frames_path)
    os.makedirs('../datasets/HiSum/features', exist_ok=True)
    video_fatures = os.listdir('../datasets/HiSum/features')
    for video in tqdm(videos):
        try:
            frame_features = {}
            if f'{video}.json' not in video_fatures:
                video_features = []
                video_path = os.path.join(frames_path, video)
                frames = os.listdir(video_path)
                for frame in frames:
                    image_path = os.path.join(video_path, frame)
                    features = extract_features_from_image(image_path)
                    video_features.append(features)
                frame_features[str(video)] = video_features
                with open(f'../datasets/HiSum/features/{video}.json', 'w') as f:
                    json.dump(frame_features, f, cls=NumpyEncoder) # Use json.dump to write directly to file
                print(f'Finished {video}')
                #frame_features = {}
        except:
            print(f'Error in {video}')
            continue