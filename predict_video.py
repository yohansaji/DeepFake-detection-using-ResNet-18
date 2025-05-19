import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import get_model
import os
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

model = get_model().to(device)
model.load_state_dict(torch.load("resnet_deepfake.pt"))
model.eval()

def predict_video(video_path, frame_skip=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    probs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.sigmoid(output).item()
                probs.append(prob)

        frame_count += 1
    cap.release()

    if not probs:
        print(f"No frames processed in '{os.path.basename(video_path)}'")
        return "No Prediction"

    video_score = np.mean(probs)
    label = "FAKE" if video_score > 0.5 else "REAL"
    print(f"\n Prediction for '{os.path.basename(video_path)}'")
    print(f"Confidence Score: {video_score:.4f}")
    print(f"Predicted Label: {label}")

    return label

def is_video_file(filename):
    return filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python predict_video.py path_to_video.mp4")
        print("  python predict_video.py video1.mp4 video2.avi ...")
        print("  python predict_video.py /path/to/video/folder/")
    else:
        path = sys.argv[1]
        if os.path.isdir(path):
            video_files = [f for f in glob.glob(os.path.join(path, "*")) if is_video_file(f)]
            print(f"\n Found {len(video_files)} video(s) in folder: '{path}'")
            if not video_files:
                print("No video files found in the folder.")
            else:
                for video_file in video_files:
                    predict_video(video_file)
        else:
            for video_file in sys.argv[1:]:
                if os.path.isfile(video_file) and is_video_file(video_file):
                    predict_video(video_file)
                else:
                    print(f" Invalid video file: {video_file}")






