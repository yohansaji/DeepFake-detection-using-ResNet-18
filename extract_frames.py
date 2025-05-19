import os
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_frames(video_path, output_folder, frame_skip=30):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    count, saved = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_skip == 0:
            out_path = os.path.join(output_folder, f"{video_name}_frame{saved}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        count += 1
    cap.release()
    return f"{video_name}: {saved} frames saved"

def extract_frames_from_videos(video_dir, output_dir, frame_skip=30, num_workers=4):
    os.makedirs(output_dir, exist_ok=True)
    jobs = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for label in ['real', 'fake']:
            input_folder = os.path.join(video_dir, label)
            output_folder = os.path.join(output_dir, label)
            os.makedirs(output_folder, exist_ok=True)

            for video_file in os.listdir(input_folder):
                if not video_file.endswith(('.mp4', '.avi', '.mov')):
                    continue
                video_path = os.path.join(input_folder, video_file)
                jobs.append(executor.submit(extract_frames, video_path, output_folder, frame_skip))

        for future in as_completed(jobs):
            print(future.result())

if __name__ == "__main__":
    extract_frames_from_videos("Data", "dataset", frame_skip=30, num_workers=8)




