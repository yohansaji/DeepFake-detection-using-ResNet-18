# Deepfake Detection Using ResNet18

This project focuses on detecting deepfake videos using a fine-tuned ResNet18 model. By classifying individual frames extracted from videos as either **real** or **fake**, the model can effectively assess the authenticity of a video.

## 📁 Project Structure

```
.
├── extract_frames.py       # Extracts frames from real/fake videos
├── train.py                # Trains the ResNet18 model
├── model.py                # Loads and modifies the ResNet18 model
├── utils.py                # Evaluation metrics (Accuracy, F1, AUC, etc.)
├── predict_video.py        # Predicts real/fake label for a video
├── requirements.txt        # Python dependencies
├── resnet_deepfake.pt      # Trained model (saved after training)
├── dataset/                # Extracted and labeled image frames
└── Data/                   # Raw video files (real/fake)
```

## 🎯 Aim

Detect whether a video is real or deepfake by analyzing extracted frames using a deep learning model.

## 🛠️ Methodology

- **Frame Extraction**: Frames are extracted every 30 frames using `extract_frames.py`.
- **Dataset**: Images are labeled into `real` and `fake` directories under `dataset/`.
- **Model**: Pretrained ResNet18 is modified to classify frames as binary output.
- **Loss Function**: `BCEWithLogitsLoss` with class weights for imbalance.
- **Training Strategy**:
  - Balanced training using `WeightedRandomSampler`
  - 10 epochs with `Adam` optimizer and learning rate `1e-4`
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, and Confusion Matrix.

## 📊 Results

**Final Test Set Evaluation:**
- Accuracy: 86.7%
- Precision: 53.9%
- Recall: 65.5%
- F1-score: 59.1%
- AUC-ROC: 91.9%
- Confusion Matrix:
  ```
  [[441, 47],
   [29,  55]]
  ```

## ▶️ Usage

### 1. Extract Frames from Videos

```bash
python extract_frames.py
```

Ensure your video files are organized as:
```
Data/
├── real/
└── fake/
```

### 2. Train the Model

```bash
python train.py
```

This saves the trained model as `resnet_deepfake.pt`.

### 3. Predict on a New Video

```bash
python predict_video.py path_to_video.mp4
```

Outputs the predicted label and confidence score.

## 📦 Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
torch
torchvision
opencv-python
scikit-learn
matplotlib
```

## Reference
-Deepfake Detection with Deep Learning: Convolutional Neural Networks versus Transformers Vrizlynn L. L. Thing 
-link:https://arxiv.org/abs/2304.03698

-Dataset used Link : https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset

## 📌 Future Work

- Use temporal models like 3D CNN or LSTM.
- Explore additional architectures (EfficientNet, ViT).
- Implement more advanced augmentation and adversarial defense.