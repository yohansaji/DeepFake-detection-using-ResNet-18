import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from model import get_model
from utils import compute_metrics
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("dataset", transform=transform)

indices = list(range(len(dataset)))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, stratify=dataset.targets, random_state=42
)

train_dataset = Subset(dataset, train_idx)
test_dataset = Subset(dataset, test_idx)

targets = np.array(dataset.targets)
class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
weights = 1. / class_sample_count
samples_weight = np.array([weights[t] for t in targets])

samples_weight = torch.from_numpy(samples_weight)
train_samples_weight = samples_weight[train_idx]
train_sampler = WeightedRandomSampler(train_samples_weight, len(train_samples_weight), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = get_model().to(device)
pos_weight = torch.tensor([len(dataset.targets) / sum(dataset.targets)]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train_and_validate(model, train_loader, test_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        # === Training ===
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total * 100
        avg_train_loss = train_loss / len(train_loader)

        # === Validation ===
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total * 100
        avg_val_loss = val_loss / len(test_loader)

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"   Train Loss: {avg_train_loss:.4f} | Accuracy: {train_acc:.2f}%")
        print(f"   Val   Loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.2f}%\n")

def evaluate(model, loader):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy().flatten())

    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    print("\n📊 Final Evaluation Metrics on Test Set:")
    for k, v in metrics.items():
        print(f"{k}: \n{v}\n")

train_and_validate(model, train_loader, test_loader, criterion, optimizer, epochs=10)
evaluate(model, test_loader)

torch.save(model.state_dict(), "resnet_deepfake.pt")

