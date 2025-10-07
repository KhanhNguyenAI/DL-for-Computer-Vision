dataset : https://www.kaggle.com/datasets/alessiocorrado99/animals10

# Deep Learning Tutorial: From Dataset to Deployment

This tutorial walks you through a simple **end-to-end Deep Learning workflow**: setup dataset → train → evaluate → deploy.

## 📂 Project Structure

```bash
├── dataset/              # Your dataset folder
│   ├── train/
│   └── test/
├── model.py              # Model definition
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── app.py                # Deployment (FastAPI)
├── requirements.txt      # Dependencies
└── README.md             # Tutorial file
```

## 🔧 1. Setup Environment

Install required packages:

```bash
pip install -r requirements.txt
```

`requirements.txt`

```txt
torch torchvision torchaudio
scikit-learn
matplotlib
pandas
fastapi uvicorn
```

## 📊 2. Prepare Dataset

Example: Image classification with `dataset/train` and `dataset/test`.

```python
# dataset.py
from torchvision import datasets, transforms

def get_dataloaders(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    train_data = datasets.ImageFolder("dataset/train", transform=transform)
    test_data = datasets.ImageFolder("dataset/test", transform=transform)
    
    from torch.utils.data import DataLoader
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(test_data, batch_size=batch_size)
    )
```

## 🧠 3. Build Model

```python
# model.py
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)
```

## 🚀 4. Train Model

```python
# train.py
import torch, torch.nn as nn, torch.optim as optim
from model import SimpleCNN
from dataset import get_dataloaders

train_loader, test_loader = get_dataloaders()
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleCNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "model.pth")
```

## 📈 5. Evaluate Model

```python
# evaluate.py
import torch
from model import SimpleCNN
from dataset import get_dataloaders
from sklearn.metrics import accuracy_score

_, test_loader = get_dataloaders()
model = SimpleCNN(num_classes=2)
model.load_state_dict(torch.load("model.pth"))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in test_loader:
        outputs = model(X)
        preds = outputs.argmax(1)
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

print("Accuracy:", accuracy_score(all_labels, all_preds))
```

## 🌐 6. Deploy with FastAPI

```python
# app.py
import torch
from fastapi import FastAPI, UploadFile
from PIL import Image
import io
from torchvision import transforms
from model import SimpleCNN

app = FastAPI()

model = SimpleCNN(num_classes=2)
model.load_state_dict(torch.load("model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        preds = model(x)
        label = preds.argmax(1).item()
    return {"prediction": label}
```

Run server:

```bash
uvicorn app:app --reload
```

Send request:

```bash
curl -X POST -F "file=@sample.jpg" http://127.0.0.1:8000/predict
```

---

## ✅ Summary

* **Dataset setup** → `dataset.py`
* **Model definition** → `model.py`
* **Training** → `train.py`
* **Evaluation** → `evaluate.py`
* **Deployment (API)** → `app.py`

This repo is a minimal but complete example for **Deep Learning workflow**. 🎉
