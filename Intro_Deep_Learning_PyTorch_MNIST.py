"""
Intro to Deep Learning with PyTorch — MNIST example
File: Intro_Deep_Learning_PyTorch_MNIST.py

What this file contains:
- Step-by-step code (with comments) to download MNIST, build a simple feedforward NN,
  train it for 5 epochs, evaluate accuracy, and optionally visualize predictions.
- Optional toggles to use Dropout or a small CNN instead of the feedforward net.
- GPU detection and safe fallback to CPU.

How to run (step-by-step):
1) Install Python 3.8+ (https://www.python.org/downloads/). Use 3.x.
2) Make a virtual environment (recommended):
   - Windows:  python -m venv venv
             
   - macOS/Linux: python3 -m venv venv
                  source venv/bin/activate
3) Upgrade pip: pip install --upgrade pip
4) Install packages:
   - CPU-only (simplest):
       pip install torch torchvision matplotlib
     (If you need GPU/CUDA support, visit https://pytorch.org/get-started/locally/ for the
      exact command that matches your CUDA version.)
5) Save this file and run:
   python Intro_Deep_Learning_PyTorch_MNIST.py


Expected output: training loss printed per epoch and final test accuracy (usually ~95-98% for
this simple network using Adam over 5 epochs). Results vary by random seed and optimizer.

====================================================================================
"""

# --------------------------- Imports & Setup ---------------------------------
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Optional: for visualizing predictions (matplotlib may not be available in some envs)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# --------------------------- Reproducibility ---------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------- Device selection --------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
if DEVICE.type == 'cuda':
    try:
        print("GPU:", torch.cuda.get_device_name(0))
    except Exception:
        pass

# --------------------------- Hyperparameters ---------------------------------
BATCH_SIZE = 64
LR = 1e-3            # learning rate (you can try 0.01 for SGD or 1e-3 for Adam)
EPOCHS = 5
NUM_WORKERS = 2      # set to 0 on Windows if you see issues

# Toggle options
USE_DROPOUT = False     # Try True to enable dropout in the feedforward net
USE_CNN = False         # Set True to use the small CNN defined below
OPTIMIZER = 'adam'      # choose 'adam' or 'sgd'

# --------------------------- Data: Transforms & Loading ----------------------
# MNIST mean and std (for 1 channel) commonly used
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download datasets (will be saved in ./data)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=NUM_WORKERS)

# --------------------------- Define Models -----------------------------------
class FeedForwardNet(nn.Module):
    """Simple feedforward neural network: 784 -> 128 -> 10"""
    def __init__(self, hidden_dim=128, use_dropout=False, p_drop=0.5):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, hidden_dim)
        self.relu = nn.ReLU()
        self.use_dropout = use_dropout
        if use_dropout:
            self.drop = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(hidden_dim, 10)
        self.logsoft = nn.LogSoftmax(dim=1)   # for use with NLLLoss

    def forward(self, x):
        x = self.flatten(x)           # shape: (batch, 784)
        x = self.fc1(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.drop(x)
        x = self.fc2(x)
        x = self.logsoft(x)
        return x


class SimpleCNN(nn.Module):
    """Small CNN: conv -> relu -> pool -> fc"""
    def __init__(self):
        super().__init__()
        # Input: 1 x 28 x 28
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # -> 16 x 28 x 28
        self.pool = nn.MaxPool2d(2,2)                                       # -> 16 x 14 x 14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # -> 32 x 14 x 14
        # after pool -> 32 x 7 x 7
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.logsoft(x)
        return x

# Choose model based on toggle
if USE_CNN:
    model = SimpleCNN()
else:
    model = FeedForwardNet(use_dropout=USE_DROPOUT)

model = model.to(DEVICE)
print(model)

# --------------------------- Loss & Optimizer --------------------------------
criterion = nn.NLLLoss()   # since last layer is LogSoftmax
if OPTIMIZER.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=LR)
else:
    optimizer = optim.SGD(model.parameters(), lr=LR)

# --------------------------- Training & Evaluation ---------------------------

def train_one_epoch(epoch_index):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # Print every 200 batches
        if (batch_idx + 1) % 200 == 0:
            print(f"Epoch {epoch_index+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {running_loss/200:.4f}")
            running_loss = 0.0


def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    acc = correct / total
    print(f"Test Accuracy: {acc*100:.2f}% ({correct}/{total})")
    return acc


# Optional: visualize a few predictions
def visualize_predictions(n=8):
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available — skipping visualization")
        return
    model.eval()
    images = []
    labels = []
    preds = []
    with torch.no_grad():
        for data, target in test_loader:
            images.append(data)
            labels.append(target)
            output = model(data.to(DEVICE))
            preds.append(output.argmax(dim=1).cpu())
            break
    imgs = images[0][:n]
    labs = labels[0][:n]
    prd  = preds[0][:n]
    imgs = imgs.cpu().numpy()

    fig, axes = plt.subplots(1, n, figsize=(n*2,2))
    for i in range(n):
        ax = axes[i]
        ax.imshow(imgs[i].squeeze(), cmap='gray')
        ax.axis('off')
        ax.set_title(f"P:{int(prd[i])}\nT:{int(labs[i])}")
    plt.show()


# --------------------------- Main execution ---------------------------------

def main():
    print("Starting training...")
    for epoch in range(EPOCHS):
        train_one_epoch(epoch)
        print(f"Completed epoch {epoch+1}/{EPOCHS}")
    print("Training finished. Evaluating on test set...")
    acc = evaluate()
    # Save model state
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("Model saved to mnist_model.pth")

    # Optional visualization
    visualize_predictions(n=8)


if __name__ == '__main__':
    main()

# --------------------------- Notes & Next steps ------------------------------
# - To enable Dropout: set USE_DROPOUT = True near the top.
# - To use the CNN model: set USE_CNN = True and increase LR/EPOCHS if desired.
# - To use GPU: ensure CUDA is installed and torch has CUDA support; the script auto-detects GPU.
# - To try different optimizers:
#       set OPTIMIZER = 'sgd' or 'adam'
# - To change batch size, learning rate, epochs: modify BATCH_SIZE, LR, EPOCHS constants.
# - If you get errors about num_workers on Windows, set NUM_WORKERS = 0.
# - For reproducible experiments, you may also call torch.backends.cudnn.deterministic = True
#   and torch.backends.cudnn.benchmark = False but this may slow down GPU performance.


