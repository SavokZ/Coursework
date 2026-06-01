import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import numpy as np
import os

SEED_FILE = 'seed.txt'
if os.path.exists(SEED_FILE):
    with open(SEED_FILE, 'r') as f:
        seed = int(f.read().strip())
else:
    seed = random.randint(0, 2**32 - 1)
    with open(SEED_FILE, 'w') as f:
        f.write(str(seed))

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('data', train=False, transform=transform)

subset = Subset(train_data, range(10000))
train_sub, val_sub = random_split(subset, [8000, 2000])
train_loader = DataLoader(train_sub, batch_size=64, shuffle=True)
val_loader = DataLoader(val_sub, batch_size=256)
test_loader = DataLoader(test_data, batch_size=256)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

criterion = nn.CrossEntropyLoss()

print("\n=== No regularization ===\n")
model = MLP()
opt = optim.Adam(model.parameters(), lr=0.001)
train_losses, val_losses = [], []

for epoch in range(30):
    model.train()
    loss_sum = 0
    for x, y in train_loader:
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        loss_sum += loss.item()
    train_losses.append(loss_sum / len(train_loader))
    
    model.eval()
    with torch.no_grad():
        val_sum = 0
        for x, y in val_loader:
            val_sum += criterion(model(x), y).item()
        val_losses.append(val_sum / len(val_loader))
    print(f"Epoch {epoch+1:3d} | Train loss: {train_losses[-1]:.4f} | Val loss: {val_losses[-1]:.4f}")

patience = 3
hold_stop = len(val_losses)
for i in range(patience, len(val_losses)):
    if all(val_losses[i-j] >= val_losses[i-j-1] for j in range(patience)):
        hold_stop = i - patience + 1
        break

ratio_stop = next((i for i in range(5, len(val_losses)) if abs(val_losses[i] - val_losses[i-3]) / val_losses[i-3] < 0.02), len(val_losses))
apriori_stop = 15

print(f"\nHold-out: epoch {hold_stop} | Ratio: epoch {ratio_stop} | Apriori: epoch {apriori_stop}")

print("\n=== L2 regularization ===\n")
model_l2 = MLP()
opt_l2 = optim.Adam(model_l2.parameters(), lr=0.001, weight_decay=1e-5)
train_losses_l2, val_losses_l2 = [], []

for epoch in range(30):
    model_l2.train()
    loss_sum = 0
    for x, y in train_loader:
        opt_l2.zero_grad()
        loss = criterion(model_l2(x), y)
        loss.backward()
        opt_l2.step()
        loss_sum += loss.item()
    train_losses_l2.append(loss_sum / len(train_loader))
    
    model_l2.eval()
    with torch.no_grad():
        val_sum = 0
        for x, y in val_loader:
            val_sum += criterion(model_l2(x), y).item()
        val_losses_l2.append(val_sum / len(val_loader))
    print(f"Epoch {epoch+1:3d} | Train loss: {train_losses_l2[-1]:.4f} | Val loss: {val_losses_l2[-1]:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(val_losses, label='No reg (val loss)')
plt.plot(val_losses_l2, label='L2 reg (val loss)')
plt.axvline(hold_stop, color='b', linestyle='--', label='Hold-out')
plt.axvline(ratio_stop, color='g', linestyle='--', label='Ratio')
plt.axvline(apriori_stop, color='r', linestyle='--', label='Apriori')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('mlp_training.png')
plt.show()
