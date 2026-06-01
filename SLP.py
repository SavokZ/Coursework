import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

n, d = 200, 20
X = np.random.randn(n, d)
U, s, Vt = np.linalg.svd(X, full_matrices=False)
s_ill = np.exp(np.linspace(0, -5, d))
X_ill = U @ np.diag(s_ill) @ Vt
w_true = np.random.randn(d)
y = X_ill @ w_true + 0.2 * np.random.randn(n)

X_t = torch.tensor(X_ill, dtype=torch.float32)
y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

n_val = n // 5
X_train, X_val = X_t[n_val:], X_t[:n_val]
y_train, y_val = y_t[n_val:], y_t[:n_val]
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

model = nn.Linear(d, 1)
opt = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

losses = []
for epoch in range(100):
    for xb, yb in loader:
        opt.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        opt.step()
    with torch.no_grad():
        losses.append(criterion(model(X_val), y_val).item())

patience = 3
hold_stop = len(losses)
for i in range(patience, len(losses)):
    if all(losses[i-j] > losses[i-j-1] for j in range(patience)):
        hold_stop = i - patience + 1
        break

ratio_stop = next((i for i in range(5, len(losses)) if abs(losses[i] - losses[i-3]) / losses[i-3] < 0.001), len(losses))
apriori_stop = 15

print(f"Hold-out (3 rises): epoch {hold_stop}, loss {losses[hold_stop-1]:.4f}")
print(f"Ratio (0.1%/3ep): epoch {ratio_stop}, loss {losses[ratio_stop-1]:.4f}")
print(f"Apriori (15): epoch {apriori_stop}, loss {losses[apriori_stop-1]:.4f}")

plt.figure(figsize=(10,5))
plt.plot(losses, 'k-', alpha=0.5, label='Val loss')
plt.axvline(hold_stop, color='b', linestyle='--', label=f'Hold-out (epoch {hold_stop})')
plt.axvline(ratio_stop, color='g', linestyle='--', label=f'Ratio (epoch {ratio_stop})')
plt.axvline(apriori_stop, color='r', linestyle='--', label=f'Apriori (epoch {apriori_stop})')
plt.xlabel('Epoch')
plt.ylabel('Val MSE')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('slp_result.png')
plt.show()
