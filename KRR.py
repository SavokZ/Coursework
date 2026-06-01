import numpy as np
import matplotlib.pyplot as plt
import random
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

n_train = 80
x_train = np.random.uniform(-3, 3, n_train)
y_train = np.sin(x_train) + 0.15 * np.random.randn(n_train)
x_test = np.linspace(-3, 3, 200)
y_true = np.sin(x_test)

def rbf_kernel(x1, x2, sigma=1.0):
    return np.exp(-(x1[:, None] - x2[None, :])**2 / (2 * sigma**2))

K = rbf_kernel(x_train, x_train)
K_test = rbf_kernel(x_test, x_train)

n_val = n_train // 5
idx = np.random.permutation(n_train)
train_idx, val_idx = idx[n_val:], idx[:n_val]
K_train = K[train_idx][:, train_idx]
K_val = K[val_idx][:, train_idx]
y_train_sub = y_train[train_idx]
y_val = y_train[val_idx]

c = np.zeros(len(train_idx))
lr = 0.05
test_losses, val_losses = [], []

for t in range(200):
    grad = K_train @ c - y_train_sub
    c -= lr * grad
    test_losses.append(np.mean((K_test[:, train_idx] @ c - y_true)**2))
    val_losses.append(np.mean((K_val @ c - y_val)**2))

patience = 3
hold_stop = len(val_losses)
for i in range(patience, len(val_losses)):
    if all(val_losses[i-j] > val_losses[i-j-1] for j in range(patience)):
        hold_stop = i - patience + 1
        break

ratio_stop = next((i for i in range(10, len(val_losses)) if abs(val_losses[i] - val_losses[i-5]) / val_losses[i-5] < 0.05), len(val_losses))
apriori_stop = 40

print(f"Hold-out (3 rises): epoch {hold_stop}, test loss {test_losses[hold_stop-1]:.4f}")
print(f"Ratio (5%/5ep): epoch {ratio_stop}, test loss {test_losses[ratio_stop-1]:.4f}")
print(f"Apriori (40): epoch {apriori_stop}, test loss {test_losses[apriori_stop-1]:.4f}")

plt.figure(figsize=(10,5))
plt.plot(test_losses, 'b-', alpha=0.7, label='Test MSE')
plt.plot(val_losses, 'orange', alpha=0.7, label='Validation MSE')
plt.axvline(hold_stop, color='b', linestyle='--', label=f'Hold-out (epoch {hold_stop})')
plt.axvline(ratio_stop, color='g', linestyle='--', label=f'Ratio (epoch {ratio_stop})')
plt.axvline(apriori_stop, color='r', linestyle='--', label=f'Apriori (epoch {apriori_stop})')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('krr_result.png')
plt.show()
