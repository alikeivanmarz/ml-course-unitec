# PyTorch Reference

PyTorch is a deep learning framework that exposes tensor computation, automatic differentiation, and neural network primitives as composable Python objects. This guide covers the building blocks needed to define models, train them, save them, and run them on GPU or Apple Silicon hardware. Code examples assume `torch` and `torchvision` are installed.

**Table of Contents**

1. [Tensors and Devices](#1-tensors-and-devices)
2. [Autograd Basics](#2-autograd-basics)
3. [Building Models with `nn.Module`](#3-building-models-with-nnmodule)
4. [Common Layer Reference](#4-common-layer-reference)
5. [Loss Functions and Optimizers](#5-loss-functions-and-optimizers)
6. [The Training Loop](#6-the-training-loop)
7. [Datasets and DataLoaders](#7-datasets-and-dataloaders)
8. [Saving and Loading Models](#8-saving-and-loading-models)
9. [Mixed Precision and GPU Memory](#9-mixed-precision-and-gpu-memory)
10. [Resources](#10-resources)

---

## 1. Tensors and Devices

### 1.1 Creating Tensors

```python
import torch

torch.tensor([1.0, 2.0, 3.0])             # from a Python list
torch.zeros(3, 4)                         # all zeros, shape (3, 4)
torch.ones(2, 2)                          # all ones
torch.randn(5, 3)                         # standard normal
torch.arange(0, 10, step=2)               # [0, 2, 4, 6, 8]
torch.eye(4)                              # 4x4 identity
```

A tensor's three defining attributes are `shape`, `dtype`, and `device`.

```python
x = torch.randn(3, 4, dtype=torch.float32)
x.shape       # torch.Size([3, 4])
x.dtype       # torch.float32
x.device      # device(type='cpu')
```

### 1.2 Device Selection

```python
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():    # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()

x = torch.randn(3, 4).to(device)
model = MyModel().to(device)
```

Tensors and models must be on the same device for operations to succeed. A `RuntimeError: Expected all tensors to be on the same device` indicates a mismatch.

### 1.3 NumPy Interoperability

```python
import numpy as np

a = np.array([[1.0, 2.0], [3.0, 4.0]])
t = torch.from_numpy(a)         # shares memory with the NumPy array
back = t.numpy()                # only valid for CPU tensors
```

Tensors created from NumPy arrays via `torch.from_numpy` share storage; modifications propagate in both directions. Use `.clone()` to break the link.

---

## 2. Autograd Basics

PyTorch builds a dynamic computation graph as operations execute. Tensors with `requires_grad=True` are tracked; calling `.backward()` computes gradients via reverse-mode automatic differentiation.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x
y.backward()
x.grad         # tensor(14.) -- dy/dx = 3x^2 + 2 evaluated at x=2
```

For vector-valued outputs, `.backward()` requires a gradient argument matching the output shape; in practice, scalar losses are summed first.

### 2.1 Disabling Gradient Tracking

```python
with torch.no_grad():
    predictions = model(x)         # no graph built; faster, less memory

# Inference shortcut
model.eval()
with torch.inference_mode():
    predictions = model(x)         # stricter than no_grad; preferred for inference
```

### 2.2 Detaching from the Graph

```python
y_detached = y.detach()           # new tensor with no grad history
```

Detaching is required when a tensor's value is needed for downstream non-differentiable use (logging, plotting, conversion to NumPy).

---

## 3. Building Models with `nn.Module`

A model is a subclass of `torch.nn.Module` with two key methods: `__init__` (declare layers) and `forward` (define the computation).

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = MLP(in_dim=20, hidden=64, out_dim=3)
out = model(torch.randn(8, 20))   # shape (8, 3)
```

Layers declared in `__init__` are automatically registered as parameters and moved with `.to(device)`. Computations placed only inside `forward` are not tracked as parameters.

### 3.1 Sequential Models

For purely sequential architectures, `nn.Sequential` removes boilerplate:

```python
model = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 3),
)
```

---

## 4. Common Layer Reference

| Layer | Purpose | Notes |
|-------|---------|-------|
| `nn.Linear(in, out)` | Fully connected | Equivalent to Keras `Dense` |
| `nn.Conv2d(in_ch, out_ch, kernel)` | 2D convolution | Input shape: `(N, C, H, W)` |
| `nn.MaxPool2d(kernel)` / `nn.AvgPool2d(kernel)` | Downsampling | Common after conv |
| `nn.BatchNorm2d(num_features)` | Channel-wise batch normalization | Use `nn.BatchNorm1d` for FC layers |
| `nn.LayerNorm(normalized_shape)` | Layer normalization | Standard in transformers |
| `nn.Dropout(p)` | Regularization | Active only in `model.train()` mode |
| `nn.Embedding(num_embeddings, dim)` | Lookup table | Used for token / categorical embeddings |
| `nn.LSTM(input_size, hidden_size)` | Recurrent layer | Returns output and `(h, c)` tuple |
| `nn.MultiheadAttention(embed_dim, num_heads)` | Scaled dot-product attention | Building block for transformers |
| `nn.Transformer(...)` | Full transformer encoder/decoder | High-level wrapper |

Activations (`F.relu`, `F.gelu`, `F.softmax`) are typically applied as functions in `forward` rather than declared as layers.

---

## 5. Loss Functions and Optimizers

### 5.1 Common Losses

| Task | Loss | Notes |
|------|------|-------|
| Binary classification | `nn.BCEWithLogitsLoss` | Combines sigmoid + BCE; numerically stable |
| Multi-class classification | `nn.CrossEntropyLoss` | Combines log-softmax + NLL; expects raw logits |
| Regression | `nn.MSELoss` or `nn.L1Loss` | L1 is robust to outliers |
| Sequence prediction | `nn.CTCLoss` | Speech, handwriting recognition |

`CrossEntropyLoss` in PyTorch expects integer class labels (not one-hot) and raw logits (not softmax outputs).

### 5.2 Optimizers

```python
from torch.optim import SGD, Adam, AdamW

optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

| Optimizer | Default choice for |
|-----------|--------------------|
| `SGD(momentum=0.9)` | Image classification with established schedules |
| `Adam` | General-purpose; common starting point |
| `AdamW` | Transformer training; correct weight decay |
| `RMSprop` | Recurrent networks |

Learning-rate schedulers live in `torch.optim.lr_scheduler` (e.g., `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`).

---

## 6. The Training Loop

PyTorch does not provide a `.fit()` method. The training loop is written explicitly, which makes the steps visible but requires more code than Keras.

```python
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()        # reset gradients
        logits = model(x)            # forward
        loss = loss_fn(logits, y)    # compute loss
        loss.backward()              # backward
        optimizer.step()             # update parameters

        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        count += x.size(0)
    return total_loss / count, correct / count
```

The four-line core — `zero_grad`, forward, `backward`, `step` — is invariant across most training scripts.

---

## 7. Datasets and DataLoaders

### 7.1 Built-In Datasets

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
```

### 7.2 Custom Dataset

```python
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

Two methods are required: `__len__` and `__getitem__`. `__getitem__` returns a single example.

### 7.3 DataLoader

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_set, batch_size=64, shuffle=True,
    num_workers=4, pin_memory=True,
)
```

`num_workers > 0` enables multi-process data loading (may not work in notebooks on Windows). `pin_memory=True` accelerates CPU-to-GPU transfer.

---

## 8. Saving and Loading Models

### 8.1 State Dict (Recommended)

```python
# Save
torch.save(model.state_dict(), "model.pt")

# Load
model = MLP(in_dim=20, hidden=64, out_dim=3)   # architecture must match
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()
```

The state dict stores parameter tensors only — not the model class. The class definition must be available when loading.

### 8.2 Full Model (Pickled)

```python
torch.save(model, "model_full.pt")
model = torch.load("model_full.pt", weights_only=False)
```

Saving the full object pickles the class definition path. Brittle across refactors; prefer state dicts.

### 8.3 TorchScript and ONNX

```python
# TorchScript: serializable representation runnable without Python
scripted = torch.jit.script(model)
scripted.save("model.ptc")

# ONNX: cross-framework export
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=17)
```

TorchScript is appropriate for production C++ deployment; ONNX is appropriate for cross-framework or hardware-accelerated inference.

---

## 9. Mixed Precision and GPU Memory

### 9.1 Automatic Mixed Precision

Mixed precision (FP16 or BF16 alongside FP32) reduces memory and accelerates training on supported hardware.

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

for x, y in loader:
    optimizer.zero_grad()
    with autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
        loss = loss_fn(logits, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 9.2 Memory Management

| Symptom | Cause | Mitigation |
|---------|-------|------------|
| `CUDA out of memory` | Batch too large | Reduce `batch_size`; use gradient accumulation |
| Memory grows across epochs | Tensors retained in Python lists | `.detach().cpu().item()` before storing |
| Slow training despite GPU | CPU-bound data loading | Increase `num_workers`; profile with `torch.profiler` |
| Memory not freed after exception | Cached references | `torch.cuda.empty_cache()` |

Gradient accumulation simulates a larger batch by accumulating gradients across mini-batches:

```python
accum_steps = 4
optimizer.zero_grad()
for i, (x, y) in enumerate(loader):
    loss = loss_fn(model(x), y) / accum_steps
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 10. Resources

- [PyTorch documentation](https://pytorch.org/docs/stable/index.html) — official API reference.
- [PyTorch tutorials](https://pytorch.org/tutorials/) — introductory and topic-specific tutorials.
- [`torch.nn` module reference](https://pytorch.org/docs/stable/nn.html) — every layer and loss class.
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) — DataLoader, AMP, and GPU optimization recipes.
- [Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library* (2019)](https://arxiv.org/abs/1912.01703) — the original PyTorch paper.
