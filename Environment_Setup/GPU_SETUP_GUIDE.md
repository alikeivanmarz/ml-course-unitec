# GPU Environment Setup Guide
**Machine Learning Course - Windows 11 + RTX 5070 Ti**

## Prerequisites

### System Requirements
- **OS**: Windows 11 (64-bit)
- **GPU**: GeForce RTX 5070 Ti 
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ free space for environment
- **NVIDIA Driver**: Latest Game Ready Driver (536.0+)

### Software Requirements
- **Anaconda/Miniconda**: Latest version
- **VSCode**: Latest version with Python extension
- **Git**: For repository management

## Step 1: NVIDIA Driver Installation

### Download and Install
1. Visit [NVIDIA GeForce Drivers](https://www.nvidia.com/drivers/)
2. Select: **GeForce RTX 5070 Ti** → **Windows 11**
3. Download **Game Ready Driver** (not Studio)
4. Run installer with **Custom Installation** → **Clean Install**
5. **Restart system** after installation

### Verification
```powershell
# Open Command Prompt and run:
nvidia-smi
```
Should display GPU information and driver version.

## Step 2: Environment Setup

### Create GPU Environment
```powershell
# Navigate to course directory
cd "path\to\Machine_Learning_Course_Unitec"

# Create environment (this may take 10-15 minutes)
conda env create -f Environment_Setup/mlcourse-gpu.yml

# Activate environment
conda activate mlcourse-gpu
```

### Register Jupyter Kernel
```powershell
# CRITICAL: Must be run after activating environment
conda activate mlcourse-gpu
python -m ipykernel install --user --name mlcourse-gpu --display-name "ML Course GPU (Python 3.11)"
```

## Step 3: Verification

### Run GPU Verification Script
```powershell
conda activate mlcourse-gpu
python Environment_Setup/gpu_verification.py
```

### Expected Output
```
🎉 SUCCESS: GPU environment is properly configured!
✅ NVIDIA Driver: Working
✅ CUDA Runtime: Available  
✅ PyTorch GPU: Functional
✅ TensorFlow GPU: Functional
🚀 Ready for GPU-accelerated deep learning!
```

## Step 4: VSCode Configuration

### Open Project
```powershell
# From course directory
code .
```

### Configure Python Interpreter
1. Press `Ctrl+Shift+P`
2. Type: **"Python: Select Interpreter"**
3. Choose: **"ML Course GPU (Python 3.11)" (mlcourse-gpu)**

### Kernel Selection in Notebooks
1. Open any `.ipynb` file
2. Click kernel selector (top right)
3. Select: **"ML Course GPU (Python 3.11)"**

## Daily Usage Workflow

### Starting Work Session
```powershell
# 1. Navigate to course directory
cd "path\to\Machine_Learning_Course_Unitec"

# 2. Get latest updates
git pull origin main

# 3. Activate GPU environment
conda activate mlcourse-gpu

# 4. Open VSCode
code .
```

### In VSCode/Jupyter
```python
# Always start notebooks with GPU verification
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Set random seeds for reproducibility
import numpy as np
import torch
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
```

## Performance Optimization

### Windows GPU Settings
1. **Windows Settings** → **System** → **Display** → **Graphics**
2. **Hardware-accelerated GPU scheduling**: **On**
3. **Variable refresh rate**: **On**

### CUDA Memory Management
```python
# At start of training scripts
import torch
torch.cuda.empty_cache()

# For large models, enable memory fraction
torch.cuda.set_per_process_memory_fraction(0.8)

# Monitor GPU memory
print(f"Memory allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
```

### Jupyter Kernel Management
```powershell
# List available kernels
jupyter kernelspec list

# Remove old kernel if needed
jupyter kernelspec remove mlcourse-gpu

# Reinstall kernel
python -m ipykernel install --user --name mlcourse-gpu --display-name "ML Course GPU (Python 3.11)"
```

## Assignment Workflow

### File Setup
```python
# Copy assignment template and rename to:
# StudentID_YourName_Assignment_X.ipynb

# Cell 1: Student Information
"""
Student Name: Your Full Name  
Student ID: 1234567
Assignment: 2 - Neural Networks
Date: 2024-XX-XX
GPU Used: RTX 5070 Ti
"""

# Cell 2: Environment Setup
import torch
import numpy as np

# Set random seed (use last 2 digits of your student ID)
SEED = 67  # Replace with YOUR last 2 digits
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# Verify GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
```

## Common GPU Workflows

### Training Neural Networks
```python
# Move model and data to GPU
model = MyModel().to(device)
data = data.to(device)
target = target.to(device)

# Training loop with GPU
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Optional: Clear cache periodically
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
```

### Computer Vision with GPU
```python
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# GPU-optimized data loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Use pin_memory for faster GPU transfer
train_loader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True, 
    pin_memory=True,  # Important for GPU
    num_workers=4
)
```

### Memory Management
```python
# Monitor memory usage
def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

# Clear cache when needed
torch.cuda.empty_cache()

# Use gradient checkpointing for large models
model = torch.utils.checkpoint.checkpoint_sequential(model, segments=2)
```

## Troubleshooting

### Environment Issues
```powershell
# Environment not working? Recreate it:
conda env remove -n mlcourse-gpu
conda clean --all
conda env create -f Environment_Setup/mlcourse-gpu.yml
```

### Kernel Issues
```powershell
# Kernel not appearing in VSCode?
conda activate mlcourse-gpu
python -m ipykernel install --user --name mlcourse-gpu --display-name "ML Course GPU (Python 3.11)" --force

# Restart VSCode
```

### GPU Not Detected
1. **Check NVIDIA Driver**: Run `nvidia-smi`
2. **Restart System**: Required after driver updates
3. **Check Windows GPU Scheduling**: Enable in Display settings
4. **Verify CUDA**: Run verification script
5. **Reinstall PyTorch**: `pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu121`

### Memory Errors
```python
# Reduce batch size
batch_size = 16  # Instead of 32

# Use gradient accumulation
accumulation_steps = 2
for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        
# Clear cache regularly
if i % 50 == 0:
    torch.cuda.empty_cache()
```

### Performance Issues
1. **Update NVIDIA Drivers**: Always use latest Game Ready
2. **Enable Hardware Acceleration**: Windows GPU scheduling
3. **Check Background Apps**: Close unnecessary programs
4. **Monitor Temperatures**: Use MSI Afterburner or similar
5. **Power Settings**: High Performance mode

## Getting Help

### Verification Commands
```powershell
# System info
systeminfo | findstr /C:"OS Name" /C:"OS Version"

# GPU info
nvidia-smi

# Environment info
conda activate mlcourse-gpu
conda list | findstr torch

# Python GPU test
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Support Resources
- **Course Discord/Forum**: Primary support channel
- **Office Hours**: Schedule with instructor
- **Documentation**: Check course repository Wiki
- **NVIDIA Developer Docs**: [developer.nvidia.com](https://developer.nvidia.com)

## Best Practices

### Code Organization
- Always verify GPU availability at start
- Use device-agnostic code: `.to(device)`
- Clear GPU cache between experiments
- Monitor memory usage during training
- Use appropriate batch sizes for your GPU

### Development Workflow
- Test small models first
- Use mixed precision training: `torch.cuda.amp`
- Save models regularly: `torch.save(model.state_dict(), 'model.pth')`
- Log GPU metrics: Use Weights & Biases or TensorBoard

### Academic Integrity
- Include GPU specifications in assignment headers
- Document any GPU-specific optimizations used
- Ensure reproducibility with proper random seeding
- Credit any external GPU resources or tutorials used

---

**Ready to accelerate your machine learning journey with GPU power! 🚀**