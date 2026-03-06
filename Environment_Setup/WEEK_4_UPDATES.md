# Environment Updates for Week 4 - Generative AI & LLMs

## Summary of Changes

Both conda environment files have been updated to include all packages required for Week 4 teaching materials and applications.

---

## Updates to `mlcourse.yml` (CPU version)

### Added Packages:

**Generative AI & Diffusion Models:**
- `diffusers` - Stable Diffusion and diffusion model pipelines
- `accelerate` - Model optimization and distributed training
- `safetensors` - Secure tensor serialization
- `compel` - Enhanced prompt weighting for Stable Diffusion

**Audio Generation:**
- `soundfile` - Audio file I/O
- `scipy` - Signal processing (already present, verified)
- `librosa` - Audio analysis and processing

### Already Present (No Changes Needed):
- ✓ `numpy` - Array operations
- ✓ `matplotlib` - Visualization
- ✓ `seaborn` - Statistical plots
- ✓ `pillow` - Image processing
- ✓ `jupyter`, `jupyterlab`, `notebook` - Notebook environment
- ✓ `ipywidgets` - Interactive controls
- ✓ `transformers` - Hugging Face models
- ✓ `torch`, `torchvision`, `torchaudio` - PyTorch
- ✓ `opencv-python` - Computer vision
- ✓ `realesrgan`, `basicsr` - Super-resolution models

---

## Updates to `mlcourse-gpu.yml` (GPU version)

### Added Packages:

**Generative AI & Diffusion Models:**
- `diffusers>=0.31` - Stable Diffusion pipelines
- `safetensors>=0.4` - Secure model weights
- `compel>=2.0` - Prompt weighting
- `invisible-watermark>=0.2` - Image watermarking (for SD XL)

**Audio Generation & Processing:**
- `soundfile>=0.12` - Audio I/O
- `librosa>=0.10` - Audio feature extraction
- `scipy>=1.15` - Signal processing (already present, verified)

### Already Present (No Changes Needed):
- ✓ `numpy>=1.23` - Array operations
- ✓ `matplotlib>=3.10` - Visualization
- ✓ `seaborn>=0.13` - Statistical plots
- ✓ `pillow>=11.0` - Image processing
- ✓ `jupyter`, `jupyterlab`, `notebook` - Notebook environment
- ✓ `ipywidgets` - Interactive controls
- ✓ `transformers>=4.55` - Hugging Face models
- ✓ `accelerate>=1.10` - Model optimization
- ✓ `torch`, `torchvision`, `torchaudio` - PyTorch with CUDA
- ✓ `opencv` - Computer vision
- ✓ `ultralytics>=8.3` - YOLO models

---

## What These Packages Enable

### For Week 4 Teaching Materials:
✅ **Notebook & Visualizations:**
- All 22 diagram generation scripts work
- Interactive widgets in notebook function properly
- No additional packages needed beyond base environment

### For Week 4 Applications:

✅ **Stable Diffusion Generator:**
- `diffusers` - Main pipeline
- `accelerate` - GPU optimization
- `safetensors` - Model loading
- `compel` - Enhanced prompts
- `invisible-watermark` - SD XL watermarking

✅ **LLaMA Text Generator:**
- `transformers` - Model loading
- `accelerate` - Memory optimization
- `torch` - Inference engine
- `safetensors` - Model weights

✅ **Text-to-Audio Generator:**
- `transformers` - Bark and MMS-TTS models
- `soundfile` - WAV file output
- `scipy` - Audio processing
- `librosa` - Feature extraction

✅ **Image/Video Upscaler:**
- `opencv-python` - Image/video I/O
- `torch` - Real-ESRGAN inference
- `pillow` - Image processing
- Already had `realesrgan` and `basicsr`

---

## Installation Instructions

### For Existing Environments:

**If you already have `mlcourse` or `mlcourse-gpu` installed:**

#### Option 1: Update existing environment
```bash
# Activate environment
conda activate mlcourse  # or mlcourse-gpu

# Install new packages
pip install diffusers accelerate safetensors compel soundfile librosa

# For GPU version also:
pip install invisible-watermark
```

#### Option 2: Recreate environment (recommended for clean install)
```bash
# Remove old environment
conda env remove -n mlcourse

# Create from updated file
conda env create -f Environment_Setup/mlcourse.yml

# Or for GPU:
conda env remove -n mlcourse-gpu
conda env create -f Environment_Setup/mlcourse-gpu.yml
```

### For New Installations:

```bash
# CPU version
conda env create -f Environment_Setup/mlcourse.yml
conda activate mlcourse

# GPU version (requires CUDA-compatible GPU)
conda env create -f Environment_Setup/mlcourse-gpu.yml
conda activate mlcourse-gpu

# Register kernel for Jupyter
python -m ipykernel install --user --name mlcourse --display-name "ML Course (CPU)"
# or
python -m ipykernel install --user --name mlcourse-gpu --display-name "ML Course (GPU)"
```

---

## Verification

### Check if Week 4 packages are installed:

```python
# Run this in Python/Jupyter
import sys

packages = {
    'diffusers': 'Stable Diffusion pipelines',
    'accelerate': 'Model optimization',
    'safetensors': 'Model loading',
    'soundfile': 'Audio I/O',
    'librosa': 'Audio processing',
}

print("Week 4 Package Check:\n")
for pkg, desc in packages.items():
    try:
        __import__(pkg)
        print(f"✓ {pkg:15s} - {desc}")
    except ImportError:
        print(f"✗ {pkg:15s} - MISSING")
```

### Test visualization packages:

```bash
cd Course_Sessions/Week_4
python check_installation.py
```

### Test diagram generation:

```bash
cd Course_Sessions/Week_4/visualization_scripts
python generate_all.py
```

---

## Package Purpose Summary

| Package | Purpose | Used In |
|---------|---------|---------|
| **diffusers** | Stable Diffusion pipelines | SD Generator |
| **accelerate** | GPU memory optimization | SD, LLM apps |
| **safetensors** | Fast, secure model loading | All apps |
| **compel** | Enhanced prompt weighting | SD Generator |
| **invisible-watermark** | SD XL watermarking | SD Generator (XL) |
| **soundfile** | Read/write audio files | TTS Generator |
| **librosa** | Audio feature extraction | TTS Generator |
| **scipy** | Signal processing | TTS, visualizations |

---

## Compatibility Notes

### Python Version:
- **mlcourse.yml**: Python 3.10
- **mlcourse-gpu.yml**: Python 3.11 (for CUDA 12.9)

### GPU Requirements:
- **mlcourse.yml**: CPU only, works on any machine
- **mlcourse-gpu.yml**: Requires NVIDIA GPU with CUDA support

### Tested Platforms:
- ✓ Windows 10/11 (native and WSL2)
- ✓ Linux (Ubuntu 20.04+)
- ✓ macOS (CPU only)

---

## Troubleshooting

### Issue: "No module named 'diffusers'"
```bash
pip install diffusers accelerate
```

### Issue: CUDA out of memory
```bash
# Use smaller models or enable optimizations
# See individual application READMEs for memory-saving tips
```

### Issue: Audio generation fails
```bash
pip install soundfile librosa scipy
```

### Issue: Package conflicts
```bash
# Recreate environment from scratch
conda env remove -n mlcourse
conda env create -f Environment_Setup/mlcourse.yml
```

---

## Size Information

**Additional disk space required:**
- Base packages (diffusers, etc.): ~500 MB
- SD 1.5 model weights: ~4 GB
- SD XL model weights: ~7 GB
- GPT-2 models: 500 MB - 3 GB
- Mistral 7B: ~15 GB

**Total for all Week 4 models:** ~30 GB

---

## Migration Guide

### From Old Environment to Updated:

1. **Export your current packages** (optional backup):
```bash
conda activate mlcourse
pip freeze > my_packages_backup.txt
```

2. **Update environment**:
```bash
conda env update -f Environment_Setup/mlcourse.yml --prune
```

3. **Or recreate** (cleaner):
```bash
conda env remove -n mlcourse
conda env create -f Environment_Setup/mlcourse.yml
```

4. **Verify**:
```bash
conda activate mlcourse
python -c "import diffusers, accelerate; print('Week 4 ready!')"
```

---

## What's NOT Included

These environment files include everything for:
- ✅ Week 4 teaching materials (notebook + visualizations)
- ✅ Week 4 applications (SD, LLM, TTS, Upscaler)
- ✅ Weeks 1-3 content
- ✅ All course notebooks and exercises

**Not included:**
- ❌ Model weights (downloaded on first use)
- ❌ Dataset files (separate downloads)
- ❌ API keys (user must provide)

---

## Summary

Both environment files are now **fully compatible** with Week 4 materials. All packages needed for:
- Teaching notebook
- 22 visualization diagrams
- 4 generative AI applications
- Interactive demonstrations

Are included in the updated environment files.

**Recommendation:** If you're setting up from scratch, use the GPU version for best performance with generative AI models. The CPU version works but is significantly slower for inference.
