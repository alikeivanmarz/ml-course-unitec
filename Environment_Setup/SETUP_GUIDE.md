# Machine Learning Course - Environment Setup Guide

## Prerequisites

- **Git installed** on your system (see [GitHub Setup Guide](../Guides/GITHUB_SETUP_GUIDE.md) for installation)
- **Anaconda or Miniconda** installed on your system
  - Download from: https://www.anaconda.com/products/anaconda
  - Or Miniconda: https://docs.conda.io/en/latest/miniconda.html
- **VSCode installed** with required extensions (see VSCode Setup section below)

## Complete Setup from GitHub (Recommended)

### Step 1: Get the Course Materials from GitHub

```bash
# Clone the repository
git clone https://github.com/alikeivanmarz/ml-course-unitec.git

# Navigate to the project folder
cd ml-course-unitec

# Navigate to Environment_Setup folder
cd Environment_Setup
```

**Alternative**: If you downloaded the ZIP file from GitHub, extract it and navigate to the `Environment_Setup` folder.

## Quick Setup (After Getting Files from GitHub)

### Step 1: Create Environment from YAML file

```bash
# Navigate to the Environment_Setup folder
cd Environment_Setup

# Create the environment (this will take a few minutes)
conda env create -f mlcourse.yml

# Activate the environment
conda activate mlcourse
```

### Step 2: Verify Installation

```bash
# Test all libraries are working
python -c "
import tensorflow as tf
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import nltk
print('All libraries imported successfully!')
print(f'TensorFlow: {tf.__version__}')
print(f'Scikit-learn: {sklearn.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'NumPy: {np.__version__}')
"
```

### Step 3: Register Jupyter Kernel

```bash
# Make sure you're in the activated environment
conda activate mlcourse

# Register the kernel for Jupyter
python -m ipykernel install --user --name mlcourse --display-name "ML Course (Python 3.10)"
```

## VSCode Setup

### Step 1: Install VSCode

**Windows:**
```bash
# Download from: https://code.visualstudio.com/
# Or use winget:
winget install Microsoft.VisualStudioCode
```

**macOS:**
```bash
# Download from: https://code.visualstudio.com/
# Or use Homebrew:
brew install visual-studio-code
```

**Linux:**
```bash
# Download from: https://code.visualstudio.com/
# Or use snap:
sudo snap install code --classic
```

### Step 2: Install Required Extensions

**Method 1: Command Line (Recommended)**
```bash
# Required extensions
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.flake8

# Recommended extensions
code --install-extension ms-vscode.vscode-json
code --install-extension redhat.vscode-yaml
code --install-extension ms-vscode.sublime-keybindings
```

**Method 2: VSCode Extensions Panel**
1. Open VSCode
2. Click Extensions icon (or Ctrl+Shift+X)
3. Search and install:
   - **Python** (by Microsoft) - Required
   - **Jupyter** (by Microsoft) - Required
   - **Flake8** (by Microsoft) - Required
   - **JSON** (by Microsoft) - Optional
   - **YAML** (by Red Hat) - Optional

### Step 3: Configure VSCode for Course

**Open Course Project:**
```bash
# Navigate to course folder and open in VSCode
cd ml-course-unitec
code .
```

**Select Python Interpreter:**
1. Open Command Palette (Ctrl+Shift+P)
2. Type "Python: Select Interpreter"
3. Choose the mlcourse environment: `~/miniconda3/envs/mlcourse/bin/python`

**Test Jupyter Integration:**
1. Open `Course_Sessions/Week_1/Session_1_Introduction.ipynb`
2. VSCode should automatically detect it as a Jupyter notebook
3. **Select the correct kernel**:
   - **Method 1**: Click the kernel name in the top-right corner of the notebook
   - **Method 2**: Press Ctrl+Shift+P → "Notebook: Select Kernel"
   - **Method 3**: Press Ctrl+Shift+P → "Python: Select Interpreter"
   - Choose "ML Course (Python 3.10)" from the list
4. Run the first cell to test

## Alternative Setup (Manual)

If the YAML file doesn't work on your system:

### Step 1: Create Environment Manually

```bash
# Create new environment with Python 3.10
conda create -n mlcourse python=3.10 -y

# Activate environment
conda activate mlcourse

# Install core ML libraries
conda install -c conda-forge scikit-learn pandas numpy scipy matplotlib seaborn plotly -y

# Install Jupyter tools
conda install -c conda-forge jupyter jupyterlab ipykernel -y

# Install deep learning and additional tools
pip install tensorflow opencv-python nltk pillow
```

### Step 2: Alternative Package Installation

If you prefer using pip only:

```bash
# Activate environment
conda activate mlcourse

# Install all packages
pip install -r requirements.txt
```

## Starting VSCode with Jupyter

### Recommended: Use VSCode with Jupyter Integration
```bash
# Navigate to course folder
cd ml-course-unitec

# Open in VSCode
code .

# VSCode will automatically detect .ipynb files and provide Jupyter functionality
```

### Alternative: Traditional Jupyter (if needed)
```bash
conda activate mlcourse
jupyter notebook
# OR
jupyter lab
```

**Note:** We recommend using VSCode for this course as it provides better integration with Git, debugging, and code editing features.

## Troubleshooting

### Issue: Environment creation fails
**Solution:** Try updating conda first:
```bash
conda update conda
conda clean --all
```

### Issue: TensorFlow warnings about CUDA
**Solution:** These warnings are normal on systems without NVIDIA GPUs. TensorFlow will use CPU automatically.

### Issue: Kernel not showing in Jupyter
**Solution:** Re-register the kernel:
```bash
conda activate mlcourse
python -m ipykernel install --user --name mlcourse --display-name "ML Course (Python 3.10)" --force
```

### Issue: Import errors in notebooks
**Solution:** Make sure you're using the correct kernel:

**In VSCode:**
1. Open the notebook file (.ipynb)
2. **Select the correct kernel** using one of these methods:
   - **Method 1**: Click the kernel name in the top-right corner
   - **Method 2**: Press Ctrl+Shift+P → "Notebook: Select Kernel"
   - **Method 3**: Press Ctrl+Shift+P → "Python: Select Interpreter"
3. Choose "ML Course (Python 3.10)" from the dropdown list

**In Traditional Jupyter:**
1. Go to Kernel → Change Kernel
2. Select "ML Course (Python 3.10)"

### Issue: VSCode not recognizing Python environment
**Solution:** 
1. Open Command Palette (Ctrl+Shift+P)
2. Type "Python: Select Interpreter"
3. Choose the mlcourse environment path

### Issue: Jupyter extension not working in VSCode
**Solution:**
1. Ensure Python and Jupyter extensions are installed
2. Reload VSCode (Ctrl+Shift+P → "Developer: Reload Window")
3. Try opening a .ipynb file again

### Issue: Can't find "ML Course (Python 3.10)" kernel
**Solution:**
1. Make sure the environment is activated: `conda activate mlcourse`
2. Re-register the kernel:
   ```bash
   python -m ipykernel install --user --name mlcourse --display-name "ML Course (Python 3.10)" --force
   ```
3. Reload VSCode (Ctrl+Shift+P → "Developer: Reload Window")
4. Try kernel selection again

## Environment Management

### Activate environment
```bash
conda activate mlcourse
```

### Deactivate environment
```bash
conda deactivate
```

### Update packages
```bash
conda activate mlcourse
conda update --all
```

### Remove environment (if needed)
```bash
conda remove --name mlcourse --all
```

## Included Libraries

### Core ML Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning algorithms
- **scipy** - Scientific computing

### Deep Learning
- **tensorflow** - Deep learning framework (includes Keras)

### Visualization
- **matplotlib** - Basic plotting
- **seaborn** - Statistical visualization
- **plotly** - Interactive plots

### Development Tools
- **jupyter** - Interactive notebooks
- **jupyterlab** - Modern Jupyter interface
- **ipykernel** - Jupyter kernel support

### Additional Tools
- **opencv-python** - Computer vision
- **nltk** - Natural language processing
- **pillow** - Image processing

## 🔄 Getting Course Updates

When your instructor releases new materials or updates:

```bash
# Navigate to your course folder
cd ml-course-unitec

# Get the latest updates from GitHub
git pull origin main

# If you get conflicts, stash your changes first:
git stash
git pull origin main
git stash pop  # Only if you want to restore your changes
```

## Next Steps

1. Environment setup complete
2. Read the Course Guide (`../Guides/COURSE_GUIDE.md`)
3. Read the GitHub Setup Guide (`../Guides/GITHUB_SETUP_GUIDE.md`) if you haven't already
4. Start with Week 1 Session 1 (`../Course_Sessions/Week_1/Session_1_Introduction.ipynb`)

## Support

If you encounter issues:
1. Check this troubleshooting section
2. Ask your instructor or TA
3. Refer to the course forum/discussion board

**Remember:** Always activate your environment before starting work!