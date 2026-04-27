# Quick Setup Guide - Machine Learning Course

##  Essential Setup Before First Session

Dear Students,

Please complete this setup **before our first session** to ensure you're ready to start coding immediately.

##  What You'll Install

1. **Git** - Version control for course materials
2. **Miniconda** - Python environment manager  
3. **VSCode** - Our coding environment
4. **Course Repository** - All materials and code

---

##  Complete Setup (Copy & Paste Method)

**If you're comfortable with command line**, use this one-command setup:

### Windows (PowerShell as Administrator)
```powershell
# Install prerequisites using winget
winget install Git.Git
winget install Anaconda.Miniconda3  
winget install Microsoft.VisualStudioCode

# Restart PowerShell, then:
git clone https://github.com/alikeivanmarz/ml-course-unitec.git
cd ml-course-unitec/Environment_Setup
conda env create -f mlcourse.yml
conda activate mlcourse
python -m ipykernel install --user --name mlcourse --display-name "ML Course (Python 3.10)"
cd .. && code .
```

### macOS (Terminal)
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install prerequisites
brew install git miniconda visual-studio-code

# Setup course
git clone https://github.com/alikeivanmarz/ml-course-unitec.git
cd ml-course-unitec/Environment_Setup
conda env create -f mlcourse.yml
conda activate mlcourse
python -m ipykernel install --user --name mlcourse --display-name "ML Course (Python 3.10)"
cd .. && code .
```

**If this works, skip to the Verification section below!**

---

##  Step-by-Step Manual Setup

### Step 1: Install Git

**Windows:**
1. Download from: https://git-scm.com/download/windows
2. Run installer with default settings
3. **Important**: Choose "Git from the command line and also from 3rd-party software"

**macOS:**
1. Install Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
2. Install Git: `brew install git`

**Test:** Open Terminal/Command Prompt and type `git --version`

### Step 2: Install Miniconda

**Windows:**
1. Download from: https://docs.conda.io/en/latest/miniconda.html
2. Run installer, check "Add to PATH" option
3. Restart Command Prompt

**macOS:**
1. Download from: https://docs.conda.io/en/latest/miniconda.html
2. Run installer and follow prompts
3. Restart Terminal

**Test:** Type `conda --version`

### Step 3: Install VSCode

**All Platforms:**
1. Download from: https://code.visualstudio.com/
2. Install with default settings

**Install Required Extensions:**
Open VSCode and press `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (macOS), then type:
```
Extensions: Install Extensions
```
Install these extensions:
- **Python** (by Microsoft)
- **Jupyter** (by Microsoft)  
- **Flake8** (by Microsoft)

### Step 4: Get Course Materials

Open Terminal/Command Prompt and run:
```bash
# Navigate to your Documents folder
cd Documents  # or cd ~/Documents on macOS

# Download course materials
git clone https://github.com/alikeivanmarz/ml-course-unitec.git

# Enter the course folder
cd ml-course-unitec
```

### Step 5: Setup Python Environment

```bash
# Navigate to environment setup
cd Environment_Setup

# Create the course environment (this takes 5-10 minutes)
conda env create -f mlcourse.yml

# Activate the environment
conda activate mlcourse

# Register Jupyter kernel for VSCode
python -m ipykernel install --user --name mlcourse --display-name "ML Course (Python 3.10)"
```

### Step 6: Open in VSCode

```bash
# Go back to main folder
cd ..

# Open project in VSCode
code .
```

---

##  Verification - Test Your Setup

### 1. Test Environment
In Terminal/Command Prompt with `mlcourse` environment activated:
```bash
python -c "import pandas as pd, numpy as np, sklearn, matplotlib.pyplot as plt; print(' All libraries working!')"
```

### 2. Test VSCode Integration
1. In VSCode, open: `Course_Sessions/Week_1/Week_1_Session_1.ipynb`
2. **CRITICAL**: Click the kernel selector in top-right corner
3. Choose **"ML Course (Python 3.10)"** from dropdown
4. Run the first cell - should execute without errors

### 3. If Kernel Not Found:
- Close VSCode completely
- In Terminal: `conda activate mlcourse`  
- Run: `code .` to reopen VSCode
- Try kernel selection again

---

##  Troubleshooting

### Git Issues
```bash
# Configure Git (replace with your info)
git config --global user.name "Your Full Name"
git config --global user.email "your.email@student.unitec.ac.nz"
```

### Conda Issues  
```bash
# If conda not found, restart Terminal/Command Prompt
# If environment creation fails:
conda update conda
conda clean --all
# Try creating environment again
```

### VSCode Issues
- **Extensions not installing?** Use VSCode Extensions panel (Ctrl+Shift+X)
- **Python not found?** Press Ctrl+Shift+P → "Python: Select Interpreter" → Choose mlcourse
- **Kernel issues?** Restart VSCode, ensure environment is activated

### Environment Creation Takes Too Long?
- This is normal (5-15 minutes depending on internet)
- If it freezes, press Ctrl+C and try again
- Check your internet connection

---

##  Quick Email Checklist

**Before first session, ensure you can:**
- [ ] Open Terminal/Command Prompt and type `conda activate mlcourse`
- [ ] Open VSCode by typing `code .` in the ml-course-unitec folder
- [ ] Open a .ipynb notebook and see "ML Course (Python 3.10)" as kernel option
- [ ] Run a simple Python cell without errors

---

##  Getting Help

### During Setup:
1. **Try the troubleshooting section above**
2. **Google the exact error message**
4. **Email me with screenshot of the error**

### Course Materials Location:
After setup, your course materials are in:
- **Windows**: `C:\Users\YourName\Documents\ml-course-unitec\`
- **macOS**: `~/Documents/ml-course-unitec/`

### Important Files:
- **Session Notebooks**: `Course_Sessions/Week_1/`
- **Assignment Templates**: `Assignments/`
- **Detailed Guides**: `Guides/` folder
- **Environment File**: `Environment_Setup/mlcourse.yml`

---

##  Tips

1. **Bookmark this guide** - you might need it later
2. **Always activate your environment** before starting work: `conda activate mlcourse`
3. **Use VSCode for everything** - it integrates perfectly with our workflow
4. **Save your work frequently** - Ctrl+S in VSCode
5. **Don't panic if something breaks** - we can fix it in class

---

##  What's Next?

1. **Complete this setup**
2. **Bring your laptop to class**
3. **We'll test everything together in the class**

**See you in class!**

---

**Questions? Email me at: [akeivanmarz@unitec.ac.nz]**

**One-time only**: This setup lasts the entire course  
**Internet required**: For downloading packages and materials
---

[← Previous: Course Guide](01_COURSE_GUIDE.md) | [Index](README.md) | [Next: GitHub Setup Guide →](03_GITHUB_SETUP_GUIDE.md)
