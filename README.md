# Machine Learning Course - Student Template

## Welcome to Your ML Journey!

This template provides everything you need to succeed in the Machine Learning course. It includes organized course materials, session notebooks, assignments, and comprehensive guides.

## Template Structure

```
STUDENT_TEMPLATE/
├── Environment_Setup/          # Get your environment ready
│   ├── SETUP_GUIDE.md         # Step-by-step setup instructions
│   ├── mlcourse.yml           # Conda environment file
│   └── requirements.txt       # Pip requirements
├── Course_Sessions/           # Weekly session notebooks
│   ├── Week_1/              
│   ├── Week_2/               
│   ├── Week_3/              
│   └── Week_4/               
├── Assignments/              # Assignment templates
│   ├── Assignment_1.ipynb
│   └── Assignment_2.ipynb
├── Datasets/                 # Course datasets (will be provided)
└── Guides/                   # Documentation & workflows
    ├── COURSE_GUIDE.md       # Complete course overview
    └── WORKFLOW_GUIDE.md     # Daily workflow instructions
```

## Quick Start

### Prerequisites Setup (First Time Only)

If you don't have these tools installed, follow these steps:

#### 1. Install Git
**Windows:**
```bash
# Download and install from: https://git-scm.com/download/windows
# Or use winget:
winget install Git.Git
```

**macOS:**
```bash
# Install using Homebrew
brew install git
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install git

# CentOS/RHEL
sudo yum install git
```

#### 2. Install Miniconda
**Windows:**
```bash
# Download from: https://docs.conda.io/en/latest/miniconda.html
# Or use winget:
winget install Anaconda.Miniconda3
```

**macOS:**
```bash
# Download from: https://docs.conda.io/en/latest/miniconda.html
# Or use Homebrew:
brew install miniconda
```

**Linux:**
```bash
# Download and run installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

#### 3. Install VSCode (Required)
**All Platforms:**
- Download from: https://code.visualstudio.com/
- Or use package managers:
  - Windows: `winget install Microsoft.VisualStudioCode`
  - macOS: `brew install visual-studio-code`
  - Linux: Follow instructions on VSCode website

**Required VSCode Extensions:**
After installing VSCode, install these extensions:
```bash
# Install extensions via command line (or use Extensions panel in VSCode)
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.flake8
```

**Recommended VSCode Extensions:**
```bash
code --install-extension ms-vscode.vscode-json
code --install-extension redhat.vscode-yaml
code --install-extension ms-vscode.sublime-keybindings
```

### Course Setup Prerequisites
- **Git installed** on your system
- **Anaconda/Miniconda** installed
- **VSCode with required extensions** installed
- **GitHub account** (free)

**New to Git/GitHub?** Read our [GitHub Setup Guide](Guides/GITHUB_SETUP_GUIDE.md) first!

## One-Command Setup (Copy & Paste)

For students: Copy and paste this entire command block into your terminal:

```bash
# Complete setup in one go
git clone https://github.com/alikeivanmarz/ml-course-unitec.git && \
cd ml-course-unitec/Environment_Setup && \
conda env create -f mlcourse.yml && \
conda init && \
conda activate mlcourse && \
python -m ipykernel install --user --name mlcourse --display-name "ML Course (Python 3.10)" && \
cd .. && \
echo "Setup complete! Opening in VSCode..." && \
code .
```

### Manual Step-by-Step (If Above Doesn't Work)

### Step 1: Get Course Materials from GitHub
```bash
# Clone the repository
git clone https://github.com/alikeivanmarz/ml-course-unitec.git

# Navigate to the project
cd ml-course-unitec
```

### Step 2: Environment Setup
```bash
# Navigate to Environment_Setup folder
cd Environment_Setup

# Create conda environment
conda env create -f mlcourse.yml

# Activate environment
conda activate mlcourse

# Register Jupyter kernel
python -m ipykernel install --user --name mlcourse --display-name \"ML Course (Python 3.10)\"
```

### Step 3: Test Your Setup
```bash
# Open project in VSCode
code .
```

**In VSCode:**
1. Open `Course_Sessions/Week_1/Session_1_Introduction.ipynb`
2. **Select the correct kernel**: Click on the kernel name in the top-right corner
3. Choose "ML Course (Python 3.10)" from the dropdown
4. Run the first few cells to verify everything works
5. If you don't see the "ML Course (Python 3.10)" close the vscode and re-open again

**Kernel Selection Help:**
- **Method 1**: Click the kernel name in top-right corner of notebook
- **Method 2**: Press Ctrl+Shift+P → "Notebook: Select Kernel" → Choose "ML Course (Python 3.10)"
- **Method 3**: Press Ctrl+Shift+P → "Python: Select Interpreter" → Choose mlcourse environment path

### Step 4: Read the Guides
- **COURSE_GUIDE.md** - Complete course overview and structure
- **WORKFLOW_GUIDE.md** - Daily workflow and best practices
- **GITHUB_SETUP_GUIDE.md** - Git and GitHub instructions

## What's Included

### Complete Environment
- **Python 3.10** with all ML/DL libraries
- **TensorFlow 2.19.0** for deep learning
- **Scikit-learn 1.6.1** for machine learning
- **Jupyter** for interactive development
- **Visualization tools** (matplotlib, seaborn, plotly)

### Structured Course Materials
- **Week 1**: Introduction to Machine Learning, Regression and Fundamentals
- **Week 2**: Classification and Ensemble Methods, Neural Networks and Deep Learning
- **Week 3**: Advanced Deep Learning and Computer Vision, Generative AI, RL and LLMs
- **Week 4**: Final Project


## Daily Workflow

1. **Get Updates**: `git pull origin main` (get latest course materials)
2. **Activate Environment**: `conda activate mlcourse`
3. **Open Session Notebook**: Navigate to current week/session
4. **Follow Structure**: Read overview → Theory → Code → Exercises
5. **Save Work**: Ctrl+S frequently

**Detailed workflow**: See [WORKFLOW_GUIDE.md](Guides/WORKFLOW_GUIDE.md) for complete Git integration steps

## Getting Help

### During Sessions
1. Read instructions carefully in each notebook
2. Try the code first, then ask for help
3. Check error messages - they often explain the problem
4. Ask classmates or instructor when stuck

### Outside Class
1. Review session notebooks and examples
2. Check the troubleshooting section in guides
3. Use online documentation (scikit-learn, TensorFlow)
4. Practice with additional datasets

## Success Tips

### Best Practices
- **Start early** on assignments
- **Practice regularly** - coding skills improve with repetition
- **Understand concepts** don't just copy code
- **Document your work** using markdown cells
- **Experiment** with different parameters and approaches

### Technical Tips
- Always activate your environment before starting
- Keep your random seed consistent for reproducible results
- Use meaningful variable names and add comments
- Test code frequently as you develop

### Environment Management
- Always use `conda activate mlcourse` before starting
- Don't modify the environment unless instructed
- If you have issues, recreate the environment from the YAML file

### Academic Integrity
- Collaborate on concepts, but submit individual work
- Cite any external resources you use
- Don't share complete assignment solutions
- Ask instructors when in doubt about collaboration policies

**Good luck! You're going to do great!**
