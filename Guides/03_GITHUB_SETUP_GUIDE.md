# GitHub Setup Guide for ML Course

## Overview

This guide will help you set up Git and GitHub to access the course materials. GitHub is a platform for hosting code repositories, and Git is the version control system that tracks changes to files.

## Prerequisites

- Computer with internet connection
- Administrator access to install software
- GitHub account (we'll create this)

## Step-by-Step Setup

### Step 1: Create a GitHub Account

1. **Go to GitHub**: Visit [https://github.com](https://github.com)
2. **Sign Up**: Click "Sign up" in the top right
3. **Choose Username**: Pick a professional username (e.g., `john-doe-unitec` or `jdoe2024`)
4. **Use School Email**: Use your student email address
5. **Verify Account**: Check your email and verify your account
6. **Choose Free Plan**: Select the free plan (sufficient for this course)

### Step 2: Install Git

#### For Windows Users:

1. **Download Git**: Go to [https://git-scm.com/download/windows](https://git-scm.com/download/windows)
2. **Run Installer**: Download and run the installer
3. **Installation Options**: 
   - Use recommended settings for most options
   - Choose "Git from the command line and also from 3rd-party software"
   - Choose "Checkout Windows-style, commit Unix-style line endings"
   - Choose "Windows Console" for terminal emulator
4. **Verify Installation**: Open Command Prompt and type:
   ```cmd
   git --version
   ```
   You should see something like `git version 2.x.x`

#### For macOS Users:

**Option 1: Using Homebrew (Recommended)**
```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Git
brew install git
```

**Option 2: Download Installer**
1. **Download**: Go to [https://git-scm.com/download/mac](https://git-scm.com/download/mac)
2. **Install**: Run the downloaded installer
3. **Verify**: Open Terminal and type `git --version`

#### For Linux Users:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install git
```

**CentOS/RHEL/Fedora:**
```bash
sudo yum install git
# or for newer versions:
sudo dnf install git
```

### Step 3: Configure Git

After installing Git, configure it with your information:

```bash
# Set your name (use your real name)
git config --global user.name "Your Full Name"

# Set your email (use the same email as your GitHub account)
git config --global user.email "your.email@student.unitec.ac.nz"

# Set default branch name
git config --global init.defaultBranch main

# Verify configuration
git config --list
```

### Step 4: Access the Course Repository

The course materials are hosted on GitHub. Here's how to get them:

#### Method 1: Clone the Repository (Recommended)

1. **Open Terminal/Command Prompt**
2. **Navigate to desired location**:
   ```bash
   # Windows
   cd C:\Users\YourName\Documents
   
   # macOS/Linux
   cd ~/Documents
   ```

3. **Clone the repository**:
   ```bash
   git clone https://github.com/alikeivanmarz/ml-course-unitec.git
   ```
   
4. **Navigate into the folder**:
   ```bash
   cd ml-course-unitec
   ```

#### Method 2: Download ZIP (Alternative)

1. **Go to repository**: Visit the GitHub repository URL
2. **Download ZIP**: Click green "Code" button → "Download ZIP"
3. **Extract**: Extract the ZIP file to your desired location
4. **Note**: This method doesn't track changes - you'll need to re-download for updates

## Getting Course Updates

### Method 1: Pull Updates (If you cloned)

When your instructor adds new materials or updates:

```bash
# Navigate to your course folder
cd path/to/ml-course-unitec

# Get the latest updates
git pull origin main
```

### Method 2: Fetch Specific Updates

If you want to see what's changed before pulling:

```bash
# Check for updates without downloading
git fetch origin

# See what's different
git log HEAD..origin/main --oneline

# Pull the updates
git pull origin main
```

## Repository Structure

After cloning, you'll have this structure:

```
ml-course-unitec/
├── Environment_Setup/
├── Course_Sessions/
├── Assignments/
├── Datasets/
├── Guides/
└── README.md
```

## Complete Setup Workflow

Here's the complete process to get started:

```bash
# 1. Clone the repository
git clone https://github.com/alikeivanmarz/ml-course-unitec.git

# 2. Navigate to the project
cd ml-course-unitec

# 3. Set up the conda environment
cd Environment_Setup
conda env create -f mlcourse.yml

# 4. Activate the environment
conda activate mlcourse

# 5. Register Jupyter kernel
python -m ipykernel install --user --name mlcourse --display-name "ML Course (Python 3.10)"

# 6. Go back to main directory
cd ..

# 7. Start Jupyter
jupyter notebook
```

## Git Commands Cheat Sheet

### Basic Commands

```bash
# Check repository status
git status

# See commit history
git log --oneline

# Check current branch
git branch

# See remote repository info
git remote -v

# Check for differences
git diff
```

### Getting Updates

```bash
# Get latest changes
git pull origin main

# Force pull (overwrites local changes)
git fetch origin
git reset --hard origin/main
```

### Working with Your Own Changes

```bash
# See what files you've changed
git status

# Add files to staging
git add filename.ipynb
# or add all changes
git add .

# Commit your changes
git commit -m "Completed Assignment 1"

# Note: You typically won't push to the course repo
# Instead, submit assignments as instructed
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: "git: command not found"
**Solution**: Git is not installed or not in PATH
- Reinstall Git following the installation steps above
- On Windows, make sure to select "Git from command line" during installation

#### Issue: "Permission denied" when cloning
**Solution**: Repository might be private or URL incorrect
- Check the repository URL
- Make sure you're using the correct GitHub repository link

#### Issue: "Your local changes would be overwritten by merge"
**Solution**: You have local changes conflicting with updates
```bash
# Save your changes first
git stash

# Pull updates
git pull origin main

# Restore your changes (if needed)
git stash pop
```

#### Issue: "fatal: not a git repository"
**Solution**: You're not in the right directory
```bash
# Navigate to your cloned repository
cd path/to/ml-course-unitec

# Verify you're in the right place
ls -la  # Should show .git folder
```

### Getting Help

#### Git Help Commands
```bash
# General help
git help

# Help for specific commands
git help clone
git help pull
git help status
```

#### Online Resources
- [Git Documentation](https://git-scm.com/doc)
- [GitHub Help](https://help.github.com/)
- [Interactive Git Tutorial](https://learngitbranching.js.org/)

## 🎓 Git Best Practices for Students

### Do's 

- **Keep your repository clean**: Don't add unnecessary files
- **Pull regularly**: Get updates before each session
- **Use meaningful commit messages**: "Completed Week 1 exercises"
- **Save your work**: Commit your assignment progress regularly
- **Ask for help**: Don't struggle alone with Git issues

### Don'ts 

- **Don't commit large files**: Avoid datasets > 100MB
- **Don't commit sensitive info**: No passwords, API keys, personal data
- **Don't force push**: You can break the shared repository
- **Don't panic**: Git problems are usually fixable

## Weekly Git Workflow

### Before Each Session
```bash
# 1. Navigate to course folder
cd ml-course-unitec

# 2. Get latest updates
git pull origin main

# 3. Start working
jupyter notebook
```

### After Each Session
```bash
# 1. Save your work in Jupyter (Ctrl+S)

# 2. Check what you've changed
git status

# 3. Commit your session work (optional)
git add .
git commit -m "Completed Week X Session Y"
```

### For Assignments
```bash
# 1. Work on assignment in your notebook

# 2. Save frequently (Ctrl+S in Jupyter)

# 3. When done, create final version with correct naming
# Copy: Assignment_1_Regression.ipynb
# To: 1234567_JohnDoe_Assignment1.ipynb

# 4. Submit as instructed by your instructor
```

## SSH Setup (Optional - Advanced)

For easier authentication without passwords:

### Generate SSH Key
```bash
# Generate new SSH key
ssh-keygen -t ed25519 -C "your.email@student.unitec.ac.nz"

# Start SSH agent
eval "$(ssh-agent -s)"

# Add key to agent
ssh-add ~/.ssh/id_ed25519
```

### Add to GitHub
1. **Copy public key**:
   ```bash
   # Linux/macOS
   cat ~/.ssh/id_ed25519.pub
   
   # Windows
   type %USERPROFILE%\.ssh\id_ed25519.pub
   ```

2. **Add to GitHub**:
   - Go to GitHub → Settings → SSH and GPG keys
   - Click "New SSH key"
   - Paste your public key

3. **Test connection**:
   ```bash
   ssh -T git@github.com
   ```

## Getting Support

### In Class
- Ask your instructor or TA for Git help
- Work with classmates to solve Git issues
- Use the troubleshooting section above

### Online Resources
- Course discussion forum
- GitHub documentation
- Git tutorial websites
- Stack Overflow (search for specific error messages)

### Emergency Recovery
If Git breaks completely:
1. **Backup your work**: Copy any important files
2. **Delete the folder**: Remove the entire repository folder
3. **Re-clone**: Start fresh with `git clone`
4. **Restore work**: Copy back your assignment files

---

**Remember**: Git might seem complex at first, but you only need a few commands for this course. Focus on `git clone`, `git pull`, and `git status` - these will handle 90% of your needs!**