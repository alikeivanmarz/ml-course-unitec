# Daily Workflow Guide

## Before Each Session

### 1. Get Latest Course Updates
```bash
# Navigate to your course folder
cd ml-course-unitec

# Get the latest updates from instructor
git pull origin main

# Check if there are any new materials
git log --oneline -5
```

### 2. Environment Check
```bash
# Activate your environment
conda activate mlcourse

# Verify it's working
python -c "import pandas as pd; import numpy as np; print('Environment ready!')"
```

### 3. Open VSCode
```bash
# Navigate to course folder and open in VSCode
cd ml-course-unitec
code .
```

### 4. Open Today's Notebook
1. In VSCode Explorer panel, navigate to: `Course_Sessions/Week_X/Session_Y_TopicName.ipynb`
2. Click to open the notebook
3. **Select the correct kernel**:
   - **Method 1**: Click the kernel name in the top-right corner
   - **Method 2**: Press Ctrl+Shift+P → "Notebook: Select Kernel"
   - **Method 3**: Press Ctrl+Shift+P → "Python: Select Interpreter"
   - Choose "ML Course (Python 3.10)" from the list

## During Each Session

### 1. Read the Overview
Each notebook starts with:
- Learning objectives
- Topics covered
- Prerequisites
- Expected time

### 2. Follow the Structure
```
Overview Cell          # Read this first
Theory Section         # Understand concepts
Code Examples          # Follow along
Practice Exercises     # Complete these
Visualization          # Create plots
Summary                # Review what you learned
```

### 3. Active Learning Tips
- **Run each cell as you go** - don't skip ahead
- **Modify parameters** - see what happens when you change values
- **Add your own markdown cells** - explain concepts in your own words
- **Take notes** - use markdown cells for observations

### 4. Complete TODO Sections
Look for cells marked:
```python
# TODO: Your task description here
# Write your code below

```

## Working with Code Cells in VSCode

### 1. Essential Keyboard Shortcuts
- **Shift + Enter**: Run cell and move to next
- **Ctrl + Enter**: Run cell and stay
- **Ctrl + Shift + P**: Open Command Palette
- **Ctrl + `**: Open integrated terminal
- **Ctrl + B**: Toggle sidebar
- **F5**: Start debugging (for Python files)

### 2. VSCode-Specific Notebook Features
- **Cell Actions**: Use the buttons that appear on hover (Run, Insert, Delete)
- **Variable Explorer**: View variables in the Python extension panel
- **Debugging**: Set breakpoints and debug notebook cells
- **IntelliSense**: Auto-completion and code suggestions
- **Git Integration**: See changes, commit, and push directly from VSCode

### 3. Managing Variables in VSCode
```python
# Check what variables exist
%whos

# Clear all variables (if needed)
%reset -f

# Time your code execution
%%time
your_code_here()
```

**VSCode Variable Inspector:**
- Variables automatically appear in the Variables panel
- Click on variables to inspect their values
- No need for print statements to see variable contents

### 4. Debugging Tips in VSCode
```python
# For detailed error information
import traceback
try:
    your_problematic_code()
except Exception as e:
    traceback.print_exc()

# Check variable values
print(f"Variable value: {your_variable}")
print(f"Variable type: {type(your_variable)}")
print(f"Variable shape: {getattr(your_variable, 'shape', 'No shape')}")
```

## Data Workflow Pattern

Each session follows this pattern:

### 1. Import and Setup
```python
# Standard imports (provided in each notebook)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
plt.rcParams['figure.dpi'] = 300
np.random.seed(42)  # UPDATE: Use your student ID's last 2 digits
```

### 2. Data Loading
```python
# Load data
df = pd.read_csv('../Datasets/datafile.csv')

# Initial exploration
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
```

### 3. Exploratory Data Analysis (EDA)
```python
# Statistical summary
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualize distributions
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()
```

### 4. Data Preprocessing
```python
# Handle missing values
# Split features and target
# Train/test split
# Feature scaling (if needed)
```

### 5. Model Building
```python
# Import required algorithms
# Fit the model
# Make predictions
# Evaluate performance
```

### 6. Visualization
```python
# Create meaningful plots
# Add titles, labels, legends
# Save important plots
```

## File Management

### 1. Saving Your Work in VSCode
- **Auto-save**: VSCode auto-saves by default
- **Manual save**: Ctrl+S
- **Git integration**: See file changes in Source Control panel
- **Notebook checkpoints**: VSCode maintains version history automatically

### 2. Exporting Notebooks in VSCode
**Method 1: VSCode Command Palette**
1. Open Command Palette (Ctrl+Shift+P)
2. Type "Jupyter: Export to HTML" or "Jupyter: Export to Python"
3. Select desired format

**Method 2: Command Line**
```bash
# Export to HTML (for viewing)
jupyter nbconvert --to html notebook_name.ipynb

# Export to Python script
jupyter nbconvert --to python notebook_name.ipynb

# Export to PDF (requires additional setup)
jupyter nbconvert --to pdf notebook_name.ipynb
```

### 3. Version Control with VSCode Git Integration
**Using VSCode GUI:**
1. Click Source Control icon (Ctrl+Shift+G)
2. Review changes in the panel
3. Stage files by clicking the + icon
4. Enter commit message
5. Click the checkmark to commit
6. Click "..." → Push to sync with remote

**Using Command Line:**
```bash
git add .
git commit -m "Completed Week X Session Y"
git push
```

## Assignment Workflow

### 1. Preparation in VSCode
1. In VSCode Explorer, navigate to `Assignments/` folder
2. Right-click on assignment template file
3. Select "Copy"
4. Right-click in the Assignments folder
5. Select "Paste"
6. Rename to: `StudentID_YourName_Assignment_X.ipynb`
7. Double-click to open the notebook

### 2. Assignment Structure
```python
# Cell 1: Student Information (REQUIRED)
"""
Student Name: Your Full Name
Student ID: 1234567
Assignment: X
Date: YYYY-MM-DD
"""

# Cell 2: Random Seed (REQUIRED)
import numpy as np
np.random.seed(67)  # Last 2 digits of YOUR student ID
```

### 3. Working Through Assignment
- Read all instructions first
- Complete TODO sections
- Test your code frequently
- Add explanations in markdown cells

### 4. Pre-Submission Checklist
- [ ] All code cells run without errors
- [ ] Random seed set correctly
- [ ] File named correctly
- [ ] All plots have titles and labels
- [ ] Explanations provided for results
- [ ] Output is clean and professional

## Troubleshooting Common Issues

### Environment Issues
```bash
# If libraries aren't found
conda activate mlcourse
conda list  # Check what's installed

# If kernel isn't found in Jupyter
python -m ipykernel install --user --name mlcourse --display-name "ML Course"
```

### VSCode Issues
```bash
# If VSCode won't start
code --version  # Check installation

# If Jupyter extension not working
code --list-extensions  # Check installed extensions
code --install-extension ms-toolsai.jupyter  # Reinstall if needed

# If kernel not found
# Open Command Palette (Ctrl+Shift+P) → "Python: Select Interpreter"
```

### Code Issues
```python
# If imports fail
import sys
print(sys.path)  # Check Python path
print(sys.executable)  # Check Python location

# If plots don't show
%matplotlib inline  # Add this magic command

# If memory issues in VSCode
# Command Palette → "Jupyter: Restart Kernel"
```

## Best Practices

### 1. Code Quality
```python
# Use meaningful variable names
student_grades = [85, 92, 78]  # Good
x = [85, 92, 78]              # Avoid

# Add comments for complex code
# Calculate polynomial features for degree 2
poly_features = PolynomialFeatures(degree=2)

# Break long lines
model = LinearRegression().fit(
    X_train_scaled, 
    y_train
)
```

### 2. Documentation
```markdown
## Analysis Summary

In this section, I explored the relationship between...

### Key Findings:
1. Feature X shows strong correlation with target (r=0.85)
2. Model achieves R² = 0.92 on test data
3. Polynomial degree 3 provides best balance of fit vs complexity
```

### 3. Visualization
```python
# Always include titles and labels
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.6, label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.title('Model Predictions vs Actual Values')
plt.xlabel('Feature Values')
plt.ylabel('Target Variable')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Time Management

### Session Time Allocation
- **5 min**: Setup and review
- **15 min**: Theory and examples
- **25 min**: Hands-on practice
- **10 min**: Exercises and questions
- **5 min**: Summary and next steps

### Weekly Study Schedule
- **Day 1**: Attend session, review concepts
- **Day 2**: Practice with additional datasets
- **Day 3**: Work on assignments
- **Day 4**: Review and prepare for next session

## Getting Unstuck

### 1. Check the Basics
- Is your environment activated?
- Are you in the right directory?
- Did you run all previous cells?
- Are there any typos in your code?

### 2. Use Built-in Help
```python
# Get help on functions
help(pd.read_csv)
?pd.read_csv  # In Jupyter

# See function signature
pd.read_csv?

# See source code
pd.read_csv??
```

### 3. Systematic Debugging
```python
# Print intermediate values
print("Step 1:", variable1)
print("Step 2:", variable2)
print("Final:", result)

# Check data types and shapes
print(f"X_train shape: {X_train.shape}")
print(f"y_train type: {type(y_train)}")
```

### 4. When to Ask for Help
- After trying for 15+ minutes
- After checking documentation
- After searching for similar errors online
- When you understand the error but not the solution

## Git Workflow Integration

### Before Each Session
```bash
# 1. Navigate to course folder
cd ml-course-unitec

# 2. Get latest updates from instructor
git pull origin main

# 3. Check for any conflicts or new files
git status

# 4. Start working
code .
```

### After Each Session
```bash
# 1. Save your work in VSCode (Ctrl+S)

# 2. Check what you've changed
git status

# 3. Commit your session work (optional - for your own tracking)
git add Course_Sessions/
git commit -m "Completed Week X Session Y exercises"

# Note: You won't push these to the course repo
```

### For Assignments
```bash
# 1. Work on assignment in your notebook

# 2. Save frequently (Ctrl+S in VSCode)

# 3. When done, create final version with correct naming
# Copy: Assignment_1_Regression.ipynb
# To: 1234567_JohnDoe_Assignment1.ipynb

# 4. Optional: Track your assignment progress
git add Assignments/1234567_JohnDoe_Assignment1.ipynb
git commit -m "Completed Assignment 1"

# 5. Submit as instructed by your instructor (usually via email or LMS)
```

### Handling Updates During Work
```bash
# If instructor releases updates while you're working:

# 1. Save your current work (Ctrl+S in VSCode)

# 2. Stash your changes temporarily
git stash push -m "Work in progress"

# 3. Pull the updates
git pull origin main

# 4. Restore your work
git stash pop

# 5. Resolve any conflicts if they occur
```

### Git Commands You'll Use Most
```bash
# Check status of files
git status

# See what changed
git log --oneline -10

# Get updates from instructor
git pull origin main

# Save your work locally
git add .
git commit -m "Your message here"

# Emergency: reset to latest version
git fetch origin
git reset --hard origin/main  # Warning: loses local changes
```

## End of Session Checklist

- [ ] All cells run successfully
- [ ] Notebook saved
- [ ] Key concepts understood
- [ ] Notes added for review
- [ ] Questions identified for next session
- [ ] Assignment progress checked (if applicable)
- [ ] Latest updates pulled from GitHub (git pull origin main)

---

**Remember: Learning ML is a journey, not a destination. Take your time and enjoy the process!**
---

[← Previous: Git Pull Guide](04_GIT_PULL_GUIDE.md) | [Index](README.md) | [Next: VSCode Tips →](06_VSCODE_TIPS_GUIDE.md)
