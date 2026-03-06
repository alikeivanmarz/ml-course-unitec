# Machine Learning Course - Student Guide

## Course Overview

This is a 4-week machine learning course covering fundamental concepts from basic supervised, unsupervised, and reinforcement learning to advanced deep learning and generative AI topics. The course culminates in Week 4 with individual final project presentations.

### Learning Objectives
By the end of this course, you will be able to:
- get to know all three types of machine learning: supervised, unsupervised, and reinforcement learning
- Implement complete ML pipelines from data preprocessing to model deployment
- Apply regression techniques (linear, polynomial, regularized) with real-world datasets
- Build classification systems using ensemble methods and advanced algorithms
- Construct and train neural networks and deep learning architectures
- Work with transformers, LLMs, and generative models
- Present technical projects

## Course Structure

### Week 1: Machine Learning Fundamentals, Regression and Classification
**Sessions:**
- Session 1: Complete Introduction to Machine Learning
  - All three ML types: supervised, unsupervised, reinforcement learning
  - ML pipeline
  - Linear regression mathematical foundations and implementation
- Session 2: Advanced Regression Techniques
  - Polynomial regression and feature engineering
  - Bias-variance tradeoff analysis and model evaluation
  - Classification: Logistic Regression

### Week 2: Classification and Neural Networks
**Sessions:**
- Session 3: Classification and Ensemble Methods
  - Support Vector Machines with kernel methods
  - k-Nearest Neighbors and decision boundaries
  - Decision trees and Random Forest ensemble methods
- Session 4: Neural Networks and Deep Learning
  - Introduction to neural networks and perceptrons
  - Multilayer perceptrons and backpropagation algorithm
  - Deep neural networks with TensorFlow/Keras
  - Gradient descent optimization and regularization techniques


### Week 3: Advanced Deep Learning and Generative AI
**Sessions:**
- Session 5: Advanced Deep Learning and Computer Vision
  - Convolutional Neural Networks (CNNs) for image processing
  - Transfer learning and pre-trained models
  - Advanced CNN architectures and computer vision applications
- Session 6: Generative AI, Transformers, LLMs, and Reinforcement Learning
  - Transformer architectures and attention mechanisms
  - Large Language Models (LLMs) and fine-tuning techniques
  - Generative Adversarial Networks (GANs) and diffusion models
  - Reinforcement Learning fundamentals and applications


### Week 4: Final Project Presentation and Interview
**Structure:**
- **Project Presentation Session** (Individual/group 15-minute presentations)
  - 10-minute technical presentation of final project
  - 5-minute Q&A and peer feedback
    - Algorithm implementation and code review
    - Theoretical knowledge assessment


**Project Requirements:**
- End-to-end ML project demonstrating course concepts
- Real-world dataset with complete preprocessing pipeline
- Advanced model implementation (neural networks, ensemble methods, or deep learning)
- model evaluation, optimisation, and experiment analysis
- Professional presentation with clear technical communication

**Q&A Components:**
- Coding: Implement ML algorithms from scratch
- Theoretical questions: Conceptual understanding
- Project deep-dive: Detailed discussion of design decisions and results
- Problem-solving and model selection

## File Organization

```
Machine_Learning_Course_Unitec/
├── Environment_Setup/          # Environment configuration
│   ├── SETUP_GUIDE.md         # Detailed setup instructions
│   ├── mlcourse.yml          # Conda environment file
│   └── requirements.txt      # Pip requirements
├── Course_Sessions/           # Weekly session notebooks
│   ├── Week_1/
│   ├── Week_2/
│   ├── Week_3/
│   └── Week_4/
│       └── Final_Project_Presentations/
├── Assignments/               # Assignment templates and submissions
│   ├── Assignment_1
│   ├── Assignment_2
│   └── Assignment_3
├── Datasets/                  # Course datasets
└── Guides/                    # Course documentation
    ├── COURSE_GUIDE.md        # This file
    ├── WORKFLOW_GUIDE.md      # Daily workflow instructions
    ├── GITHUB_SETUP_GUIDE.md  # Git setup and usage
    └── QUICK_SETUP_GUIDE.md   # Quick start instructions
```

## Workflow for Each Session

### 1. Environment Setup (First time only)
```bash
# Navigate to Environment_Setup folder
cd Environment_Setup

# Follow SETUP_GUIDE.md instructions
conda env create -f mlcourse.yml
conda activate mlcourse
```

### 2. Starting a Session
```bash
# Activate environment
conda activate mlcourse

# Open VSCode in course directory
code .
```

**In VSCode:**
1. Navigate to the appropriate Week/Session notebook
2. **Select the correct kernel**:
   - Click the kernel name in the top-right corner of the notebook
   - Choose "ML Course (Python 3.10)" from the dropdown
   - Alternative: Press Ctrl+Shift+P → "Notebook: Select Kernel"

### 3. Working with Notebooks in VSCode
- Each notebook has pre-imported libraries
- Follow the markdown instructions
- Complete the code cells marked with `# TODO:`
- Run cells using Shift+Enter or click the Run button
- Use VSCode's variable inspector to examine data
- Leverage IntelliSense for code completion

### 4. Saving Your Work in VSCode
- VSCode auto-saves by default
- Manual save: Ctrl+S
- Git integration shows unsaved changes
- For assignments, save with your student ID: `StudentID_Assignment1.ipynb`

## Assignment Submission Guidelines

### File Naming Convention
```
{StudentID}_{YourName}_Assignment{N}.ipynb
```
**Example:** `1234567_JohnDoe_Assignment1.ipynb`

### Assignment Requirements
1. **Complete all TODO sections**
2. **Include your student ID** in the first cell
3. **Add markdown explanations** for your code
4. **Include visualizations** where requested
5. **Test your code** before submission

### Submission Checklist
- [ ] All code cells run without errors
- [ ] Plots are clearly labeled with titles and axis labels
- [ ] Random seed is set using last 2 digits of your student ID
- [ ] File naming convention followed
- [ ] Output is clear and well-formatted

## Code Standards

### Imports
Each notebook includes standard imports at the top:
```python
# Standard imports (already included in notebooks)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed (UPDATE THIS!)
np.random.seed(42)  # Change to last 2 digits of your student ID
```

### Plotting Standards
```python
# Set high DPI for better quality plots
plt.rcParams['figure.dpi'] = 300

# Always include titles and labels
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Clear, Descriptive Title')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Data Analysis Standards
```python
# Always explore your data first
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nStatistical summary:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())
```

## Getting Help

### During Sessions
1. **Read the instructions carefully** in each notebook
2. **Try the code first** before asking for help
3. **Check the error messages** - they often tell you what's wrong
4. **Ask your neighbor** or work in pairs when appropriate

### Outside Class
1. **Review the session notebooks** and try the examples
2. **Practice with different datasets** from the Datasets folder
3. **Read the documentation** for libraries (pandas, scikit-learn, etc.)
4. **Use online resources** like Stack Overflow for specific errors

### Common Issues
- **ImportError:** Make sure your environment is activated
- **Kernel died:** Restart the kernel and run cells again
- **Plots not showing:** Make sure you have `%matplotlib inline` or call `plt.show()`
- **File not found:** Check you're in the correct directory

## Assessment

### Session Participation and Labs (30%)
- Active participation in sessions (Sessions 1-6)
- Completion of hands-on lab exercises
- Peer collaboration and problem-solving demonstrations
- In-class coding exercises and concept applications

### Progressive Assignments (40%)
- **Assignment 1: ML Pipeline and Regression** (Due end of Week 1)
  - Complete ML pipeline implementation with real datasets
  - Linear and polynomial regression analysis
  - Logistic regression for classification
  - Model evaluation and statistical analysis
- **Assignment 2: Classification and Neural Networks** (Due end of Week 2)
  - Ensemble methods (SVM, k-NN, Random Forest) comparison
  - Neural network implementation and optimization
  - Performance metrics analysis and model selection
- **Assignment 3: Advanced Deep Learning** (Due beginning of Week 4)
  - CNN implementation for computer vision tasks
  - Advanced deep learning architecture design
  - Model evaluation, optimization, and experiment analysis

### Final Project and Presentation (30%)
- **Final Project (25%)**
  - End-to-end ML solution demonstrating course concepts
  - Real-world dataset with complete preprocessing pipeline
  - Advanced model implementation (neural networks, ensemble methods, or deep learning)
  - Model evaluation, optimization, and experiment analysis
  - Professional documentation and technical communication
- **Project Presentation and Q&A (5%)**
  - 10-minute technical presentation of project results
  - 5-minute Q&A covering:
    - Algorithm implementation and code review
    - Theoretical knowledge and conceptual understanding
    - Project design decisions and results discussion
    - Problem-solving and model selection rationale

## Tips for Success

1. **Practice regularly** - coding skills improve with practice
2. **Understand the concepts** - don't just copy code
3. **Experiment** - try different parameters and datasets
4. **Document your work** - use markdown cells to explain your thinking
5. **Ask questions** - there are no stupid questions in learning
6. **Start assignments early** - they take longer than you think
7. **Collaborate** - discuss concepts with classmates
8. **Keep a learning journal** - note what you've learned each session

## Resources

### Essential Documentation
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)

### Recommended Reading
- "Hands-On Machine Learning" by Aurélien Géron (available in course materials)
- [Kaggle Learn](https://www.kaggle.com/learn) - Free micro-courses
- [Machine Learning Mastery](https://machinelearningmastery.com/) - Practical tutorials

### Practice Platforms
- [Kaggle](https://www.kaggle.com/) - Competitions and datasets
- [Google Colab](https://colab.research.google.com/) - Free cloud notebooks
- [Jupyter nbviewer](https://nbviewer.jupyter.org/) - View notebooks online

## Course Schedule Template

### Week 1: Machine Learning Fundamentals, Regression and Classification
- **Session 1:** Complete Introduction to Machine Learning
  - All three ML types: supervised, unsupervised, reinforcement learning
  - ML pipeline implementation
  - Linear regression mathematical foundations and implementation
- **Session 2:** Advanced Regression Techniques
  - Polynomial regression and feature engineering
  - Bias-variance tradeoff analysis and model evaluation
  - Classification: Logistic regression introduction
- **Assignment 1 Released:** ML Pipeline and Regression (Due end of week)

### Week 2: Classification and Neural Networks
- **Session 3:** Classification and Ensemble Methods
  - Support Vector Machines with kernel methods
  - k-Nearest Neighbors and decision boundaries
  - Decision trees and Random Forest ensemble methods
- **Session 4:** Neural Networks and Deep Learning
  - Introduction to neural networks and perceptrons
  - Multilayer perceptrons and backpropagation algorithm
  - Deep neural networks with TensorFlow/Keras
  - Gradient descent optimization and regularization techniques
- **Assignment 1 Due, Assignment 2 Released:** Classification and Neural Networks (Due end of week)

### Week 3: Advanced Deep Learning and Generative AI
- **Session 5:** Advanced Deep Learning and Computer Vision
  - Convolutional Neural Networks (CNNs) for image processing
  - Transfer learning and pre-trained models
  - Advanced CNN architectures and computer vision applications
- **Session 6:** Generative AI, Transformers, LLMs, and Reinforcement Learning
  - Transformer architectures and attention mechanisms
  - Large Language Models (LLMs) and fine-tuning techniques
  - Generative Adversarial Networks (GANs) and diffusion models
  - Reinforcement Learning fundamentals and applications
- **Assignment 2 Due, Assignment 3 Released:** Advanced Deep Learning (Due beginning Week 4)

### Week 4: Final Project Presentation
- **Assignment 3 Due:** Advanced Deep Learning Application submissions
- **Final Project Presentations:** Individual/group 15-minute presentations
  - 10-minute technical presentation of final project
  - 5-minute Q&A covering algorithm implementation, theoretical knowledge, and project decisions
- **Course Completion:** Portfolio review and next steps discussion

---

**Good luck with your machine learning journey!**