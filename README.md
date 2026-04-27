# Machine Learning Course

## Quick Start

### Prerequisites
- **Git** installed on your system
- **Miniconda** installed ([download](https://docs.conda.io/en/latest/miniconda.html))
- **VSCode** installed ([download](https://code.visualstudio.com/)) with Python and Jupyter extensions

### Setup

```bash
# Clone the repository
git clone https://github.com/alikeivanmarz/ml-course-unitec.git
cd ml-course-unitec

# Create and activate the environment
conda env create -f Environment_Setup/mlcourse.yml
conda activate mlcourse

# Register Jupyter kernel
python -m ipykernel install --user --name mlcourse --display-name "ML Course (Python 3.10)"

# Open in VSCode
code .
```

### Verify Setup
1. Open `Course_Sessions/Week_1/Week_1_Session_1.ipynb` in VSCode
2. Select kernel: **ML Course (Python 3.10)** (top-right corner)
3. Run the first cell

## Repository Structure

```
Machine_Learning_Course_Unitec/
├── Environment_Setup/          # Environment configuration
│   ├── SETUP_GUIDE.md         # Detailed setup instructions
│   └── mlcourse.yml           # Conda environment file
├── Course_Sessions/           # Weekly session notebooks
│   ├── Week_1/
│   ├── Week_2/
│   ├── Week_3/
│   └── Week_4/
├── Assignments/               # Assignment templates
│   ├── Assignment_1/
│   ├── Assignment_2/
│   └── Assignment_3/
├── Datasets/                  # Course datasets
└── Guides/                    # Course documentation (numbered in recommended reading order)
    ├── 01_COURSE_GUIDE.md             # Course overview and assessment
    ├── 02_QUICK_SETUP_GUIDE.md        # Quick start instructions
    ├── 03_GITHUB_SETUP_GUIDE.md       # Git setup and usage
    ├── 04_GIT_PULL_GUIDE.md           # Pulling latest course updates
    ├── 05_WORKFLOW_GUIDE.md           # Daily workflow instructions
    ├── 06_PYTHON_ESSENTIALS_FOR_ML.md # Python, NumPy, Pandas, plotting
    ├── 07_DATA_PREPROCESSING_GUIDE.md # Cleaning, scaling, encoding, pipelines
    ├── 08_ML_PIPELINE_GUIDE.md        # End-to-end ML pipeline examples
    ├── 09_MODEL_EVALUATION_GUIDE.md   # Metrics, CV, hyperparameter tuning
    ├── 10_ML_DEBUGGING_GUIDE.md       # Overfitting, leakage, NaN, shapes
    ├── 11_DEEP_LEARNING_KERAS_GUIDE.md # Neural networks with Keras
    ├── 12_COMPUTER_VISION_GUIDE.md    # CNNs, transfer learning, YOLO
    ├── 13_NLP_TRANSFORMERS_GUIDE.md   # Text processing and HuggingFace
    └── 14_GENERATIVE_AI_GUIDE.md      # GANs, LLMs, diffusion models
```

## Course Structure

- **Week 1**: Machine Learning Fundamentals, Regression and Classification
- **Week 2**: Classification, Ensemble Methods, Neural Networks and Deep Learning
- **Week 3**: Advanced Deep Learning, Computer Vision, Generative AI, RL and LLMs
- **Week 4**: Final Project Presentations

## Environment

The `mlcourse.yml` file works on all platforms (Windows, macOS, Linux). It includes:
- **Core ML**: scikit-learn, xgboost, lightgbm, catboost
- **Deep Learning**: TensorFlow, PyTorch, Keras
- **Computer Vision**: OpenCV, Ultralytics (YOLO)
- **NLP/LLM**: Transformers, LangChain, OpenAI, Anthropic
- **Generative AI**: Diffusers, Real-ESRGAN
- **Deployment**: Streamlit, Gradio, FastAPI

PyTorch and TensorFlow will automatically detect and use your GPU if available.

## Daily Workflow

```bash
git pull origin main          # Get latest materials
conda activate mlcourse       # Activate environment
code .                        # Open in VSCode
```

## Guides

| Guide | Description |
|-------|-------------|
| [Setup Guide](Environment_Setup/SETUP_GUIDE.md) | Detailed environment setup |
| [01 Course Guide](Guides/01_COURSE_GUIDE.md) | Course overview and assessment — start here |
| [02 Quick Setup](Guides/02_QUICK_SETUP_GUIDE.md) | Fast setup instructions |
| [03 GitHub Guide](Guides/03_GITHUB_SETUP_GUIDE.md) | Git setup and usage |
| [04 Git Pull Guide](Guides/04_GIT_PULL_GUIDE.md) | Pulling latest updates each session |
| [05 Workflow Guide](Guides/05_WORKFLOW_GUIDE.md) | Daily workflow |
| [06 Python Essentials](Guides/06_PYTHON_ESSENTIALS_FOR_ML.md) | Python, NumPy, Pandas, plotting |
| [07 Data Preprocessing](Guides/07_DATA_PREPROCESSING_GUIDE.md) | Cleaning, scaling, encoding, pipelines |
| [08 ML Pipeline](Guides/08_ML_PIPELINE_GUIDE.md) | End-to-end pipeline examples |
| [09 Model Evaluation](Guides/09_MODEL_EVALUATION_GUIDE.md) | Metrics, CV, hyperparameter tuning |
| [10 ML Debugging](Guides/10_ML_DEBUGGING_GUIDE.md) | Overfitting, leakage, NaN, shapes |
| [11 Deep Learning with Keras](Guides/11_DEEP_LEARNING_KERAS_GUIDE.md) | Neural networks with Keras |
| [12 Computer Vision](Guides/12_COMPUTER_VISION_GUIDE.md) | CNNs, transfer learning, YOLO |
| [13 NLP & Transformers](Guides/13_NLP_TRANSFORMERS_GUIDE.md) | Text processing and HuggingFace |
| [14 Generative AI](Guides/14_GENERATIVE_AI_GUIDE.md) | GANs, LLMs, diffusion models |

## Getting Help

1. Check the [Setup Guide](Environment_Setup/SETUP_GUIDE.md) troubleshooting section
2. Ask your instructor
3. Email: akeivanmarz@unitec.ac.nz
