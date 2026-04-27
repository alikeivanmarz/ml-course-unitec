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
└── Guides/                    # 34 reference guides — see Guides/README.md for the full index
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

The full guide index — 34 numbered reference guides in recommended reading order — lives at [Guides/README.md](Guides/README.md). Two essential entry points:

- [Setup Guide](Environment_Setup/SETUP_GUIDE.md) — detailed environment setup
- [Course Guide](Guides/01_COURSE_GUIDE.md) — course overview and assessment

## Getting Help

1. Check the [Setup Guide](Environment_Setup/SETUP_GUIDE.md) troubleshooting section
2. Ask your instructor
3. Email: akeivanmarz@unitec.ac.nz
