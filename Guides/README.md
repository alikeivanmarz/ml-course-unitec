# Guides

Reference material for the Machine Learning course. Files are numbered in the recommended reading order — start with `01_` and work your way down. You don't need to read everything top-to-bottom; once you're set up, treat guides 06–14 as a **lookup library** you return to whenever a session or assignment touches that topic.

## How to use these guides

- **Before the course starts** → read 01–05 in order. These get you set up and ready to attend.
- **During Week 1** → keep 06 (Python) and 07 (Preprocessing) open as you work through the notebooks.
- **From Week 2 onwards** → open the guide that matches the day's topic. Each guide has a table of contents you can jump straight into.
- **When something breaks** → 10 (Debugging) is your first stop.
- **For assignments and the final project** → 08 (Pipeline) shows how the pieces fit together end-to-end.

## Reading order

### Setup & orientation — read before Week 1

| # | Guide | What it covers |
|---|-------|----------------|
| 01 | [Course Guide](01_COURSE_GUIDE.md) | Course overview, learning objectives, assessment |
| 02 | [Quick Setup Guide](02_QUICK_SETUP_GUIDE.md) | Install Git, Miniconda, VSCode, and the course env |
| 03 | [GitHub Setup Guide](03_GITHUB_SETUP_GUIDE.md) | Create a GitHub account and clone the repo |
| 04 | [Git Pull Guide](04_GIT_PULL_GUIDE.md) | Pulling the latest course materials each session |
| 05 | [Workflow Guide](05_WORKFLOW_GUIDE.md) | Daily workflow once everything is installed |

### ML foundations — Weeks 1–2

| # | Guide | What it covers |
|---|-------|----------------|
| 06 | [Python Essentials for ML](06_PYTHON_ESSENTIALS_FOR_ML.md) | Core Python, NumPy, Pandas, matplotlib/seaborn |
| 07 | [Data Preprocessing Guide](07_DATA_PREPROCESSING_GUIDE.md) | Cleaning, scaling, encoding, sklearn Pipelines |
| 08 | [ML Pipeline Guide](08_ML_PIPELINE_GUIDE.md) | End-to-end worked examples (regression and classification) |
| 09 | [Model Evaluation Guide](09_MODEL_EVALUATION_GUIDE.md) | Metrics, cross-validation, hyperparameter tuning |
| 10 | [ML Debugging Guide](10_ML_DEBUGGING_GUIDE.md) | Overfitting, data leakage, NaN/Inf, shape errors |

### Deep learning — Weeks 3–4

| # | Guide | What it covers |
|---|-------|----------------|
| 11 | [Deep Learning with Keras](11_DEEP_LEARNING_KERAS_GUIDE.md) | Building, training, and saving neural networks |
| 12 | [Computer Vision Guide](12_COMPUTER_VISION_GUIDE.md) | CNNs, transfer learning, object detection with YOLO |
| 13 | [NLP & Transformers Guide](13_NLP_TRANSFORMERS_GUIDE.md) | Text preprocessing, HuggingFace, fine-tuning |
| 14 | [Generative AI Guide](14_GENERATIVE_AI_GUIDE.md) | LLMs, prompt engineering, Stable Diffusion, LangChain |

### Reference & advanced topics — read on demand

These guides extend the reference library beyond the weekly progression. Each is self-contained and can be opened directly when its topic comes up — they are not intended to be read in order.

| # | Guide | What it covers |
|---|-------|----------------|
| 15 | [PyTorch Reference](15_PYTORCH_GUIDE.md) | Tensors, autograd, `nn.Module`, training loop, saving — sister to the Keras guide |
| 16 | [Exploratory Data Analysis](16_EDA_GUIDE.md) | Workflow for inspecting an unfamiliar tabular dataset before modelling |
| 17 | [Unsupervised Learning](17_UNSUPERVISED_LEARNING_GUIDE.md) | Clustering, dimensionality reduction, anomaly detection, evaluation |
| 18 | [Building with LLMs](18_LLM_PROMPTING_GUIDE.md) | Structured output, tool calling, RAG, evaluation, cost — applied LLM craft |
| 19 | [Model Deployment](19_DEPLOYMENT_GUIDE.md) | Serialization, Streamlit / Gradio / FastAPI, containerization, monitoring |
| 20 | [ML Project Structure](20_PROJECT_STRUCTURE_GUIDE.md) | Repository layout, dependency management, configs, reproducibility |
| 21 | [Presenting Technical ML Projects](21_PRESENTATION_GUIDE.md) | Talk structure, slide design, honest metrics, demos, Q&A |
| 22 | [Model Interpretability](22_INTERPRETABILITY_GUIDE.md) | Permutation importance, partial dependence, SHAP, LIME, counterfactuals |
| 23 | [Mathematics for ML](23_MATH_FOR_ML_GUIDE.md) | Linear algebra, calculus, probability, optimization, information theory |
| 24 | [Dataset Sourcing and Loading](24_DATASETS_GUIDE.md) | Sources, file formats, splitting, sampling, licensing, documentation |
| 25 | [VSCode for Python and Jupyter](25_VSCODE_TIPS_GUIDE.md) | Extensions, kernels, debugging, settings, keyboard shortcuts |

## Conventions

- All code examples assume the **`mlcourse`** conda environment is active and the **ML Course (Python 3.10)** Jupyter kernel is selected.
- Datasets referenced in the guides live in [`../Datasets/`](../Datasets/).
- Code snippets use `pd.read_csv('../Datasets/filename.csv')` — paths are relative to a notebook in `Course_Sessions/Week_X/`.
