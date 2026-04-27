# Guides

Thirty-four reference guides covering the full path from setup, through ML foundations, modelling, deep learning, engineering, and finally research communication. Files are numbered in a logical reading order, but most guides are self-contained — once setup is complete, treat the rest as a **lookup library** to consult when a topic comes up.

## How to use these guides

- **Before starting** → read 01–06 in order. Setup, workflow, and the IDE are prerequisites for everything else.
- **Foundations (07–11)** → Python, math, statistics, datasets, EDA. The prerequisites every modelling decision rests on.
- **Core ML workflow (12–17)** → preprocessing → pipeline → evaluation → debugging → testing → interpretability. The lifecycle of every supervised ML project.
- **Other modelling families (18–20)** → unsupervised learning, time series, reinforcement learning. Reference when a project's structure does not fit standard supervised regression or classification.
- **Deep learning (21–26)** → frameworks, modalities, generative models, LLM applications.
- **Engineering & deployment (27–28)** → packaging an ML project for re-use and serving.
- **Research & communication (29–34)** → reading, reviewing, proposing, writing, and presenting research.

## Reading order

### Phase 1 — Orientation & Setup

| # | Guide | What it covers |
|---|-------|----------------|
| 01 | [Course Guide](01_COURSE_GUIDE.md) | Course overview, learning objectives, assessment |
| 02 | [Quick Setup Guide](02_QUICK_SETUP_GUIDE.md) | Install Git, Miniconda, VSCode, and the course env |
| 03 | [GitHub Setup Guide](03_GITHUB_SETUP_GUIDE.md) | Create a GitHub account and clone the repo |
| 04 | [Git Pull Guide](04_GIT_PULL_GUIDE.md) | Pulling the latest course materials each session |
| 05 | [Workflow Guide](05_WORKFLOW_GUIDE.md) | Daily workflow once everything is installed |
| 06 | [VSCode for Python and Jupyter](06_VSCODE_TIPS_GUIDE.md) | Extensions, kernels, debugging, settings, keyboard shortcuts |

### Phase 2 — Foundations

| # | Guide | What it covers |
|---|-------|----------------|
| 07 | [Python Essentials for ML](07_PYTHON_ESSENTIALS_FOR_ML.md) | Core Python, NumPy, Pandas, matplotlib/seaborn |
| 08 | [Mathematics for ML](08_MATH_FOR_ML_GUIDE.md) | Linear algebra, calculus, probability, optimization, information theory |
| 09 | [Statistics for ML](09_STATISTICS_FOR_ML_GUIDE.md) | Hypothesis testing, common tests, effect sizes, bootstrap, multiple comparisons |
| 10 | [Dataset Sourcing and Loading](10_DATASETS_GUIDE.md) | Sources, file formats, splitting, sampling, licensing, documentation |
| 11 | [Exploratory Data Analysis](11_EDA_GUIDE.md) | Workflow for inspecting an unfamiliar tabular dataset before modelling |

### Phase 3 — Core ML Workflow

| # | Guide | What it covers |
|---|-------|----------------|
| 12 | [Data Preprocessing Guide](12_DATA_PREPROCESSING_GUIDE.md) | Cleaning, scaling, encoding, sklearn Pipelines |
| 13 | [ML Pipeline Guide](13_ML_PIPELINE_GUIDE.md) | End-to-end worked examples (regression and classification) |
| 14 | [Model Evaluation Guide](14_MODEL_EVALUATION_GUIDE.md) | Metrics, cross-validation, hyperparameter tuning |
| 15 | [ML Debugging Guide](15_ML_DEBUGGING_GUIDE.md) | Overfitting, data leakage, NaN/Inf, shape errors |
| 16 | [Testing ML Code](16_TESTING_ML_CODE_GUIDE.md) | pytest patterns, data tests, model contracts, property-based testing, CI |
| 17 | [Model Interpretability](17_INTERPRETABILITY_GUIDE.md) | Permutation importance, partial dependence, SHAP, LIME, counterfactuals |

### Phase 4 — Modelling Beyond Standard Supervised

| # | Guide | What it covers |
|---|-------|----------------|
| 18 | [Unsupervised Learning](18_UNSUPERVISED_LEARNING_GUIDE.md) | Clustering, dimensionality reduction, anomaly detection, evaluation |
| 19 | [Time Series and Forecasting](19_TIME_SERIES_GUIDE.md) | ARIMA family, exponential smoothing, Prophet, ML and DL approaches, evaluation |
| 20 | [Reinforcement Learning](20_REINFORCEMENT_LEARNING_GUIDE.md) | MDPs, Q-learning, DQN, policy gradients, actor-critic, Gymnasium |

### Phase 5 — Deep Learning

| # | Guide | What it covers |
|---|-------|----------------|
| 21 | [Deep Learning with Keras](21_DEEP_LEARNING_KERAS_GUIDE.md) | Building, training, and saving neural networks |
| 22 | [PyTorch Reference](22_PYTORCH_GUIDE.md) | Tensors, autograd, `nn.Module`, training loop, saving |
| 23 | [Computer Vision Guide](23_COMPUTER_VISION_GUIDE.md) | CNNs, transfer learning, object detection with YOLO |
| 24 | [NLP & Transformers Guide](24_NLP_TRANSFORMERS_GUIDE.md) | Text preprocessing, HuggingFace, fine-tuning |
| 25 | [Generative AI Guide](25_GENERATIVE_AI_GUIDE.md) | GANs, LLMs, prompt engineering, Stable Diffusion, LangChain |
| 26 | [Building with LLMs](26_LLM_PROMPTING_GUIDE.md) | Structured output, tool calling, RAG, evaluation, cost — applied LLM craft |

### Phase 6 — Engineering & Deployment

| # | Guide | What it covers |
|---|-------|----------------|
| 27 | [ML Project Structure](27_PROJECT_STRUCTURE_GUIDE.md) | Repository layout, dependency management, configs, reproducibility |
| 28 | [Model Deployment](28_DEPLOYMENT_GUIDE.md) | Serialization, Streamlit / Gradio / FastAPI, containerization, monitoring |

### Phase 7 — Research & Communication

| # | Guide | What it covers |
|---|-------|----------------|
| 29 | [Reading ML Research Papers](29_READING_ML_PAPERS_GUIDE.md) | Three-pass method, paper anatomy, decoding notation, evaluating claims |
| 30 | [Literature Review](30_LITERATURE_REVIEW_GUIDE.md) | Search strategies, snowballing, triage, synthesis tables, common pitfalls |
| 31 | [Research Proposal Writing](31_RESEARCH_PROPOSAL_GUIDE.md) | Structure, research questions, scope, evaluation plans, what reviewers look for |
| 32 | [Technical Report and Paper Writing](32_REPORT_AND_PAPER_WRITING_GUIDE.md) | Section anatomy, abstract patterns, figures, tables, citation mechanics |
| 33 | [Academic Writing Style](33_ACADEMIC_WRITING_STYLE_GUIDE.md) | Tense, voice, hedging, signposting, sentence-level style, common pitfalls |
| 34 | [Presenting Technical ML Projects](34_PRESENTATION_GUIDE.md) | Talk structure, slide design, honest metrics, demos, Q&A |

## Conventions

- Setup guides 01–05 assume the course conda environment named `mlcourse` and the **ML Course (Python 3.10)** Jupyter kernel.
- Reference guides 06–34 are written to be standalone — code uses generic data (`make_classification`, public sources) and generic placeholders (`X`, `y`, `df`, `model`) so they can be used outside this course.
