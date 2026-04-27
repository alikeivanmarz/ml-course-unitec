# Model Deployment

A trained model is useful only when something can call it. Deployment is the discipline of converting a model artefact and its inference code into a runnable interface — a notebook button, a web app, an HTTP endpoint, a containerized service. This guide covers serialization formats, three deployment frameworks (Streamlit, Gradio, FastAPI), containerization basics, environment pinning, and lightweight monitoring.

**Table of Contents**

1. [Serialization Formats](#1-serialization-formats)
2. [Inference-Only Loading](#2-inference-only-loading)
3. [Streamlit — Single-File UIs](#3-streamlit--single-file-uis)
4. [Gradio — Demo-First Interfaces](#4-gradio--demo-first-interfaces)
5. [FastAPI — Programmatic Endpoints](#5-fastapi--programmatic-endpoints)
6. [Containerization](#6-containerization)
7. [Environment Pinning](#7-environment-pinning)
8. [Lightweight Monitoring](#8-lightweight-monitoring)
9. [Resources](#9-resources)

---

## 1. Serialization Formats

| Format | Library | Strengths | Weaknesses |
|--------|---------|-----------|------------|
| `joblib` | scikit-learn | Compact, fast for NumPy-heavy objects | Python-only; brittle across sklearn versions |
| `pickle` | Standard library | Universal Python serialization | Insecure with untrusted data; version-fragile |
| `.keras` | Keras 3 | Native Keras format; portable across backends | Keras-specific |
| `torch.save` | PyTorch | Native PyTorch format; supports state dict + full model | PyTorch-specific |
| ONNX | `onnx`, `onnxruntime` | Cross-framework; hardware-accelerated inference | Conversion may fail on custom layers |
| SavedModel | TensorFlow | Production-grade TF format; serving-server compatible | TensorFlow-specific |
| TorchScript | PyTorch | Runnable without Python; deployable to C++ | PyTorch-specific |
| Safetensors | `safetensors` | Safe (no arbitrary code execution); fast load | Tensors only; no Python objects |

### 1.1 sklearn — joblib

```python
import joblib

# Save
joblib.dump(model, "model.joblib")

# Load
model = joblib.load("model.joblib")
```

`joblib` outperforms `pickle` for objects containing large NumPy arrays. Pin the sklearn version in deployment to match the training environment.

### 1.2 PyTorch — State Dict

```python
import torch

# Save weights only
torch.save(model.state_dict(), "model.pt")

# Load: re-instantiate the model class, then load weights
model = ModelClass(...)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()
```

### 1.3 Keras — Native Format

```python
# Save
model.save("model.keras")

# Load
import keras
model = keras.models.load_model("model.keras")
```

### 1.4 ONNX — Cross-Framework

```python
import torch.onnx
torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)
```

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
output = session.run(None, {"input": x.numpy()})[0]
```

ONNX Runtime supports CPU, CUDA, CoreML, and TensorRT execution providers, and is typically 2–10× faster than the source framework for inference.

---

## 2. Inference-Only Loading

Training and inference code paths should be separated. The inference code should not import training utilities, optimizers, or validation datasets.

A typical structure:

```
inference/
├── model_io.py         # load model artefact
├── preprocessing.py    # input normalization, encoding
├── predict.py          # single function: input → output
└── requirements.txt    # only inference dependencies
```

```python
# inference/predict.py
import joblib
from .preprocessing import preprocess

_model = joblib.load("artefacts/model.joblib")

def predict(payload: dict) -> dict:
    features = preprocess(payload)
    proba = _model.predict_proba([features])[0]
    return {"class": int(proba.argmax()), "confidence": float(proba.max())}
```

The model is loaded once at import time, not on every request. Stateless prediction functions parallelize trivially behind a web server.

---

## 3. Streamlit — Single-File UIs

Streamlit is appropriate for internal tools, dashboards, and demos with form-style inputs. Each interaction reruns the script top-to-bottom; widgets retain their state.

```python
# app.py
import streamlit as st
import joblib

st.title("Churn Predictor")

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

model = load_model()

age = st.slider("Age", 18, 90, 35)
tenure_months = st.number_input("Tenure (months)", min_value=0, value=12)

if st.button("Predict"):
    proba = model.predict_proba([[age, tenure_months]])[0, 1]
    st.metric("Churn probability", f"{proba:.1%}")
```

Run with `streamlit run app.py`. Deploys to Streamlit Community Cloud, Hugging Face Spaces, or any container host.

### 3.1 Caching

| Decorator | Use |
|-----------|-----|
| `@st.cache_resource` | Long-lived objects (models, DB connections) |
| `@st.cache_data` | Expensive computations on serializable data |

Without caching, the model is reloaded on every interaction.

---

## 4. Gradio — Demo-First Interfaces

Gradio is appropriate for ML model demos with media inputs (image, audio, video) or chat interfaces. Auto-generates a UI from a function signature.

```python
# app.py
import gradio as gr
import joblib

model = joblib.load("model.joblib")

def predict(text: str) -> dict:
    proba = model.predict_proba([text])[0]
    return {label: float(p) for label, p in zip(model.classes_, proba)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, label="Review text"),
    outputs=gr.Label(num_top_classes=3),
    title="Sentiment Classifier",
    examples=[["The film was a triumph."], ["A complete waste of time."]],
)

demo.launch()
```

Gradio integrates natively with Hugging Face Spaces (zero-config deployment) and provides shareable public URLs via `demo.launch(share=True)`.

---

## 5. FastAPI — Programmatic Endpoints

FastAPI is appropriate for production HTTP APIs consumed by other services rather than humans. Provides automatic OpenAPI documentation and Pydantic-based validation.

```python
# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Churn Predictor", version="1.0.0")
model = joblib.load("model.joblib")


class PredictionRequest(BaseModel):
    age: int
    tenure_months: int


class PredictionResponse(BaseModel):
    churn_probability: float
    decision: str


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    proba = float(model.predict_proba([[req.age, req.tenure_months]])[0, 1])
    return PredictionResponse(
        churn_probability=proba,
        decision="churn" if proba > 0.5 else "retain",
    )


@app.get("/health")
def health():
    return {"status": "ok"}
```

Run with `uvicorn main:app --host 0.0.0.0 --port 8000`. Interactive documentation auto-generated at `/docs`.

### 5.1 Decision Table

| Need | Framework |
|------|-----------|
| Internal dashboard with form inputs | Streamlit |
| Public demo of a media-input model | Gradio |
| Programmatic API consumed by other services | FastAPI |
| Chatbot-style UI | Gradio (`ChatInterface`) or Streamlit |
| High-throughput, low-latency serving | FastAPI behind a process manager (Gunicorn + Uvicorn workers) or a dedicated server (Triton, BentoML, Ray Serve) |

---

## 6. Containerization

A container packages the model, inference code, dependencies, and runtime into a portable image. Minimal Dockerfile for a FastAPI service:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies separately for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and model artefacts
COPY src/ src/
COPY artefacts/model.joblib artefacts/

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t churn-api:1.0.0 .
docker run -p 8000:8000 churn-api:1.0.0
```

### 6.1 Image Size

Common reductions:

| Technique | Effect |
|-----------|--------|
| `python:3.11-slim` over `python:3.11` | ~700 MB → ~150 MB base |
| Multi-stage build | Strip build tools from final image |
| `--no-cache-dir` on `pip install` | Saves ~50–200 MB |
| Wheel-based dependencies | Faster builds; smaller layers |
| `.dockerignore` for `__pycache__`, `.git`, datasets | Smaller build context |

For large model artefacts (>1 GB), consider mounting from a volume or pulling from object storage at startup rather than baking into the image.

---

## 7. Environment Pinning

Reproducible deployment requires deterministic dependency versions.

### 7.1 pip — `requirements.txt`

```bash
pip freeze > requirements.txt
```

`pip freeze` captures exact versions of every installed package. Restoration:

```bash
pip install -r requirements.txt
```

### 7.2 conda — `environment.yml`

```bash
conda env export --from-history > environment.yml      # only explicitly installed packages
conda env export > environment.yml                      # full lock
```

The `--from-history` form is portable across platforms; the full lock is reproducible only on the same OS and architecture.

### 7.3 Lockfiles

For stricter reproducibility, use a lockfile that records the full dependency tree:

| Tool | Lockfile |
|------|----------|
| `pip-tools` | `requirements.txt` compiled from `requirements.in` |
| `poetry` | `poetry.lock` |
| `uv` | `uv.lock` |
| `pipenv` | `Pipfile.lock` |
| `conda-lock` | `conda-lock.yml` |

Lockfiles record transitive dependencies and hashes, eliminating "works on my machine" failures caused by upstream version drift.

---

## 8. Lightweight Monitoring

Even a small deployed service benefits from minimal monitoring. Three signals to capture from the start:

### 8.1 Request Logging

```python
import logging
import time

logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round(duration_ms, 2),
        },
    )
    return response
```

Structured logs (JSON) integrate with downstream aggregators (Loki, Datadog, CloudWatch).

### 8.2 Metrics to Track

| Metric | Why |
|--------|-----|
| Request count | Traffic volume; baseline for alerting |
| Latency (p50, p95, p99) | Tail latency reveals contention or cold starts |
| Error rate | Operational health; alerting threshold |
| Prediction distribution | Drift detection — input or output distribution shift over time |
| Model version in use | Trace which model produced which prediction |

`prometheus_client` exposes counters, histograms, and gauges with minimal overhead. Aggregate with Prometheus + Grafana for visualization.

### 8.3 Drift Detection

| Signal | Detection |
|--------|-----------|
| Input drift | Distribution comparison (KL divergence, PSI) between recent inputs and training data |
| Prediction drift | Same comparison, on outputs |
| Performance drift | Requires labelled feedback; sample requests for human review |

Drift detection rarely justifies full real-time monitoring at small scale; daily batch comparison against a reference window is usually sufficient.

---

## 9. Resources

- [Streamlit documentation](https://docs.streamlit.io/) — widgets, layout, caching, deployment.
- [Gradio documentation](https://www.gradio.app/docs) — interfaces, blocks, chat interface.
- [FastAPI documentation](https://fastapi.tiangolo.com/) — request handling, validation, dependency injection.
- [ONNX Runtime](https://onnxruntime.ai/docs/) — inference across frameworks and hardware.
- [Docker — Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/) — image size, layer caching, security.
- [`uv` documentation](https://docs.astral.sh/uv/) — modern Python packaging and lockfiles.
- [Sculley et al., *Hidden Technical Debt in Machine Learning Systems* (2015)](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) — operational considerations beyond model code.
