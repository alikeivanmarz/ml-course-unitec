# 🛰️ YOLO Vision Lab

An interactive Streamlit dashboard for exploring and **comparing YOLO models** —
the dashboard companion to the `Bonus_Live_CV_YOLO.ipynb` notebook.

Point your webcam at the screen, upload an image or video, tune the thresholds,
and watch detections update live. Switch to **compare mode** to run two models
(e.g. `nano` vs `medium`) on the same frame and see the speed/accuracy trade-off.

![tasks](https://img.shields.io/badge/tasks-detect%20%7C%20segment%20%7C%20pose-00D8A0)
![models](https://img.shields.io/badge/models-YOLOv8%20%E2%86%92%20YOLO11-7C5CFF)

---

## Features

| | |
|---|---|
| 🎯 **Single & compare modes** | Run one model, or two side-by-side with a head-to-head summary |
| 📷 **5 input sources** | Live webcam · webcam snapshot · image upload · video upload · sample images |
| 🧠 **Many models** | YOLOv8/v9/v10/YOLO11 in nano→large, plus segmentation & pose (auto-downloaded) |
| 🔀 **Object tracking** | Toggle on to give each object a stable ID as it moves (Live & Video) |
| 🎚️ **Live controls** | Confidence, IoU, image size, device, and per-class filtering |
| 📊 **Rich readouts** | FPS / latency / object metrics, class + confidence charts, downloadable detections table |

---

## How to run

Copy-paste these four steps into a terminal:

```bash
# 1. Activate the course conda environment
conda activate mlcourse

# 2. Go to the dashboard folder (adjust the start of the path if needed)
cd "Course_Sessions/Week_1/yolo_dashboard"

# 3. Install anything missing (the last two packages enable the live webcam)
pip install streamlit-webrtc av
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app.py
```

Your browser opens automatically at **<http://localhost:8501>**. If it doesn't,
open that address yourself. The first run downloads the model weights (a few MB)
— after that they're cached and start-up is instant.

**To stop the app:** press `Ctrl + C` in the terminal.

**To re-run later:** you only need steps 1, 2 and 4 (skip the install).

> 💡 **Fastest path to a working demo:** once it's open, start with
> **🧪 Sample image** in the sidebar, then try **📸 Webcam (snapshot)**. Both
> work everywhere with zero extra setup.

> ⚠️ **No conda?** Any Python 3.10+ environment works — replace step 1 with your
> own `venv` and run `pip install -r requirements.txt`.

---

## The input sources

- **📷 Webcam (live)** — continuous stream via `streamlit-webrtc`. Click **START**
  and allow camera access. Sidebar changes apply to the live feed. If the
  optional packages aren't installed, the app tells you and the other sources
  still work.
- **📸 Webcam (snapshot)** — take a single photo. The most reliable option and
  great for classrooms / locked-down networks.
- **🖼️ Image upload** / **🎬 Video upload** — drop in your own files. Video is
  processed frame-by-frame with a "process every Nth frame" speed control.
- **🧪 Sample image** — built-in Ultralytics test images.

---

## Compare mode 💡

Pick **⚖️ Compare two models** in the sidebar, choose **Model A** and **Model B**,
and run any source. You get both annotated views, per-model metrics/charts, and a
**head-to-head table** showing which model was faster and how many objects each
found. Try `YOLOv8n` vs `YOLOv8m` — the classic size-vs-speed lesson.

---

## How it's organised (easy to modify)

```
yolo_dashboard/
├── app.py                 # Streamlit UI + live-webcam processor
├── yolo_utils.py          # model registry, inference, stats  ← edit this to add models
├── requirements.txt
├── README.md
└── .streamlit/config.toml # colours / theme  ← edit this to rebrand
```

**Add a model:** open `yolo_utils.py` and add one line to `MODEL_REGISTRY`, e.g.

```python
"YOLO11l · large": "yolo11l.pt",
```

It appears in the dropdowns automatically.

**Rebrand:** change `primaryColor` in `.streamlit/config.toml` and the `ACCENT`
constants at the top of `app.py`.

---

## Troubleshooting

- **Live webcam won't start** — some browsers block camera access on `http`
  over a network. Use `localhost`, or fall back to snapshot mode.
- **Slow / low FPS** — use a `nano` model, pick a smaller **Image size** (e.g.
  480), and prefer GPU/MPS via the **Device** selector if available.
- **`ModuleNotFoundError: streamlit_webrtc`** — `pip install streamlit-webrtc av`
  (only needed for the live source).
- **Apple Silicon** — if MPS gives odd errors in live mode, set **Device → CPU**.
