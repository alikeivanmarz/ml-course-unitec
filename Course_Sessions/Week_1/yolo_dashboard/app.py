"""
YOLO Vision Lab  ·  Streamlit dashboard
=======================================
A dashboard for exploring and comparing YOLO models
(Unitec ML Course, Week 1 bonus).

Run:
    streamlit run app.py

Features
--------
* Single-model and side-by-side compare modes
* Input sources: live webcam, webcam snapshot, image upload, video upload, samples
* Model / task selector across YOLOv8–YOLO11 (auto-downloads weights)
* Confidence, IoU, image-size, device and per-class filtering
* Live FPS / latency / object metrics, class + confidence charts, detections table

Everything model-related lives in `yolo_utils.py` — add a model there and it
shows up in the dropdowns automatically.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import yolo_utils as yu

# ---------------------------------------------------------------------------
# Page config + theme accents  (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="YOLO Vision Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#00D8A0"       # primary accent (teal-green)
ACCENT_ALT = "#7C5CFF"   # secondary accent for the "B" model in compare mode
SAMPLE_IMAGES = {
    "Bus  ·  street scene": "https://ultralytics.com/images/bus.jpg",
    "Zidane  ·  people": "https://ultralytics.com/images/zidane.jpg",
}

# Optional live-video support. If streamlit-webrtc / av are missing the app
# still runs — the live tab just shows an install hint.
try:
    import av
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    WEBRTC_OK = True
except Exception:  # pragma: no cover - depends on optional deps
    WEBRTC_OK = False


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        /* ---- header banner ---- */
        .vl-header {{
            padding: 1.1rem 1.4rem;
            border-radius: 16px;
            background: linear-gradient(120deg, #10221d 0%, #141821 55%, #1a1330 100%);
            border: 1px solid rgba(255,255,255,0.06);
            margin-bottom: 1.1rem;
        }}
        .vl-title {{
            font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em; margin: 0;
            background: linear-gradient(90deg, {ACCENT} 0%, {ACCENT_ALT} 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .vl-sub {{ color: #9aa4b2; font-size: 0.95rem; margin-top: 0.15rem; }}
        .vl-pill {{
            display:inline-block; padding: 2px 10px; border-radius: 999px;
            font-size: 0.72rem; font-weight: 600; letter-spacing:.03em;
            background: rgba(0,216,160,0.12); color:{ACCENT};
            border:1px solid rgba(0,216,160,0.35); margin-right:.4rem;
        }}
        /* ---- metric cards ---- */
        div[data-testid="stMetric"] {{
            background: #161a22; border: 1px solid rgba(255,255,255,0.07);
            padding: 14px 16px; border-radius: 14px;
        }}
        div[data-testid="stMetric"] label p {{ color:#8b95a5 !important; font-weight:600; }}
        div[data-testid="stMetricValue"] {{ color: #f4f6fa; }}
        /* ---- misc ---- */
        .stImage img {{ border-radius: 12px; border:1px solid rgba(255,255,255,0.06); }}
        section[data-testid="stSidebar"] {{ border-right:1px solid rgba(255,255,255,0.06); }}
        .vl-caption {{ color:#7d8798; font-size:0.8rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header() -> None:
    st.markdown(
        """
        <div class="vl-header">
            <p class="vl-title">YOLO Vision Lab</p>
            <div class="vl-sub">
                <span class="vl-pill">REAL-TIME</span>
                <span class="vl-pill">COMPARE MODELS</span>
                Interactive object-detection dashboard · Unitec ML Course
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(weights: str):
    """Load (and cache) a YOLO model. First call downloads the weights."""
    from ultralytics import YOLO

    return YOLO(weights)


@st.cache_data(show_spinner=False)
def fetch_sample_bytes(url: str) -> bytes:
    import urllib.request

    with urllib.request.urlopen(url, timeout=20) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def file_to_bgr(uploaded) -> np.ndarray | None:
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def label_frame(img: np.ndarray, text: str, color=(0, 216, 160)) -> np.ndarray:
    """Draw a small banner label onto a frame (used in compare views)."""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (max(140, 12 * len(text)), 34), (20, 22, 28), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def metric_row(stats: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Objects", stats["objects"])
    c2.metric("Classes", stats["classes"])
    c3.metric("Latency", f"{stats['latency_ms']:.0f} ms")
    c4.metric("Speed", f"{stats['fps']:.1f} FPS")


def _style_fig(fig, height=240):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=8, r=8, t=30, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d2df", size=12),
        showlegend=False,
    )
    return fig


def class_bar(counts: Counter, color=ACCENT):
    if not counts:
        return None
    df = (
        pd.DataFrame({"class": list(counts.keys()), "count": list(counts.values())})
        .sort_values("count")
    )
    fig = px.bar(df, x="count", y="class", orientation="h", title="Detections by class")
    fig.update_traces(marker_color=color)
    return _style_fig(fig, height=max(200, 40 + 26 * len(df)))


def confidence_hist(df: pd.DataFrame, color=ACCENT):
    if df.empty:
        return None
    fig = px.histogram(df, x="confidence", nbins=10, title="Confidence distribution")
    fig.update_traces(marker_color=color)
    fig.update_layout(bargap=0.05)
    return _style_fig(fig, height=240)


def charts_and_table(df: pd.DataFrame, stats: dict, color=ACCENT, key: str = ""):
    fig1 = class_bar(stats["class_counts"], color)
    fig2 = confidence_hist(df, color)
    if fig1:
        st.plotly_chart(fig1, use_container_width=True,
                        config={"displayModeBar": False}, key=f"clsbar_{key}")
    if fig2:
        st.plotly_chart(fig2, use_container_width=True,
                        config={"displayModeBar": False}, key=f"confhist_{key}")
    if not df.empty:
        st.dataframe(df, use_container_width=True, height=240, hide_index=True)
        st.download_button(
            "Download detections (CSV)",
            df.to_csv(index=False).encode(),
            file_name="detections.csv",
            mime="text/csv",
            key=f"dl_{key}",
        )
    else:
        st.info("No objects detected — try lowering the confidence threshold.")


# ---------------------------------------------------------------------------
# Sidebar controls -> config dict
# ---------------------------------------------------------------------------
def sidebar_config() -> dict:
    st.sidebar.markdown("### Controls")

    mode = st.sidebar.radio(
        "Mode", ["Single model", "Compare two models"], index=0
    )
    compare = mode == "Compare two models"

    task = st.sidebar.selectbox("Task", list(yu.MODEL_REGISTRY.keys()), index=0)
    model_names = list(yu.models_for_task(task).keys())

    if compare:
        cfg_a = st.sidebar.selectbox("Model A", model_names, index=0)
        default_b = 2 if len(model_names) > 2 else len(model_names) - 1
        cfg_b = st.sidebar.selectbox("Model B", model_names, index=default_b)
    else:
        cfg_a = st.sidebar.selectbox("Model", model_names, index=0)
        cfg_b = None

    source = st.sidebar.radio(
        "Input source",
        [
            "Webcam (live)",
            "Webcam (snapshot)",
            "Image upload",
            "Video upload",
            "Sample image",
        ],
        index=0,
    )

    st.sidebar.markdown("### Detection settings")
    conf = st.sidebar.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
    iou = st.sidebar.slider("IoU (NMS)", 0.10, 0.95, 0.45, 0.05)
    imgsz = st.sidebar.select_slider(
        "Image size", options=[320, 480, 640, 800, 960, 1280], value=640
    )
    device_label = st.sidebar.selectbox("Device", yu.available_devices(), index=0)

    # Populate the class filter from model A's names (loaded + cached anyway).
    weights_a = yu.models_for_task(task)[cfg_a]
    names = load_model(weights_a).names
    idx_to_name = {int(i): n for i, n in names.items()}
    sel_names = st.sidebar.multiselect(
        "Filter classes (empty = all)",
        options=[idx_to_name[i] for i in sorted(idx_to_name)],
        default=[],
        help="Only show these object classes.",
    )
    name_to_idx = {n: i for i, n in idx_to_name.items()}
    classes = [name_to_idx[n] for n in sel_names] if sel_names else None

    track = st.sidebar.checkbox(
        "Track objects across frames (assign IDs)",
        value=False,
        help="Give each object a stable ID as it moves. Works with the Live and "
             "Video sources; ignored for single images.",
    )

    with st.sidebar.expander("Help", expanded=False):
        st.markdown(
            "- **Live** streams the webcam continuously.\n"
            "- **Snapshot** takes one photo — most reliable everywhere.\n"
            "- **Compare** runs two models on the same input.\n"
            "- **Track** keeps an ID on each object as it moves (Live / Video).\n"
            "- Lower **confidence** → more (but noisier) detections.\n"
            "- Add models by editing `yolo_utils.py`."
        )

    return {
        "compare": compare,
        "task": task,
        "weights_a": weights_a,
        "weights_b": yu.models_for_task(task)[cfg_b] if compare else None,
        "name_a": cfg_a,
        "name_b": cfg_b,
        "source": source,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "device": yu.resolve_device(device_label),
        "classes": classes,
        "names": idx_to_name,
        "track": track,
    }


# ---------------------------------------------------------------------------
# Static image rendering (image upload / snapshot / sample)
# ---------------------------------------------------------------------------
def render_static_single(image_bgr, cfg):
    model = load_model(cfg["weights_a"])
    with st.spinner("Detecting…"):
        result, latency = yu.run_inference(
            model, image_bgr, cfg["conf"], cfg["iou"],
            cfg["imgsz"], cfg["classes"], cfg["device"],
        )
    df = yu.detections_dataframe(result, cfg["names"])
    stats = yu.summarize(df, latency)
    annotated = result.plot()

    col_img, col_side = st.columns([3, 2], gap="large")
    with col_img:
        st.image(bgr_to_rgb(annotated), use_container_width=True,
                 caption=cfg["name_a"])
    with col_side:
        metric_row(stats)
        charts_and_table(df, stats, ACCENT, key="single")


def render_static_compare(image_bgr, cfg):
    model_a = load_model(cfg["weights_a"])
    model_b = load_model(cfg["weights_b"])
    with st.spinner("Running both models…"):
        res_a, lat_a = yu.run_inference(
            model_a, image_bgr, cfg["conf"], cfg["iou"],
            cfg["imgsz"], cfg["classes"], cfg["device"])
        res_b, lat_b = yu.run_inference(
            model_b, image_bgr, cfg["conf"], cfg["iou"],
            cfg["imgsz"], cfg["classes"], cfg["device"])

    df_a = yu.detections_dataframe(res_a, cfg["names"])
    df_b = yu.detections_dataframe(res_b, cfg["names"])
    st_a, st_b = yu.summarize(df_a, lat_a), yu.summarize(df_b, lat_b)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown(f"#### {cfg['name_a']}")
        st.image(bgr_to_rgb(res_a.plot()), use_container_width=True)
        metric_row(st_a)
        charts_and_table(df_a, st_a, ACCENT, key="cmp_a")
    with col_b:
        st.markdown(f"#### {cfg['name_b']}")
        st.image(bgr_to_rgb(res_b.plot()), use_container_width=True)
        metric_row(st_b)
        charts_and_table(df_b, st_b, ACCENT_ALT, key="cmp_b")

    # Head-to-head summary
    st.markdown("#### Head-to-head")
    faster = cfg["name_a"] if lat_a <= lat_b else cfg["name_b"]
    speedup = (max(lat_a, lat_b) / min(lat_a, lat_b)) if min(lat_a, lat_b) > 0 else 1
    summary = pd.DataFrame(
        {
            "Metric": ["Objects found", "Latency (ms)", "Speed (FPS)", "Avg confidence"],
            cfg["name_a"]: [st_a["objects"], f"{lat_a:.0f}", f"{st_a['fps']:.1f}",
                            f"{st_a['avg_confidence']:.2f}"],
            cfg["name_b"]: [st_b["objects"], f"{lat_b:.0f}", f"{st_b['fps']:.1f}",
                            f"{st_b['avg_confidence']:.2f}"],
        }
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.success(f"**{faster}** was faster (~{speedup:.1f}× lower latency on this frame).")


def render_static(image_bgr, cfg):
    if image_bgr is None:
        st.warning("Could not read the image.")
        return
    if cfg.get("track"):
        st.caption("Tracking needs a sequence of frames — it applies to the "
                   "**Live** and **Video** sources. Showing single-frame detection here.")
    if cfg["compare"]:
        render_static_compare(image_bgr, cfg)
    else:
        render_static_single(image_bgr, cfg)


# ---------------------------------------------------------------------------
# Input source: snapshot / upload / sample  ->  static image
# ---------------------------------------------------------------------------
def source_snapshot(cfg):
    shot = st.camera_input("Take a photo")
    if shot is not None:
        render_static(file_to_bgr(shot), cfg)


def source_image_upload(cfg):
    up = st.file_uploader("Upload an image",
                          type=["jpg", "jpeg", "png", "bmp", "webp"])
    if up is not None:
        render_static(file_to_bgr(up), cfg)
    else:
        st.info("Upload an image to run detection.")


def source_sample(cfg):
    choice = st.selectbox("Sample image", list(SAMPLE_IMAGES.keys()))
    try:
        raw = fetch_sample_bytes(SAMPLE_IMAGES[choice])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        render_static(img, cfg)
    except Exception as e:
        st.error(f"Could not download the sample image: {e}")


# ---------------------------------------------------------------------------
# Input source: video upload  ->  frame loop
# ---------------------------------------------------------------------------
def source_video(cfg):
    up = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    skip = st.slider("Process every Nth frame (higher = faster)", 1, 10, 2)
    if up is None:
        st.info("Upload a short clip to run detection frame-by-frame.")
        return

    import tempfile

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(up.read())
    cap = cv2.VideoCapture(tfile.name)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    stframe = st.empty()
    progress = st.progress(0.0)
    mcols = st.columns(4)
    m_obj, m_cls, m_lat, m_fps = (c.empty() for c in mcols)

    model_a = load_model(cfg["weights_a"])
    model_b = load_model(cfg["weights_b"]) if cfg["compare"] else None
    track = cfg.get("track", False)
    agg_counts: Counter = Counter()
    lat_hist: deque = deque(maxlen=30)
    seen_ids: set = set()
    track_ph = st.empty()

    idx, processed = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % skip == 0:
            # Reset the tracker on the first processed frame of this video.
            persist = processed > 0
            res_a, lat_a = yu.run_inference(
                model_a, frame, cfg["conf"], cfg["iou"],
                cfg["imgsz"], cfg["classes"], cfg["device"],
                track=track, persist=persist)
            view = res_a.plot()
            if cfg["compare"] and model_b is not None:
                res_b, _ = yu.run_inference(
                    model_b, frame, cfg["conf"], cfg["iou"],
                    cfg["imgsz"], cfg["classes"], cfg["device"],
                    track=track, persist=persist)
                view = np.hstack([
                    label_frame(res_a.plot(), cfg["name_a"], (0, 216, 160)),
                    label_frame(res_b.plot(), cfg["name_b"], (124, 92, 255)),
                ])
            df = yu.detections_dataframe(res_a, cfg["names"])
            agg_counts.update(df["class"].tolist())
            lat_hist.append(lat_a)
            if track:
                seen_ids |= yu.unique_track_ids(res_a)

            stframe.image(bgr_to_rgb(view), use_container_width=True)
            m_obj.metric("Objects (frame)", len(df))
            m_cls.metric("Classes seen", len(agg_counts))
            m_lat.metric("Latency", f"{np.mean(lat_hist):.0f} ms")
            m_fps.metric("Speed", f"{1000 / max(np.mean(lat_hist), 1e-6):.1f} FPS")
            if track:
                track_ph.caption(f"Unique objects tracked so far: **{len(seen_ids)}**")
            processed += 1
        idx += 1
        if total:
            progress.progress(min(idx / total, 1.0))

    cap.release()
    progress.empty()
    st.success(f"Processed {idx} frames.")
    fig = class_bar(agg_counts, ACCENT)
    if fig:
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False}, key="video_agg")


# ---------------------------------------------------------------------------
# Input source: live webcam (streamlit-webrtc)
# ---------------------------------------------------------------------------
class YOLOProcessor:
    """Runs YOLO on each webcam frame in a background thread.

    Config is pushed in from the main thread every rerun (so moving a slider
    updates the live stream). Latest stats are read back under a lock.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.model_b = None
        self.compare = False
        self.conf = 0.25
        self.iou = 0.45
        self.imgsz = 640
        self.classes = None
        self.device = "cpu"
        self.names = {}
        self.track = False
        self.seen_ids: set = set()
        self._first_track = True  # reset the tracker on the first tracked frame
        self.fps = 0.0
        self._prev_t = None
        self.stats = {"objects": 0, "classes": 0, "latency_ms": 0.0, "fps": 0.0,
                      "class_counts": Counter(), "tracking": False, "unique_ids": 0}
        self.fps_history: deque = deque(maxlen=120)

    def update(self, **kw):
        with self.lock:
            # Switching tracking on (fresh) restarts IDs from scratch.
            if kw.get("track") and not self.track:
                self._first_track = True
                self.seen_ids = set()
            for k, v in kw.items():
                setattr(self, k, v)

    def get_stats(self):
        with self.lock:
            s = dict(self.stats)
            s["class_counts"] = Counter(self.stats["class_counts"])
            s["fps_history"] = list(self.fps_history)
            return s

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            model, model_b, compare = self.model, self.model_b, self.compare
            conf, iou, imgsz = self.conf, self.iou, self.imgsz
            classes, device, names = self.classes, self.device, self.names
            track = self.track
            persist = not self._first_track     # first tracked frame resets IDs
            if track:
                self._first_track = False

        if model is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        res, latency = yu.run_inference(model, img, conf, iou, imgsz, classes,
                                        device, track=track, persist=persist)
        view = res.plot()

        if compare and model_b is not None:
            res_b, _ = yu.run_inference(model_b, img, conf, iou, imgsz, classes,
                                        device, track=track, persist=persist)
            view = np.hstack([
                label_frame(res.plot(), "A", (0, 216, 160)),
                label_frame(res_b.plot(), "B", (124, 92, 255)),
            ])

        # Smoothed display FPS (wall-clock between frames).
        now = time.perf_counter()
        if self._prev_t is not None:
            dt = now - self._prev_t
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self._prev_t = now

        df = yu.detections_dataframe(res, names)
        with self.lock:
            if track:
                self.seen_ids |= yu.unique_track_ids(res)
            self.stats = {
                "objects": int(len(df)),
                "classes": int(df["class"].nunique()) if len(df) else 0,
                "latency_ms": latency,
                "fps": self.fps,
                "class_counts": Counter(df["class"]) if len(df) else Counter(),
                "tracking": track,
                "unique_ids": len(self.seen_ids),
            }
            self.fps_history.append(self.fps)

        return av.VideoFrame.from_ndarray(view, format="bgr24")


def source_live(cfg):
    if not WEBRTC_OK:
        st.warning(
            "Live webcam needs the optional packages **streamlit-webrtc** and "
            "**av**.\n\n```bash\npip install streamlit-webrtc av\n```\n"
            "Meanwhile, use **Webcam (snapshot)** — it works everywhere."
        )
        return

    st.caption(
        "Click **START**, allow camera access, then adjust the sidebar — "
        "changes apply to the live stream. In compare mode both models run "
        "on each frame (A left / B right), so FPS roughly halves."
    )
    if cfg.get("track"):
        st.caption("**Tracking on** — each object keeps its ID as it moves "
                   "around the frame.")

    ctx = webrtc_streamer(
        key="yolo-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLOProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )

    # Push current settings into the live processor.
    if ctx.video_processor:
        ctx.video_processor.update(
            model=load_model(cfg["weights_a"]),
            model_b=load_model(cfg["weights_b"]) if cfg["compare"] else None,
            compare=cfg["compare"],
            conf=cfg["conf"], iou=cfg["iou"], imgsz=cfg["imgsz"],
            classes=cfg["classes"], device=cfg["device"], names=cfg["names"],
            track=cfg["track"],
        )

    st.divider()
    auto = st.checkbox("Auto-refresh live metrics (pauses the controls)", value=False)
    # The whole stats block lives in one placeholder and is fully replaced on
    # each redraw — so metrics don't stack up and the chart doesn't collide.
    stats_ph = st.empty()

    def _draw_stats(tick: int = 0):
        if not ctx.video_processor:
            return
        s = ctx.video_processor.get_stats()
        with stats_ph.container():
            metric_row(s)
            if s.get("tracking"):
                st.caption(f"Unique objects tracked so far: **{s.get('unique_ids', 0)}**")
            fig = class_bar(s["class_counts"], ACCENT)
            if fig:
                # Unique key per redraw so repeated draws never reuse an element ID.
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False},
                                key=f"live_chart_{tick}")
            else:
                st.caption("No objects detected yet — point the camera at something.")

    if auto and ctx.state.playing:
        tick = 0
        while ctx.state.playing and ctx.video_processor:
            _draw_stats(tick)
            tick += 1
            time.sleep(0.4)
    else:
        _draw_stats()
        if ctx.state.playing:
            st.button("Refresh stats")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    inject_css()
    header()

    try:
        cfg = sidebar_config()
    except Exception as e:
        st.error(f"Could not load the selected model: {e}")
        st.stop()

    source = cfg["source"]
    if source == "Webcam (live)":
        source_live(cfg)
    elif source == "Webcam (snapshot)":
        source_snapshot(cfg)
    elif source == "Image upload":
        source_image_upload(cfg)
    elif source == "Video upload":
        source_video(cfg)
    elif source == "Sample image":
        source_sample(cfg)

    st.markdown(
        "<p class='vl-caption'>Tip: start with the sample image, then try the "
        "webcam. Compare a <b>nano</b> vs <b>medium</b> model to see the "
        "speed/accuracy trade-off.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
