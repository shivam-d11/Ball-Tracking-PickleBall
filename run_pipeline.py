"""
TrackNet-Pickleball: Complete Inference Pipeline
=================================================
Run ball detection on a pickleball video, generate trajectory overlay,
and plot detection confidence / accuracy statistics.

Usage (Colab — from GCS bucket):
    !pip install -q opencv-python-headless pillow scikit-learn gdown matplotlib
    !python run_pipeline.py --video gs://pickle_testing_bucket/ball_tracking/testBall.mp4

Usage (Colab — local file):
    !python run_pipeline.py --video /content/your_video.mp4

Usage (Local with GPU):
    python run_pipeline.py --video /path/to/your_video.mp4 --weights /path/to/weights_dir

The --video flag accepts:
  - Local file path:  /content/video.mp4
  - GCS URI:          gs://bucket_name/path/to/video.mp4

Optional: pass --labels /path/to/labels.csv for ground-truth evaluation.

**F1 score** is computed only when ``--labels`` is provided. Without annotations,
TP/FP/FN are undefined, so F1 cannot be computed (only proxy stats like detection rate).

Use ``--eval-frame-range START END`` (inclusive, 0-based) or ``--eval-frames-file`` to score
only frames you labeled; inference and trajectory output still cover the **full** video.

By default the pipeline uses **one video pass** (inference + trajectory together).
Use ``--separate-trajectory-pass`` to decode the file twice (old behavior).
"""

import argparse
import csv
import os
import sys
import time
import math
import re
import queue
from collections import deque
from datetime import datetime
import shutil
import subprocess
import threading

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Helps TrackNet NCHW MaxPool on some CPU builds (Intel oneDNN). Set to 0 to disable.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEIGHT = 288
WIDTH = 512
TRAIL_LENGTH = 8

# Named BGR colors (OpenCV). Default: magenta — high contrast on green court.
BALL_COLOR_PRESETS = {
    "yellow": (0, 255, 255),
    "magenta": (255, 0, 255),
    "cyan": (255, 255, 0),
    "lime": (0, 255, 0),
    "orange": (0, 165, 255),
    "red": (0, 0, 255),
    "white": (255, 255, 255),
    "blue": (255, 0, 0),
}


class PipelineSettings:
    """Mutable runtime options (CLI → main() assigns fields)."""
    heatmap_threshold = 0.5
    min_confidence = 0.0
    smooth_window = 0
    max_jump_frac = 0.0
    video_diagonal = 1.0
    ball_bgr = (255, 0, 255)
    marker_radius_base = 8
    marker_radius_per_trail = 2
    bbox_half_side = 16
    draw_bbox = True


PIPE = PipelineSettings()


def parse_ball_color(spec):
    """Return BGR tuple from preset name or #RRGGBB hex."""
    if not spec:
        return PIPE.ball_bgr
    s = spec.strip().lower()
    if s in BALL_COLOR_PRESETS:
        return BALL_COLOR_PRESETS[s]
    m = re.match(r"^#?([0-9a-fA-F]{6})$", s.replace(" ", ""))
    if m:
        hx = m.group(1)
        r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
        return (b, g, r)
    raise ValueError(f"Unknown --ball-color {spec!r}. Use a preset {list(BALL_COLOR_PRESETS)} or #RRGGBB")


def decode_ball_from_heatmap(y_pred_ch, ratio):
    """Single output channel → visibility, cx, cy, peak confidence (model space)."""
    raw_conf = float(np.amax(y_pred_ch))
    mask = (y_pred_ch > PIPE.heatmap_threshold).astype(np.float32)
    hmap = (mask * 255).astype(np.uint8)
    if np.amax(hmap) <= 0:
        return 0, -1, -1, raw_conf
    cnts, _ = cv2.findContours(hmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0, -1, -1, raw_conf
    rects = [cv2.boundingRect(c) for c in cnts]
    areas = [r[2] * r[3] for r in rects]
    target = rects[np.argmax(areas)]
    cx = int(ratio * (target[0] + target[2] / 2))
    cy = int(ratio * (target[1] + target[3] / 2))
    return 1, cx, cy, raw_conf


class InferenceRefiner:
    """Optional temporal smoothing + spike rejection (no retraining)."""

    def __init__(self):
        self.last_good = None
        maxlen = PIPE.smooth_window if PIPE.smooth_window >= 3 else 3
        self.buf = deque(maxlen=maxlen)

    def refine(self, vis, cx, cy, conf):
        if vis == 1 and conf < PIPE.min_confidence:
            vis, cx, cy = 0, -1, -1
        self.buf.append((vis, cx, cy))

        if PIPE.smooth_window >= 3 and vis == 1:
            pts = [(b[1], b[2]) for b in self.buf if b[0] == 1 and b[1] > 0 and b[2] > 0]
            if len(pts) >= 1:
                cx = int(np.median([p[0] for p in pts]))
                cy = int(np.median([p[1] for p in pts]))

        if (
            vis == 1
            and PIPE.max_jump_frac > 0
            and self.last_good is not None
            and PIPE.video_diagonal > 1
        ):
            d = math.hypot(cx - self.last_good[0], cy - self.last_good[1])
            if d > PIPE.max_jump_frac * PIPE.video_diagonal:
                vis, cx, cy = 0, -1, -1

        if vis == 1:
            self.last_good = (cx, cy)
        return vis, cx, cy, conf


# ---------------------------------------------------------------------------
# GCS bucket config
# ---------------------------------------------------------------------------
GCS_BUCKET = "pickle_testing_bucket"
GCS_VIDEO_FOLDER = "ball_tracking"

# ---------------------------------------------------------------------------
# Google Drive weight URLs
# ---------------------------------------------------------------------------
NEW_WEIGHTS_GDRIVE_FOLDER = (
    "https://drive.google.com/drive/folders/1EGsddY1fgEJ5ITrfF32aPCn6nml2Anzr"
)


# ===================================================================
# GCS HELPERS
# ===================================================================

def download_from_gcs(gcs_uri, local_dir="/tmp/tracknet_videos"):
    """Download a file from Google Cloud Storage.
    Accepts gs://bucket/path/to/file.mp4 or just the filename (uses default bucket).
    Returns the local file path.
    """
    os.makedirs(local_dir, exist_ok=True)

    if gcs_uri.startswith("gs://"):
        src = gcs_uri
        filename = os.path.basename(gcs_uri)
    else:
        src = f"gs://{GCS_BUCKET}/{GCS_VIDEO_FOLDER}/{gcs_uri}"
        filename = os.path.basename(gcs_uri)

    local_path = os.path.join(local_dir, filename)
    if os.path.isfile(local_path):
        print(f"[GCS] File already downloaded: {local_path}")
        return local_path

    print(f"[GCS] Downloading {src} -> {local_path}")
    result = subprocess.run(["gsutil", "cp", src, local_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[GCS] gsutil failed: {result.stderr.strip()}")
        print("[GCS] Trying gcloud auth...")
        subprocess.run(["gcloud", "auth", "login", "--no-launch-browser"], check=False)
        result = subprocess.run(["gsutil", "cp", src, local_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[GCS] ERROR: Could not download {src}")
            print(f"[GCS] stderr: {result.stderr.strip()}")
            sys.exit(1)

    print(f"[GCS] Downloaded successfully: {local_path}")
    return local_path


def list_gcs_videos(bucket=GCS_BUCKET, folder=GCS_VIDEO_FOLDER):
    """List all video files in the GCS bucket folder."""
    uri = f"gs://{bucket}/{folder}/"
    print(f"[GCS] Listing files in {uri}")
    result = subprocess.run(["gsutil", "ls", uri], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[GCS] Could not list bucket: {result.stderr.strip()}")
        return []
    files = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
    video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    return video_files


def upload_to_gcs(local_path, bucket=GCS_BUCKET, folder="ball_tracking/results"):
    """Upload a local file to the GCS bucket."""
    filename = os.path.basename(local_path)
    dest = f"gs://{bucket}/{folder}/{filename}"
    print(f"[GCS] Uploading {local_path} -> {dest}")
    result = subprocess.run(["gsutil", "cp", local_path, dest], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[GCS] Upload failed: {result.stderr.strip()}")
        return None
    print(f"[GCS] Uploaded: {dest}")
    return dest


# ===================================================================
# 1. SETUP — directories, weights download
# ===================================================================

def setup_workspace(base_dir):
    dirs = {}
    for name in ["predictions", "trajectories", "weights", "plots"]:
        d = os.path.join(base_dir, name)
        os.makedirs(d, exist_ok=True)
        dirs[name] = d
    return dirs


def download_weights(weights_dir):
    """Download pre-trained pickleball weights from Google Drive."""
    target = os.path.join(weights_dir, "new_weights")
    if os.path.exists(target) and os.listdir(target):
        print(f"[SETUP] Weights already exist at {target}")
        return target

    print("[SETUP] Downloading pre-trained pickleball weights from Google Drive...")
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown

    gdown.download_folder(NEW_WEIGHTS_GDRIVE_FOLDER, output=target, quiet=False)
    print(f"[SETUP] Weights downloaded to {target}")
    return target


# ===================================================================
# 2. MODEL LOADING
# ===================================================================

# Default tuned for smaller GPUs (e.g. T4 under memory pressure). Use --batch-size 16–32 on large GPUs.
INFERENCE_BATCH_SIZE = 8

def _ensure_tf_keras():
    """Install tf-keras if not available (Keras 2 compat for TF 2.16+)."""
    try:
        import tf_keras
        return tf_keras
    except ImportError:
        print("[MODEL] Installing tf-keras (Keras 2 compatibility layer)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tf-keras"])
        import tf_keras
        return tf_keras


def custom_loss_fn(y_true, y_pred):
    """Focal-style loss (needed for loading the saved model)."""
    import keras.backend as K
    loss = (-1) * (
        K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) +
        K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1))
    )
    return K.mean(loss)


def load_model_tfsm(weights_path):
    """Load legacy Keras 2 SavedModel using tf-keras compatibility package."""
    print(f"[MODEL] Loading weights from {weights_path}")

    tf_keras = _ensure_tf_keras()
    model = tf_keras.models.load_model(
        weights_path,
        custom_objects={'custom_loss': custom_loss_fn}
    )
    print(f"[MODEL] Model loaded successfully. Output shape: {model.output_shape}")
    return model


def _tensor_to_numpy(out):
    if hasattr(out, "numpy"):
        return out.numpy()
    return np.asarray(out)


def _is_cpu_nchw_pool_error(err):
    msg = str(err).lower()
    return "nhwc" in msg or "maxpool" in msg or "channels_first" in msg


def build_predict_fn(model):
    """Return a callable(tf.constant batch) -> numpy.

    Order: ``@tf.function`` (fast on GPU) → eager ``model(x)`` → ``model.predict`` (numpy).
    macOS / CPU often hits \"MaxPoolingOp only supports NHWC on CPU\" inside ``tf.function``;
    later paths usually still work. Set ``TF_ENABLE_ONEDNN_OPTS=1`` before TF import (done above).

    """
    dummy = tf.zeros((1, 9, HEIGHT, WIDTH), dtype=tf.float32)
    dummy_np = np.zeros((1, 9, HEIGHT, WIDTH), dtype=np.float32)

    @tf.function(reduce_retracing=True)
    def predict_fn_tf(x):
        return model(x, training=False)

    try:
        print("[MODEL] Warming up tf.function with dummy input...")
        _ = predict_fn_tf(dummy)
        print("[MODEL] tf.function compiled and ready.")

        def predict_fn(x):
            return _tensor_to_numpy(predict_fn_tf(x))

        return predict_fn
    except (tf.errors.InvalidArgumentError, ValueError) as e:
        if not _is_cpu_nchw_pool_error(e):
            raise
        print(
            "[MODEL] tf.function failed (common on CPU + channels-first TrackNet). "
            "Trying eager inference…"
        )

    try:
        _ = _tensor_to_numpy(model(dummy, training=False))
        print("[MODEL] Eager tensor inference ready.")

        def predict_fn(x):
            return _tensor_to_numpy(model(x, training=False))

        return predict_fn
    except (tf.errors.InvalidArgumentError, ValueError) as e:
        if not _is_cpu_nchw_pool_error(e):
            raise
        print("[MODEL] Eager tensor failed; using model.predict() path…")

    out = model.predict(dummy_np, batch_size=1, verbose=0)
    assert out is not None
    print("[MODEL] model.predict() path ready (CPU-friendly).")

    def predict_fn(x):
        t = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
        return model.predict(t, batch_size=t.shape[0], verbose=0)

    return predict_fn


def run_batched_inference(predict_fn, arr):
    """Run ``predict_fn`` on batch array (N, 9, H, W). On GPU OOM, split recursively.

    Same venv / two script copies do not cause OOM; peak VRAM is set by **batch size**
    and model activations (e.g. shape [N, 64, 288, 512] for batch N).
    """
    n = arr.shape[0]
    if n == 0:
        return np.zeros((0, 3, HEIGHT, WIDTH), dtype=np.float32)
    try:
        return predict_fn(tf.constant(arr))
    except tf.errors.ResourceExhaustedError:
        if n <= 1:
            print(
                "[PREDICT] FATAL: GPU OOM even with batch size 1. Close other GPU processes, "
                "or run with CUDA_VISIBLE_DEVICES= and accept CPU (very slow)."
            )
            raise
        mid = n // 2
        print(
            f"[PREDICT] GPU out of memory on batch {n} — splitting into {mid} + {n - mid}. "
            f"Tip: pass --batch-size {mid} (or lower) to avoid retries."
        )
        y1 = run_batched_inference(predict_fn, np.ascontiguousarray(arr[:mid]))
        y2 = run_batched_inference(predict_fn, np.ascontiguousarray(arr[mid:]))
        return np.concatenate([y1, y2], axis=0)


# ===================================================================
# 3. OPTIMIZED PREDICTION PIPELINE
# ===================================================================

def preprocess_frame(frame):
    """Convert a BGR frame to CHW float32 normalized array using pure OpenCV/numpy."""
    resized = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw


def build_window(f1, f2, f3):
    """Stack 3 preprocessed CHW frames into a single (9, H, W) tensor."""
    return np.concatenate([f1, f2, f3], axis=0)


class VideoReaderThread:
    """Background thread that reads and preprocesses frames ahead of inference."""

    def __init__(self, video_path, prefetch_size=64):
        self.cap = cv2.VideoCapture(video_path)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.ratio = None
        self._queue = queue.Queue(maxsize=prefetch_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _read_loop(self):
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self._queue.put(None)
                break
            if self.ratio is None:
                self.ratio = frame.shape[0] / HEIGHT
            preprocessed = preprocess_frame(frame)
            self._queue.put(preprocessed)
        self.cap.release()

    def get(self, timeout=10):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)


def make_predictions(video_path, model, output_csv):
    """Optimized ball detection: batched inference, threaded I/O, compiled tf.function."""
    print(f"\n[PREDICT] Running optimized inference on {os.path.basename(video_path)}")
    print(f"[PREDICT] Batch size: {INFERENCE_BATCH_SIZE}, threaded reader enabled")
    start = time.time()

    predict_fn = build_predict_fn(model)

    reader = VideoReaderThread(video_path, prefetch_size=INFERENCE_BATCH_SIZE * 6)
    reader.start()

    # Read first 3 frames to get ratio
    frames_buffer = []
    for _ in range(3):
        f = reader.get()
        if f is None:
            print("[PREDICT] ERROR: Could not read video frames.")
            reader.stop()
            return None
        frames_buffer.append(f)

    ratio = reader.ratio
    num_frames = reader.num_frames
    count = 0
    rows = []
    refiner = InferenceRefiner()

    batch_buffer = np.zeros((INFERENCE_BATCH_SIZE, 9, HEIGHT, WIDTH), dtype=np.float32)
    batch_count = 0
    windows_in_batch = []
    done_reading = False

    while True:
        # Fill batch
        while batch_count < INFERENCE_BATCH_SIZE and not done_reading:
            if len(frames_buffer) >= 3:
                window = build_window(frames_buffer[0], frames_buffer[1], frames_buffer[2])
                batch_buffer[batch_count] = window
                batch_count += 1
                windows_in_batch.append(True)

                # Slide window: consume all 3, read next 3
                frames_buffer.clear()
                for _ in range(3):
                    f = reader.get()
                    if f is None:
                        done_reading = True
                        break
                    frames_buffer.append(f)
            else:
                done_reading = True
                break

        if batch_count == 0:
            break

        # Run batched inference (splits automatically on GPU OOM)
        y_pred = run_batched_inference(predict_fn, batch_buffer[:batch_count])

        for b in range(batch_count):
            for ch in range(3):
                vis, cx, cy, raw_conf = decode_ball_from_heatmap(y_pred[b, ch], ratio)
                vis, cx, cy, raw_conf = refiner.refine(vis, cx, cy, raw_conf)
                rows.append((count, vis, cx, cy, raw_conf))
                count += 1

        if count % 900 == 0 or done_reading:
            elapsed = time.time() - start
            fps = count / elapsed if elapsed > 0 else 0
            pct = count / num_frames * 100
            print(f"  {count}/{num_frames} frames ({pct:.1f}%) — {elapsed:.1f}s — {fps:.1f} FPS")

        batch_count = 0
        windows_in_batch.clear()

        if done_reading:
            break

    reader.stop()
    elapsed = time.time() - start
    fps = count / elapsed if elapsed > 0 else 0
    print(f"[PREDICT] Done! {count}/{num_frames} frames in {elapsed:.1f}s ({fps:.1f} FPS)")

    df = pd.DataFrame(rows, columns=["Frame", "Visibility", "X", "Y", "Confidence"])
    df.to_csv(output_csv, index=False)
    print(f"[PREDICT] Predictions saved to {output_csv}")
    return df


def inference_plus_trajectory(video_path, model, pred_csv, output_video):
    """One pass over the source video: run TrackNet inference and write trajectory MP4.

    Avoids a second full decode of the file (and avoids per-frame ``seek``, which is
    ~1 FPS on typical H.264/MP4). Same CSV columns as ``make_predictions``.
    Returns ``(pred_df, output_path)`` or ``(None, None)`` on failure.
    """
    print(f"\n[COMBINED] Single-pass inference + trajectory video (one video read)")
    print(f"[COMBINED] Batch size: {INFERENCE_BATCH_SIZE}")
    start = time.time()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[COMBINED] ERROR: could not open video.")
        return None, None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps < 1e-3:
        fps = 30.0
        print(f"[COMBINED] WARNING: FPS unknown — using {fps}")
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        output, actual_path = _open_trajectory_writer(output_video, fps, out_w, out_h)
    except RuntimeError as e:
        cap.release()
        print(e)
        return None, None

    predict_fn = build_predict_fn(model)
    q = deque([None] * TRAIL_LENGTH)
    rows = []
    count = 0
    ratio = None

    frames_small = []
    frames_full = []
    while len(frames_small) < 3:
        ret, fr = cap.read()
        if not ret or fr is None:
            cap.release()
            output.release()
            print("[COMBINED] ERROR: not enough frames for first window.")
            return None, None
        if ratio is None:
            ratio = fr.shape[0] / HEIGHT
        frames_small.append(preprocess_frame(fr))
        frames_full.append(fr)

    refiner = InferenceRefiner()
    batch_buffer = np.zeros((INFERENCE_BATCH_SIZE, 9, HEIGHT, WIDTH), dtype=np.float32)
    batch_count = 0
    pending_windows = []
    done_reading = False

    while True:
        while batch_count < INFERENCE_BATCH_SIZE and not done_reading:
            if len(frames_small) >= 3:
                window = build_window(frames_small[0], frames_small[1], frames_small[2])
                batch_buffer[batch_count] = window
                pending_windows.append(
                    (frames_full[0], frames_full[1], frames_full[2])
                )
                batch_count += 1
                frames_small = []
                frames_full = []
                while len(frames_small) < 3 and not done_reading:
                    ret, fr = cap.read()
                    if not ret or fr is None:
                        done_reading = True
                        break
                    frames_small.append(preprocess_frame(fr))
                    frames_full.append(fr)
            else:
                done_reading = True
                break

        if batch_count == 0:
            break

        y_pred = run_batched_inference(predict_fn, batch_buffer[:batch_count])

        for b in range(batch_count):
            f0, f1, f2 = pending_windows[b]
            for ch, img_src in enumerate((f0, f1, f2)):
                img = np.ascontiguousarray(img_src.copy())
                vis, cx, cy, raw_conf = decode_ball_from_heatmap(y_pred[b, ch], ratio)
                vis, cx, cy, raw_conf = refiner.refine(vis, cx, cy, raw_conf)
                rows.append((count, vis, cx, cy, raw_conf))
                _write_frame_trajectory_opencv(output, q, count, vis, cx, cy, img)
                count += 1

        if count % 900 == 0 or done_reading:
            elapsed = time.time() - start
            fps_eff = count / elapsed if elapsed > 0 else 0
            pct = count / max(num_frames, 1) * 100
            print(f"  {count}/{num_frames} frames ({pct:.1f}%) — {elapsed:.1f}s — {fps_eff:.1f} FPS (decode+infer+write)")

        batch_count = 0
        pending_windows.clear()
        if done_reading:
            break

    cap.release()
    output.release()

    elapsed = time.time() - start
    fps_eff = count / elapsed if elapsed > 0 else 0
    print(f"[COMBINED] Done! {count}/{num_frames} frames in {elapsed:.1f}s ({fps_eff:.1f} FPS)")

    df = pd.DataFrame(rows, columns=["Frame", "Visibility", "X", "Y", "Confidence"])
    df.to_csv(pred_csv, index=False)
    print(f"[COMBINED] Predictions saved to {pred_csv}")

    final_path = _trajectory_ffmpeg_post(actual_path)
    return df, final_path


# ===================================================================
# 4. TRAJECTORY — overlay ball positions on video
# ===================================================================

def _open_trajectory_writer(output_video, fps, out_w, out_h):
    """Open a cv2.VideoWriter; fall back to MJPEG AVI if mp4 fails."""

    def _mk(path, codec, rate):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        return cv2.VideoWriter(path, fourcc, rate, (out_w, out_h))

    w = _mk(output_video, 'mp4v', fps)
    path = output_video
    if not w.isOpened():
        avi_path = os.path.splitext(output_video)[0] + '_trajectory.avi'
        print(f"[TRAJECTORY] mp4 writer failed — trying MJPEG AVI: {avi_path}")
        w = _mk(avi_path, 'MJPG', fps)
        path = avi_path
    if not w.isOpened():
        raise RuntimeError(
            "[TRAJECTORY] Could not open VideoWriter. Check codecs and disk space."
        )
    return w, path


def _write_frame_trajectory_opencv(output, q, frame_idx, vis, cx, cy, img_bgr):
    """Write one BGR frame; frames 0–1 raw, then trail (OpenCV — fast)."""
    if frame_idx < 2:
        output.write(np.ascontiguousarray(img_bgr))
        return
    if vis == 1 and cx > 0 and cy > 0:
        q.appendleft([int(cx), int(cy)])
    else:
        q.appendleft(None)
    q.pop()
    col = PIPE.ball_bgr
    outline = (255, 255, 255)
    for i in range(TRAIL_LENGTH):
        if q[i] is not None:
            dx, dy = q[i][0], q[i][1]
            age = i
            r = PIPE.marker_radius_base + (TRAIL_LENGTH - 1 - age) * PIPE.marker_radius_per_trail
            r = max(6, int(r))
            cv2.circle(img_bgr, (dx, dy), r, col, -1, lineType=cv2.LINE_AA)
            cv2.circle(img_bgr, (dx, dy), r, outline, 2, lineType=cv2.LINE_AA)
            if PIPE.draw_bbox and PIPE.bbox_half_side > 0 and age == 0:
                h = int(PIPE.bbox_half_side)
                cv2.rectangle(
                    img_bgr, (dx - h, dy - h), (dx + h, dy + h), col, 3, lineType=cv2.LINE_AA
                )
    output.write(np.ascontiguousarray(img_bgr))


def _trajectory_ffmpeg_post(actual_path):
    """Optional H.264 transcode for smaller, QuickTime-friendly MP4."""
    h264_path = os.path.splitext(actual_path)[0] + '_h264.mp4'
    try:
        r = subprocess.run(
            [
                'ffmpeg', '-y', '-loglevel', 'error',
                '-i', actual_path,
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23',
                '-movflags', '+faststart',
                h264_path,
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        r = None

    if r is not None and r.returncode == 0 and os.path.isfile(h264_path) and os.path.getsize(h264_path) > 0:
        print(f"[TRAJECTORY] H.264 copy (recommended for playback): {h264_path}")
        try:
            if actual_path != h264_path and os.path.isfile(actual_path):
                os.remove(actual_path)
                print(f"[TRAJECTORY] Removed intermediate file: {actual_path}")
        except OSError:
            pass
        return h264_path
    if shutil.which('ffmpeg'):
        print(
            "[TRAJECTORY] ffmpeg present but transcode failed; keeping raw. "
            f"Manual: ffmpeg -y -i \"{actual_path}\" -c:v libx264 -pix_fmt yuv420p "
            f"-crf 23 -movflags +faststart \"{h264_path}\""
        )
    else:
        print(
            "[TRAJECTORY] Tip: install ffmpeg for smaller H.264 MP4s: "
            "sudo apt-get install -y ffmpeg"
        )
    return actual_path


def generate_trajectory(pred_csv, video_path, output_video):
    """Second pass: CSV + video → trajectory clip. **Slow path avoided in default pipeline.**

    Uses **sequential** ``read()`` only (no per-frame seek — seeking H.264 is ~1 FPS).
    Same trail rules as the combined pass (OpenCV drawing).
    """
    print(f"\n[TRAJECTORY] Second pass (sequential read + OpenCV) — use combined mode to skip this")
    preds = pd.read_csv(pred_csv)
    x_coords = preds['X'].values.astype(np.float64)
    y_coords = preds['Y'].values.astype(np.float64)
    num_preds = len(x_coords)

    q = deque([None] * TRAIL_LENGTH)
    video = cv2.VideoCapture(video_path)
    fps = float(video.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps < 1e-3:
        fps = 30.0
        print(f"[TRAJECTORY] WARNING: FPS unknown — using {fps}")
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[TRAJECTORY] num_frames={num_frames}, num_preds={num_preds}, fps={fps}, size={out_w}x{out_h}")

    output, actual_path = _open_trajectory_writer(output_video, fps, out_w, out_h)
    current = 0
    while current < num_preds and current < num_frames:
        ret, img = video.read()
        if not ret or img is None:
            break
        vis = 1 if (x_coords[current] > 0 and y_coords[current] > 0) else 0
        cx = int(x_coords[current]) if vis else -1
        cy = int(y_coords[current]) if vis else -1
        _write_frame_trajectory_opencv(output, q, current, vis, cx, cy, img)
        current += 1
        if current % 500 == 0:
            print(f"  Processed {current}/{min(num_preds, num_frames)} frames")

    video.release()
    output.release()
    final_path = _trajectory_ffmpeg_post(actual_path)
    print(f"[TRAJECTORY] Done! Saved to {final_path} ({current} frames)")
    return final_path


# ===================================================================
# 5. DETECTION STATS — plots and overall accuracy (no ground truth)
# ===================================================================

def plot_detection_stats(pred_df, plots_dir, video_name):
    """Generate detection statistics plots from predictions (no labels needed)."""
    print(f"\n[STATS] Generating detection statistics...")

    total = len(pred_df)
    detected = pred_df[pred_df['Visibility'] == 1]
    not_detected = pred_df[pred_df['Visibility'] == 0]
    n_detected = len(detected)
    n_not_detected = len(not_detected)
    detection_rate = n_detected / total * 100 if total > 0 else 0

    print(f"\n{'='*55}")
    print(f"  DETECTION SUMMARY — {video_name}")
    print(f"{'='*55}")
    print(f"  Total frames analyzed : {total}")
    print(f"  Ball detected         : {n_detected} ({detection_rate:.1f}%)")
    print(f"  Ball NOT detected     : {n_not_detected} ({100-detection_rate:.1f}%)")
    if 'Confidence' in pred_df.columns:
        mean_conf = pred_df['Confidence'].mean() * 100
        det_conf = detected['Confidence'].mean() * 100 if n_detected > 0 else 0
        print(f"  Mean confidence (all) : {mean_conf:.1f}%")
        print(f"  Mean confidence (det) : {det_conf:.1f}%")
    print(f"{'='*55}")
    print(f"  Overall Detection Rate: {detection_rate:.2f}%")
    print(f"{'='*55}")
    print("  Note: this is NOT F1 — use --labels for Precision / Recall / F1 vs ground truth.")
    print(f"{'='*55}\n")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"TrackNet Pickleball Detection — {video_name}", fontsize=16, fontweight='bold')

    # --- Plot 1: Detection pie chart ---
    ax1 = axes[0, 0]
    colors = ['#2ecc71', '#e74c3c']
    ax1.pie(
        [n_detected, n_not_detected],
        labels=[f'Detected\n{n_detected} frames', f'Not Detected\n{n_not_detected} frames'],
        colors=colors, autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 12}, pctdistance=0.75
    )
    ax1.set_title('Ball Detection Rate', fontsize=14, fontweight='bold')

    # --- Plot 2: Detection over time (sliding window) ---
    ax2 = axes[0, 1]
    window = max(30, total // 50)
    rolling_det = pred_df['Visibility'].rolling(window=window, min_periods=1).mean() * 100
    ax2.plot(pred_df['Frame'], rolling_det, color='#3498db', linewidth=1.5)
    ax2.axhline(y=detection_rate, color='#e74c3c', linestyle='--', alpha=0.7, label=f'Overall: {detection_rate:.1f}%')
    ax2.fill_between(pred_df['Frame'], rolling_det, alpha=0.15, color='#3498db')
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('Detection Rate (%)', fontsize=12)
    ax2.set_title(f'Detection Rate Over Time (window={window} frames)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Confidence distribution ---
    ax3 = axes[1, 0]
    if 'Confidence' in pred_df.columns:
        conf_values = pred_df['Confidence'].values * 100
        ax3.hist(conf_values, bins=50, color='#9b59b6', alpha=0.75, edgecolor='white')
        ax3.axvline(x=50, color='red', linestyle='--', linewidth=1.5, label='Threshold (50%)')
        ax3.set_xlabel('Max Heatmap Confidence (%)', fontsize=12)
        ax3.set_ylabel('Number of Frames', fontsize=12)
        ax3.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No confidence data', ha='center', va='center', fontsize=14)
        ax3.set_title('Confidence Distribution', fontsize=14, fontweight='bold')

    # --- Plot 4: Ball position scatter ---
    ax4 = axes[1, 1]
    if n_detected > 0:
        sc = ax4.scatter(
            detected['X'], detected['Y'],
            c=detected['Frame'], cmap='viridis', s=3, alpha=0.5
        )
        ax4.set_xlabel('X (pixels)', fontsize=12)
        ax4.set_ylabel('Y (pixels)', fontsize=12)
        ax4.set_title('Detected Ball Positions (colored by time)', fontsize=14, fontweight='bold')
        ax4.invert_yaxis()
        plt.colorbar(sc, ax=ax4, label='Frame Number')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No detections', ha='center', va='center', fontsize=14)
        ax4.set_title('Detected Ball Positions', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f"{video_name}_detection_stats.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[STATS] Plot saved to {plot_path}")
    return plot_path, detection_rate


# ===================================================================
# 6. GROUND-TRUTH EVALUATION (optional — only if labels CSV provided)
# ===================================================================

def _load_ground_truth_labels(labels_path, video_path):
    """Build pixel ``X``, ``Y`` columns for eval.

    Supports:
    - **Step2 labeller export**: ``Frame,Ball,x,y`` with ``x,y`` normalized to [0,1]
      (and ``-1`` when no ball). Converted to pixel coordinates using video size.
    - **Eval-ready CSV**: columns ``X``, ``Y`` already in **pixels**; ball absent should
      be ``X < 0`` and ``Y < 0`` (e.g. ``-1``), not ``0,0``.
    """
    raw = pd.read_csv(labels_path)
    colmap = {c.lower(): c for c in raw.columns}

    cap = cv2.VideoCapture(video_path)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if "ball" in colmap and "x" in colmap and "y" in colmap:
        print(
            "[EVAL] Detected Step2 labeller format (Frame,Ball,x,y with normalized x,y) — "
            "converting to pixel X,Y."
        )
        bc, xc, yc = colmap["ball"], colmap["x"], colmap["y"]
        xs, ys = [], []
        for _, row in raw.iterrows():
            b = int(row[bc])
            xf, yf = float(row[xc]), float(row[yc])
            if b == 0 or xf < 0 or yf < 0:
                xs.append(-1.0)
                ys.append(-1.0)
            else:
                xs.append(xf * vid_w)
                ys.append(yf * vid_h)
        out = raw.copy()
        out["X"] = xs
        out["Y"] = ys
        return out

    if "x" in colmap and "y" in colmap and "X" not in raw.columns:
        xc, yc = colmap["x"], colmap["y"]
        out = raw.copy()
        out["X"] = raw[xc].astype(float)
        out["Y"] = raw[yc].astype(float)
        return out

    if "X" not in raw.columns or "Y" not in raw.columns:
        raise ValueError(
            "Labels CSV must have either (Frame,Ball,x,y) from Step2 labelling_tool, "
            "or columns X,Y in **pixels** (negative = no ball)."
        )
    return raw


def _frame_index_for_row(df, row_i):
    if "Frame" in df.columns:
        return int(df["Frame"].iloc[row_i])
    return row_i


def parse_eval_frame_subset(eval_range, eval_frames_file):
    """Return a set of 0-based frame indices to score, or None = score all frames."""
    if eval_range is None and not eval_frames_file:
        return None
    if eval_range is not None and eval_frames_file:
        raise ValueError("Use only one of --eval-frame-range or --eval-frames-file.")
    if eval_range is not None:
        lo, hi = int(eval_range[0]), int(eval_range[1])
        if lo > hi:
            raise ValueError(f"--eval-frame-range invalid: start {lo} > end {hi}")
        return set(range(lo, hi + 1))
    indices = set()
    with open(eval_frames_file, "r") as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            indices.add(int(line))
    if not indices:
        raise ValueError(f"No frame indices found in {eval_frames_file}")
    return indices


def evaluate_with_labels(
    pred_df, labels_path, video_path, plots_dir, video_name, eval_frames=None
):
    """Compare predictions vs ground truth labels and plot accuracy.

    If ``eval_frames`` is a set of 0-based frame indices, only those frames are scored
    (F1 / accuracy). Inference and trajectory video are unchanged — still full clip.
    """
    print(f"\n[EVAL] Evaluating against ground truth: {labels_path}")

    labels = _load_ground_truth_labels(labels_path, video_path)
    cap = cv2.VideoCapture(video_path)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    tol_fraction = 0.0075
    scaled_tol = tol_fraction * math.sqrt(vid_h**2 + vid_w**2)

    label_x = labels["X"].values
    label_y = labels["Y"].values
    pred_x = pred_df["X"].values
    pred_y = pred_df["Y"].values

    n = min(len(label_x), len(pred_x))
    per_frame = []
    eval_frame_nums = []
    TP = TN = FP1 = FP2 = FN = 0

    for i in range(n):
        fr = _frame_index_for_row(labels, i)
        if eval_frames is not None and fr not in eval_frames:
            continue
        xp, yp = float(pred_x[i]), float(pred_y[i])
        xt, yt = float(label_x[i]), float(label_y[i])

        eval_frame_nums.append(fr)

        if xp < 0 and yp < 0 and xt < 0 and yt < 0:
            TN += 1; per_frame.append("TN")
        elif xp > 0 and yp > 0 and xt < 0 and yt < 0:
            FP2 += 1; per_frame.append("FP2")
        elif xp < 0 and yp < 0 and xt > 0 and yt > 0:
            FN += 1; per_frame.append("FN")
        elif xt > 0 and yt > 0 and xp > 0 and yp > 0:
            dist = math.sqrt((xp - xt) ** 2 + (yp - yt) ** 2)
            if dist > scaled_tol:
                FP1 += 1; per_frame.append("FP1")
            else:
                TP += 1; per_frame.append("TP")
        else:
            per_frame.append("UNK")

    total = TP + TN + FP1 + FP2 + FN
    if total == 0:
        raise ValueError(
            "No frames matched the evaluation filter (check --eval-frame-range / "
            "--eval-frames-file vs label CSV Frame column)."
        )
    acc = (TP + TN) / total * 100 if total > 0 else 0
    prec = TP / (TP + FP1 + FP2) * 100 if (TP + FP1 + FP2) > 0 else 0
    rec = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0

    print(f"\n{'='*55}")
    print(f"  GROUND TRUTH EVALUATION — {video_name}")
    print(f"{'='*55}")
    if eval_frames is not None:
        print(
            f"  Eval subset       : {total} frames (video has {n} rows aligned; "
            f"scoring only selected indices)"
        )
    print(f"  Frames evaluated  : {total}")
    print(f"  Tolerance          : {scaled_tol:.2f} px")
    print(f"  TP (correct detect): {TP} ({TP/total*100:.1f}%)")
    print(f"  TN (correct miss)  : {TN} ({TN/total*100:.1f}%)")
    print(f"  FP1 (wrong loc)    : {FP1} ({FP1/total*100:.1f}%)")
    print(f"  FP2 (false detect) : {FP2} ({FP2/total*100:.1f}%)")
    print(f"  FN (missed ball)   : {FN} ({FN/total*100:.1f}%)")
    print(f"  --------------------------")
    print(f"  Accuracy           : {acc:.2f}%")
    print(f"  Precision          : {prec:.2f}%")
    print(f"  Recall             : {rec:.2f}%")
    p = prec / 100.0
    r = rec / 100.0
    f1 = 100.0 * (2 * p * r / (p + r)) if (p + r) > 1e-12 else 0.0
    print(f"  F1 score           : {f1:.2f}%  (harmonic mean of precision & recall on ball localization)")
    print(f"{'='*55}\n")

    # --- Plot ground truth metrics ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Ground Truth Evaluation — {video_name}", fontsize=16, fontweight='bold')

    # Confusion breakdown pie
    ax1 = axes[0]
    sizes = [TP, TN, FP1, FP2, FN]
    labels_pie = ['TP', 'TN', 'FP1\n(wrong loc)', 'FP2\n(false det)', 'FN\n(missed)']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
    nonzero = [(s, l, c) for s, l, c in zip(sizes, labels_pie, colors) if s > 0]
    if nonzero:
        ax1.pie(
            [x[0] for x in nonzero],
            labels=[x[1] for x in nonzero],
            colors=[x[2] for x in nonzero],
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11}
        )
    ax1.set_title('Confusion Breakdown', fontsize=14, fontweight='bold')

    # Bar chart (includes F1 when labels exist)
    ax2 = axes[1]
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    metrics_vals = [acc, prec, rec, f1]
    bar_colors = ['#2ecc71', '#3498db', '#f39c12', '#9b59b6']
    bars = ax2.bar(metrics_names, metrics_vals, color=bar_colors, edgecolor='white', width=0.55)
    for bar, val in zip(bars, metrics_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
                 ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.set_ylabel('%', fontsize=12)
    ax2.set_title('Model Performance (incl. F1)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Rolling accuracy (over evaluated frames only)
    ax3 = axes[2]
    correct = pd.Series([1 if c in ("TP", "TN") else 0 for c in per_frame])
    n_ev = len(per_frame)
    window = max(5, min(60, max(n_ev // 10, 3)))
    rolling_acc = correct.rolling(window=window, min_periods=1).mean() * 100
    ax3.plot(eval_frame_nums, rolling_acc, color="#2ecc71", linewidth=1.5)
    ax3.axhline(y=acc, color="red", linestyle="--", alpha=0.7, label=f"Overall: {acc:.1f}%")
    ax3.fill_between(eval_frame_nums, rolling_acc, alpha=0.15, color="#2ecc71")
    ax3.set_xlabel("Video frame index", fontsize=12)
    ax3.set_ylabel("Accuracy (%)", fontsize=12)
    ax3.set_title(f"Accuracy over eval frames (window={window})", fontsize=14, fontweight="bold")
    ax3.set_ylim(0, 105)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f"{video_name}_ground_truth_eval.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[EVAL] Ground truth plot saved to {plot_path}")

    metrics_csv = os.path.join(plots_dir, f"{video_name}_ground_truth_metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "video",
                "frames_scored",
                "tolerance_px",
                "TP",
                "TN",
                "FP1",
                "FP2",
                "FN",
                "accuracy_pct",
                "precision_pct",
                "recall_pct",
                "f1_pct",
                "eval_subset",
            ]
        )
        subset_note = (
            f"{len(eval_frames)}_indices_in_filter" if eval_frames is not None else "all_frames"
        )
        w.writerow(
            [
                video_name,
                total,
                f"{scaled_tol:.4f}",
                TP,
                TN,
                FP1,
                FP2,
                FN,
                f"{acc:.4f}",
                f"{prec:.4f}",
                f"{rec:.4f}",
                f"{f1:.4f}",
                subset_note,
            ]
        )
    print(f"[EVAL] Metrics CSV saved to {metrics_csv}")

    return acc, prec, rec, f1


def print_f1_without_labels_notice():
    print(
        "\n[METRICS] F1 score is **not** computed without ground-truth labels.\n"
        "          F1 needs TP / FP / FN, which require (x, y) or visibility per frame from a human.\n"
        "          Pass  --labels path/to/labels.csv  (same video, aligned rows) to get Precision, Recall, F1.\n"
        "          The detection-rate plot is only a proxy (how often the model fires), not accuracy vs truth."
    )


def print_finetune_workflow():
    print(
        "\n[FINETUNE] Improving weights with more footage (high level):\n"
        "  1. Step1-Frames — extract frames from new videos (video_to_frames.py).\n"
        "  2. Step2-Labelling — mark ball (x, y) per frame (labelling_tool.py); export CSV.\n"
        "  3. Step3-FixLabels — convert label format if needed (fix_labels.py).\n"
        "  4. Step4-BatchData — build .npy training batches (gen_data.py) from frames + CSV.\n"
        "  5. Step5-TransferLearning — transfer-learning.ipynb: load pretrained TrackNet weights,\n"
        "     train on your batches (more diverse clips → better generalization), export SavedModel.\n"
        "  6. Point this pipeline at new weights:  --weights /path/to/saved_model_dir\n"
        "  Tip: mix lighting, camera angles, indoor/outdoor; label occlusions as no-ball where appropriate."
    )


# ===================================================================
# MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="TrackNet Pickleball — Inference Pipeline")
    parser.add_argument("--video", required=False, default=None,
                        help="Path to video: local path OR gs://bucket/path/video.mp4")
    parser.add_argument("--weights", default=None,
                        help="Path to SavedModel weights dir (auto-downloads if not provided)")
    parser.add_argument(
        "--labels",
        default=None,
        help="Ground-truth CSV (Frame,X,Y,... per labeled frame). Enables Accuracy, Precision, Recall, F1.",
    )
    parser.add_argument(
        "--eval-frame-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Inclusive 0-based frame range for F1/accuracy only (e.g. 0 1029). Full video still inferred.",
    )
    parser.add_argument(
        "--eval-frames-file",
        default=None,
        help="Text file: one 0-based frame index per line (# comments OK). Scores only those frames.",
    )
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <video_name>_output)")
    parser.add_argument("--upload-results", action="store_true",
                        help="Upload results (trajectory video, CSV, plots) back to GCS bucket")
    parser.add_argument("--list-bucket", action="store_true",
                        help="List all videos in the GCS bucket and exit")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Inference batch size — larger = faster but more GPU VRAM. "
                             "If OOM, the run auto-splits batches; try 4 or 2 on small GPUs (default: 8)")
    parser.add_argument(
        "--separate-trajectory-pass",
        action="store_true",
        help="Read the video twice: inference then trajectory (slower). Default is one combined pass.",
    )
    parser.add_argument(
        "--ball-color",
        default="magenta",
        help="Ball/trail color: preset name (yellow,magenta,cyan,lime,orange,red,white,blue) or #RRGGBB",
    )
    parser.add_argument("--heatmap-threshold", type=float, default=0.5,
                        help="Heatmap binarization threshold (lower → more detections, more FP). Default: 0.5")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Drop detections with peak heatmap value below this (0–1). Try 0.1–0.25 to cut FPs.")
    parser.add_argument("--smooth-window", type=int, default=0,
                        help="Temporal median window (odd, ≥3). E.g. 5 smooths jitter without retraining. 0=off.")
    parser.add_argument("--max-jump-frac", type=float, default=0.0,
                        help="Reject detections jumping more than this × video diagonal vs previous hit (kills teleports). Try 0.12–0.2. 0=off.")
    parser.add_argument("--marker-radius-base", type=int, default=8,
                        help="Base filled-circle radius for newest trail point (pixels).")
    parser.add_argument("--marker-radius-per-trail", type=int, default=2,
                        help="Extra radius step per older trail point.")
    parser.add_argument("--bbox-half", type=int, default=16,
                        help="Half side length of square box drawn on current ball position (pixels).")
    parser.add_argument("--no-bbox", action="store_true", help="Draw only circles, no square box.")
    parser.add_argument("--no-timestamp", action="store_true",
                        help="Use plain *_trajectory.mp4 name without date-time suffix.")
    args = parser.parse_args()

    if (args.eval_frame_range is not None or args.eval_frames_file) and not args.labels:
        parser.error("--eval-frame-range / --eval-frames-file require --labels")

    # --- List bucket videos ---
    if args.list_bucket:
        videos = list_gcs_videos()
        if videos:
            print(f"\nVideos in gs://{GCS_BUCKET}/{GCS_VIDEO_FOLDER}/:")
            for v in videos:
                print(f"  {v}")
        else:
            print("No videos found (or bucket not accessible).")
        sys.exit(0)

    # --- Require --video unless --list-bucket ---
    if not args.list_bucket and not args.video:
        parser.error("--video is required (unless using --list-bucket)")

    # --- Resolve video path (GCS or local) ---
    video_input = args.video
    if video_input.startswith("gs://"):
        video_path = download_from_gcs(video_input)
    elif not os.path.isfile(video_input):
        print(f"[INFO] Local file not found, trying GCS: gs://{GCS_BUCKET}/{GCS_VIDEO_FOLDER}/{video_input}")
        video_path = download_from_gcs(video_input)
    else:
        video_path = os.path.abspath(video_input)

    if not os.path.isfile(video_path):
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    # Video geometry for jump rejection + PIPE setup
    _cap_probe = cv2.VideoCapture(video_path)
    _vw = int(_cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    _vh = int(_cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _cap_probe.release()
    PIPE.video_diagonal = math.hypot(_vw, _vh) if _vw > 0 and _vh > 0 else 1920.0

    try:
        PIPE.ball_bgr = parse_ball_color(args.ball_color)
    except ValueError as e:
        parser.error(str(e))
    PIPE.heatmap_threshold = args.heatmap_threshold
    PIPE.min_confidence = args.min_confidence
    sw = args.smooth_window
    if sw > 0 and sw % 2 == 0:
        sw += 1
    PIPE.smooth_window = sw
    PIPE.max_jump_frac = args.max_jump_frac
    PIPE.marker_radius_base = args.marker_radius_base
    PIPE.marker_radius_per_trail = args.marker_radius_per_trail
    PIPE.bbox_half_side = 0 if args.no_bbox else args.bbox_half
    PIPE.draw_bbox = not args.no_bbox

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_base = args.output_dir or os.path.join(os.path.dirname(video_path), f"{video_name}_output")
    dirs = setup_workspace(output_base)

    # --- GPU check ---
    gpus = tf.config.list_physical_devices('GPU')
    print(f"[SETUP] TensorFlow {tf.__version__} | GPU: {'YES — ' + gpus[0].name if gpus else 'NO (will be slow)'}")

    # --- Download / locate weights ---
    if args.weights and os.path.isdir(args.weights):
        weights_path = args.weights
    else:
        weights_path = download_weights(dirs["weights"])

    # --- Set batch size from CLI arg ---
    global INFERENCE_BATCH_SIZE
    INFERENCE_BATCH_SIZE = args.batch_size

    # --- Load model ---
    model = load_model_tfsm(weights_path)

    pred_csv = os.path.join(dirs["predictions"], f"{video_name}_predictions.csv")
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.no_timestamp:
        traj_video = os.path.join(dirs["trajectories"], f"{video_name}_trajectory.mp4")
    else:
        traj_video = os.path.join(dirs["trajectories"], f"{video_name}_trajectory_{_ts}.mp4")

    if args.separate_trajectory_pass:
        pred_df = make_predictions(video_path, model, pred_csv)
        if pred_df is None:
            sys.exit(1)
        traj_video = generate_trajectory(pred_csv, video_path, traj_video)
    else:
        pred_df, traj_video = inference_plus_trajectory(
            video_path, model, pred_csv, traj_video
        )
        if pred_df is None:
            sys.exit(1)

    # --- Detection stats (always runs — no labels needed) ---
    plot_path, det_rate = plot_detection_stats(pred_df, dirs["plots"], video_name)

    gt_f1 = None
    metrics_csv_path = None
    # --- Ground truth evaluation (F1 only possible with labels) ---
    if args.labels:
        if not os.path.isfile(args.labels):
            print(f"\n[WARN] --labels file not found: {args.labels} — skipping F1 / ground-truth eval.")
            print_f1_without_labels_notice()
            print_finetune_workflow()
        else:
            eval_subset = None
            if args.eval_frame_range is not None or args.eval_frames_file:
                try:
                    eval_subset = parse_eval_frame_subset(
                        args.eval_frame_range, args.eval_frames_file
                    )
                except ValueError as ex:
                    parser.error(str(ex))
                print(
                    f"[EVAL] Scoring {len(eval_subset)} frame indices only "
                    f"(inference + trajectory still use full video)."
                )
            acc, prec, rec, gt_f1 = evaluate_with_labels(
                pred_df,
                args.labels,
                video_path,
                dirs["plots"],
                video_name,
                eval_frames=eval_subset,
            )
            metrics_csv_path = os.path.join(dirs["plots"], f"{video_name}_ground_truth_metrics.csv")
    else:
        print("\n[INFO] No ground truth labels provided. Detection stats only (no F1).")
        print_f1_without_labels_notice()
        print_finetune_workflow()

    print(
        "\n[TUNING] Without retraining you can try:\n"
        "  --heatmap-threshold 0.45   (more recall) or 0.55 (stricter)\n"
        "  --min-confidence 0.15        (drop weak heatmap peaks / glare FPs)\n"
        "  --smooth-window 5            (median filter on positions)\n"
        "  --max-jump-frac 0.15         (reject impossible jumps vs previous detection)\n"
        "  Lighting / motion blur and court color still limit ceiling without new labels + fine-tune."
    )

    # --- Upload results to GCS ---
    if args.upload_results:
        print(f"\n[GCS] Uploading results to gs://{GCS_BUCKET}/ball_tracking/results/")
        upload_to_gcs(pred_csv)
        upload_to_gcs(traj_video)
        upload_to_gcs(plot_path)
        gt_plot = os.path.join(dirs["plots"], f"{video_name}_ground_truth_eval.png")
        if os.path.isfile(gt_plot):
            upload_to_gcs(gt_plot)
        if metrics_csv_path and os.path.isfile(metrics_csv_path):
            upload_to_gcs(metrics_csv_path)

    # --- Summary ---
    print(f"\n{'='*55}")
    print(f"  ALL OUTPUTS")
    print(f"{'='*55}")
    print(f"  Predictions CSV  : {pred_csv}")
    print(f"  Trajectory video : {traj_video}")
    print(f"  Stats plot       : {plot_path}")
    print(f"  Detection rate   : {det_rate:.2f}% (model fire rate — not F1)")
    if gt_f1 is not None:
        print(f"  F1 (with labels) : {gt_f1:.2f}%")
        if metrics_csv_path:
            print(f"  Metrics CSV      : {metrics_csv_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
