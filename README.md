# TrackNet-Pickleball

Real-time ball tracking for pickleball using **TrackNetV2** (U-Net encoder-decoder). Given a video, the system detects the ball in every frame, outputs a prediction CSV with pixel coordinates, and generates a trajectory overlay video.

## Model Architecture

![model architecture](https://github.com/AndrewDettor/TrackNet-Pickleball/blob/main/tnv2%20architecture%20picture.png)

**Input:** 3 consecutive RGB frames stacked channel-wise &rarr; `(9, 288, 512)`
**Output:** 3 heatmaps (one per frame) &rarr; `(3, 288, 512)`

Pre-trained on badminton (TrackNetV2), then fine-tuned on pickleball footage via transfer learning.

---

## Quick Start

### Option A: Google Colab Notebook

Open **`TrackNet_Pickleball_Colab.ipynb`** in [Google Colab](https://colab.research.google.com) with a **GPU runtime** (T4 is fine).

The notebook supports two modes:

| Mode | When to use | What you need |
|------|-------------|---------------|
| **Inference Only** (`SKIP_TO_INFERENCE = True`) | You just have a video (e.g. from YouTube) and no labels | A `.mp4` video |
| **Full Training** (`SKIP_TO_INFERENCE = False`) | You have a video **and** a hand-labeled CSV with ball positions | A `.mp4` video + label CSV |

**Inference-only flow:**
1. Run Sections 0-2 (setup, clone repo, download weights)
2. Upload your video (Section 3)
3. Extract frames (Step 1)
4. Set `SKIP_TO_INFERENCE = True`
5. Run Step 6 (predictions) &rarr; Step 8 (trajectory video)
6. Download results (Section 9)

**Full training flow:**
1. Run all sections in order
2. Upload both video and label CSV
3. Training fine-tunes the last 14 conv layers of the pre-trained model

### Option B: Local / VM Pipeline (`run_pipeline.py`)

Single-script pipeline optimized for GPU inference with batching, `@tf.function` compilation, and threaded video reading.

```bash
pip install opencv-python-headless pillow scikit-learn gdown matplotlib pandas tf-keras

python run_pipeline.py \
  --video /path/to/your_video.mp4 \
  --batch-size 16
```

With ground-truth labels for F1 evaluation:

```bash
python run_pipeline.py \
  --video /path/to/your_video.mp4 \
  --labels /path/to/labels.csv \
  --eval-frame-range 0 1029 \
  --batch-size 16
```

The script accepts local files or GCS URIs (`gs://bucket/path/video.mp4`). Weights are auto-downloaded from Google Drive if `--weights` is not provided.

**Key flags:**

| Flag | Description |
|------|-------------|
| `--video` | Path to video (local or `gs://`) |
| `--weights` | Path to SavedModel weights dir (auto-downloads if omitted) |
| `--labels` | Ground-truth CSV &mdash; enables Precision, Recall, F1 |
| `--eval-frame-range START END` | Score only these frames (0-based inclusive); full video still inferred |
| `--batch-size N` | Inference batch size (default 8; use 16-32 on large GPUs) |
| `--heatmap-threshold` | Heatmap binarization threshold (default 0.5) |
| `--min-confidence` | Drop weak detections (try 0.1-0.25) |
| `--smooth-window` | Temporal median filter (odd, e.g. 5) |
| `--max-jump-frac` | Reject impossible jumps (try 0.12-0.2) |
| `--ball-color` | Trail color: `yellow`, `magenta`, `cyan`, `#RRGGBB`, etc. |
| `--separate-trajectory-pass` | Two video reads instead of one combined pass |

**Output structure:**

```
<video_name>_output/
  predictions/    # <video>_predictions.csv (Frame, Visibility, X, Y, Confidence)
  trajectories/   # <video>_trajectory_<timestamp>.mp4
  plots/          # detection stats, F1 plots, ground_truth_metrics.csv
  weights/        # auto-downloaded model weights
```

---

## Labelling Tool

The labelling tool (`Step2-Labelling/`) is an OpenCV GUI for manually marking ball positions frame-by-frame. It outputs a CSV with columns `Frame, Ball, x, y` (normalized coordinates).

### Setup

1. Edit `Step2-Labelling/parser.py`:
   - Set `--label_video_path` to your video file (line 44)
   - Optionally set `--csv_path` to resume labelling from an existing CSV (line 46)

2. Run:
```bash
cd Step2-Labelling
python3 labelling_tool.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| **Left click** | Mark ball center |
| **Middle click** | Mark "no ball" for current frame |
| `n` | Next frame |
| `p` | Previous frame |
| `>` | Skip forward 36 frames |
| `<` | Skip backward 36 frames |
| `f` | Jump to first frame |
| `l` | Jump to last frame |
| `s` | Save CSV |
| `e` | Exit |

**Important:** Keyboard focus must be on the OpenCV window (click on the video window), not the terminal.

The tool writes `<video_name>.csv` in the working directory. This CSV is used as input for Step 3 (fix labels) and for `run_pipeline.py --labels`.

---

## Pipeline Steps

| Step | Folder | Description |
|------|--------|-------------|
| 1 | `Step1-Frames/` | Extract video into individual PNG frames |
| 2 | `Step2-Labelling/` | Hand-label ball positions (OpenCV GUI) |
| 3 | `Step3-FixLabels/` | Convert label CSV format (normalized &rarr; pixel coords) |
| 4 | `Step4-BatchData/` | Generate `.npy` training arrays from frames + labels |
| 5 | `Step5-TransferLearning/` | Fine-tune TrackNetV2 on pickleball data |
| 6 | `Step6-Predict/` | Run inference to get prediction CSV |
| 7 | `Step7-Performance/` | Evaluate model accuracy, precision, recall |
| 8 | `Step8-Trajectory/` | Overlay ball trajectory on video |

All steps are also available as cells in the Colab notebook.

---

## Weights

Model weights are too large for GitHub. Download from Google Drive:

- [Old Weights](https://drive.google.com/file/d/16ZnOljaxW6zM4bP7TTo1t81gaty7Egts/view?usp=sharing) (TrackNetV2 pre-trained on badminton)
- [New Weights](https://drive.google.com/drive/folders/1EGsddY1fgEJ5ITrfF32aPCn6nml2Anzr?usp=sharing) (fine-tuned on pickleball)

`run_pipeline.py` auto-downloads the new weights via `gdown` if `--weights` is not specified.

---

## Mandatory Files to Upload

```
TrackNet-Pickleball/
  README.md
  TrackNet_Pickleball_Colab.ipynb
  run_pipeline.py
  Step1-Frames/
    README.md
  Step2-Labelling/
    README.md
    labelling_tool.py
    parser.py
    utils.py
  Step3-FixLabels/
    README.md
    (fix_labels script)
  Step4-BatchData/
    README.md
    (batch data script)
  Step5-TransferLearning/
    README.md
    (training script)
  Step6-Predict/
    README.md
    (predict script)
  Step7-Performance/
    README.md
    model_performance.ipynb
  Step8-Trajectory/
    README.md
    show_trajectory.ipynb
  Presentation/
    *.jpg
  Flow Chart.jpg
  tnv2 architecture picture.png
  ProjectReport.pdf
```

### Do NOT upload

- `.DS_Store`
- `requirements.txt` (the existing one is a Kaggle image dump with 700+ packages; not useful as-is)
- `run_pipeline_backup.py` (development artifact)
- `test.csv` / any label CSVs with your data
- Model weights (use Google Drive links)
- Output directories (`*_output/`)
- Video files (`.mp4`)
- `.npy` training data

---

## System Requirements

- **GPU** required for training (Step 5) and fast inference (Step 6 / `run_pipeline.py`)
- **CPU** works for labelling, frame extraction, and trajectory generation (inference will be slow)
- Python 3.8+
- TensorFlow 2.x with GPU support
- Key packages: `opencv-python`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `gdown`, `tf-keras`

## Sources

- [Labelling Tool (original)](https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2)
- [TrackNetV2](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2)
