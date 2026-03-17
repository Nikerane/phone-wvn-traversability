# Phone-Video Traversability Estimation

Adapted from [Wild Visual Navigation (WVN)](https://github.com/leggedrobotics/wild_visual_navigation) by ETH Zurich / University of Oxford.

> **Original paper:** Mattamala & Frey et al., ["Wild Visual Navigation: Fast Traversability Learning via Pre-Trained Models and Online Self-Supervision"](https://arxiv.org/abs/2404.07110), Autonomous Robots 2025.

This project reproduces the **WVN perception-learning stack** with phone video, replacing robot sensors and ROS transport with an offline phone-video pipeline.

---

## Demo Results

### DINOv2 + SLIC

https://github.com/user-attachments/assets/demo_dinov2_slic.mp4

> Left: **Input Video** | Right: **DINOv2 + SLIC** (pretrained head, segment-level prediction)

### STEGO features + STEGO segmentation

https://github.com/user-attachments/assets/demo_stego.mp4

> Left: **Input Video** | Right: **STEGO feat + seg** (pretrained head, segment-level prediction)

### Side-by-side: DINOv2+SLIC vs STEGO

https://github.com/user-attachments/assets/demo_dinov2_vs_stego.mp4

> Left: **Input Video** | Middle: **DINOv2 + SLIC** | Right: **STEGO feat + seg**

All use pretrained heads only (no adaptation on this video). The comparison shows which feature+segmentation combination produces better traversability maps out of the box.

**Finding:** DINOv2+SLIC gives cleaner, more locally grounded predictions. STEGO's semantic grouping over-propagates — if one region is mislabeled, all visually similar regions across the image inherit that error.

---

## Architecture

```
Phone camera video
  → Extract frames
  → Pretrained feature extraction (DINOv2, frozen)
  → Region segmentation (SLIC / Grid / STEGO)
  → Pool features per region (average DINOv2 features within each segment)
  → Traversability prediction per region (lightweight MLP head)
  → Weak supervision from walking demonstration priors
  → Chunked online adaptation of traversability head
  → Traversability overlay on video (green = safe, red = avoid)
```

This follows the same core idea as the WVN paper: **pretrained self-supervised features + online self-supervision** for fast traversability learning — but adapted for offline phone video without robot hardware.

---

## How DINOv2 and SLIC Are Combined

DINOv2 and SLIC run **independently** on the same image, then their outputs are merged:

```
Input frame (224x224 RGB)
  │
  ├──→ DINOv2 (frozen ViT backbone)
  │      Produces a dense feature map: one 384-dim vector per image patch
  │      These features encode texture, shape, and implicit semantic meaning
  │
  └──→ SLIC (classical algorithm on raw RGB pixels)
         Groups nearby pixels with similar color into ~120 superpixels
         Each superpixel is a small coherent region (e.g., a patch of grass, a stone)
         SLIC does NOT use any neural network — it only sees pixel colors and positions
  │
  ▼
  Combine: for each SLIC superpixel, average all the DINOv2 feature vectors
           of the pixels that belong to that superpixel
  │
  ▼
  Result: one 384-dim feature vector per superpixel region
  │
  ▼
  Traversability head (MLP): predicts a traversability score (0–1) per region
  │
  ▼
  Project back to pixels: each pixel gets the score of its superpixel
  │
  ▼
  Color overlay: green (safe) → yellow (uncertain) → red (avoid)
```

**Why this combination works well:**
- **DINOv2** provides rich, high-level features that capture what a region *is* (grass vs trunk vs path) — trained on millions of images via self-supervised learning.
- **SLIC** provides clean, boundary-aware grouping that respects edges in the image — so the prediction doesn't bleed across object boundaries.
- Together: smart features + clean regions = locally accurate traversability maps.

**In contrast, STEGO** does both steps internally (features + semantic segmentation from the same neural network). This can be powerful but also over-propagates errors: if one region is mislabeled, all visually similar regions across the entire image inherit that error.

---

## What is kept from WVN

- Feature extraction backbone (DINOv2 / DINO / STEGO)
- Segmentation strategies (SLIC, Grid, STEGO, Random)
- Traversability head (SimpleMLP)
- Online/chunked adaptation loop
- Confidence computation

## What is replaced

- ROS 1 transport → offline frame processing
- Robot proprioceptive supervision → weak pseudo-labels from video priors
- Onboard deployment → local GPU inference

---

## Pipeline Modes

### 1. Pretrain (build a traversability prior)

Train the head on diverse outdoor clips to create a generic prior checkpoint.

```bash
python phone_wvn/scripts/chunk_adapt_no_ckpt.py \
  --mode pretrain \
  --pretrain_frames_root phone_wvn/data/frames_pretrain \
  --save_head_ckpt phone_wvn/outputs/pretrain/head_prior.pt \
  --feature_type dinov2 \
  --segmentation_type slic \
  --prediction_granularity segment \
  --chunk_size 120 \
  --epochs_per_chunk 1 \
  --learning_rate 1e-4
```

### 2. Adapt (fine-tune on a target video)

Load the prior and adapt on a new clip.

```bash
python phone_wvn/scripts/chunk_adapt_no_ckpt.py \
  --mode adapt \
  --input_frames_dir phone_wvn/data/frames/my_video \
  --init_head_ckpt phone_wvn/outputs/pretrain/head_prior.pt \
  --output_dir phone_wvn/outputs/my_video_run \
  --feature_type dinov2 \
  --segmentation_type slic \
  --prediction_granularity segment \
  --chunk_size 100 \
  --epochs_per_chunk 1 \
  --learning_rate 1e-4
```

### 3. Generate comparison video

```bash
ffmpeg -y \
  -i phone_wvn/data/videos/my_video.mp4 \
  -i phone_wvn/outputs/my_video_run/before.mp4 \
  -i phone_wvn/outputs/my_video_run/after.mp4 \
  -filter_complex "[0:v]fps=30,scale=224:224,drawtext=text='RAW':x=20:y=20:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=8[v0];[1:v]drawtext=text='Before':x=20:y=20:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=8[v1];[2:v]drawtext=text='After':x=20:y=20:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=8[v2];[v0][v1][v2]hstack=inputs=3[v]" \
  -map "[v]" -c:v libx264 -pix_fmt yuv420p output_triptych.mp4
```

---

## Robustness Guards

- **Confidence-gated training**: only high-confidence pseudo-labels contribute to loss
- **Temporal consistency**: labels are stabilized across adjacent frames
- **Obstacle mask**: vertical edge structures are suppressed as non-traversable
- **Per-chunk rollback**: if a chunk worsens performance, weights revert
- **Gradient clipping**: prevents catastrophic weight updates
- **Conservative inference**: obstacle penalty + score sharpening reduce false positives

---

## Segmentation Types

| Type | How it groups pixels | Strengths |
|------|---------------------|-----------|
| **SLIC** | Color + spatial proximity (superpixels) | Good boundary alignment, paper-faithful |
| **Grid** | Fixed rectangular cells | Fast, stable, deterministic |
| **STEGO** | Learned semantic clustering | Can discover object-level regions |
| **Random** | Random pixel sampling | Baseline/ablation only |

---

## Key CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `adapt` | `pretrain` or `adapt` |
| `--feature_type` | `dinov2` | `dino`, `dinov2`, `stego` |
| `--segmentation_type` | `grid` | `slic`, `grid`, `random`, `stego` |
| `--prediction_granularity` | `segment` | `segment` (region-level) or `pixel` |
| `--init_head_ckpt` | | Path to pretrained head checkpoint |
| `--chunk_size` | `100` | Frames per adaptation chunk |
| `--learning_rate` | `3e-4` | Head optimizer learning rate |
| `--confidence_threshold` | `0.7` | Minimum confidence to use a label |
| `--rollback_factor` | `1.2` | Rollback if chunk loss exceeds this multiple |
| `--inference_sharpen_gamma` | `1.5` | Sharpen traversability scores (higher = more decisive) |
| `--inference_obstacle_penalty` | `0.7` | Suppress obstacle-like regions during inference |

---

## Project Structure

```
phone_wvn/
  scripts/
    chunk_adapt_no_ckpt.py    # Main pipeline script
  data/
    videos/                   # Source video files (gitignored)
    frames/                   # Extracted frames (gitignored)
    frames_pretrain/          # Pretrain clip frames (gitignored)
  outputs/                    # Overlays, checkpoints, videos (gitignored)
  demo/
    demo_slic.mp4             # SLIC demo video
    demo_stego.mp4            # STEGO demo video
  .gitignore
  README.md
```

---

## Requirements

- Python 3.10+
- CUDA-enabled GPU (tested on NVIDIA RTX 3000 Ada, 8 GB VRAM)
- PyTorch 2.0+
- Key packages: `torchvision`, `kornia`, `matplotlib`, `omegaconf`, `fast-slic`

### Setup

```bash
# Clone this repo
git clone https://github.com/Nikerane/phone-wvn-traversability.git
cd phone-wvn-traversability

# Also clone the STEGO reimplementation (needed for feature extraction)
git clone https://github.com/leggedrobotics/self_supervised_segmentation.git ../self_supervised_segmentation

# Create venv and install
python3.10 -m venv ~/venv/wvn
source ~/venv/wvn/bin/activate
pip install -e . --no-deps
pip install -e ../self_supervised_segmentation --no-deps
pip install torch torchvision kornia matplotlib pillow omegaconf tqdm \
  scikit-image seaborn pandas pytictac torchmetrics pytorch-lightning wandb fast-slic
```

---

## Credits

This project is based on the [Wild Visual Navigation](https://github.com/leggedrobotics/wild_visual_navigation) system by ETH Zurich and University of Oxford.

```bibtex
@article{mattamala25wild,
  title = {Wild Visual Navigation: Fast Traversability Learning via Pre-Trained Models and Online Self-Supervision},
  author = {Mattamala, Matias and Frey, Jonas and Libera, Piotr and Chebrolu, Nived and Martius, Georg and Cadena, Cesar and Hutter, Marco and Fallon, Maurice},
  journal = {Autonomous Robots},
  volume = {49},
  number = {3},
  pages = {19},
  year = {2025},
  doi = {10.1007/s10514-025-10202-x},
}
```

## License

MIT License (same as original WVN repository)
