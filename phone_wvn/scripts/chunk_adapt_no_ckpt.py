#!/usr/bin/env python3
#
# WVN-style head training/adaptation with robust weak supervision:
# - freeze feature extraction (DINO/DINOv2/STEGO)
# - train only traversability head
# - confidence gating + ignore masks
# - temporal consistency via optical flow (if cv2 available)
# - conservative updates + per-chunk rollback
#
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import colormaps

from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.feature_extractor import FeatureExtractor

try:
    import cv2
except Exception:
    cv2 = None


class Data:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class SimpleMLP(torch.nn.Module):
    def __init__(self, input_size=64, hidden_sizes=None, reconstruction=False):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 1]
        cfg = list(hidden_sizes)
        self.nr_sigmoid_layers = cfg[-1]
        if reconstruction:
            cfg[-1] = cfg[-1] + input_size
        layers = []
        for hs in cfg[:-1]:
            layers.append(torch.nn.Linear(input_size, hs))
            layers.append(torch.nn.ReLU())
            input_size = hs
        layers.append(torch.nn.Linear(input_size, cfg[-1]))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, data: Data) -> torch.Tensor:
        x = self.layers(data.x)
        x[:, : self.nr_sigmoid_layers] = torch.sigmoid(x[:, : self.nr_sigmoid_layers])
        return x


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--mode", default="adapt", choices=["adapt", "pretrain"])
    parser.add_argument(
        "--input_frames_dir",
        default="/home/nikhil.rane/repos/wild_visual_navigation/phone_wvn/data/frames/test_clip",
        help="Used in adapt mode: path to one clip frames folder.",
    )
    parser.add_argument(
        "--pretrain_frames_root",
        default="/home/nikhil.rane/repos/wild_visual_navigation/phone_wvn/data/frames",
        help="Used in pretrain mode: root directory containing one subfolder per clip.",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/nikhil.rane/repos/wild_visual_navigation/phone_wvn/outputs/chunk_adapt_no_ckpt",
        help="Output folder for overlays/logs/checkpoints.",
    )
    parser.add_argument("--feature_type", default="dinov2", choices=["dino", "dinov2", "stego"])
    parser.add_argument("--segmentation_type", default="grid", choices=["slic", "grid", "random", "stego"])
    parser.add_argument("--dino_patch_size", type=int, default=8, choices=[8, 16])
    parser.add_argument("--dino_backbone", default="vit_small", choices=["vit_small"])
    parser.add_argument("--slic_num_components", type=int, default=100)
    parser.add_argument("--network_input_image_height", type=int, default=224)
    parser.add_argument("--network_input_image_width", type=int, default=224)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--epochs_per_chunk", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--confidence_threshold", type=float, default=0.7)
    parser.add_argument("--rollback_factor", type=float, default=1.2)
    parser.add_argument("--max_frames", type=int, default=0, help="0 means use all frames (adapt mode).")
    parser.add_argument("--max_frames_per_clip", type=int, default=0, help="0 means use all frames (pretrain mode).")
    parser.add_argument("--max_clips", type=int, default=0, help="0 means use all clip folders (pretrain mode).")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init_head_ckpt", default="", help="Optional path to initial head checkpoint.")
    parser.add_argument("--save_head_ckpt", default="", help="Optional path for final head checkpoint.")
    parser.add_argument("--use_temporal_consistency", action="store_true")
    parser.add_argument("--no-use_temporal_consistency", dest="use_temporal_consistency", action="store_false")
    parser.set_defaults(use_temporal_consistency=True)
    parser.add_argument("--use_optical_flow", action="store_true")
    parser.add_argument("--no-use_optical_flow", dest="use_optical_flow", action="store_false")
    parser.set_defaults(use_optical_flow=True)
    parser.add_argument(
        "--prediction_granularity",
        default="segment",
        choices=["segment", "pixel"],
        help="segment uses region features (slic/grid) and makes segmentation type matter.",
    )
    parser.add_argument(
        "--inference_conservative_mode",
        action="store_true",
        help="Apply conservative post-processing to reduce false-positive traversability.",
    )
    parser.add_argument(
        "--no-inference_conservative_mode",
        dest="inference_conservative_mode",
        action="store_false",
    )
    parser.set_defaults(inference_conservative_mode=True)
    parser.add_argument(
        "--inference_obstacle_penalty",
        type=float,
        default=0.7,
        help="How strongly obstacle-like pixels are suppressed at inference (0..1).",
    )
    parser.add_argument(
        "--inference_bottom_bias",
        type=float,
        default=0.2,
        help="Bias toward lower image regions being more traversable (0..1).",
    )
    parser.add_argument(
        "--inference_sharpen_gamma",
        type=float,
        default=1.5,
        help="Exponent for conservative calibration (>1 lowers mid/high scores).",
    )
    return parser.parse_args()


def resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return Path(WVN_ROOT_DIR) / p


def list_frames(input_dir: Path) -> List[Path]:
    frames = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png")))
    if len(frames) == 0:
        raise ValueError(f"No frames found in {input_dir}")
    return frames


def load_and_resize(image_path: Path, new_h: int, new_w: int, device: torch.device) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    torch_image = torch.from_numpy(np.array(img)).to(device).permute(2, 0, 1).float() / 255.0
    return F.interpolate(torch_image[None], size=(new_h, new_w), mode="bilinear", align_corners=False)[0]


def basic_seeds(h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    yf = yy.float() / float(max(h - 1, 1))
    xf = xx.float() / float(max(w - 1, 1))

    # Trapezoid-like bottom center prior.
    center_width = 0.12 + 0.18 * ((yf - 0.6).clamp(min=0.0) / 0.4).clamp(max=1.0)
    positive = (yf > 0.6) & (torch.abs(xf - 0.5) < center_width)
    negative = (yf < 0.3) | (xf < 0.06) | (xf > 0.94)
    return positive, negative


def obstacle_mask_from_image(cur_img: torch.Tensor) -> torch.Tensor:
    gray = cur_img.mean(dim=0, keepdim=True)[None]  # 1x1xHxW
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=cur_img.device)[None, None]
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=cur_img.device)[None, None]
    gx = F.conv2d(gray, kx, padding=1)[0, 0]
    gy = F.conv2d(gray, ky, padding=1)[0, 0]
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    # Prefer vertical structures.
    vertical = torch.abs(gx) > (1.1 * torch.abs(gy))
    thr = torch.quantile(mag.reshape(-1), 0.88)
    return (mag > thr) & vertical


def temporal_warp_and_motion(
    prev_img: torch.Tensor,
    cur_img: torch.Tensor,
    prev_pos_mask: torch.Tensor,
    use_optical_flow: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w = prev_pos_mask.shape
    if (cv2 is None) or (not use_optical_flow):
        diff = (cur_img - prev_img).abs().mean(dim=0)
        diff_norm = diff / (diff.max() + 1e-6)
        return prev_pos_mask.clone(), diff_norm

    prev_gray = (prev_img.mean(dim=0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    cur_gray = (cur_img.mean(dim=0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x - flow[:, :, 0]
    map_y = grid_y - flow[:, :, 1]
    prev_mask_u8 = (prev_pos_mask.detach().cpu().numpy().astype(np.uint8) * 255)
    warped = cv2.remap(prev_mask_u8, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    warped = torch.from_numpy((warped > 127).astype(np.bool_)).to(prev_pos_mask.device)

    mag = np.sqrt((flow[:, :, 0] ** 2) + (flow[:, :, 1] ** 2))
    mag = torch.from_numpy(mag).to(prev_pos_mask.device).float()
    mag = mag / (torch.quantile(mag.reshape(-1), 0.95) + 1e-6)
    mag = mag.clamp(0.0, 1.0)
    return warped, mag


def build_weak_labels(
    cur_img: torch.Tensor,
    prev_img: Optional[torch.Tensor],
    prev_pos_mask: Optional[torch.Tensor],
    use_temporal_consistency: bool,
    use_optical_flow: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h, w = cur_img.shape[1], cur_img.shape[2]
    seed_pos, seed_neg = basic_seeds(h, w, cur_img.device)
    obstacle = obstacle_mask_from_image(cur_img)

    labels = torch.full((h, w), -1.0, device=cur_img.device)
    conf = torch.zeros((h, w), device=cur_img.device)

    pos = seed_pos.clone()
    stable = torch.zeros_like(seed_pos)
    motion_norm = torch.zeros((h, w), device=cur_img.device)
    if use_temporal_consistency and (prev_img is not None) and (prev_pos_mask is not None):
        warped_prev, motion_norm = temporal_warp_and_motion(prev_img, cur_img, prev_pos_mask, use_optical_flow)
        stable = warped_prev & (motion_norm < 0.18)
        pos = pos | stable

    neg = seed_neg | obstacle
    conflict = pos & neg
    pos = pos & (~conflict)
    neg = neg & (~conflict)

    labels[pos] = 1.0
    labels[neg] = 0.0

    conf[pos] = 0.82
    conf[neg] = 0.78
    conf[stable] = 0.94
    conf[conflict] = 0.0

    # Penalize high-motion areas unless they are stable positives.
    conf = torch.where(motion_norm > 0.5, conf * 0.6, conf)
    conf = conf.clamp(0.0, 1.0)
    return labels, conf, pos


@torch.no_grad()
def extract_features_bundle(
    feature_extractor: FeatureExtractor,
    frame_path: Path,
    h: int,
    w: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cur_img = load_and_resize(frame_path, h, w, device)
    _, feat, seg, _, dense_feat = feature_extractor.extract(
        img=cur_img[None],
        return_centers=False,
        return_dense_features=True,
        n_random_pixels=100,
    )
    return cur_img, feat, seg, dense_feat


def segment_supervision_from_pixels(
    seg: torch.Tensor,
    labels: torch.Tensor,
    conf: torch.Tensor,
    majority_threshold: float = 0.6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seg_flat = seg.reshape(-1).long()
    y_flat = labels.reshape(-1)
    c_flat = conf.reshape(-1)

    valid_seg = seg_flat >= 0
    if int(valid_seg.sum()) == 0:
        return torch.empty((0,), device=seg.device), torch.empty((0,), device=seg.device)

    seg_flat = seg_flat[valid_seg]
    y_flat = y_flat[valid_seg]
    c_flat = c_flat[valid_seg]
    n_seg = int(seg_flat.max().item()) + 1

    total_count = torch.bincount(seg_flat, minlength=n_seg).float()
    pos_mask = y_flat == 1
    neg_mask = y_flat == 0
    pos_count = torch.bincount(seg_flat[pos_mask], minlength=n_seg).float()
    neg_count = torch.bincount(seg_flat[neg_mask], minlength=n_seg).float()
    labeled_count = pos_count + neg_count

    seg_labels = torch.full((n_seg,), -1.0, device=seg.device)
    pos_ratio = pos_count / (total_count + 1e-6)
    neg_ratio = neg_count / (total_count + 1e-6)
    seg_labels[pos_ratio >= majority_threshold] = 1.0
    seg_labels[(neg_ratio >= majority_threshold) & (seg_labels < 0)] = 0.0

    conf_sum = torch.bincount(seg_flat[y_flat >= 0], weights=c_flat[y_flat >= 0], minlength=n_seg)
    seg_conf = conf_sum / (labeled_count + 1e-6)
    seg_conf[labeled_count == 0] = 0.0
    return seg_labels, seg_conf


@torch.no_grad()
def infer_traversability(
    frame_path: Path,
    feature_extractor: FeatureExtractor,
    model: torch.nn.Module,
    h: int,
    w: int,
    device: torch.device,
    prediction_granularity: str,
    conservative_mode: bool,
    obstacle_penalty: float,
    bottom_bias: float,
    sharpen_gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cur_img, feat, seg, dense_feat = extract_features_bundle(feature_extractor, frame_path, h, w, device)
    if prediction_granularity == "segment":
        pred_seg = model(Data(x=feat))[:, 0].clamp(0, 1)
        pred = torch.zeros_like(seg, dtype=torch.float32)
        valid = seg >= 0
        pred[valid] = pred_seg[seg[valid]]
    else:
        x = dense_feat[0].permute(1, 2, 0).reshape(-1, dense_feat.shape[1])
        pred = model(Data(x=x))[:, 0].reshape(h, w).clamp(0, 1)

    if conservative_mode:
        # Make uncertain positives less optimistic.
        pred = pred.clamp(0.0, 1.0) ** max(1.0, float(sharpen_gamma))

        # Suppress obstacle-like image structures.
        obstacle = obstacle_mask_from_image(cur_img)
        penalty = float(np.clip(obstacle_penalty, 0.0, 1.0))
        pred = pred * torch.where(obstacle, 1.0 - penalty, 1.0)

        # Weak geometric prior: lower part of the image is more likely traversable.
        yy = torch.linspace(0.0, 1.0, h, device=pred.device)[:, None]
        bias = float(np.clip(bottom_bias, 0.0, 1.0))
        pred = pred * (1.0 - bias + bias * yy)
        pred = pred.clamp(0.0, 1.0)

    return cur_img, pred


def save_overlay(torch_image: torch.Tensor, trav_map: torch.Tensor, target_path: Path):
    img = (torch_image.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    cmap = colormaps["RdYlGn"]
    trav_rgb = (cmap(trav_map.detach().cpu().numpy())[:, :, :3] * 255.0).astype(np.uint8)
    overlay = ((1.0 - 0.45) * img + 0.45 * trav_rgb).astype(np.uint8)
    Image.fromarray(overlay).save(target_path)


def load_head_checkpoint(model: torch.nn.Module, ckpt_path: Path):
    payload = torch.load(ckpt_path, map_location="cpu")
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded head checkpoint: {ckpt_path}")


def gather_sequences(args) -> List[List[Path]]:
    if args.mode == "adapt":
        frames = list_frames(resolve_path(args.input_frames_dir))
        if args.max_frames > 0:
            frames = frames[: args.max_frames]
        return [frames]

    root = resolve_path(args.pretrain_frames_root)
    if not root.exists():
        raise ValueError(f"Pretrain frames root not found: {root}")
    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if args.max_clips > 0:
        subdirs = subdirs[: args.max_clips]
    sequences = []
    for d in subdirs:
        frames = list_frames(d)
        if args.max_frames_per_clip > 0:
            frames = frames[: args.max_frames_per_clip]
        if len(frames) > 0:
            sequences.append(frames)
    if len(sequences) == 0:
        raise ValueError(f"No pretrain clips with frames found in {root}")
    return sequences


def flatten_sequences(sequences: Sequence[Sequence[Path]]) -> List[Tuple[int, int, Path]]:
    items = []
    for clip_id, seq in enumerate(sequences):
        for frame_id, frame_path in enumerate(seq):
            items.append((clip_id, frame_id, frame_path))
    return items


def train_head(
    args,
    feature_extractor: FeatureExtractor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sequences: List[List[Path]],
    output_dir: Path,
    ckpt_dir: Path,
    device: torch.device,
) -> List[str]:
    bce = torch.nn.BCELoss()
    log_lines = [
        "mode,chunk,epoch,mean_loss,used_pixels,total_pixels,used_ratio,mean_conf,rolled_back"
    ]

    items = flatten_sequences(sequences)
    total_items = len(items)
    chunk_size = max(1, args.chunk_size)
    n_chunks = int(np.ceil(total_items / chunk_size))
    prev_chunk_loss = None

    for chunk_id in range(n_chunks):
        s = chunk_id * chunk_size
        e = min(total_items, (chunk_id + 1) * chunk_size)
        chunk_items = items[s:e]
        print(f"Chunk {chunk_id + 1}/{n_chunks}: items {s}..{e - 1}")

        chunk_start_state = deepcopy(model.state_dict())
        epoch_losses = []
        rolled_back = 0

        for epoch in range(args.epochs_per_chunk):
            model.train()
            loss_sum = 0.0
            loss_count = 0
            used_pixels = 0
            total_pixels = 0
            conf_sum = 0.0
            conf_count = 0

            prev_clip_id = None
            prev_img = None
            prev_pos = None

            for clip_id, _, frame_path in chunk_items:
                with torch.no_grad():
                    cur_img, feat, seg, dense_feat = extract_features_bundle(
                        feature_extractor,
                        frame_path,
                        args.network_input_image_height,
                        args.network_input_image_width,
                        device,
                    )

                    same_clip = (prev_clip_id == clip_id)
                    labels, conf, pos_mask = build_weak_labels(
                        cur_img=cur_img,
                        prev_img=prev_img if same_clip else None,
                        prev_pos_mask=prev_pos if same_clip else None,
                        use_temporal_consistency=args.use_temporal_consistency,
                        use_optical_flow=args.use_optical_flow,
                    )

                if args.prediction_granularity == "segment":
                    x = feat
                    y, c = segment_supervision_from_pixels(seg=seg, labels=labels, conf=conf)
                else:
                    x = dense_feat[0].permute(1, 2, 0).reshape(-1, dense_feat.shape[1])
                    y = labels.reshape(-1)
                    c = conf.reshape(-1)

                valid = (y >= 0) & (c >= args.confidence_threshold)
                total_pixels += int((y >= 0).sum().item())
                if int(valid.sum()) == 0:
                    prev_clip_id = clip_id
                    prev_img = cur_img
                    prev_pos = pos_mask
                    continue

                pred = model(Data(x=x))[:, 0]
                loss = bce(pred[valid], y[valid])

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                optimizer.step()

                loss_sum += float(loss.item())
                loss_count += 1
                used_pixels += int(valid.sum().item())
                conf_sum += float(c[valid].sum().item())
                conf_count += int(valid.sum().item())

                prev_clip_id = clip_id
                prev_img = cur_img
                prev_pos = pos_mask

            mean_loss = loss_sum / max(loss_count, 1)
            mean_conf = conf_sum / max(conf_count, 1)
            used_ratio = used_pixels / max(total_pixels, 1)
            epoch_losses.append(mean_loss)
            log_lines.append(
                f"{args.mode},{chunk_id},{epoch},{mean_loss:.6f},{used_pixels},{total_pixels},{used_ratio:.6f},{mean_conf:.6f},0"
            )
            print(
                f"  epoch {epoch + 1}/{args.epochs_per_chunk} - mean_loss={mean_loss:.6f}, "
                f"used_ratio={used_ratio:.3f}, mean_conf={mean_conf:.3f}"
            )

        chunk_loss = float(np.mean(epoch_losses)) if len(epoch_losses) > 0 else 0.0
        if prev_chunk_loss is not None and chunk_loss > (args.rollback_factor * prev_chunk_loss):
            model.load_state_dict(chunk_start_state, strict=True)
            rolled_back = 1
            print(
                f"  rollback: chunk_loss={chunk_loss:.6f} > "
                f"{args.rollback_factor:.2f} * prev_chunk_loss={prev_chunk_loss:.6f}"
            )
        else:
            prev_chunk_loss = chunk_loss

        ckpt_path = ckpt_dir / f"head_chunk_{chunk_id + 1:03d}.pt"
        torch.save({"model_state_dict": model.state_dict(), "chunk_id": chunk_id, "mode": args.mode}, ckpt_path)
        if rolled_back == 1:
            log_lines.append(f"{args.mode},{chunk_id},-1,{chunk_loss:.6f},0,0,0.000000,0.000000,1")

    return log_lines


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = resolve_path(args.output_dir)
    before_dir = output_dir / "before"
    after_dir = output_dir / "after"
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "adapt":
        before_dir.mkdir(parents=True, exist_ok=True)
        after_dir.mkdir(parents=True, exist_ok=True)

    sequences = gather_sequences(args)
    if args.mode == "adapt":
        print(f"Using {len(sequences[0])} frames from {resolve_path(args.input_frames_dir)}")
    else:
        total_frames = sum(len(s) for s in sequences)
        print(f"Using {len(sequences)} clips and {total_frames} total frames from {resolve_path(args.pretrain_frames_root)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if cv2 is None and args.use_optical_flow:
        print("Warning: OpenCV not available; falling back to frame-difference temporal cue.")

    feature_extractor = FeatureExtractor(
        device=device,
        segmentation_type=args.segmentation_type,
        feature_type=args.feature_type,
        patch_size=args.dino_patch_size,
        backbone_type=args.dino_backbone,
        input_size=args.network_input_image_height,
        slic_num_components=args.slic_num_components,
    )

    model = SimpleMLP(
        input_size=feature_extractor.feature_dim,
        hidden_sizes=[256, 32, 1],
        reconstruction=True,
    ).to(device)

    if args.init_head_ckpt.strip():
        load_head_checkpoint(model, resolve_path(args.init_head_ckpt))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Baseline inference only in adapt mode.
    if args.mode == "adapt":
        print("Running baseline inference...")
        model.eval()
        with torch.no_grad():
            for i, frame_path in enumerate(sequences[0]):
                cur_img, trav_map = infer_traversability(
                    frame_path=frame_path,
                    feature_extractor=feature_extractor,
                    model=model,
                    h=args.network_input_image_height,
                    w=args.network_input_image_width,
                    device=device,
                    prediction_granularity=args.prediction_granularity,
                    conservative_mode=args.inference_conservative_mode,
                    obstacle_penalty=args.inference_obstacle_penalty,
                    bottom_bias=args.inference_bottom_bias,
                    sharpen_gamma=args.inference_sharpen_gamma,
                )
                save_overlay(cur_img, trav_map, before_dir / f"{i:05d}.png")

    print(f"Starting head training in mode={args.mode}...")
    log_lines = train_head(
        args=args,
        feature_extractor=feature_extractor,
        model=model,
        optimizer=optimizer,
        sequences=sequences,
        output_dir=output_dir,
        ckpt_dir=ckpt_dir,
        device=device,
    )

    # Post-adaptation inference only in adapt mode.
    if args.mode == "adapt":
        print("Running post-adaptation inference...")
        model.eval()
        with torch.no_grad():
            for i, frame_path in enumerate(sequences[0]):
                cur_img, trav_map = infer_traversability(
                    frame_path=frame_path,
                    feature_extractor=feature_extractor,
                    model=model,
                    h=args.network_input_image_height,
                    w=args.network_input_image_width,
                    device=device,
                    prediction_granularity=args.prediction_granularity,
                    conservative_mode=args.inference_conservative_mode,
                    obstacle_penalty=args.inference_obstacle_penalty,
                    bottom_bias=args.inference_bottom_bias,
                    sharpen_gamma=args.inference_sharpen_gamma,
                )
                save_overlay(cur_img, trav_map, after_dir / f"{i:05d}.png")

    final_ckpt = resolve_path(args.save_head_ckpt) if args.save_head_ckpt.strip() else (output_dir / "head_final.pt")
    torch.save({"model_state_dict": model.state_dict(), "mode": args.mode}, final_ckpt)
    print(f"Saved final head checkpoint: {final_ckpt}")

    (output_dir / "train_log.csv").write_text("\n".join(log_lines) + "\n")
    print(f"Done. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
