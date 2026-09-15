#!/usr/bin/env python3
"""Render a complete Waymo scene as a clean RGB MP4.

The defaults match the 080 experiment and its ``eval_28999.txt`` result.  The
script deliberately writes only rendered RGB frames (no depth/flow panels),
and shifts the final inference window backwards so every source frame is
emitted exactly once.
"""

import argparse
import json
import logging
import os
import re
import time
from bisect import bisect_right
from types import SimpleNamespace

# data_utils conditionally exposes semantic text features under FEAT_DIST.
# The saved experiment uses these features, so make the import deterministic
# instead of requiring callers to export the training-time environment flag.
os.environ.setdefault("FEAT_DIST", "1")

import numpy as np
import torch

from engine_tools import build_model
from main_slarm import get_args_parser
from src.dataset.constants import DATASET_DICT
from src.dataset.data_utils import prepare_inputs_and_targets, to_batch_tensor
from src.dataset.datasets import SingleSequenceDataset
from src.utils.logging import setup_logging
from tools.lseg_feat_extractor import LSegFeatureExtractor


def _load_config(config_path, overrides):
    with open(config_path, "r", encoding="utf-8") as handle:
        values = json.load(handle)
    # Keep the trained experiment configuration intact; only visualization
    # controls are intentionally overlaid from the command line.
    for key in ("scene_id", "start", "end", "output", "camera"):
        value = getattr(overrides, key, None)
        if value is not None:
            values[key] = value
    # args.json stores dataset as a list, while argparse also accepts a list.
    return SimpleNamespace(**values)


def _rgb_frames(prediction, camera="center", sigmoid_rgb=False):
    rendered = prediction["render_results"]
    key = rendered["rgb_key"]
    rgb = rendered[key][0].detach().float().cpu().numpy()  # T,V,H,W,C
    if not np.isfinite(rgb).all():
        raise ValueError("Rendered RGB contains NaN or infinity")
    print(
        f"[RGB] raw range [{rgb.min():.6f}, {rgb.max():.6f}], "
        f"mean {rgb.mean():.6f}",
        flush=True,
    )
    # This checkpoint was trained with sigmoid_rgb=True, so rendered RGB is in
    # [0, 1]. Rasterization can introduce tiny outliers, making range-based
    # detection unreliable; use the saved model configuration explicitly.
    if sigmoid_rgb:
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb = np.clip((rgb + 1.0) * 127.5, 0, 255).astype(np.uint8)
    if camera == "all":
        return [np.concatenate(tuple(frame), axis=1) for frame in rgb]
    view = rgb.shape[1] // 2 if camera == "center" else int(camera)
    if view < 0 or view >= rgb.shape[1]:
        raise ValueError(f"camera index {view} is outside [0, {rgb.shape[1]})")
    return list(rgb[:, view])


@torch.no_grad()
def render_scene(
    dataset, model, device, scene_id, start, end, feat_extractor, camera, sigmoid_rgb
):
    scene = dataset.annotations[scene_id % len(dataset.annotations)]
    end = min(end, int(scene["num_timesteps"]))
    interval = dataset.get_interval(int(scene["fps"]))
    output = []
    windows = list(range(start, end, interval))
    for window_index, requested_start in enumerate(windows, 1):
        print(
            f"[RGB] window {window_index}/{len(windows)} "
            f"(frames {requested_start}:{min(requested_start + interval, end)})...",
            flush=True,
        )
        started = time.perf_counter()
        segment_start = requested_start
        if segment_start + interval > end:
            segment_start = max(start, end - interval)
        data = to_batch_tensor(dataset.get_segment(scene_id, segment_start, return_all=True))
        inputs, _ = prepare_inputs_and_targets(
            data, device, v=int(data["num_max_cams"]), feat_extractor=feat_extractor, is_vis=True
        )
        # Match inference.py's mixed-precision behavior without requiring CUDA.
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(inputs)
        else:
            prediction = model(inputs)
        frames = _rgb_frames(prediction, camera, sigmoid_rgb=sigmoid_rgb)
        relative_start = requested_start - segment_start
        output.extend(frames[relative_start:])
        print(
            f"[RGB] window {window_index}/{len(windows)} done "
            f"({len(frames[relative_start:])} frames, {time.perf_counter() - started:.1f}s)",
            flush=True,
        )
    return output, int(scene["fps"]), end - start


def main():
    base = get_args_parser()
    base.add_argument("--config", default="work_dirs/slarm/waymo_scene_080_test/args.json")
    base.add_argument("--checkpoint", default=None)
    base.add_argument("--scene-id", type=int, default=0)
    base.add_argument("--annotation", default="scene_list/waymo_val.txt")
    base.add_argument("--all-scenes", action="store_true")
    base.add_argument("--start", type=int, default=0)
    base.add_argument("--end", type=int, default=None)
    base.add_argument("--output", default="work_dirs/slarm/waymo_scene_080_test/videos/scene080_rgb.mp4")
    base.add_argument("--camera", choices=("center", "all"), default="center")
    cli = base.parse_args()
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise SystemExit("Video writing requires imageio; install requirements.txt first.") from exc
    args = _load_config(cli.config, cli)
    if cli.device != "cuda":
        args.device = cli.device
    if cli.data_root != "./data/SLARM_data":
        args.data_root = cli.data_root
    args.scene_id = cli.scene_id
    args.start = cli.start
    args.end = cli.end
    args.output = cli.output
    args.camera = cli.camera

    if cli.checkpoint:
        checkpoint = cli.checkpoint
    else:
        config_name = os.path.basename(cli.config)
        if config_name.startswith("eval_"):
            step = config_name[len("eval_"):].split(".")[0]
        else:
            # args.json belongs to the 080 experiment evaluated at step 28999.
            step = "28999"
        experiment_dir = (
            os.path.dirname(os.path.dirname(cli.config))
            if config_name.startswith("eval_")
            else os.path.dirname(cli.config)
        )
        checkpoint = os.path.join(experiment_dir, "checkpoints", f"ckpt_{int(step):06d}.pth")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    annotation = cli.annotation
    if not os.path.isabs(annotation):
        annotation = os.path.join(args.data_root, annotation)
    if not os.path.exists(annotation):
        raise FileNotFoundError(f"Annotation list not found: {annotation}")
    args.ckpt_dir = os.path.dirname(checkpoint)
    args.load_from = None
    args.resume_from = checkpoint
    args.auto_resume = False
    setup_logging(output=os.path.dirname(args.output) or ".", level=logging.INFO)
    device = torch.device(args.device)
    model = build_model(args).to(device).eval()
    # The checkpoint was trained with online LSeg features; retain that path by default.
    feat_extractor = None
    if getattr(args, "online_feat", False):
        feat_extractor = LSegFeatureExtractor(
            args.lseg_model_pretrained_path,
            args.lseg_model_scratch_path,
            dtype=torch.float16 if os.environ.get("DISABLE_BFLOAT") else torch.bfloat16,
        )
    from src.utils.misc import load_model
    load_model(args, model)
    dataset = SingleSequenceDataset(
        data_root=args.data_root,
        annotation_txt_file_list=annotation,
        target_size=tuple(args.input_size),
        num_context_timesteps=args.num_context_timesteps,
        num_target_timesteps=args.num_target_timesteps,
        timespan=args.timespan,
        num_max_cams=args.num_max_cameras,
        load_depth=args.load_depth,
        load_flow=args.load_flow,
        load_semantic_label=args.load_semantic_label,
        online_feat=args.online_feat,
        img_norm_for_online_feat=args.img_norm_for_online_feat,
    )
    if cli.all_scenes:
        if os.path.splitext(args.output)[1].lower() in {".mp4", ".mov"}:
            raise ValueError("--output must be a directory when --all-scenes is used")
        output_dir = args.output
        scene_ids = range(len(dataset.annotations))
    else:
        output_dir = os.path.dirname(args.output) or "."
        scene_ids = [args.scene_id]

    os.makedirs(output_dir, exist_ok=True)
    for scene_id in scene_ids:
        scene = dataset.annotations[scene_id]
        scene_end = args.end if args.end is not None else int(scene["num_timesteps"])
        frames, fps, expected = render_scene(
            dataset, model, device, scene_id, args.start, scene_end,
            feat_extractor, args.camera, sigmoid_rgb=args.sigmoid_rgb,
        )
        if len(frames) != expected:
            raise RuntimeError(f"Rendered {len(frames)} frames, expected {expected}")
        if cli.all_scenes:
            scene_name = scene.get("scene_name", f"scene_{scene_id:03d}")
            scene_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(scene_name))
            output_path = os.path.join(output_dir, f"{scene_id:03d}_{scene_name}.mp4")
        else:
            output_path = args.output
        imageio.mimsave(output_path, frames, fps=fps, macro_block_size=1)
        print(
            f"Saved scene {scene_id} ({len(frames)} RGB frames, "
            f"{len(frames) / fps:.2f}s at {fps} FPS) to {output_path}", flush=True,
        )


if __name__ == "__main__":
    main()
