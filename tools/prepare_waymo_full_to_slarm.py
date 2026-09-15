#!/usr/bin/env python3
"""Adapt the complete processed Waymo train/validation set for SLARM."""

import argparse
import json
import os
from pathlib import Path
from typing import Any


REQUIRED_ANNOTATION_KEYS = (
    "dataset",
    "scene_id",
    "scene_name",
    "num_timesteps",
    "camera_list",
    "normalized_time",
    "normalized_intrinsics",
    "camera_to_ego",
    "ego_to_world",
    "camera_to_world",
    "original_image_size",
    "relative_image_path",
    "fps",
)
SPLITS = (("training", "train", 798), ("validation", "val", 202))


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Create a lightweight SLARM data root for all Waymo scenes."
    )
    parser.add_argument(
        "--source-data-root",
        type=Path,
        default=Path(
            "/inspire/hdd/global_user/guoluosong-253108120129/aaai/long_dggt/data"
        ),
        help="Root containing waymo/processed and annotations/waymo.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=repo_root / "data/SLARM_data_full"
    )
    parser.add_argument(
        "--scene-list-root",
        type=Path,
        default=repo_root / "data/dataset_scene_list",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_directory_link(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    if destination.is_symlink():
        if destination.resolve(strict=True) != source:
            raise FileExistsError(
                f"Refusing to replace {destination} -> {os.readlink(destination)}"
            )
        return
    if destination.exists():
        raise FileExistsError(
            f"Refusing to replace existing path {destination}; use another output root"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=True)


def load_annotation(path: Path, scene_id: int, scene_name: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    missing = [key for key in REQUIRED_ANNOTATION_KEYS if key not in source]
    if missing:
        raise ValueError(f"{path} is missing keys: {', '.join(missing)}")
    if source["dataset"] != "waymo":
        raise ValueError(f"{path} has dataset={source['dataset']!r}, expected 'waymo'")
    if int(source["scene_id"]) != scene_id or source["scene_name"] != scene_name:
        raise ValueError(f"Scene metadata does not match list entry {scene_id}: {scene_name}")
    return {key: source[key] for key in REQUIRED_ANNOTATION_KEYS}


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle)
        handle.write("\n")
    temporary.replace(path)


def prepare_split(
    source_data_root: Path,
    output_root: Path,
    scene_list_root: Path,
    split: str,
    short_split: str,
    expected_count: int,
    overwrite: bool,
) -> None:
    canonical_list = scene_list_root / f"waymo_{short_split}_list.txt"
    scene_names = [
        line.strip()
        for line in canonical_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(scene_names) != expected_count:
        raise ValueError(
            f"{canonical_list} contains {len(scene_names)} scenes; expected {expected_count}"
        )

    source_annotations = source_data_root / "annotations/waymo" / split
    source_scenes = source_data_root / "waymo/processed" / split
    output_annotations = output_root / "annotations/waymo" / split
    output_annotations.mkdir(parents=True, exist_ok=True)
    list_entries = []

    for scene_id, scene_name in enumerate(scene_names):
        scene_directory = source_scenes / f"{scene_id:03d}"
        source_annotation = source_annotations / f"{scene_name}.json"
        if not scene_directory.is_dir():
            raise FileNotFoundError(f"Processed scene is missing: {scene_directory}")
        if not source_annotation.is_file():
            raise FileNotFoundError(f"Annotation is missing: {source_annotation}")

        output_annotation = output_annotations / source_annotation.name
        if overwrite or not output_annotation.is_file():
            annotation = load_annotation(source_annotation, scene_id, scene_name)
            write_json_atomic(output_annotation, annotation)
        list_entries.append(f"annotations/waymo/{split}/{source_annotation.name}")

    output_list = output_root / "scene_list" / f"waymo_{short_split}.txt"
    output_list.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_list.with_suffix(output_list.suffix + ".tmp")
    temporary.write_text("\n".join(list_entries) + "\n", encoding="utf-8")
    temporary.replace(output_list)
    print(f"Prepared {len(list_entries)} {split} scenes: {output_list}")


def main() -> None:
    args = parse_args()
    source_data_root = args.source_data_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    scene_list_root = args.scene_list_root.resolve(strict=True)

    ensure_directory_link(
        source_data_root / "waymo/processed", output_root / "datasets/waymo"
    )
    for split, short_split, expected_count in SPLITS:
        prepare_split(
            source_data_root,
            output_root,
            scene_list_root,
            split,
            short_split,
            expected_count,
            args.overwrite,
        )
    print(f"SLARM data root: {output_root}")


if __name__ == "__main__":
    main()
