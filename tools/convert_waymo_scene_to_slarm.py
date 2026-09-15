#!/usr/bin/env python3
"""Create a lightweight SLARM data root for one processed Waymo scene.

The source scene is never modified.  Image/depth/mask data are exposed through
a symbolic link, while the annotation and scene-list files are written under a
new SLARM data root.
"""

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

REQUIRED_MODALITIES = (
    ("images", ".jpg"),
    ("images_4", ".jpg"),
    ("depth_flows_4", ".npy"),
    ("ground_label_4", ".png"),
    ("sky_masks", ".png"),
    ("dynamic_masks", ".png"),
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_source = Path(
        "/inspire/hdd/global_user/guoluosong-253108120129/aaai/long_dggt/"
        "data/waymo/processed/training/621"
    )
    parser = argparse.ArgumentParser(
        description="Adapt one processed Waymo scene to SLARM without changing it."
    )
    parser.add_argument("--source-scene", type=Path, default=default_source)
    parser.add_argument(
        "--output-root", type=Path, default=repo_root / "data/SLARM_data_621"
    )
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=repo_root / "data/dataset_scene_list/waymo_train_list.txt",
        help="One scene name per line; the numeric scene id is a zero-based index.",
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=None,
        help="Defaults to <data-root>/annotations/waymo/training.",
    )
    return parser.parse_args()


def count_flat_files(directory: Path, suffix: str) -> int:
    return sum(path.is_file() for path in directory.glob(f"*{suffix}"))


def load_and_validate_annotation(
    annotation_path: Path, scene_id: int, source_scene: Path
) -> dict[str, Any]:
    with annotation_path.open("r", encoding="utf-8") as handle:
        source_annotation = json.load(handle)

    missing = [key for key in REQUIRED_ANNOTATION_KEYS if key not in source_annotation]
    if missing:
        raise ValueError(f"Annotation is missing keys: {', '.join(missing)}")
    if source_annotation["dataset"] != "waymo":
        raise ValueError(f"Expected Waymo annotation, got {source_annotation['dataset']!r}")
    if int(source_annotation["scene_id"]) != scene_id:
        raise ValueError(
            f"Annotation scene_id={source_annotation['scene_id']} does not match {scene_id}"
        )

    num_timesteps = int(source_annotation["num_timesteps"])
    cameras = [str(camera) for camera in source_annotation["camera_list"]]
    expected_camera_files = num_timesteps * len(cameras)
    for modality, suffix in REQUIRED_MODALITIES:
        directory = source_scene / modality
        if not directory.is_dir():
            raise FileNotFoundError(f"Required modality directory is missing: {directory}")
        actual = count_flat_files(directory, suffix)
        if actual != expected_camera_files:
            raise ValueError(
                f"{directory} has {actual} flat {suffix} files; "
                f"expected {expected_camera_files}"
            )

    for camera in cameras:
        relative_paths = source_annotation["relative_image_path"].get(camera, [])
        if len(relative_paths) != num_timesteps:
            raise ValueError(
                f"Camera {camera} has {len(relative_paths)} image paths; "
                f"expected {num_timesteps}"
            )
        expected_prefix = f"training/{scene_id:03d}/images/"
        if not all(path.startswith(expected_prefix) for path in relative_paths):
            raise ValueError(
                f"Camera {camera} image paths do not start with {expected_prefix!r}"
            )

    # Instance tracks can make these JSON files several MB, but SLARM's dataset
    # loader only consumes the fields below.
    return {key: source_annotation[key] for key in REQUIRED_ANNOTATION_KEYS}


def ensure_scene_link(source_scene: Path, destination: Path) -> None:
    source_scene = source_scene.resolve(strict=True)
    if destination.is_symlink():
        if destination.resolve(strict=True) != source_scene:
            raise FileExistsError(
                f"Refusing to replace existing link {destination} -> {os.readlink(destination)}"
            )
        return
    if destination.exists():
        raise FileExistsError(
            f"Refusing to replace existing path: {destination}. "
            "Choose another --output-root."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source_scene, target_is_directory=True)


def main() -> None:
    args = parse_args()
    source_scene = args.source_scene.resolve(strict=True)
    try:
        scene_id = int(source_scene.name)
    except ValueError as exc:
        raise ValueError(
            f"Source scene directory must have a numeric name: {source_scene}"
        ) from exc

    with args.scene_list.open("r", encoding="utf-8") as handle:
        scene_names = [line.strip() for line in handle if line.strip()]
    if not 0 <= scene_id < len(scene_names):
        raise IndexError(
            f"Scene id {scene_id} is outside scene list range 0..{len(scene_names) - 1}"
        )
    scene_name = scene_names[scene_id]

    data_root = source_scene.parents[3]
    annotation_root = (
        args.annotation_root
        if args.annotation_root is not None
        else data_root / "annotations/waymo/training"
    )
    source_annotation_path = annotation_root / f"{scene_name}.json"
    annotation = load_and_validate_annotation(
        source_annotation_path, scene_id, source_scene
    )
    if annotation["scene_name"] != scene_name:
        raise ValueError(
            f"Scene-list name {scene_name!r} does not match annotation name "
            f"{annotation['scene_name']!r}"
        )

    output_root = args.output_root.resolve()
    scene_destination = (
        output_root / f"datasets/waymo/training/{scene_id:03d}"
    )
    ensure_scene_link(source_scene, scene_destination)

    annotation_directory = output_root / "annotations/waymo/training"
    annotation_directory.mkdir(parents=True, exist_ok=True)
    output_annotation_path = annotation_directory / f"{scene_name}.json"
    with output_annotation_path.open("w", encoding="utf-8") as handle:
        json.dump(annotation, handle)
        handle.write("\n")

    relative_annotation = f"annotations/waymo/training/{scene_name}.json"
    scene_list_directory = output_root / "scene_list"
    scene_list_directory.mkdir(parents=True, exist_ok=True)
    for filename in ("waymo_train.txt", "waymo_train_1_scene.txt"):
        (scene_list_directory / filename).write_text(
            relative_annotation + "\n", encoding="utf-8"
        )

    print(f"SLARM data root: {output_root}")
    print(f"Scene data link: {scene_destination} -> {source_scene}")
    print(f"Annotation: {output_annotation_path}")
    print(
        f"Validated {annotation['num_timesteps']} timesteps x "
        f"{len(annotation['camera_list'])} cameras."
    )


if __name__ == "__main__":
    main()
