#!/usr/bin/env python

"""Convert dual-arm UMI recordings to the VISTA LeRobot v3.0 format."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

DEFAULT_INPUT_ROOT = Path("/home/jianan/workspace/data/0616_dex")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/jianan/workspace/data/0616_dex_vista_lerobot_v30/task_001_umi_episode"
)
DEFAULT_REPO_ID = "local/0616-dex-vista-v30"
DEFAULT_TASK = "umi episode"
DEFAULT_FPS = 30

SIDES = ("l", "r")
POSE_DIR = "localization/pose/pika_{side}"
GRIPPER_DIR = "gripper/encoder/pika_{side}"
FISHEYE_DIR = "camera/color/pikaFisheyeCamera_{side}"
IMAGE_KEYS = {
    "l": "observation.images.robot_0",
    "r": "observation.images.robot_1",
}
ARM_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper_width")
STATE_NAMES = tuple(f"robot_{robot_index}_{name}" for robot_index in (0, 1) for name in ARM_NAMES)


@dataclass(frozen=True)
class EpisodeFiles:
    path: Path
    pose: dict[str, list[Path]]
    gripper: dict[str, list[Path]]
    fisheye: dict[str, list[Path]]

    @property
    def frame_count(self) -> int:
        return len(self.pose["l"])


def load_synced_files(directory: Path) -> list[Path]:
    sync_path = directory / "sync.txt"
    if not sync_path.is_file():
        raise FileNotFoundError(f"Required sync file not found: {sync_path}")

    files = []
    for line in sync_path.read_text().splitlines():
        filename = line.strip()
        if not filename:
            continue
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"File listed in sync.txt does not exist: {path}")
        try:
            float(path.stem)
        except ValueError as exc:
            raise ValueError(f"Synced filename must use a numeric timestamp: {path.name}") from exc
        files.append(path)

    if not files:
        raise ValueError(f"No files listed in sync file: {sync_path}")
    return files


def discover_episode_dirs(input_root: Path) -> list[Path]:
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")

    indexed_dirs = []
    for path in input_root.iterdir():
        if not path.is_dir() or not path.name.startswith("episode"):
            continue
        suffix = path.name.removeprefix("episode")
        if suffix.isdigit():
            indexed_dirs.append((int(suffix), path))

    if not indexed_dirs:
        raise ValueError(f"No episode<number> directories found under: {input_root}")
    return [path for _, path in sorted(indexed_dirs)]


def discover_episode_files(episode_dir: Path) -> EpisodeFiles:
    pose = {
        side: load_synced_files(episode_dir / POSE_DIR.format(side=side)) for side in SIDES
    }
    gripper = {
        side: load_synced_files(episode_dir / GRIPPER_DIR.format(side=side)) for side in SIDES
    }
    fisheye = {
        side: load_synced_files(episode_dir / FISHEYE_DIR.format(side=side)) for side in SIDES
    }

    lengths = {
        f"{modality}_{side}": len(files)
        for modality, side_files in (
            ("pose", pose),
            ("gripper", gripper),
            ("fisheye", fisheye),
        )
        for side, files in side_files.items()
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Synced modality counts do not match in {episode_dir}: {lengths}")

    return EpisodeFiles(path=episode_dir, pose=pose, gripper=gripper, fisheye=fisheye)


def load_arm_state(pose_path: Path, gripper_path: Path) -> np.ndarray:
    pose = json.loads(pose_path.read_text())
    gripper = json.loads(gripper_path.read_text())

    position = [pose[name] for name in ("x", "y", "z")]
    euler_xyz = [pose[name] for name in ("roll", "pitch", "yaw")]
    quaternion_xyzw = Rotation.from_euler("xyz", euler_xyz).as_quat()
    gripper_width_mm = float(gripper["distance"]) * 1000.0

    state = np.asarray([*position, *quaternion_xyzw, gripper_width_mm], dtype=np.float32)
    if not np.isfinite(state).all():
        raise ValueError(f"Non-finite arm state from {pose_path} and {gripper_path}")
    return state


def load_fisheye_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        image = image.crop((left, top, left + crop_size, top + crop_size))
        return image.resize((224, 224), Image.Resampling.LANCZOS)


def build_features() -> dict[str, dict[str, Any]]:
    arm_feature = {
        "dtype": "float32",
        "shape": (8,),
        "names": list(ARM_NAMES),
    }
    state_feature = {
        "dtype": "float32",
        "shape": (16,),
        "names": list(STATE_NAMES),
    }
    image_feature = {
        "dtype": "video",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channels"],
    }
    return {
        "observation.state": state_feature,
        "action": state_feature.copy(),
        "robot_0_action": arm_feature,
        "robot_1_action": arm_feature.copy(),
        IMAGE_KEYS["l"]: image_feature,
        IMAGE_KEYS["r"]: image_feature.copy(),
    }


def create_dataset(output_root: Path, repo_id: str, fps: int) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    create_kwargs = {
        "repo_id": repo_id,
        "fps": fps,
        "features": build_features(),
        "root": output_root,
        "robot_type": "fastumi",
        "use_videos": True,
        "image_writer_processes": 0,
        "image_writer_threads": 4,
    }
    if "vcodec" in inspect.signature(LeRobotDataset.create).parameters:
        create_kwargs["vcodec"] = "h264"

    return LeRobotDataset.create(**create_kwargs)


def add_episode(dataset: Any, episode: EpisodeFiles, task: str) -> None:
    for frame_index in range(episode.frame_count):
        robot_0_action = load_arm_state(
            episode.pose["l"][frame_index], episode.gripper["l"][frame_index]
        )
        robot_1_action = load_arm_state(
            episode.pose["r"][frame_index], episode.gripper["r"][frame_index]
        )
        action = np.concatenate((robot_0_action, robot_1_action)).astype(np.float32)
        dataset.add_frame(
            {
                "task": task,
                "observation.state": action.copy(),
                "action": action,
                "robot_0_action": robot_0_action,
                "robot_1_action": robot_1_action,
                IMAGE_KEYS["l"]: load_fisheye_image(episode.fisheye["l"][frame_index]),
                IMAGE_KEYS["r"]: load_fisheye_image(episode.fisheye["r"][frame_index]),
            }
        )
    dataset.save_episode()


def convert_dataset(
    input_root: Path,
    output_root: Path,
    repo_id: str,
    task: str,
    fps: int,
    overwrite: bool,
) -> dict[str, Any]:
    episode_dirs = discover_episode_dirs(input_root)
    episodes = [discover_episode_files(path) for path in episode_dirs]

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    dataset = create_dataset(output_root=output_root, repo_id=repo_id, fps=fps)
    try:
        for episode_index, episode in enumerate(episodes):
            logging.info(
                "Converting episode %s/%s: %s (%s frames)",
                episode_index + 1,
                len(episodes),
                episode.path.name,
                episode.frame_count,
            )
            add_episode(dataset, episode=episode, task=task)
        dataset.finalize()
    except Exception:
        dataset.finalize()
        raise

    return {
        "repo_id": repo_id,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "task": task,
        "fps": fps,
        "total_episodes": len(episodes),
        "total_frames": sum(episode.frame_count for episode in episodes),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output-root if it already exists.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    manifest = convert_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        repo_id=args.repo_id,
        task=args.task,
        fps=args.fps,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
