import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random


def adapt_lerobot_batch_sarm(
    batch: Dict[str, Any],
    camera_names: List[str] = ["top_camera-images-rgb"],
    eval_video: bool = False,
) -> Dict[str, Any]:
    """
    Convert a batch to the (multi-stage) LeRobot-compatible format.

    When eval_video=True, wrap single-example tensors with a leading batch dim
    and wrap the scalar task into a single-item list to mimic batched inputs.
    """
    def maybe_unsqueeze(x):
        return x.unsqueeze(0) if eval_video else x

    result = {
        "image_frames": {},
        "targets": maybe_unsqueeze(batch["targets"]),
        "lengths": maybe_unsqueeze(batch["lengths"]),
        "tasks": [batch["task"]] if eval_video else batch["task"],
        "state": maybe_unsqueeze(batch["state"]),
        "frame_relative_indices": maybe_unsqueeze(batch["frame_relative_indices"]),
    }

    for cam_name in camera_names:
        result["image_frames"][cam_name] = maybe_unsqueeze(batch[cam_name])

    return result


def adapt_lerobot_batch_rewind(
    batch: dict,
    camera_names: List[str] = ["top_camera-images-rgb"],
    eval_video: bool = False
) -> dict:
    """Convert to lerobot-compatible batch format.
    
    Args:
        batch: Input batch dictionary.
        camera_names: List of camera keys to include.
        eval_video: If True, wrap tensors with an additional batch dimension.
    """
    def maybe_unsqueeze(x):
        return x.unsqueeze(0) if eval_video else x

    result = {
        "image_frames": {},
        "targets": maybe_unsqueeze(batch["targets"]),
        "lengths": maybe_unsqueeze(batch["lengths"]),
        "tasks": batch["task"],
        "state": maybe_unsqueeze(batch["state"]),
        "frame_relative_indices": maybe_unsqueeze(batch["frame_relative_indices"]),
    }

    for cam_name in camera_names:
        result["image_frames"][cam_name] = maybe_unsqueeze(batch[cam_name])

    return result



def get_valid_episodes(repo_id: str) -> List[int]:
    """
    Collects valid episode indices under the lerobot cache for the given repo_id.

    Args:
        repo_id (str): HuggingFace repo ID, 

    Returns:
        List[int]: Sorted list of valid episode indices (e.g., [0, 1, 5, 7, ...])
    """
    repo_path = Path(repo_id)
    if repo_path.is_absolute():
        base_path = repo_path / "data"
    else:
        base_path = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id / "data"
    episode_pattern = re.compile(r"episode_(\d+)\.parquet")

    valid_episodes = []

    if not base_path.exists():
        raise FileNotFoundError(f"Data directory not found: {base_path}")

    for chunk_dir in base_path.glob("chunk-*"):
        if not chunk_dir.is_dir():
            continue
        for file in chunk_dir.glob("episode_*.parquet"):
            match = episode_pattern.match(file.name)
            if match:
                ep_idx = int(match.group(1))
                valid_episodes.append(ep_idx)

    return sorted(valid_episodes)

def split_train_eval_episodes(valid_episodes: List[int], train_ratio: float = 0.9, seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Randomly split valid episodes into training and evaluation sets.

    Args:
        valid_episodes (List[int]): List of valid episode indices.
        train_ratio (float): Fraction of episodes to use for training (default: 0.9).
        seed (int): Random seed for reproducibility (default: 42).

    Returns:
        Tuple[List[int], List[int]]: (train_episodes, eval_episodes)
    """
    random.seed(seed)
    episodes = valid_episodes.copy()
    random.shuffle(episodes)

    split_index = int(len(episodes) * train_ratio)
    train_episodes = episodes[:split_index]
    eval_episodes = episodes[split_index:]

    return train_episodes, eval_episodes

