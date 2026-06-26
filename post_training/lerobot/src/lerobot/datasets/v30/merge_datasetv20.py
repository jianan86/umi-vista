import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import (
    write_episode,
    write_episode_stats,
    write_info,
    write_task,
)


def validate_all_metadata(all_metadata: list[LeRobotDatasetMetadata]):
    """Validate metadata consistency across datasets (fps, robot_type, features)"""
    fps = all_metadata[0].fps
    robot_type = all_metadata[0].robot_type
    features = all_metadata[0].features

    for meta in tqdm(all_metadata, desc="Validate all meta data"):
        if fps != meta.fps:
            raise ValueError(f"Same fps is expected, but got fps={meta.fps} instead of {fps}.")
        if robot_type != meta.robot_type:
            raise ValueError(f"Same robot_type is expected, but got robot_type={meta.robot_type} instead of {robot_type}.")
        if features != meta.features:
            raise ValueError(f"Same features is expected, but got features={meta.features} instead of {features}.")

    return fps, robot_type, features


def merge_lerobot_datasets(
    source_dirs: list[Path],
    output_dir: Path,
    repo_id: str = None,
):
    """
    Merge multiple LeRobot v2.x datasets
    
    Args:
        source_dirs: List of source dataset directories
        output_dir: Merged output directory
        repo_id: Name of the merged dataset, for example "user/merged_dataset"
    """
    
    # 1. Load all metadata
    all_metadata = [LeRobotDatasetMetadata("", root=src_dir) for src_dir in source_dirs]
    
    # 2. Validate consistency
    fps, robot_type, features = validate_all_metadata(all_metadata)
    
    # 3. Clean and create the output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # 4. Create merged metadata
    merged_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id or f"{output_dir.parent.name}/{output_dir.name}",
        root=output_dir,
        fps=fps,
        robot_type=robot_type,
        features=features,
    )
    
    # 5. Merge task-index mappings
    datasets_task_index_to_merged_task_index = {}
    merged_task_index = 0
    
    for dataset_idx, meta in enumerate(tqdm(all_metadata, desc="Merge tasks index")):
        task_index_to_merged_task_index = {}
        
        for task_idx, task in meta.tasks.items():
            if task not in merged_meta.task_to_task_index:
                # Add new tasks to the merged mapping
                merged_meta.tasks[merged_task_index] = task
                merged_meta.task_to_task_index[task] = merged_task_index
                merged_task_index += 1
            
            task_index_to_merged_task_index[task_idx] = merged_meta.task_to_task_index[task]
        
        datasets_task_index_to_merged_task_index[dataset_idx] = task_index_to_merged_task_index
    
    # 6. Merge episode indices and global index offsets
    datasets_ep_idx_to_merged_ep_idx = {}
    datasets_merged_episode_index_shift = {}
    datasets_merged_index_shift = {}
    merged_episode_index_shift = 0
    merged_frame_index_shift = 0
    
    for dataset_idx, meta in enumerate(tqdm(all_metadata, desc="Merge episodes and global index")):
        ep_idx_to_merged_ep_idx = {}
        
        # Build the episode-index mapping
        for episode_idx in range(meta.total_episodes):
            merged_ep_idx = episode_idx + merged_episode_index_shift
            ep_idx_to_merged_ep_idx[episode_idx] = merged_ep_idx
        
        datasets_ep_idx_to_merged_ep_idx[dataset_idx] = ep_idx_to_merged_ep_idx
        datasets_merged_episode_index_shift[dataset_idx] = merged_episode_index_shift
        datasets_merged_index_shift[dataset_idx] = merged_frame_index_shift
        
        # Populate episode information
        for episode_idx, episode_dict in meta.episodes.items():
            merged_ep_idx = episode_idx + merged_episode_index_shift
            episode_dict["episode_index"] = merged_ep_idx
            merged_meta.episodes[merged_ep_idx] = episode_dict
        
        # Populate episode stats
        for episode_idx, episode_stats in meta.episodes_stats.items():
            merged_ep_idx = episode_idx + merged_episode_index_shift
            merged_meta.episodes_stats[merged_ep_idx] = episode_stats
        
        # Update info statistics
        merged_meta.info["total_episodes"] += meta.total_episodes
        merged_meta.info["total_frames"] += meta.total_frames
        merged_meta.info["total_videos"] += len(merged_meta.video_keys) * meta.total_episodes
        
        # Update offsets
        merged_episode_index_shift += meta.total_episodes
        merged_frame_index_shift += meta.total_frames
    
    # 7. Write merged metadata
    merged_meta.info["total_tasks"] = len(merged_meta.tasks)
    merged_meta.info["total_chunks"] = merged_meta.get_episode_chunk(merged_episode_index_shift - 1)
    merged_meta.info["splits"] = {"train": f"0:{merged_meta.info['total_episodes']}"}
    
    # Write episodes.jsonl
    for episode_dict in tqdm(merged_meta.episodes.values(), desc="Write episodes info"):
        write_episode(episode_dict, merged_meta.root)
    
    # # Write episodes_stats.jsonl
    # for episode_idx, episode_stats in tqdm(merged_meta.episodes_stats.items(), desc="Write episodes stats info"):
    #     write_episode_stats(episode_idx, episode_stats, merged_meta.root)
    
    # Write tasks.jsonl
    for task_idx, task in tqdm(merged_meta.tasks.items(), desc="Write tasks info"):
        write_task(task_idx, task, merged_meta.root)
    
    # Write info.json
    write_info(merged_meta.info, merged_meta.root)
    
    # 8. Copy and update data files (parquet and video)
    for dataset_idx, meta in enumerate(tqdm(all_metadata, desc="Copy data files")):
        merged_episode_index_shift = datasets_merged_episode_index_shift[dataset_idx]
        merged_frame_index_shift = datasets_merged_index_shift[dataset_idx]
        task_index_to_merged_task_index = datasets_task_index_to_merged_task_index[dataset_idx]
        
        # Copy and update parquet data
        for episode_idx in range(meta.total_episodes):
            merged_ep_idx = datasets_ep_idx_to_merged_ep_idx[dataset_idx][episode_idx]
            
            data_path = meta.root / meta.get_data_file_path(episode_idx)
            merged_data_path = merged_meta.root / merged_meta.get_data_file_path(merged_ep_idx)
            merged_data_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Update index, episode_index, and task_index
            df = pd.read_parquet(data_path)
            df["index"] += merged_frame_index_shift
            df["episode_index"] = merged_ep_idx  # Set directly to the new episode_index
            df["task_index"] = df["task_index"].map(task_index_to_merged_task_index)
            df.to_parquet(merged_data_path)
        
        # Copy video files
        for episode_idx in range(meta.total_episodes):
            merged_ep_idx = episode_idx + merged_episode_index_shift
            
            for vid_key in meta.video_keys:
                video_path = meta.root / meta.get_video_file_path(episode_idx, vid_key)
                merged_video_path = merged_meta.root / merged_meta.get_video_file_path(merged_ep_idx, vid_key)
                merged_video_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(video_path, merged_video_path)
    
    print(f"✅ Merge completed. Output directory: {output_dir}")
    print(f"   - Total episodes: {merged_meta.info['total_episodes']}")
    print(f"   - Total frames: {merged_meta.info['total_frames']}")
    print(f"   - Total tasks: {merged_meta.info['total_tasks']}")
    
    return merged_meta


# ============ Usage example ============

if __name__ == "__main__":
    # Example: merge multiple datasets
    source_dirs = [
        Path("/path/to/dataset1"),
        Path("/path/to/dataset2"),
        Path("/path/to/dataset3"),
    ]
    output_dir = Path("/path/to/merged_dataset")
    repo_id = "username/merged_dataset"
    
    # Run the merge
    merged_meta = merge_lerobot_datasets(
        source_dirs=source_dirs,
        output_dir=output_dir,
        repo_id=repo_id,
    )