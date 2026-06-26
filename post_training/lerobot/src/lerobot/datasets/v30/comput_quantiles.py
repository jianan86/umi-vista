#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Brute-force exact q01/q99 computation for action/state on full dataset.

This script loads every sample from a LeRobot v3.0 dataset for:
- action
- observation.state

Then computes exact q01/q99 using numpy quantile on the full arrays.
No episode-level aggregation and no histogram approximation are used.
"""

import argparse
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_stats
from lerobot.utils.utils import init_logging

ACTION_KEY = "action"
STATE_KEY = "observation.state"
TARGET_KEYS = [ACTION_KEY, STATE_KEY]


class _ActionStateOnlyDataset(Dataset):
	"""Read only action/state directly from hf_dataset for faster loading."""

	def __init__(self, dataset: LeRobotDataset):
		self.hf_dataset = dataset.hf_dataset

	def __len__(self) -> int:
		return len(self.hf_dataset)

	def __getitem__(self, idx: int) -> dict[str, torch.Tensor | np.ndarray]:
		item = self.hf_dataset[idx]
		return {
			ACTION_KEY: item[ACTION_KEY],
			STATE_KEY: item[STATE_KEY],
		}


def _to_numpy_2d(value: torch.Tensor | np.ndarray) -> np.ndarray:
	"""Convert one sample tensor/array into shape (1, D) or (1, 1)."""
	if isinstance(value, torch.Tensor):
		arr = value.detach().cpu().numpy()
	elif isinstance(value, np.ndarray):
		arr = value
	else:
		arr = np.array(value)

	arr = np.asarray(arr)
	if arr.ndim == 0:
		return arr.reshape(1, 1)
	if arr.ndim == 1:
		return arr.reshape(1, -1)
	return arr.reshape(1, -1)


def _compute_exact_stats_from_full_array(full_array: np.ndarray) -> dict[str, np.ndarray]:
	"""Compute exact stats for vectors shaped as (N, D)."""
	# Use float64 for numerically stable quantile computation.
	x = full_array.astype(np.float64, copy=False)

	stats = {
		"min": np.min(x, axis=0),
		"max": np.max(x, axis=0),
		"mean": np.mean(x, axis=0),
		"std": np.std(x, axis=0),
		"count": np.array([x.shape[0]]),
		"q01": np.quantile(x, 0.01, axis=0, method="linear"),
		"q99": np.quantile(x, 0.99, axis=0, method="linear"),
	}
	return stats


def _collate_action_state(batch: list[dict[str, torch.Tensor | np.ndarray]]) -> dict[str, torch.Tensor]:
	"""Batch action/state and convert to torch tensors for efficient concatenation."""
	actions = [torch.as_tensor(sample[ACTION_KEY]) for sample in batch]
	states = [torch.as_tensor(sample[STATE_KEY]) for sample in batch]
	return {
		ACTION_KEY: torch.stack(actions, dim=0),
		STATE_KEY: torch.stack(states, dim=0),
	}


def compute_quantiles_bruteforce(
	dataset: LeRobotDataset,
	batch_size: int,
	num_workers: int,
	prefetch_factor: int,
) -> dict[str, dict[str, np.ndarray]]:
	"""Load all samples for action/state and compute exact q01/q99."""
	missing_keys = [k for k in TARGET_KEYS if k not in dataset.features]
	if missing_keys:
		raise KeyError(f"Dataset missing required keys: {missing_keys}")

	logging.info(
		"Loading full dataset with DataLoader for keys: %s (batch_size=%d, num_workers=%d, prefetch_factor=%d)",
		TARGET_KEYS,
		batch_size,
		num_workers,
		prefetch_factor,
	)
	buffers: dict[str, list[np.ndarray]] = {k: [] for k in TARGET_KEYS}
	total_samples = len(dataset.hf_dataset)
	total_batches = math.ceil(total_samples / batch_size)

	source_dataset = _ActionStateOnlyDataset(dataset)
	loader_kwargs = {
		"dataset": source_dataset,
		"batch_size": batch_size,
		"shuffle": False,
		"num_workers": num_workers,
		"collate_fn": _collate_action_state,
		"pin_memory": False,
	}
	if num_workers > 0:
		loader_kwargs["persistent_workers"] = True
		loader_kwargs["prefetch_factor"] = prefetch_factor

	loader = DataLoader(**loader_kwargs)

	loaded_samples = 0
	for batch in tqdm(loader, total=total_batches, desc="Loading action/state", unit="batch"):
		for key in TARGET_KEYS:
			value = batch[key]
			if isinstance(value, torch.Tensor):
				arr = value.detach().cpu().numpy()
			else:
				arr = np.asarray(value)

			if arr.ndim == 1:
				arr = arr.reshape(-1, 1)
			else:
				arr = arr.reshape(arr.shape[0], -1)
			buffers[key].append(arr)

		loaded_samples += int(batch[ACTION_KEY].shape[0])
		if loaded_samples % 10_000 == 0:
			logging.info("Loaded %d/%d samples", loaded_samples, total_samples)

	result: dict[str, dict[str, np.ndarray]] = {}
	for key in TARGET_KEYS:
		full_array = np.concatenate(buffers[key], axis=0)
		logging.info("Computing exact quantiles for %s with shape %s", key, tuple(full_array.shape))
		result[key] = _compute_exact_stats_from_full_array(full_array)

	return result


def main() -> None:
	parser = argparse.ArgumentParser(description="Brute-force exact q01/q99 for action/state")
	parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo id")
	parser.add_argument("--root", type=str, default=None, help="Local dataset root")
	parser.add_argument("--revision", type=str, default=None, help="Dataset revision")
	parser.add_argument("--batch-size", type=int, default=2048, help="DataLoader batch size")
	parser.add_argument(
		"--num-workers",
		type=int,
		default=max(1, min(8, (os.cpu_count() or 4) // 2)),
		help="DataLoader worker processes",
	)
	parser.add_argument(
		"--prefetch-factor",
		type=int,
		default=4,
		help="Batches prefetched per worker (only when num_workers > 0)",
	)
	args = parser.parse_args()

	init_logging()

	root = Path(args.root) if args.root else None
	dataset = LeRobotDataset(repo_id=args.repo_id, root=root, revision=args.revision)

	exact_stats = compute_quantiles_bruteforce(
		dataset=dataset,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		prefetch_factor=args.prefetch_factor,
	)

	if dataset.meta.stats is None:
		dataset.meta.stats = {}

	for key, stats in exact_stats.items():
		if key not in dataset.meta.stats:
			dataset.meta.stats[key] = {}
		dataset.meta.stats[key].update(stats)

	write_stats(dataset.meta.stats, dataset.meta.root)
	logging.info("Done. Updated exact q01/q99 (and basic stats) for %s", TARGET_KEYS)


if __name__ == "__main__":
	main()
