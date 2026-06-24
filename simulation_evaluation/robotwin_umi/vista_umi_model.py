from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.transforms import AbsoluteActionTransform
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


class LerobotVistaUMI:
    """VISTA inference wrapper for Robotwin UMI/endpose evaluation."""

    def __init__(
        self,
        task_name: str,
        pretrained_checkpoint_path: str,
        image_channel_order: str = "rgb",
        debug_image_dir: str | None = None,
        save_debug_images: bool = True,
    ):
        self.task_name = task_name
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.image_channel_order = str(image_channel_order).lower()
        if self.image_channel_order not in {"rgb", "bgr"}:
            raise ValueError(f"image_channel_order must be rgb or bgr, got {image_channel_order!r}")

        print(f"[vista-umi] loading config: {pretrained_checkpoint_path}", flush=True)
        policy_config = PreTrainedConfig.from_pretrained(
            pretrained_name_or_path=pretrained_checkpoint_path,
            local_files_only=True,
        )
        policy_cls = get_policy_class(policy_config.type)
        print(f"[vista-umi] loading policy class={policy_cls}", flush=True)
        model = policy_cls.from_pretrained(
            pretrained_name_or_path=pretrained_checkpoint_path,
            local_files_only=True,
        )
        model.eval()

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=pretrained_checkpoint_path,
        )
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.policy = model

        self.img_size = (224, 224)
        self.observation_window = None
        self.instruction = None
        self.pi0_step = int(getattr(model.config, "n_action_steps", 50) or 50)
        self.enable_fisheye = True
        self.fisheye_strength = 1.8
        self._fisheye_maps = {}
        self.test_num = 1
        self.debug_image_dir = Path(debug_image_dir) if debug_image_dir else None
        self.save_debug_images = save_debug_images
        self._debug_image_step = 0

        self.action_dim = self._feature_dim(getattr(model.config, "output_features", {}), "action", 16)
        self.state_dim = self._feature_dim(getattr(model.config, "input_features", {}), "observation.state", 16)
        self.image_feature_keys = [
            key
            for key, ft in getattr(model.config, "input_features", {}).items()
            if key.startswith("observation.images.")
        ]
        print(
            "[vista-umi] ready "
            f"task={task_name} state_dim={self.state_dim} action_dim={self.action_dim} "
            f"image_features={self.image_feature_keys} step={self.pi0_step} image_order={self.image_channel_order}",
            flush=True,
        )

        if getattr(model.config, "use_delta_action", False):
            device = getattr(model.config, "device", "cuda")
            self.action_mask = torch.ones(self.action_dim, dtype=torch.bool, device=device)
            if self.action_dim > 7:
                self.action_mask[7] = False
            if self.action_dim > 15:
                self.action_mask[15] = False
            self.delta2abs = AbsoluteActionTransform(action_mask=self.action_mask)

    @staticmethod
    def _feature_dim(features, key: str, default: int) -> int:
        try:
            feat = features.get(key)
        except AttributeError:
            feat = None
        shape = getattr(feat, "shape", None)
        if shape is None and isinstance(feat, dict):
            shape = feat.get("shape")
        if shape:
            return int(shape[0])
        return int(default)

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        self.test_num += 1
        print("[vista-umi] reset observation window", flush=True)

    def set_language(self, instruction: str):
        self.instruction = instruction
        print(f"[vista-umi] instruction: {instruction}", flush=True)

    def build_fisheye_maps(self, width: int, height: int, strength: float = 1.8):
        center_x = width / 2.0
        center_y = height / 2.0
        radius = min(center_x, center_y)
        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        norm_x = (grid_x - center_x) / radius
        norm_y = (grid_y - center_y) / radius
        radial = np.sqrt(norm_x * norm_x + norm_y * norm_y)
        source_radius = np.tan(radial * np.arctan(strength)) / strength
        scale = np.zeros_like(source_radius)
        valid = radial > 1e-8
        scale[valid] = source_radius[valid] / radial[valid]
        map_x = center_x + norm_x * radius * scale
        map_y = center_y + norm_y * radius * scale
        outside = radial > 1.0
        map_x[outside] = -1.0
        map_y[outside] = -1.0
        return map_x.astype(np.float32), map_y.astype(np.float32)

    def _to_hwc_uint8(self, image):
        img = np.asarray(image)
        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.ndim != 3 or img.shape[-1] < 3:
            raise ValueError(f"Expected HWC image with 3 channels, got shape={img.shape}")
        img = img[..., :3]
        if self.image_channel_order == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating):
                scale = 255.0 if np.nanmax(img) <= 1.0 else 1.0
                img = np.clip(img * scale, 0, 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def apply_fisheye(self, image, camera_name: str):
        hwc = self._to_hwc_uint8(image)
        height, width = hwc.shape[:2]
        map_key = (camera_name, width, height, float(self.fisheye_strength))
        if map_key not in self._fisheye_maps:
            self._fisheye_maps[map_key] = self.build_fisheye_maps(width, height, self.fisheye_strength)
        map_x, map_y = self._fisheye_maps[map_key]
        fisheye = cv2.remap(
            hwc,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return cv2.resize(fisheye, self.img_size, interpolation=cv2.INTER_AREA)

    def _save_debug(self, name: str, image):
        if not self.save_debug_images or self.debug_image_dir is None or self._debug_image_step >= 8:
            return
        self.debug_image_dir.mkdir(parents=True, exist_ok=True)
        img = np.asarray(image)
        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            scale = 255.0 if np.nanmax(img) <= 1.0 else 1.0
            img = np.clip(img * scale, 0, 255).astype(np.uint8)
        Image.fromarray(img[..., :3]).save(self.debug_image_dir / f"{self._debug_image_step:04d}_{name}.png")
        self._debug_image_step += 1

    def _chw_tensor(self, image, dtype):
        img = np.asarray(image, dtype=np.float32) / 255.0
        img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
        return torch.from_numpy(img).unsqueeze(0).to(dtype)

    def update_observation_window(self, img_arr, state):
        _, img_right, img_left = img_arr[0], img_arr[1], img_arr[2]
        if self.enable_fisheye:
            img_left = self.apply_fisheye(img_left, "left_wrist")
            img_right = self.apply_fisheye(img_right, "right_wrist")
        else:
            img_left = cv2.resize(self._to_hwc_uint8(img_left), self.img_size, interpolation=cv2.INTER_AREA)
            img_right = cv2.resize(self._to_hwc_uint8(img_right), self.img_size, interpolation=cv2.INTER_AREA)

        self._save_debug("left_wrist_eval_rgb.png", img_left)
        self._save_debug("right_wrist_eval_rgb.png", img_right)

        para = next(self.policy.parameters())
        dtype = para.dtype
        state = np.asarray(state, dtype=np.float32)
        self.observation_window = {
            "observation.state": torch.from_numpy(state).unsqueeze(0).to(dtype),
            "observation.images.left_wrist": self._chw_tensor(img_left, dtype),
            "observation.images.right_wrist": self._chw_tensor(img_right, dtype),
            "task": [self.instruction or "Execute the robot action."],
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first"
        with torch.no_grad():
            inputs = {
                key: value.to(self.policy.config.device) if isinstance(value, torch.Tensor) else value
                for key, value in self.observation_window.items()
            }
            state_raw = inputs["observation.state"].clone()
            inputs = self.preprocessor(inputs)
            actions = self.policy.predict_action_chunk(inputs)
            actions_un = self.postprocessor(actions)

            if getattr(self.policy.config, "use_delta_action", False):
                state_raw = state_raw.to(actions_un.device)
                if hasattr(self, "action_mask"):
                    self.action_mask = self.action_mask.to(actions_un.device)
                if hasattr(self, "delta2abs") and hasattr(self.delta2abs, "action_mask"):
                    self.delta2abs.action_mask = self.delta2abs.action_mask.to(actions_un.device)
                actions_un = self.delta2abs(
                    {
                        "observation.state": state_raw,
                        "action": actions_un,
                    }
                )["action"]

        return actions_un[0].to(torch.float32).cpu().numpy()
