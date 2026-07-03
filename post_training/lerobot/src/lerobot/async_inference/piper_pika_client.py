# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Dual-arm Piper + Pika async-inference client for Vista checkpoints."""

from __future__ import annotations

import json
import logging
import pickle  # nosec
import threading
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Any

import draccus
import grpc
import numpy as np
import torch
from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from .constants import DEFAULT_FPS
from .helpers import RemotePolicyConfig, TimedAction, TimedObservation, get_logger

ARM_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper_width")
STATE_NAMES = tuple(f"robot_{robot_index}_{name}" for robot_index in (0, 1) for name in ARM_NAMES)
LEFT_SLICE = slice(0, 8)
RIGHT_SLICE = slice(8, 16)
_PIPER_FATAL_STATUS_TERMS = (
    "NO_SOLUTION",
    "SINGULARITY_POINT",
    "TARGET_POS_EXCEEDS_LIMIT",
    "EMERGENCY_STOP",
    "JOINT_COMMUNICATION_ERR",
    "JOINT_BRAKE_NOT_RELEASED",
    "COLLISION_OCCURRED",
    "JOINT_STATUS_ERR",
    "OTHER_ERR",
    "MAIN_CONTROLLER_NTC_OVER_TEMPERATURE",
    "RELEASE_RESISTOR_NTC_OVER_TEMPERATURE",
)

R_EE_TCP = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
T_EE_TCP = np.eye(4, dtype=np.float32)
T_EE_TCP[:3, :3] = R_EE_TCP
T_EE_TCP[:3, 3] = np.array([0.0, 0.0, 0.1943], dtype=np.float32)
T_TCP_EE = np.linalg.inv(T_EE_TCP).astype(np.float32)


def euler_xyz_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float32,
    )


def matrix_to_euler_xyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    sy = float(np.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0]))
    if sy >= 1e-6:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        pitch = np.arctan2(-matrix[2, 0], sy)
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = np.arctan2(-matrix[1, 2], matrix[1, 1])
        pitch = np.arctan2(-matrix[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float32)


def normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), eps)


def matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat_wxyz = np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        )
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            quat_wxyz = np.array(
                [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
            )
        elif idx == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            quat_wxyz = np.array(
                [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
            )
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            quat_wxyz = np.array(
                [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
            )
    quat_wxyz = normalize(quat_wxyz)
    return quat_wxyz[[1, 2, 3, 0]].astype(np.float32)


def quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize(np.asarray(quat_xyzw, dtype=np.float32))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def pose7_rpy_to_mat(pose7: np.ndarray) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float32)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = euler_xyz_to_matrix(pose7[3:6])
    mat[:3, 3] = pose7[:3]
    return mat


def mat_to_pose7_rpy(mat: np.ndarray, gripper_mm: float) -> np.ndarray:
    pose = np.empty((7,), dtype=np.float32)
    pose[:3] = mat[:3, 3]
    pose[3:6] = matrix_to_euler_xyz(mat[:3, :3])
    pose[6] = np.float32(gripper_mm)
    return pose


def ee_rpy_to_tcp_quat_state(ee_pose7: np.ndarray) -> np.ndarray:
    base_to_tcp = pose7_rpy_to_mat(ee_pose7) @ T_EE_TCP
    return np.concatenate(
        (base_to_tcp[:3, 3], matrix_to_quat_xyzw(base_to_tcp[:3, :3]), [ee_pose7[6]])
    ).astype(np.float32)


def tcp_quat_action_to_ee_rpy(tcp_action8: np.ndarray) -> np.ndarray:
    tcp_action8 = np.asarray(tcp_action8, dtype=np.float32)
    base_to_tcp = np.eye(4, dtype=np.float32)
    base_to_tcp[:3, 3] = tcp_action8[:3]
    base_to_tcp[:3, :3] = quat_xyzw_to_matrix(tcp_action8[3:7])
    base_to_ee = base_to_tcp @ T_TCP_EE
    return mat_to_pose7_rpy(base_to_ee, float(tcp_action8[7]))


def ee_rpy_to_tcp_rpy(ee_pose7: np.ndarray) -> np.ndarray:
    base_to_tcp = pose7_rpy_to_mat(ee_pose7) @ T_EE_TCP
    return mat_to_pose7_rpy(base_to_tcp, float(ee_pose7[6]))


def tcp_rpy_to_ee_rpy(tcp_pose7: np.ndarray) -> np.ndarray:
    base_to_ee = pose7_rpy_to_mat(tcp_pose7) @ T_TCP_EE
    return mat_to_pose7_rpy(base_to_ee, float(tcp_pose7[6]))


def center_crop_resize_rgb(image: np.ndarray, size: int = 224) -> np.ndarray:
    import cv2

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image, got shape {image.shape}")
    h, w = image.shape[:2]
    crop_size = min(h, w)
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    image = image[top : top + crop_size, left : left + crop_size]
    if image.shape[0] != size or image.shape[1] != size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image.astype(np.uint8, copy=False))


def limit_tcp_rpy_step(current: np.ndarray, target: np.ndarray, cfg: "PiperPikaClientConfig") -> np.ndarray:
    target = np.asarray(target, dtype=np.float32).copy()
    current = np.asarray(current, dtype=np.float32)
    max_pos_step = np.inf if cfg.max_pos_step is None else cfg.max_pos_step
    max_rot_step = np.inf if cfg.max_rot_step is None else cfg.max_rot_step
    max_gripper_step_mm = np.inf if cfg.max_gripper_step_mm is None else cfg.max_gripper_step_mm
    target[:3] = current[:3] + np.clip(target[:3] - current[:3], -max_pos_step, max_pos_step)
    target[3:6] = current[3:6] + np.clip(target[3:6] - current[3:6], -max_rot_step, max_rot_step)
    target[6] = current[6] + float(
        np.clip(target[6] - current[6], -max_gripper_step_mm, max_gripper_step_mm)
    )
    target[6] = float(np.clip(target[6], cfg.min_gripper_mm, cfg.max_gripper_mm))
    return target


def make_vista_lerobot_features() -> dict[str, dict[str, Any]]:
    image_feature = {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channels"]}
    state_feature = {"dtype": "float32", "shape": (16,), "names": list(STATE_NAMES)}
    return {
        OBS_STATE: state_feature,
        f"{OBS_IMAGES}.robot_0": image_feature,
        f"{OBS_IMAGES}.robot_1": image_feature.copy(),
    }


@dataclass
class PiperPikaClientConfig:
    policy_type: str = "vista"
    pretrained_name_or_path: str = ""
    actions_per_chunk: int = 50
    task: str = ""
    server_address: str = "localhost:8080"
    policy_device: str = "cpu"
    chunk_size_threshold: float = 0.5
    fps: int = DEFAULT_FPS
    left_piper_can: str = "can_left"
    right_piper_can: str = "can_right"
    left_pika_port: str = "/dev/ttyUSB0"
    right_pika_port: str = "/dev/ttyUSB1"
    left_fisheye_device: int | str = 0
    right_fisheye_device: int | str = 1
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = DEFAULT_FPS
    image_size: int = 224
    max_pos_step: float | None = None
    max_rot_step: float | None = None
    max_gripper_step_mm: float | None = None
    min_gripper_mm: float = 0.0
    max_gripper_mm: float = 90.0

    @property
    def environment_dt(self) -> float:
        return 1 / self.fps

    def __post_init__(self) -> None:
        if self.policy_type != "vista":
            raise ValueError("Piper/Pika async client only supports policy_type='vista'")
        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")
        if self.actions_per_chunk <= 0:
            raise ValueError("actions_per_chunk must be positive")
        if not 0 <= self.chunk_size_threshold <= 1:
            raise ValueError("chunk_size_threshold must be between 0 and 1")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        step_limits = (self.max_pos_step, self.max_rot_step, self.max_gripper_step_mm)
        if any(limit is not None and limit < 0 for limit in step_limits):
            raise ValueError("step limits must be non-negative")


class PiperArm:
    def __init__(self, side: str, can_name: str):
        self.side = side
        self.can_name = can_name
        from piper_sdk import C_PiperInterface

        self.robot = C_PiperInterface(can_name=can_name)
        self.robot.ConnectPort()
        while not self.robot.EnablePiper():
            time.sleep(0.01)
        self.robot.MotionCtrl_2(0x01, 0x00, 100, 0x00)

    def read_ee_pose6(self) -> np.ndarray:
        pose = self.robot.GetArmEndPoseMsgs().end_pose
        xyz = np.array([pose.X_axis, pose.Y_axis, pose.Z_axis], dtype=np.float32) / 1_000_000.0
        rpy = np.deg2rad(np.array([pose.RX_axis, pose.RY_axis, pose.RZ_axis], dtype=np.float32) / 1000.0)
        return np.concatenate([xyz, rpy.astype(np.float32)])

    def command_ee_pose6(self, ee_pose6: np.ndarray) -> None:
        x, y, z, roll, pitch, yaw = np.asarray(ee_pose6, dtype=np.float32).tolist()
        self.robot.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.robot.EndPoseCtrl(
            int(round(x * 1_000_000.0)),
            int(round(y * 1_000_000.0)),
            int(round(z * 1_000_000.0)),
            int(round(np.degrees(roll) * 1000.0)),
            int(round(np.degrees(pitch) * 1000.0)),
            int(round(np.degrees(yaw) * 1000.0)),
        )
        self.raise_for_status()

    def raise_for_status(self) -> None:
        snapshots = []
        for name in ("GetArmStatus", "GetArmStatusMsgs"):
            method = getattr(self.robot, name, None)
            if method is not None:
                snapshots.append(str(method()))
        text = json.dumps(snapshots)
        errors = [term for term in _PIPER_FATAL_STATUS_TERMS if term in text]
        if "TARGET_POS_EXCEEDS_LIMIT" in errors:
            logging.warning(
                "Piper %s reported TARGET_POS_EXCEEDS_LIMIT; continuing", self.side
            )
            errors = [error for error in errors if error != "TARGET_POS_EXCEEDS_LIMIT"]
        if errors:
            raise RuntimeError(f"Piper {self.side} reported fatal status: {errors}")


class PikaGripper:
    def __init__(self, side: str, port: str, fisheye_device: int | str, width: int, height: int, fps: int):
        self.side = side
        from pika.gripper import Gripper

        self.device = Gripper(port)
        if not self.device.connect():
            raise RuntimeError(f"failed to connect Pika {side} gripper: {port}")
        self.device.enable()
        self.device.set_camera_param(width, height, fps)
        self.device.set_fisheye_camera_index(fisheye_device)
        self.fisheye = self.device.get_fisheye_camera()
        if not getattr(self.fisheye, "is_connected", False):
            raise RuntimeError(f"failed to connect Pika {side} fisheye camera: {fisheye_device}")

    def close(self) -> None:
        self.device.disconnect()

    def read_gripper_mm(self) -> float:
        return max(float(self.device.get_gripper_distance()), 0.0)

    def command_gripper_mm(self, width_mm: float) -> None:
        self.device.set_gripper_distance(float(width_mm))

    def read_fisheye_rgb(self, width: int, height: int) -> np.ndarray:
        import cv2

        ok, frame = self.fisheye.get_frame()
        if not ok or frame is None:
            raise RuntimeError(f"failed to read Pika {self.side} fisheye frame")
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class PiperPikaHardware:
    def __init__(self, cfg: PiperPikaClientConfig):
        self.cfg = cfg
        self.left_arm = PiperArm("left", cfg.left_piper_can)
        self.right_arm = PiperArm("right", cfg.right_piper_can)
        self.left_gripper = PikaGripper(
            "left",
            cfg.left_pika_port,
            cfg.left_fisheye_device,
            cfg.camera_width,
            cfg.camera_height,
            cfg.camera_fps,
        )
        self.right_gripper = PikaGripper(
            "right",
            cfg.right_pika_port,
            cfg.right_fisheye_device,
            cfg.camera_width,
            cfg.camera_height,
            cfg.camera_fps,
        )

    def close(self) -> None:
        self.left_gripper.close()
        self.right_gripper.close()

    def read_state(self) -> np.ndarray:
        left_ee = np.concatenate(
            [self.left_arm.read_ee_pose6(), [self.left_gripper.read_gripper_mm()]]
        ).astype(np.float32)
        right_ee = np.concatenate(
            [self.right_arm.read_ee_pose6(), [self.right_gripper.read_gripper_mm()]]
        ).astype(np.float32)
        return np.concatenate(
            [ee_rpy_to_tcp_quat_state(left_ee), ee_rpy_to_tcp_quat_state(right_ee)]
        ).astype(np.float32)

    def read_observation(self, task: str) -> dict[str, Any]:
        left_image = center_crop_resize_rgb(
            self.left_gripper.read_fisheye_rgb(self.cfg.camera_width, self.cfg.camera_height),
            self.cfg.image_size,
        )
        right_image = center_crop_resize_rgb(
            self.right_gripper.read_fisheye_rgb(self.cfg.camera_width, self.cfg.camera_height),
            self.cfg.image_size,
        )
        state = self.read_state()
        obs = {name: float(state[i]) for i, name in enumerate(STATE_NAMES)}
        obs["robot_0"] = left_image
        obs["robot_1"] = right_image
        obs["task"] = task
        return obs

    def execute_action(self, action16: np.ndarray, previous_tcp_rpy16: np.ndarray | None) -> np.ndarray:
        action16 = np.asarray(action16, dtype=np.float32)
        if action16.shape != (16,):
            raise ValueError(f"expected 16D Vista action, got {action16.shape}")

        left_ee_target = tcp_quat_action_to_ee_rpy(action16[LEFT_SLICE])
        right_ee_target = tcp_quat_action_to_ee_rpy(action16[RIGHT_SLICE])
        target_tcp_rpy = np.concatenate(
            [ee_rpy_to_tcp_rpy(left_ee_target), ee_rpy_to_tcp_rpy(right_ee_target)]
        )
        if previous_tcp_rpy16 is not None:
            left = limit_tcp_rpy_step(previous_tcp_rpy16[:7], target_tcp_rpy[:7], self.cfg)
            right = limit_tcp_rpy_step(previous_tcp_rpy16[7:], target_tcp_rpy[7:], self.cfg)
            target_tcp_rpy = np.concatenate([left, right]).astype(np.float32)
            left_ee_target = tcp_rpy_to_ee_rpy(target_tcp_rpy[:7])
            right_ee_target = tcp_rpy_to_ee_rpy(target_tcp_rpy[7:])

        self.left_arm.command_ee_pose6(left_ee_target[:6])
        self.right_arm.command_ee_pose6(right_ee_target[:6])
        self.left_gripper.command_gripper_mm(left_ee_target[6])
        self.right_gripper.command_gripper_mm(right_ee_target[6])
        return target_tcp_rpy


class PiperPikaClient:
    logger = get_logger("piper_pika_client")

    def __init__(self, cfg: PiperPikaClientConfig):
        self.cfg = cfg
        self.hardware = PiperPikaHardware(cfg)
        self.policy_config = RemotePolicyConfig(
            cfg.policy_type,
            cfg.pretrained_name_or_path,
            make_vista_lerobot_features(),
            cfg.actions_per_chunk,
            cfg.policy_device,
        )
        self.channel = grpc.insecure_channel(
            cfg.server_address, grpc_channel_options(initial_backoff=f"{cfg.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.shutdown_event = threading.Event()
        self.action_queue: list[TimedAction] = []
        self.action_queue_lock = threading.Lock()
        self.latest_executed_timestep = -1
        self.action_chunk_size = cfg.actions_per_chunk
        self.last_tcp_rpy_target: np.ndarray | None = None
        self.previous_state: torch.Tensor | None = None

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    def start(self) -> None:
        self.previous_state = None
        self.stub.Ready(services_pb2.Empty())
        self.stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(self.policy_config)))
        self.shutdown_event.clear()

    def stop(self) -> None:
        self.shutdown_event.set()
        self.hardware.close()
        self.channel.close()

    def _queue_ready_for_observation(self) -> bool:
        with self.action_queue_lock:
            return len(self.action_queue) / max(self.action_chunk_size, 1) <= self.cfg.chunk_size_threshold

    def _replace_queue_with_future_actions(self, actions: list[TimedAction]) -> None:
        future = [a for a in actions if a.get_timestep() > self.latest_executed_timestep]
        with self.action_queue_lock:
            self.action_queue = future
            if future:
                self.action_chunk_size = max(self.action_chunk_size, len(actions))

    def send_observation(self) -> None:
        raw_observation = self.hardware.read_observation(self.cfg.task)
        current_state = torch.tensor(
            [raw_observation[name] for name in STATE_NAMES],
            dtype=torch.float32,
        )
        obs = TimedObservation(
            timestamp=time.time(),
            timestep=max(self.latest_executed_timestep, 0),
            observation=raw_observation,
            must_go=True,
            previous_state=self.previous_state,
        )
        payload = pickle.dumps(obs)
        iterator = send_bytes_in_chunks(
            payload, services_pb2.Observation, log_prefix="[PIPER_PIKA] Observation", silent=True
        )
        self.stub.SendObservations(iterator)
        self.previous_state = current_state

    def receive_actions_loop(self) -> None:
        while self.running:
            try:
                chunk = self.stub.GetActions(services_pb2.Empty())
                if not chunk.data:
                    continue
                actions = pickle.loads(chunk.data)  # nosec
                for action in actions:
                    if tuple(action.get_action().shape) != (16,):
                        raise ValueError(f"Vista action must be 16D, got {action.get_action().shape}")
                self._replace_queue_with_future_actions(actions)
            except grpc.RpcError as exc:
                self.logger.error(f"Error receiving actions: {exc}")

    def control_loop(self) -> None:
        while self.running:
            started = time.perf_counter()
            if self._queue_ready_for_observation():
                self.send_observation()

            action = None
            with self.action_queue_lock:
                if self.action_queue:
                    action = self.action_queue.pop(0)

            if action is not None and action.get_timestep() > self.latest_executed_timestep:
                self.last_tcp_rpy_target = self.hardware.execute_action(
                    action.get_action().detach().cpu().numpy(), self.last_tcp_rpy_target
                )
                self.latest_executed_timestep = action.get_timestep()

            time.sleep(max(0.0, self.cfg.environment_dt - (time.perf_counter() - started)))


def _run_client(cfg: PiperPikaClientConfig) -> None:
    logging.info(pformat(asdict(cfg)))
    client = PiperPikaClient(cfg)
    receiver = threading.Thread(target=client.receive_actions_loop, name="piper-pika-actions", daemon=True)
    try:
        client.start()
        receiver.start()
        client.control_loop()
    finally:
        client.stop()
        receiver.join(timeout=2.0)


@draccus.wrap()
def async_client(cfg: PiperPikaClientConfig) -> None:
    _run_client(cfg)


if __name__ == "__main__":
    async_client()
