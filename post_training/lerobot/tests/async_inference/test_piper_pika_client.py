from __future__ import annotations

import logging
import threading
import time

import numpy as np
import pytest
import torch

from lerobot.async_inference.helpers import TimedAction
from lerobot.async_inference.piper_pika_client import (
    PiperArm,
    PiperPikaClient,
    PiperPikaClientConfig,
    center_crop_resize_rgb,
    ee_rpy_to_tcp_quat_state,
    limit_tcp_rpy_step,
    tcp_quat_action_to_ee_rpy,
)


class FakePiperStatus:
    def __init__(self, status: str):
        self.status = status

    def GetArmStatus(self):
        return self.status


def test_target_limit_status_warns_and_continues(caplog):
    arm = object.__new__(PiperArm)
    arm.side = "right"
    arm.robot = FakePiperStatus("TARGET_POS_EXCEEDS_LIMIT")

    with caplog.at_level(logging.WARNING):
        arm.raise_for_status()

    assert "TARGET_POS_EXCEEDS_LIMIT; continuing" in caplog.text


def test_other_fatal_status_still_raises_with_target_limit(caplog):
    arm = object.__new__(PiperArm)
    arm.side = "right"
    arm.robot = FakePiperStatus("TARGET_POS_EXCEEDS_LIMIT EMERGENCY_STOP")

    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="EMERGENCY_STOP"):
        arm.raise_for_status()

    assert "TARGET_POS_EXCEEDS_LIMIT; continuing" in caplog.text


def test_step_limits_are_disabled_by_default():
    cfg = PiperPikaClientConfig(pretrained_name_or_path="test")
    current = np.zeros(7, dtype=np.float32)
    target = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 100.0], dtype=np.float32)

    limited = limit_tcp_rpy_step(current, target, cfg)

    np.testing.assert_allclose(limited[:6], target[:6])
    assert limited[6] == pytest.approx(90.0)


def test_configured_step_limits_are_enforced():
    cfg = PiperPikaClientConfig(
        pretrained_name_or_path="test",
        max_pos_step=0.01,
        max_rot_step=0.05,
        max_gripper_step_mm=5.0,
    )
    current = np.zeros(7, dtype=np.float32)
    target = np.ones(7, dtype=np.float32) * 100

    limited = limit_tcp_rpy_step(current, target, cfg)

    np.testing.assert_allclose(limited[:3], 0.01)
    np.testing.assert_allclose(limited[3:6], 0.05)
    assert limited[6] == pytest.approx(5.0)


def test_negative_step_limit_is_rejected():
    with pytest.raises(ValueError, match="step limits must be non-negative"):
        PiperPikaClientConfig(pretrained_name_or_path="test", max_pos_step=-0.01)


def test_ee_tcp_quaternion_roundtrip():
    ee_pose = np.array([0.2, -0.1, 0.3, 0.1, -0.2, 0.3, 42.0], dtype=np.float32)

    tcp_state = ee_rpy_to_tcp_quat_state(ee_pose)
    recovered = tcp_quat_action_to_ee_rpy(tcp_state)

    np.testing.assert_allclose(recovered[:3], ee_pose[:3], atol=1e-6)
    np.testing.assert_allclose(recovered[3:6], ee_pose[3:6], atol=1e-5)
    assert recovered[6] == np.float32(42.0)


def test_center_crop_resize_rgb_uses_center_square():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, 0] = [255, 0, 0]
    image[:, -1] = [0, 255, 0]
    image[:, 1:5] = [10, 20, 30]

    cropped = center_crop_resize_rgb(image, size=4)

    assert cropped.shape == (4, 4, 3)
    assert np.all(cropped == np.array([10, 20, 30], dtype=np.uint8))


def test_piper_pika_queue_filters_executed_actions():
    client = object.__new__(PiperPikaClient)
    client.latest_executed_timestep = 4
    client.action_queue = []
    client.action_queue_lock = threading.Lock()
    client.action_chunk_size = -1

    actions = [
        TimedAction(timestamp=time.time(), timestep=t, action=torch.zeros(16))
        for t in range(3, 8)
    ]

    client._replace_queue_with_future_actions(actions)

    assert [action.get_timestep() for action in client.action_queue] == [5, 6, 7]
    assert client.action_chunk_size == 5


def test_piper_pika_sends_previous_absolute_state(monkeypatch):
    import pickle
    import lerobot.async_inference.piper_pika_client as client_module

    class Hardware:
        def __init__(self):
            self.offset = 0.0

        def read_observation(self, task):
            obs = {name: float(i) + self.offset for i, name in enumerate(client_module.STATE_NAMES)}
            obs["task"] = task
            self.offset += 10.0
            return obs

    class Stub:
        def __init__(self):
            self.observations = []

        def SendObservations(self, payload):
            self.observations.append(pickle.loads(payload))

    client = object.__new__(PiperPikaClient)
    client.hardware = Hardware()
    client.cfg = type("Cfg", (), {"task": "test"})()
    client.stub = Stub()
    client.latest_executed_timestep = -1
    client.previous_state = None
    monkeypatch.setattr(client_module, "send_bytes_in_chunks", lambda payload, *args, **kwargs: payload)

    client.send_observation()
    client.send_observation()

    assert client.stub.observations[0].previous_state is None
    torch.testing.assert_close(
        client.stub.observations[1].previous_state,
        torch.arange(16, dtype=torch.float32),
    )
