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
"""Unit-tests for the `PolicyServer` core logic.
Monkey-patch the `policy` attribute with a stub so that no real model inference is performed.
"""

from __future__ import annotations

import time

import pytest
import torch

from lerobot.configs.types import PolicyFeature
from lerobot.utils.constants import OBS_STATE
from tests.utils import require_package

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


class MockPolicy:
    """A minimal mock for an actual policy, returning zeros.
    Refer to tests/policies for tests of the individual policies supported."""

    class _Config:
        robot_type = "dummy_robot"

        @property
        def image_features(self) -> dict[str, PolicyFeature]:
            """Empty image features since this test doesn't use images."""
            return {}

    def predict_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a chunk of 20 dummy actions."""
        batch_size = len(observation[OBS_STATE])
        return torch.zeros(batch_size, 20, 6)

    def __init__(self):
        self.config = self._Config()

    def to(self, *args, **kwargs):
        # The server calls `policy.to(device)`. This stub ignores it.
        return self

    def model(self, batch: dict) -> torch.Tensor:
        # Return a chunk of 20 dummy actions.
        batch_size = len(batch["robot_type"])
        return torch.zeros(batch_size, 20, 6)


@pytest.fixture
@require_package("grpc")
def policy_server():
    """Fresh `PolicyServer` instance with a stubbed-out policy model."""
    # Import only when the test actually runs (after decorator check)
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import PolicyServer

    test_config = PolicyServerConfig(host="localhost", port=9999)
    server = PolicyServer(test_config)
    # Replace the real policy with our fast, deterministic stub.
    server.policy = MockPolicy()
    server.actions_per_chunk = 20
    server.device = "cpu"

    # Add mock lerobot_features that the observation similarity functions need
    server.lerobot_features = {
        OBS_STATE: {
            "dtype": "float32",
            "shape": [6],
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        }
    }

    return server


# -----------------------------------------------------------------------------
# Helper utilities for tests
# -----------------------------------------------------------------------------


def _make_obs(state: torch.Tensor, timestep: int = 0, must_go: bool = False):
    """Create a TimedObservation with a given state vector."""
    # Import only when needed
    from lerobot.async_inference.helpers import TimedObservation

    return TimedObservation(
        observation={
            "joint1": state[0].item() if len(state) > 0 else 0.0,
            "joint2": state[1].item() if len(state) > 1 else 0.0,
            "joint3": state[2].item() if len(state) > 2 else 0.0,
            "joint4": state[3].item() if len(state) > 3 else 0.0,
            "joint5": state[4].item() if len(state) > 4 else 0.0,
            "joint6": state[5].item() if len(state) > 5 else 0.0,
        },
        timestamp=time.time(),
        timestep=timestep,
        must_go=must_go,
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_time_action_chunk(policy_server):
    """Verify that `_time_action_chunk` assigns correct timestamps and timesteps."""
    start_ts = time.time()
    start_t = 10
    # A chunk of 3 action tensors.
    action_tensors = [torch.randn(6) for _ in range(3)]

    timed_actions = policy_server._time_action_chunk(start_ts, action_tensors, start_t)

    assert len(timed_actions) == 3
    # Check timesteps
    assert [ta.get_timestep() for ta in timed_actions] == [10, 11, 12]
    # Check timestamps
    expected_timestamps = [
        start_ts,
        start_ts + policy_server.config.environment_dt,
        start_ts + 2 * policy_server.config.environment_dt,
    ]
    for ta, expected_ts in zip(timed_actions, expected_timestamps, strict=True):
        assert abs(ta.get_timestamp() - expected_ts) < 1e-6


def test_maybe_enqueue_observation_must_go(policy_server):
    """An observation with `must_go=True` is always enqueued."""
    obs = _make_obs(torch.zeros(6), must_go=True)
    assert policy_server._enqueue_observation(obs) is True
    assert policy_server.observation_queue.qsize() == 1
    assert policy_server.observation_queue.get_nowait() is obs


def test_maybe_enqueue_observation_dissimilar(policy_server):
    """A dissimilar observation (not `must_go`) is enqueued."""
    # Set a last predicted observation.
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    # Create a new, dissimilar observation.
    new_obs = _make_obs(torch.ones(6) * 5)  # High norm difference

    assert policy_server._enqueue_observation(new_obs) is True
    assert policy_server.observation_queue.qsize() == 1


def test_maybe_enqueue_observation_is_skipped(policy_server):
    """A similar observation (not `must_go`) is skipped."""
    # Set a last predicted observation.
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    # Create a new, very similar observation.
    new_obs = _make_obs(torch.zeros(6) + 1e-4)

    assert policy_server._enqueue_observation(new_obs) is False
    assert policy_server.observation_queue.empty() is True


def test_obs_sanity_checks(policy_server):
    """Unit-test the private `_obs_sanity_checks` helper."""
    prev = _make_obs(torch.zeros(6), timestep=0)

    # Case 1 – timestep already predicted
    policy_server._predicted_timesteps.add(1)
    obs_same_ts = _make_obs(torch.ones(6), timestep=1)
    assert policy_server._obs_sanity_checks(obs_same_ts, prev) is False

    # Case 2 – observation too similar
    policy_server._predicted_timesteps.clear()
    obs_similar = _make_obs(torch.zeros(6) + 1e-4, timestep=2)
    assert policy_server._obs_sanity_checks(obs_similar, prev) is False

    # Case 3 – genuinely new & dissimilar observation passes
    obs_ok = _make_obs(torch.ones(6) * 5, timestep=3)
    assert policy_server._obs_sanity_checks(obs_ok, prev) is True


def test_predict_action_chunk(monkeypatch, policy_server):
    """End-to-end test of `_predict_action_chunk` with a stubbed _get_action_chunk."""
    # Import only when needed
    from lerobot.async_inference.policy_server import PolicyServer

    # Force server to act-style policy; patch method to return deterministic tensor
    policy_server.policy_type = "act"
    # NOTE(Steven): Smelly tests as the Server is a state machine being partially mocked. Adding these processors as a quick fix.
    policy_server.preprocessor = lambda obs: obs
    policy_server.postprocessor = lambda tensor: tensor
    action_dim = 6
    batch_size = 1
    actions_per_chunk = policy_server.actions_per_chunk

    def _fake_get_action_chunk(_self, _obs, _type="act"):
        return torch.zeros(batch_size, actions_per_chunk, action_dim)

    monkeypatch.setattr(PolicyServer, "_get_action_chunk", _fake_get_action_chunk, raising=True)

    obs = _make_obs(torch.zeros(6), timestep=5)
    timed_actions = policy_server._predict_action_chunk(obs)

    assert len(timed_actions) == actions_per_chunk
    assert [ta.get_timestep() for ta in timed_actions] == list(range(5, 5 + actions_per_chunk))

    for i, ta in enumerate(timed_actions):
        expected_ts = obs.get_timestamp() + i * policy_server.config.environment_dt
        assert abs(ta.get_timestamp() - expected_ts) < 1e-6


def test_vista_postprocess_delta_action_to_absolute(policy_server):
    """Vista delta actions are converted back to absolute actions, except gripper dims."""
    policy_server.policy_type = "vista"
    policy_server.policy.config.use_delta_action = True
    policy_server.postprocessor = lambda tensor: tensor

    state = torch.arange(16, dtype=torch.float32).unsqueeze(0)
    delta_chunk = torch.ones(1, 2, 16, dtype=torch.float32)

    action = policy_server._postprocess_action_chunk(delta_chunk, raw_state=state)

    expected = torch.ones(2, 16, dtype=torch.float32) + state.squeeze(0)
    expected[:, [7, 15]] = 1.0
    torch.testing.assert_close(action, expected)


def test_vista_postprocess_requires_16d_actions(policy_server):
    policy_server.policy_type = "vista"
    policy_server.policy.config.use_delta_action = False
    policy_server.postprocessor = lambda tensor: tensor

    with pytest.raises(ValueError, match="expects 16D actions"):
        policy_server._postprocess_action_chunk(torch.zeros(1, 2, 14), raw_state=torch.zeros(1, 16))


def test_relative_state_uses_delta_mask():
    from lerobot.datasets.transforms import make_relative_state

    current = torch.arange(16, dtype=torch.float32).unsqueeze(0)
    previous = current + 2
    mask = torch.ones(16, dtype=torch.bool)
    mask[[7, 15]] = False

    relative = make_relative_state(previous, current, mask)

    expected = torch.full_like(current, 2.0)
    expected[:, [7, 15]] = previous[:, [7, 15]]
    torch.testing.assert_close(relative, expected)


def test_vista_relative_state_preprocess_and_absolute_restore(monkeypatch, policy_server):
    from lerobot.async_inference.helpers import TimedObservation

    names = [f"joint{i}" for i in range(16)]
    policy_server.lerobot_features = {
        OBS_STATE: {"dtype": "float32", "shape": [16], "names": names}
    }
    policy_server.policy_type = "vista"
    policy_server.policy.config.use_relative_state = True
    policy_server.policy.config.use_delta_action = True
    policy_server.actions_per_chunk = 2

    captured = {}

    def preprocess(observation):
        captured["state"] = observation[OBS_STATE].clone()
        return observation

    policy_server.preprocessor = preprocess
    policy_server.postprocessor = lambda tensor: tensor
    monkeypatch.setattr(
        policy_server,
        "_get_action_chunk",
        lambda observation: torch.zeros(1, 2, 16),
    )

    current = torch.arange(16, dtype=torch.float32)
    previous = current + 2
    observation = TimedObservation(
        timestamp=time.time(),
        timestep=1,
        observation={**{name: current[i].item() for i, name in enumerate(names)}, "task": "test"},
        previous_state=previous,
    )

    actions = policy_server._predict_action_chunk(observation)

    expected_state = torch.full((1, 16), 2.0)
    expected_state[:, [7, 15]] = previous[[7, 15]]
    torch.testing.assert_close(captured["state"], expected_state)
    expected_action = current.clone()
    expected_action[[7, 15]] = 0
    torch.testing.assert_close(actions[0].get_action(), expected_action)


def test_vista_relative_state_skips_first_observation(policy_server):
    policy_server.policy_type = "vista"
    policy_server.policy.config.use_relative_state = True
    names = [f"joint{i}" for i in range(16)]
    policy_server.lerobot_features = {
        OBS_STATE: {"dtype": "float32", "shape": [16], "names": names}
    }
    observation = _make_obs(torch.zeros(6), timestep=0)
    observation.observation = {**{name: 0.0 for name in names}, "task": "test"}

    assert policy_server._predict_action_chunk(observation) == []
