import torch

from lerobot.configs.default import DatasetConfig
from lerobot.scripts.lerobot_train_umi import UmiTrainPipelineConfig, maybe_zero_observation_state
from lerobot.utils.constants import OBS_STATE


def test_zero_observation_state_is_disabled_by_default():
    cfg = UmiTrainPipelineConfig(dataset=DatasetConfig(repo_id="test/dataset"))

    assert cfg.zero_observation_state is False

    enabled_cfg = UmiTrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/dataset"), zero_observation_state=True
    )
    assert enabled_cfg.zero_observation_state is True


def test_maybe_zero_observation_state_disabled_keeps_state():
    state = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    batch = {OBS_STATE: state, "task": ["test"]}

    result = maybe_zero_observation_state(batch, enabled=False)

    assert result[OBS_STATE] is state
    assert result["task"] == ["test"]


def test_maybe_zero_observation_state_enabled_replaces_state_with_zeros():
    state = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    batch = {OBS_STATE: state, "task": ["test"]}

    result = maybe_zero_observation_state(batch, enabled=True)

    assert result[OBS_STATE].shape == state.shape
    assert result[OBS_STATE].dtype == state.dtype
    assert torch.count_nonzero(result[OBS_STATE]) == 0
    assert result["task"] == ["test"]
