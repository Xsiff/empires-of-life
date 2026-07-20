import pytest

torch = pytest.importorskip("torch")

from eol.micro.model import DirectionPolicy
from eol.micro.reward import RewardConfig, compute_reward
from eol.micro.train import TrainingConfig, encode_state, train_policy


def test_encode_state_returns_expected_shape() -> None:
    state = encode_state((1, 2), (3, 4), grid_size=5)

    assert state.shape == (6,)
    assert state.dtype == torch.float32


def test_reward_improves_when_agent_moves_closer() -> None:
    reward = compute_reward(
        previous_distance=4,
        new_distance=2,
        reached_target=False,
        hit_obstacle=False,
        hit_wall=False,
        config=RewardConfig(),
    )

    assert reward > 0


def test_train_policy_returns_torch_model(tmp_path) -> None:
    model = train_policy(
        config=TrainingConfig(
            grid_size=4,
            obstacle_count=1,
            episodes=2,
            max_steps=4,
            print_every=1,
            save_path=tmp_path / "policy.pt",
        )
    )

    assert isinstance(model, DirectionPolicy)
    assert (tmp_path / "policy.pt").exists()
