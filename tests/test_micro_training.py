import torch

from eol.environment import Action, Agent2D, Environment
from eol.micro.inference import evaluate_policy
from eol.micro.model import DirectionPolicy
from eol.micro.reward import compute_reward
from eol.micro.train import (
    EpisodeTrajectory,
    TrainingConfig,
    collect_episode,
    discounted_returns,
    encode_state,
    get_valid_action_mask,
    mask_action_logits,
    select_action,
    train_policy,
)


def test_encode_state_returns_expected_shape_and_dtype() -> None:
    environment = Environment(
        width=5,
        height=5,
        agents={Agent2D((1, 2))},
        target_position=(3, 4),
        obstacle_positions={(0, 2)},
    )

    state = encode_state(environment, next(iter(environment.agents)))

    assert state.shape == (11,)
    assert state.dtype == torch.float32


def test_valid_action_mask_blocks_walls_and_obstacles() -> None:
    environment = Environment(
        width=3,
        height=3,
        agents={Agent2D((0, 0))},
        target_position=(2, 2),
        obstacle_positions={(1, 0)},
    )

    mask = get_valid_action_mask(environment, next(iter(environment.agents)))

    assert mask.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_mask_action_logits_removes_invalid_moves() -> None:
    logits = torch.tensor([[10.0, 5.0, 2.0, 1.0, 7.0]])
    mask = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])

    masked_logits = mask_action_logits(logits, mask)

    assert torch.argmax(masked_logits, dim=1).item() == 4


def test_select_action_returns_valid_index_and_state() -> None:
    environment = Environment(
        width=3,
        height=3,
        agents={Agent2D((0, 0))},
        target_position=(2, 2),
        obstacle_positions={(1, 0)},
    )
    policy = DirectionPolicy(input_dim=11, hidden_dim=8, action_dim=5)

    action_index, log_prob, state, valid_action_mask = select_action(
        policy, environment, next(iter(environment.agents))
    )

    assert state.shape == (11,)
    assert valid_action_mask.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert action_index in {3, 4}
    assert log_prob.ndim == 0


def test_collect_episode_reaches_target_with_deterministic_policy() -> None:
    environment = Environment(
        width=3,
        height=3,
        agents={Agent2D((0, 0))},
        target_position=(0, 2),
        obstacle_positions=(),
    )
    agent = next(iter(environment.agents))
    policy = DirectionPolicy(input_dim=11, hidden_dim=8, action_dim=5)

    def forward_override(_state: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[-10.0, -10.0, -10.0, 10.0, -10.0]], dtype=torch.float32)

    policy.forward = forward_override  # type: ignore[method-assign]

    trajectory = collect_episode(policy, environment, agent, max_steps=4)

    assert isinstance(trajectory, EpisodeTrajectory)
    assert trajectory.reached_target is True
    assert trajectory.termination_reason == "target_reached"
    assert len(trajectory.steps) == 2
    assert trajectory.steps[-1].done is True
    assert agent.position == (0, 2)


def test_collect_episode_counts_halt_when_it_is_the_only_valid_action() -> None:
    environment = Environment(
        width=2,
        height=2,
        agents={Agent2D((0, 0))},
        target_position=(1, 1),
        obstacle_positions={(0, 1), (1, 0)},
    )
    agent = next(iter(environment.agents))
    policy = DirectionPolicy(input_dim=11, hidden_dim=8, action_dim=5)

    trajectory = collect_episode(policy, environment, agent, max_steps=4)

    assert len(trajectory.steps) == 4
    assert all(step.action is Action.HALT for step in trajectory.steps)
    assert trajectory.reached_target is False
    assert trajectory.termination_reason == "max_steps"


def test_halt_action_is_valid_and_keeps_position() -> None:
    environment = Environment(
        width=3,
        height=3,
        agents={Agent2D((1, 1))},
        target_position=(2, 2),
        obstacle_positions=(),
    )
    agent = next(iter(environment.agents))

    mask = get_valid_action_mask(environment, agent)
    before = agent.position
    environment.move_agent((agent, Action.HALT))

    assert mask.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert agent.position == before


def test_reward_improves_when_agent_moves_closer() -> None:
    environment = Environment(
        width=5,
        height=5,
        agents={Agent2D((2, 2))},
        target_position=(2, 4),
        obstacle_positions=(),
    )
    agent = next(iter(environment.agents))
    environment.move_agent((agent, Action.RIGHT))

    reward = compute_reward(
        environment,
        agent,
        3,
        position_before=(2, 2),
        reached_target=False,
        episode_finished=False,
    )

    assert reward > 0


def test_discounted_returns_computes_expected_sequence() -> None:
    trajectory = EpisodeTrajectory(
        steps=(
            type("Step", (), {"reward": 1.0})(),
            type("Step", (), {"reward": 2.0})(),
            type("Step", (), {"reward": 3.0})(),
        ),
        total_reward=6.0,
        reached_target=False,
        termination_reason="max_steps",
    )

    returns = discounted_returns(trajectory, gamma=0.5)

    assert torch.allclose(returns, torch.tensor([2.75, 3.5, 3.0]))


def test_train_policy_returns_torch_model_and_saves_weights(tmp_path) -> None:
    model = train_policy(
        TrainingConfig(
            grid_size=4,
            obstacle_count=1,
            episodes=2,
            max_steps=4,
            hidden_dim=8,
            print_every=1,
            save_path=tmp_path / "policy.pt",
        )
    )

    assert isinstance(model, DirectionPolicy)
    assert (tmp_path / "policy.pt").exists()


def test_evaluate_policy_runs_on_fresh_environments(tmp_path) -> None:
    policy = train_policy(
        TrainingConfig(
            grid_size=4,
            obstacle_count=1,
            episodes=1,
            max_steps=3,
            hidden_dim=8,
            print_every=1,
            evaluation_episodes=2,
            save_path=tmp_path / "policy.pt",
        )
    )

    results = evaluate_policy(policy, TrainingConfig(evaluation_episodes=2))

    assert len(results) == 2
    assert all(isinstance(result, bool) for result in results)
