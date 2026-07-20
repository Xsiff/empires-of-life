import torch
from eol.environment import Environment
from eol.micro.model import DirectionPolicy
from eol.micro.reward import RewardConfig, compute_reward
from eol.micro.train import (
    TrainingConfig,
    encode_state,
    get_valid_action_mask,
    mask_action_logits,
    train_policy,
)
from eol.visualization import (
    build_environment_frame,
    render_grid,
    select_greedy_action,
)


def test_encode_state_returns_expected_shape() -> None:
    state = encode_state((1, 2), (3, 4), grid_size=5, obstacle_positions={(0, 2)})

    assert state.shape == (10,)
    assert state.dtype == torch.float32


def test_encode_state_marks_blocked_neighbor_cells() -> None:
    state = encode_state(
        (0, 0),
        (2, 2),
        grid_size=3,
        obstacle_positions={(1, 0)},
    )

    assert state[-4:].tolist() == [1.0, 1.0, 1.0, 0.0]


def test_reward_improves_when_agent_moves_closer() -> None:
    reward = compute_reward(
        previous_distance=4,
        new_distance=2,
        reached_target=False,
        hit_obstacle=False,
        hit_wall=False,
        revisited_position=False,
        episode_finished=False,
        config=RewardConfig(),
    )

    assert reward > 0


def test_reward_penalizes_hovering_near_target() -> None:
    reward = compute_reward(
        previous_distance=1,
        new_distance=1,
        reached_target=False,
        hit_obstacle=False,
        hit_wall=False,
        revisited_position=True,
        episode_finished=False,
        config=RewardConfig(),
    )

    assert reward < -1.0


def test_reward_strongly_prefers_reaching_target() -> None:
    reached_target_reward = compute_reward(
        previous_distance=1,
        new_distance=0,
        reached_target=True,
        hit_obstacle=False,
        hit_wall=False,
        revisited_position=False,
        episode_finished=True,
        config=RewardConfig(),
    )
    orbit_reward = compute_reward(
        previous_distance=1,
        new_distance=1,
        reached_target=False,
        hit_obstacle=False,
        hit_wall=False,
        revisited_position=True,
        episode_finished=False,
        config=RewardConfig(),
    )

    assert reached_target_reward > orbit_reward


def test_reward_penalizes_failing_to_reach_target_by_episode_end() -> None:
    reward = compute_reward(
        previous_distance=2,
        new_distance=1,
        reached_target=False,
        hit_obstacle=False,
        hit_wall=False,
        revisited_position=False,
        episode_finished=True,
        config=RewardConfig(),
    )

    assert reward < 0


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


def test_render_grid_contains_agent_target_and_obstacles() -> None:
    environment = Environment(
        size=3,
        agent_position=(0, 0),
        target_position=(2, 2),
        obstacle_positions=((1, 1),),
    )
    frame = build_environment_frame(environment, agent_position=(0, 0))
    rendered = render_grid(frame)

    assert "A" in rendered
    assert "T" in rendered
    assert "#" in rendered


def test_valid_action_mask_blocks_walls_and_obstacles() -> None:
    mask = get_valid_action_mask(
        agent_position=(0, 0),
        grid_size=3,
        obstacle_positions={(1, 0)},
    )

    assert mask.tolist() == [0.0, 0.0, 0.0, 1.0]


def test_mask_action_logits_removes_invalid_moves() -> None:
    logits = torch.tensor([[10.0, 5.0, 2.0, 1.0]])
    mask = torch.tensor([0.0, 0.0, 0.0, 1.0])

    masked_logits = mask_action_logits(logits, mask)

    assert torch.argmax(masked_logits, dim=1).item() == 3


def test_select_greedy_action_avoids_visited_positions_when_possible() -> None:
    policy = DirectionPolicy(hidden_dim=8)

    def forward_override(_state: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[1.0, 4.0, 2.0, 3.0]], dtype=torch.float32)

    policy.forward = forward_override  # type: ignore

    action = select_greedy_action(
        policy=policy,
        agent_position=(1, 1),
        target_position=(2, 2),
        grid_size=4,
        obstacle_positions=set(),
        visited_positions={(2, 1), (1, 0)},
    )

    assert action == 3


def test_select_greedy_action_allows_revisits_when_no_alternative_exists() -> None:
    policy = DirectionPolicy(hidden_dim=8)

    def forward_override(_state: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[1.0, 4.0, 2.0, 3.0]], dtype=torch.float32)

    policy.forward = forward_override  # type: ignore

    action = select_greedy_action(
        policy=policy,
        agent_position=(0, 0),
        target_position=(2, 2),
        grid_size=3,
        obstacle_positions={(0, 1)},
        visited_positions={(1, 0)},
    )

    assert action == 1
