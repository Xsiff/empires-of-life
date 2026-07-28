import torch

from eol.environment import Action, Agent2D, Environment
from eol.micro.algorithms import (
    build_ppo_batch,
    compute_gae,
    discounted_returns,
    ppo_losses,
    train_policy,
    train_ppo_policy,
)
from eol.micro.config import PPOTrainingConfig, TrainingConfig
from eol.micro.curriculum import (
    CURRICULUM_STAGES,
    build_curriculum_scenario_factory,
    get_curriculum_stage,
)
from eol.micro.expert import (
    a_star_path,
    collect_expert_samples,
    pretrain_policy_with_astar,
)
from eol.micro.features import (
    encode_state,
    get_valid_action_mask,
    mask_action_logits,
    resolve_action,
)
from eol.micro.inference import evaluate_policy, select_greedy_action
from eol.micro.model import ActorCriticPolicy, DirectionPolicy
from eol.micro.reward import compute_reward
from eol.micro.rollout import (
    EpisodeStep,
    EpisodeTrajectory,
    collect_episode,
    collect_ppo_episode,
    select_action,
)


class RightOnlyPolicy(DirectionPolicy):
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        del state
        return torch.tensor([[-10.0, -10.0, -10.0, 10.0, -10.0]], dtype=torch.float32)


class DownOnlyPolicy(DirectionPolicy):
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        del state
        return torch.tensor([[-10.0, 10.0, -10.0, -10.0, -10.0]], dtype=torch.float32)


class DownOnlyActorCritic(ActorCriticPolicy):
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = state.shape[0]
        return (
            torch.tensor(
                [[-10.0, 10.0, -10.0, -10.0, -10.0]],
                dtype=torch.float32,
            ).repeat(batch_size, 1),
            torch.full((batch_size,), 0.5, dtype=torch.float32),
        )


def test_actor_critic_policy_returns_logits_and_values() -> None:
    policy = ActorCriticPolicy(input_dim=31, hidden_dim=8, action_dim=5)

    logits, values = policy(torch.zeros((2, 31)))

    assert logits.shape == (2, 5)
    assert values.shape == (2,)


def test_a_star_path_finds_a_valid_route() -> None:
    environment = Environment(
        width=4,
        height=4,
        agents={Agent2D((0, 0))},
        target_position=(0, 3),
        obstacle_positions={(0, 1), (1, 1)},
    )

    path = a_star_path(environment, (0, 0), (0, 3))

    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (0, 3)
    assert len(path) > 1


def test_curriculum_stage_schedule_matches_requested_progression() -> None:
    stages = [
        get_curriculum_stage(1, 8),
        get_curriculum_stage(2, 8),
        get_curriculum_stage(3, 8),
        get_curriculum_stage(4, 8),
        get_curriculum_stage(5, 8),
        get_curriculum_stage(6, 8),
        get_curriculum_stage(7, 8),
        get_curriculum_stage(8, 8),
    ]

    assert stages == list(CURRICULUM_STAGES)


def test_curriculum_scenario_factory_grows_environment() -> None:
    factory = build_curriculum_scenario_factory(TrainingConfig(episodes=8))

    stage_one_environment, _ = factory(1, 11)
    stage_eight_environment, _ = factory(8, 17)

    assert stage_one_environment.width == 5
    assert stage_one_environment.height == 5
    assert len(stage_one_environment.obstacle_positions) == 1
    assert stage_eight_environment.width == 20
    assert stage_eight_environment.height == 20
    assert len(stage_eight_environment.obstacle_positions) == 40


def test_encode_state_returns_expected_shape_and_dtype() -> None:
    environment = Environment(
        width=5,
        height=5,
        agents={Agent2D((1, 2))},
        target_position=(3, 4),
        obstacle_positions={(0, 2)},
    )

    state = encode_state(environment, next(iter(environment.agents)))

    assert state.shape == (31,)
    assert state.dtype == torch.float32


def test_encode_state_includes_centered_5x5_local_view() -> None:
    environment = Environment(
        width=5,
        height=5,
        agents={Agent2D((1, 2))},
        target_position=(2, 3),
        obstacle_positions={(0, 2), (1, 1)},
    )

    state = encode_state(environment, next(iter(environment.agents)))

    assert state[6:].tolist() == [
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.25,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


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
    policy = DirectionPolicy(input_dim=31, hidden_dim=8, action_dim=5)

    action_index, log_prob, state, valid_action_mask = select_action(
        policy, environment, next(iter(environment.agents))
    )

    assert state.shape == (31,)
    assert valid_action_mask.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert 0 <= action_index < 5
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
    policy = RightOnlyPolicy(input_dim=31, hidden_dim=8, action_dim=5)

    trajectory = collect_episode(policy, environment, agent, max_steps=4)

    assert isinstance(trajectory, EpisodeTrajectory)
    assert trajectory.reached_target is True
    assert trajectory.termination_reason == "target_reached"
    assert len(trajectory.steps) == 2
    assert trajectory.steps[-1].done is True
    assert agent.position == (0, 2)


def test_resolve_action_falls_back_to_halt_for_blocked_move() -> None:
    environment = Environment(
        width=3,
        height=3,
        agents={Agent2D((0, 0))},
        target_position=(2, 2),
        obstacle_positions={(1, 0)},
    )
    agent = next(iter(environment.agents))

    assert resolve_action(environment, agent, 1) is Action.HALT
    assert resolve_action(environment, agent, 3) is Action.RIGHT


def test_collect_episode_counts_halt_when_it_is_the_only_valid_action() -> None:
    environment = Environment(
        width=2,
        height=2,
        agents={Agent2D((0, 0))},
        target_position=(1, 1),
        obstacle_positions={(0, 1), (1, 0)},
    )
    agent = next(iter(environment.agents))
    policy = DirectionPolicy(input_dim=31, hidden_dim=8, action_dim=5)

    trajectory = collect_episode(policy, environment, agent, max_steps=4)

    assert len(trajectory.steps) == 4
    assert all(step.action is Action.HALT for step in trajectory.steps)
    assert trajectory.reached_target is False
    assert trajectory.termination_reason == "max_steps"


def test_select_greedy_action_returns_halt_for_blocked_suggestion() -> None:
    environment = Environment(
        width=3,
        height=3,
        agents={Agent2D((0, 0))},
        target_position=(2, 2),
        obstacle_positions={(1, 0)},
    )
    agent = next(iter(environment.agents))
    policy = DownOnlyPolicy(input_dim=31, hidden_dim=8, action_dim=5)

    action_index = select_greedy_action(policy, environment, agent)

    assert action_index == 4


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


def test_reward_is_negative_when_target_is_not_reached() -> None:
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

    assert reward < 0.0


def test_reward_is_positive_when_target_is_reached() -> None:
    environment = Environment(
        width=5,
        height=5,
        agents={Agent2D((2, 3))},
        target_position=(2, 4),
        obstacle_positions=(),
    )
    agent = next(iter(environment.agents))
    environment.move_agent((agent, Action.RIGHT))

    reward = compute_reward(
        environment,
        agent,
        1,
        position_before=(2, 3),
        reached_target=True,
        episode_finished=True,
    )

    assert reward > 5.0


def test_discounted_returns_computes_expected_sequence() -> None:
    trajectory = EpisodeTrajectory(
        steps=(
            EpisodeStep(
                state=torch.zeros(31),
                valid_action_mask=torch.ones(5),
                action_index=0,
                action=Action.UP,
                log_prob=torch.tensor(0.0),
                reward=1.0,
                done=False,
                position_before=(0, 0),
                position_after=(0, 0),
            ),
            EpisodeStep(
                state=torch.zeros(31),
                valid_action_mask=torch.ones(5),
                action_index=0,
                action=Action.UP,
                log_prob=torch.tensor(0.0),
                reward=2.0,
                done=False,
                position_before=(0, 0),
                position_after=(0, 0),
            ),
            EpisodeStep(
                state=torch.zeros(31),
                valid_action_mask=torch.ones(5),
                action_index=0,
                action=Action.UP,
                log_prob=torch.tensor(0.0),
                reward=3.0,
                done=True,
                position_before=(0, 0),
                position_after=(0, 0),
            ),
        ),
        total_reward=6.0,
        reached_target=False,
        termination_reason="max_steps",
    )

    returns = discounted_returns(trajectory, gamma=0.5)

    assert torch.allclose(returns, torch.tensor([2.75, 3.5, 3.0]))


def test_compute_gae_returns_expected_sequence() -> None:
    advantages, returns = compute_gae(
        rewards=torch.tensor([1.0, 2.0, 3.0]),
        values=torch.tensor([0.5, 0.5, 0.5]),
        dones=torch.tensor([0.0, 0.0, 1.0]),
        gamma=0.5,
        gae_lambda=1.0,
    )

    assert torch.allclose(advantages, torch.tensor([2.25, 3.0, 2.5]))
    assert torch.allclose(returns, torch.tensor([2.75, 3.5, 3.0]))


def test_collect_expert_samples_returns_valid_actions() -> None:
    samples = collect_expert_samples(
        PPOTrainingConfig(
            grid_size=4,
            obstacle_count=1,
            pretraining_episodes=3,
            seed=7,
        )
    )

    assert len(samples) > 0
    assert all(sample.state.shape == (31,) for sample in samples)
    assert all(sample.valid_action_mask.shape == (5,) for sample in samples)
    assert all(
        sample.valid_action_mask[sample.action_index].item() == 1.0
        for sample in samples
    )


def test_collect_ppo_episode_stores_old_log_probs_and_values() -> None:
    environment = Environment(
        width=3,
        height=3,
        agents={Agent2D((0, 0))},
        target_position=(2, 2),
        obstacle_positions={(1, 0)},
    )
    agent = next(iter(environment.agents))
    policy = DownOnlyActorCritic(input_dim=31, hidden_dim=8, action_dim=5)

    trajectory = collect_ppo_episode(policy, environment, agent, max_steps=2)

    assert len(trajectory.steps) == 2
    assert trajectory.steps[0].action in {Action.RIGHT, Action.HALT}
    assert all(step.old_log_prob.ndim == 0 for step in trajectory.steps)
    assert all(step.value.ndim == 0 for step in trajectory.steps)


def test_build_ppo_batch_and_losses_have_expected_shapes() -> None:
    environment = Environment(
        width=3,
        height=3,
        agents={Agent2D((0, 0))},
        target_position=(2, 2),
        obstacle_positions={(1, 0)},
    )
    agent = next(iter(environment.agents))
    policy = ActorCriticPolicy(input_dim=31, hidden_dim=8, action_dim=5)
    trajectory = collect_ppo_episode(policy, environment, agent, max_steps=3)

    batch = build_ppo_batch((trajectory,), gamma=0.99, gae_lambda=0.95)
    total_loss, policy_loss, value_loss, entropy = ppo_losses(
        policy,
        batch,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
    )

    assert batch.states.shape[1] == 31
    assert batch.valid_action_masks.shape[1] == 5
    assert batch.action_indices.ndim == 1
    assert batch.old_log_probs.ndim == 1
    assert batch.returns.ndim == 1
    assert batch.advantages.ndim == 1
    assert torch.isfinite(total_loss)
    assert torch.isfinite(policy_loss)
    assert torch.isfinite(value_loss)
    assert torch.isfinite(entropy)


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


def test_train_policy_supports_curriculum(tmp_path) -> None:
    model = train_policy(
        TrainingConfig(
            episodes=4,
            max_steps=4,
            hidden_dim=8,
            print_every=1,
            save_path=tmp_path / "policy_curriculum.pt",
            curriculum=True,
        )
    )

    assert isinstance(model, DirectionPolicy)
    assert (tmp_path / "policy_curriculum.pt").exists()


def test_train_ppo_policy_returns_torch_model_and_saves_weights(tmp_path) -> None:
    model = train_ppo_policy(
        PPOTrainingConfig(
            grid_size=4,
            obstacle_count=1,
            episodes=1,
            max_steps=4,
            hidden_dim=8,
            rollout_episodes_per_update=2,
            ppo_epochs=2,
            minibatch_size=4,
            print_every=1,
            save_path=tmp_path / "policy_ppo.pt",
        )
    )

    assert isinstance(model, ActorCriticPolicy)
    assert (tmp_path / "policy_ppo.pt").exists()


def test_train_ppo_policy_supports_curriculum(tmp_path) -> None:
    model = train_ppo_policy(
        PPOTrainingConfig(
            episodes=1,
            max_steps=4,
            hidden_dim=8,
            rollout_episodes_per_update=2,
            ppo_epochs=2,
            minibatch_size=4,
            print_every=1,
            save_path=tmp_path / "policy_ppo_curriculum.pt",
            curriculum=True,
        )
    )

    assert isinstance(model, ActorCriticPolicy)
    assert (tmp_path / "policy_ppo_curriculum.pt").exists()


def test_pretrain_policy_with_astar_keeps_actor_critic_callable() -> None:
    policy = ActorCriticPolicy(input_dim=31, hidden_dim=8, action_dim=5)
    config = PPOTrainingConfig(
        grid_size=4,
        obstacle_count=1,
        pretraining_episodes=4,
        pretraining_epochs=2,
        pretraining_batch_size=4,
        hidden_dim=8,
    )

    pretrain_policy_with_astar(policy, config)
    logits, values = policy(torch.zeros((1, 31)))

    assert logits.shape == (1, 5)
    assert values.shape == (1,)


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


def test_evaluate_policy_runs_on_fresh_environments_for_ppo(tmp_path) -> None:
    policy = train_ppo_policy(
        PPOTrainingConfig(
            grid_size=4,
            obstacle_count=1,
            episodes=1,
            max_steps=3,
            hidden_dim=8,
            rollout_episodes_per_update=2,
            ppo_epochs=2,
            minibatch_size=4,
            print_every=1,
            evaluation_episodes=2,
            save_path=tmp_path / "policy_ppo.pt",
        )
    )

    results = evaluate_policy(policy, PPOTrainingConfig(evaluation_episodes=2))

    assert len(results) == 2
    assert all(isinstance(result, bool) for result in results)
