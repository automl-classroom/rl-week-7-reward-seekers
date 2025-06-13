"""
Deep Q-Learning with RND implementation.
"""

from typing import Any, Dict, List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.week_4.dqn import DQNAgent, set_seed


class RNDNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int):
        super().__init__()
        layers = []
        current_size = input_size

        for _ in range(n_layers):
            layers.extend([nn.Linear(current_size, hidden_size), nn.ReLU()])
            current_size = hidden_size

        # Final layer to embedding size
        layers.append(nn.Linear(current_size, hidden_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class RNDDQNAgent(DQNAgent):
    """
    Deep Q-Learning agent with ε-greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 0,
        rnd_hidden_size: int = 128,
        rnd_lr: float = 1e-3,
        rnd_update_freq: int = 1000,
        rnd_n_layers: int = 2,
        rnd_reward_weight: float = 0.1,
    ) -> None:
        """
        Initialize replay buffer, Q-networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini-batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target-network syncs.
        seed : int
            RNG seed.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )
        self.seed = seed
        # TODO: initialize the RND networks
        self.rnd_reward_weight = rnd_reward_weight
        self.rnd_update_freq = rnd_update_freq

        # Get input size from environment observation space
        if isinstance(env.observation_space, gym.spaces.Box):
            input_size = np.prod(env.observation_space.shape)
        else:
            input_size = env.observation_space.n

        # Initialize the RND networks
        self.rnd_target = RNDNetwork(input_size, rnd_hidden_size, rnd_n_layers)
        self.rnd_predictor = RNDNetwork(input_size, rnd_hidden_size, rnd_n_layers)

        # Freeze target network
        for param in self.rnd_target.parameters():
            param.requires_grad = False

        # RND optimizer
        self.rnd_optimizer = optim.Adam(self.rnd_predictor.parameters(), lr=rnd_lr)
        self.rnd_criterion = nn.MSELoss()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rnd_target.to(device)
        self.rnd_predictor.to(device)

        self.rnd_losses = []

    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Convert state to tensor and flatten if needed."""
        device = next(self.rnd_predictor.parameters()).device

        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).to(device)
        else:
            state_tensor = torch.FloatTensor([state]).to(device)

        if len(state_tensor.shape) > 1:
            state_tensor = state_tensor.flatten()

        return state_tensor.unsqueeze(0)

    def update_rnd(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on the RND network on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).
        """
        # TODO: get states and next_states from the batch
        states = []
        next_states = []

        for transition in training_batch:
            state, _, _, next_state, done, _ = transition
            states.append(state)
            if not done:
                next_states.append(next_state)

        device = next(self.rnd_predictor.parameters()).device
        all_states = states + next_states

        if not all_states:
            return 0.0

        # Preprocess all states
        state_tensors = []
        for state in all_states:
            state_tensor = self._preprocess_state(state)
            state_tensors.append(state_tensor)

        batch_states = torch.cat(state_tensors, dim=0)

        # TODO: compute the MSE
        with torch.no_grad():
            target_features = self.rnd_target(batch_states)

        predicted_features = self.rnd_predictor(batch_states)
        rnd_loss = self.rnd_criterion(predicted_features, target_features)

        # TODO: update the RND network
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()

        loss_value = rnd_loss.item()
        self.rnd_losses.append(loss_value)
        return loss_value

    def get_rnd_bonus(self, state: np.ndarray) -> float:
        """Compute the RND bonus for a given state.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        float
            The RND bonus for the state.
        """
        # TODO: predict embeddings
        # TODO: get error
        with torch.no_grad():
            state_tensor = self._preprocess_state(state)

            # Predict embeddings
            target_features = self.rnd_target(state_tensor)
            predicted_features = self.rnd_predictor(state_tensor)

            error = torch.mean((predicted_features - target_features) ** 2)

            return error.item()

    def train(self, num_frames: int, eval_interval: int = 1000) -> None:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        ep_intrinsic_reward = 0.0
        recent_rewards: List[float] = []
        recent_intrinsic_rewards: List[float] = []
        episode_rewards = []
        episode_intrinsic_rewards = []
        steps = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, info = self.env.step(action)

            # TODO: apply RND bonus
            rnd_bonus = self.get_rnd_bonus(state)
            intrinsic_reward = self.rnd_reward_weight * rnd_bonus
            total_reward = reward + intrinsic_reward

            # store and step
            self.buffer.add(
                state, action, total_reward, next_state, done or truncated, info
            )
            state = next_state
            ep_reward += reward
            ep_intrinsic_reward += intrinsic_reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

                if self.total_steps % self.rnd_update_freq == 0:
                    rnd_loss = self.update_rnd(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                recent_intrinsic_rewards.append(ep_intrinsic_reward)
                episode_rewards.append(ep_reward)
                episode_intrinsic_rewards.append(ep_intrinsic_reward)
                steps.append(frame)
                ep_reward = 0.0
                ep_intrinsic_reward = 0.0

                # Logging
                if len(recent_rewards) % 10 == 0:
                    avg_ext = np.mean(recent_rewards)
                    avg_int = np.mean(recent_intrinsic_rewards)
                    print(
                        f"Frame {frame}, AvgExtReward(10): {avg_ext:.2f}, "
                        f"AvgIntReward(10): {avg_int:.2f}, ε={self.epsilon():.3f}"
                    )

        # Saving to .csv for simplicity
        # Could also be e.g. npz
        print("Training complete.")
        training_data = pd.DataFrame(
            {
                "steps": steps,
                "rewards": episode_rewards,
                "intrinsic_rewards": episode_intrinsic_rewards,
            }
        )
        training_data.to_csv(f"training_data_seed_{self.seed}.csv", index=False)

        if self.rnd_losses:
            rnd_data = pd.DataFrame({"rnd_loss": self.rnd_losses})
            rnd_data.to_csv(f"rnd_losses_seed_{self.seed}.csv", index=False)


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    # 1) build env
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    # 3) TODO: instantiate & train the agent
    agent = RNDDQNAgent(
        env=env,
        buffer_capacity=cfg.get("buffer_capacity", 10000),
        batch_size=cfg.get("batch_size", 32),
        lr=cfg.get("lr", 1e-3),
        gamma=cfg.get("gamma", 0.99),
        epsilon_start=cfg.get("epsilon_start", 1.0),
        epsilon_final=cfg.get("epsilon_final", 0.01),
        epsilon_decay=cfg.get("epsilon_decay", 500),
        target_update_freq=cfg.get("target_update_freq", 1000),
        seed=cfg.seed,
        rnd_hidden_size=cfg.get("rnd_hidden_size", 128),
        rnd_lr=cfg.get("rnd_lr", 1e-3),
        rnd_update_freq=cfg.get("rnd_update_freq", 1000),
        rnd_n_layers=cfg.get("rnd_n_layers", 2),
        rnd_reward_weight=cfg.get("rnd_reward_weight", 0.1),
    )
    agent.train(cfg.get("num_frames", 50000))


if __name__ == "__main__":
    main()
