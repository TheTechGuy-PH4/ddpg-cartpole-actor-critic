import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dm_control import suite


###############################################
# 1. Utilities: seeding & observation handling
###############################################

def set_seed(seed: int):
    """Set Python, NumPy, and PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_observation(time_step):
    """
    dm_control returns an observation dict.
    We flatten & concatenate all entries into a single np.array.
    """
    obs_dict = time_step.observation
    keys = sorted(obs_dict.keys())
    return np.concatenate([obs_dict[k].ravel() for k in keys], axis=0)


###############################################
# 2. Replay Buffer
###############################################

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size: int):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.max_size = size
        self.ptr = 0
        self.size = 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


###############################################
# 3. Actor & Critic Networks
###############################################

class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_low, act_high, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], act_dim),
            nn.Tanh(),  # output in [-1, 1]
        )
        # action bounds as buffers (not trainable, move with device)
        self.register_buffer("act_low", torch.tensor(act_low, dtype=torch.float32))
        self.register_buffer("act_high", torch.tensor(act_high, dtype=torch.float32))

    def forward(self, obs):
        # obs: (batch, obs_dim)
        x = self.net(obs)  # in [-1, 1]
        # scale to [act_low, act_high]
        return self.act_low + (x + 1.0) * 0.5 * (self.act_high - self.act_low)


class MLPQCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

    def forward(self, obs, act):
        # obs: (batch, obs_dim), act: (batch, act_dim)
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


###############################################
# 4. DDPG Agent (Actor-Critic Policy Gradient)
###############################################

class DDPGAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_low,
        act_high,
        gamma=0.99,
        tau=0.005,
        actor_lr=1e-3,
        critic_lr=1e-3,
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Online networks
        self.actor = MLPActor(obs_dim, act_dim, act_low, act_high).to(self.device)
        self.critic = MLPQCritic(obs_dim, act_dim).to(self.device)

        # Target networks
        self.actor_target = MLPActor(obs_dim, act_dim, act_low, act_high).to(self.device)
        self.critic_target = MLPQCritic(obs_dim, act_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau

        # action bounds as numpy (for clipping noise-perturbed actions)
        self.act_low = act_low
        self.act_high = act_high

    @torch.no_grad()
    def select_action(self, obs, noise_scale=0.0):
        """
        Deterministic policy + optional exploration noise
        obs: np.array (obs_dim,)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(obs_t).cpu().numpy()[0]

        if noise_scale > 0.0:
            noise = noise_scale * np.random.randn(*action.shape)
            action = action + noise

        return np.clip(action, self.act_low, self.act_high)

    def update(self, replay_buffer: ReplayBuffer, batch_size=128):
        batch = replay_buffer.sample_batch(batch_size)
        obs = batch["obs"].to(self.device)
        obs2 = batch["obs2"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device)
        done = batch["done"].to(self.device)

        # ----- Critic update -----
        with torch.no_grad():
            next_actions = self.actor_target(obs2)
            target_q = self.critic_target(obs2, next_actions)
            # y = r + gamma * (1 - done) * Q_target(s', π_target(s'))
            y = rews + self.gamma * (1.0 - done) * target_q

        current_q = self.critic(obs, acts)
        critic_loss = ((current_q - y) ** 2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----- Actor update (policy gradient via critic) -----
        # maximize Q(s, π(s))  <=>  minimize -Q(s, π(s))
        actor_loss = -self.critic(obs, self.actor(obs)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----- Soft updates -----
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

        return actor_loss.item(), critic_loss.item()

    def soft_update(self, main_net, target_net):
        with torch.no_grad():
            for p, p_targ in zip(main_net.parameters(), target_net.parameters()):
                p_targ.data.mul_(1.0 - self.tau)
                p_targ.data.add_(self.tau * p.data)

    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])


###############################################
# 5. Environment construction (Cartpole/Balance)
###############################################

def make_env_cartpole_balance(seed: int):
    """
    DeepMind Control Suite Cartpole Balance.
    'random' seed controls the task stochasticity.
    """
    env = suite.load(
        domain_name="cartpole",
        task_name="balance",
        task_kwargs={"random": seed},
    )
    return env


###############################################
# 6. Training for one seed
###############################################

def train_ddpg_for_seed(
    seed: int,
    num_episodes=300,
    max_steps_per_episode=1000,
    replay_size=int(1e6),
    batch_size=128,
    start_steps=1000,       # purely random steps before using the policy
    update_after=1000,      # start learning after this many env steps
    update_every=50,        # perform this many gradient steps every 'update_every' env steps
    actor_lr=1e-3,
    critic_lr=1e-3,
    gamma=0.99,
    tau=0.005,
    noise_scale_initial=0.2,
    noise_scale_final=0.05,
):
    set_seed(seed)
    env = make_env_cartpole_balance(seed)

    # Get obs & action dimensions
    time_step = env.reset()
    obs = flatten_observation(time_step)
    obs_dim = obs.shape[0]
    action_spec = env.action_spec()
    act_dim = action_spec.shape[0]
    act_low = action_spec.minimum
    act_high = action_spec.maximum

    agent = DDPGAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_low=act_low,
        act_high=act_high,
        gamma=gamma,
        tau=tau,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
    )
    replay_buffer = ReplayBuffer(obs_dim, act_dim, size=replay_size)

    total_steps = 0
    episode_returns = []

    for ep in range(num_episodes):
        time_step = env.reset()
        obs = flatten_observation(time_step)
        ep_return = 0.0

        # decay exploration noise over episodes
        frac = ep / max(1, num_episodes - 1)
        noise_scale = noise_scale_initial + frac * (noise_scale_final - noise_scale_initial)

        for step in range(max_steps_per_episode):
            total_steps += 1

            # Action selection
            if total_steps < start_steps:
                # pure random actions initially
                action = np.random.uniform(act_low, act_high, size=act_dim)
            else:
                action = agent.select_action(obs, noise_scale=noise_scale)

            # Step environment
            time_step = env.step(action)
            next_obs = flatten_observation(time_step)
            reward = float(time_step.reward) if time_step.reward is not None else 0.0
            done = bool(time_step.last())

            replay_buffer.store(obs, action, reward, next_obs, float(done))

            ep_return += reward
            obs = next_obs

            # Learning updates
            if (total_steps >= update_after) and (total_steps % update_every == 0):
                for _ in range(update_every):
                    if replay_buffer.size >= batch_size:
                        agent.update(replay_buffer, batch_size=batch_size)

            if done:
                break

        episode_returns.append(ep_return)
        print(f"[Seed {seed}] Episode {ep+1}/{num_episodes} | Return: {ep_return:.2f}")

    return agent, np.array(episode_returns, dtype=np.float32)


###############################################
# 7. Evaluation on one seed (seed = 10 per PDF)
###############################################

def evaluate_agent(agent: DDPGAgent, seed=10, num_episodes=10, max_steps_per_episode=1000):
    set_seed(seed)
    env = make_env_cartpole_balance(seed)

    returns = []

    for ep in range(num_episodes):
        time_step = env.reset()
        obs = flatten_observation(time_step)
        ep_return = 0.0

        for step in range(max_steps_per_episode):
            # no exploration noise during evaluation
            action = agent.select_action(obs, noise_scale=0.0)
            time_step = env.step(action)
            next_obs = flatten_observation(time_step)
            reward = float(time_step.reward) if time_step.reward is not None else 0.0
            done = bool(time_step.last())

            ep_return += reward
            obs = next_obs

            if done:
                break

        returns.append(ep_return)
        print(f"[Eval seed {seed}] Episode {ep+1}/{num_episodes} | Return: {ep_return:.2f}")

    return np.array(returns, dtype=np.float32)


###############################################
# 8. Main: Train with seeds 0,1,2; Eval with seed 10
###############################################

def main():
    output_dir = "ddpg_results"
    os.makedirs(output_dir, exist_ok=True)

    train_seeds = [0, 1, 2]
    all_train_returns = []

    # ---- Training for each seed ----
    for seed in train_seeds:
        print(f"\n========== Training DDPG with seed {seed} ==========\n")
        agent, train_returns = train_ddpg_for_seed(seed)

        # save per-seed training returns
        train_path = os.path.join(output_dir, f"train_returns_seed{seed}.npy")
        np.save(train_path, train_returns)
        all_train_returns.append(train_returns)
        print(f"Saved training returns to {train_path}")

        # save model checkpoint for this seed
        model_path = os.path.join(output_dir, f"ddpg_cartpole_seed{seed}.pth")
        agent.save(model_path)
        print(f"Saved model to {model_path}")

    # ---- Compute mean ± std across seeds ----
    min_len = min(len(ret) for ret in all_train_returns)
    aligned = np.stack([ret[:min_len] for ret in all_train_returns], axis=0)  # shape: (3, min_len)
    mean_train = aligned.mean(axis=0)
    std_train = aligned.std(axis=0)
    np.save(os.path.join(output_dir, "train_mean.npy"), mean_train)
    np.save(os.path.join(output_dir, "train_std.npy"), std_train)
    print("\nSaved train_mean.npy and train_std.npy")

    # ---- Evaluation on seed 10 (per homework spec) ----
    # You can choose which seed's model to evaluate; here we reuse last 'agent',
    # but you could also reload a specific one from disk.
    print("\n========== Evaluating final agent on seed 10 ==========\n")
    eval_returns = evaluate_agent(agent, seed=10, num_episodes=10)
    np.save(os.path.join(output_dir, "eval_returns_seed10.npy"), eval_returns)
    print("Saved eval_returns_seed10.npy")

    print("\nAll done. Use these .npy files to plot mean ± std curves for your report.")


if __name__ == "__main__":
    main()
