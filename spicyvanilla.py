import gymnasium as gym
from stable_baselines3 import PPO
import torch.nn as nn
import numpy as np
import torch
import mujoco
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from scipy.stats import exponnorm

"""
DEFAULT
# self._ctrl_cost_weigh = 0.1
# self._contact_cost_weight = 5e-7
# self._contact_cost_range = (-np.inf, 10.0)
# self._healthy_z_range = (1.0, 2.0)
# self._healthy_reward = 5.0
# self._forward_reward_weight = 1.25
# self._reset_noise_scale = 1e-2
# self._exclude_current_positions_from_observation = (True)


n_steps: int = 2048
batch_size: int = 64
n_epochs: int = 10
"""

# Callback to add noise to the policy parameters at the end of each rollout
class ParamNoiseCallback(BaseCallback):
    def __init__(self, noise_std, verbose=0):
        super(ParamNoiseCallback, self).__init__(verbose)
        self.noise_std = noise_std

    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        # Add Gaussian noise to each parameter of the policy
        with torch.no_grad():
            for name, param in self.model.policy.named_parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * self.noise_std
                    param.add_(noise)
        if self.verbose > 0:
            print("Parameter noise added after rollout.")

# A custom callback to create a live-updating matplotlib plot of training metrics.
class LivePlotCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LivePlotCallback, self).__init__(verbose)
        # Create the figure and 2x2 subplots:
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 10))
        plt.ion()  # Turn interactive mode on
        self.fig.show()
        self.fig.canvas.draw()

        # Lists to store metrics over time
        self.train_steps = []             # for training update metrics
        self.explained_variances = []
        self.clip_fractions = []

        self.rollout_indices = []         # for episode-level metrics
        self.ep_rewards = []
        self.ep_lengths = []

    # Minimal implementation of the abstract method _on_step
    def _on_step(self) -> bool:
        # This callback does not need to take any action on every step.
        return True

    def on_train_batch_end(self, _locals, _globals):
        """
        Called after each gradient update.
        We try to grab the training metrics (if they are computed by PPO).
        """
        explained_variance = _locals.get("explained_variance", None)
        clip_fraction = _locals.get("clip_fraction", None)
        if explained_variance is not None:
            step = len(self.train_steps) + 1
            self.train_steps.append(step)
            self.explained_variances.append(explained_variance)
            self.clip_fractions.append(clip_fraction if clip_fraction is not None else np.nan)
            self._update_training_plots()

    def on_rollout_end(self):
        """
        Called at the end of each rollout (after data collection).
        Here we compute the average episode reward and length from the Monitor info.
        """
        if len(self.model.ep_info_buffer) > 0:
            rewards = [info.get("r", 0) for info in self.model.ep_info_buffer]
            lengths = [info.get("l", 0) for info in self.model.ep_info_buffer]
            mean_reward = np.mean(rewards)
            mean_length = np.mean(lengths)
        else:
            mean_reward = np.nan
            mean_length = np.nan

        index = len(self.rollout_indices) + 1
        self.rollout_indices.append(index)
        self.ep_rewards.append(mean_reward)
        self.ep_lengths.append(mean_length)
        self._update_episode_plots()

    def _update_training_plots(self):
        """Update the subplots for training metrics: explained variance and clip fraction."""
        # Bottom-left: explained variance
        ax_ev = self.axs[1, 0]
        ax_ev.clear()
        ax_ev.plot(self.train_steps, self.explained_variances, label="Explained Variance")
        ax_ev.set_title("Explained Variance")
        ax_ev.legend()

        # Bottom-right: clip fraction
        ax_cf = self.axs[1, 1]
        ax_cf.clear()
        ax_cf.plot(self.train_steps, self.clip_fractions, label="Clip Fraction")
        ax_cf.set_title("Clip Fraction")
        ax_cf.legend()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update_episode_plots(self):
        """Update the subplots for episode metrics: reward and episode length."""
        # Top-right: episode reward mean
        ax_rew = self.axs[0, 1]
        ax_rew.clear()
        ax_rew.plot(self.rollout_indices, self.ep_rewards, label="Episode Reward Mean")
        ax_rew.set_title("Episode Reward Mean")
        ax_rew.legend()

        # Top-left: episode length mean
        ax_len = self.axs[0, 0]
        ax_len.clear()
        ax_len.plot(self.rollout_indices, self.ep_lengths, label="Episode Length Mean")
        ax_len.set_title("Episode Length Mean")
        ax_len.legend()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# A custom reward wrapper (unchanged from your code)
class CustomReward(gym.Wrapper):

    def __init__(self, env, torso_z_min):
        super().__init__(env)
        self.z_min = torso_z_min
        # Initialize last_x_velocity to 0.0 (or None if you prefer a conditional check)
        self.last_x_velocity = 0.0

    def reset(self, **kwargs):
        # When resetting the environment, also reset the last_x_velocity
        # Note: gymnasium's reset returns (obs, info)
        obs, info = self.env.reset(**kwargs)
        self.last_x_velocity = 0.0  # or you can set this to info["x_velocity"] if available
        return obs, info

    def step(self, action):
        # Step the environment and get the info dictionary
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Get the simulation timestep (dt) from the MuJoCo model options.
        # This assumes you are using a MuJoCo-based environment.
        dt = self.unwrapped.model.opt.timestep
        
        # Calculate x acceleration as the change in x_velocity divided by dt.
        x_acceleration = (info["x_velocity"] - self.last_x_velocity) / dt
        self.last_x_velocity = info["x_velocity"]

        # Continue with your existing reward modifications.
        body_id = mujoco.mj_name2id(
            self.env.unwrapped.model,
            mujoco.mjtObj.mjOBJ_BODY.value,
            "pelvis"
        )
        quat = self.env.unwrapped.data.xquat[body_id]
        pelvis_tilt = np.linalg.norm(quat[1:])

        qpos = self.unwrapped.data.qpos
        torso_z = qpos[2]

        survive = info["reward_survive"]
        forward = info["reward_forward"]
        ctrl = info["reward_ctrl"]
        contact = info["reward_contact"]
        x_position = info["x_position"]

        # For example, you might choose to define the reward as:
        reward = (0 * survive + 10 * forward)
        reward += 10 * x_position
        reward += 10 * (0.3 - pelvis_tilt)
        reward += 100 * exponnorm.pdf(x_acceleration, K=2, loc=15, scale=4)
    
        # Optionally, override the done flag if torso_z is above your minimum
        if torso_z >= self.z_min:
            done = False

        return obs, reward, done, truncated, info

def train_humanoid():
    env = gym.make("Humanoid-v5", 
                   render_mode="human",
                   reset_noise_scale=1e-2)
    
    env = CustomReward(env, torso_z_min=0.85)

    # Instantiate your callbacks
    live_plot_callback = LivePlotCallback()
    callback = CallbackList([live_plot_callback])

    # Uncomment the following line if you wish to add the parameter noise callback
    #param_noise_callback = ParamNoiseCallback(noise_std=5e-4, verbose=1)
    #callback = CallbackList([live_plot_callback, param_noise_callback])
    # Combine callbacks into a CallbackList (you can add more if needed)
    

    
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={
            "net_arch": [512, 512, 256],
            "activation_fn": nn.LeakyReLU  # Changing activation function
        },
        verbose=1,
        learning_rate=1e-4, 
        n_steps=2048,
        batch_size=256,
        ent_coef=0.3,
        clip_range=0.4,
        vf_coef=0.5, 
        gamma=0.99,
        gae_lambda=0.97,
        tensorboard_log="./logs/"
    )
    
    try:
        model.learn(total_timesteps=1_500_000, callback=callback)
        model.save("spicy_walker2")
        print("training complete!")
    except KeyboardInterrupt:
        model.save("spicy_walker2")
        print("midpoint_saved")
    finally:
        env.close()
        plt.ioff()  # Turn off interactive mode when done
        plt.show()  # Keep the final plot visible

if __name__ == "__main__":
    train_humanoid()