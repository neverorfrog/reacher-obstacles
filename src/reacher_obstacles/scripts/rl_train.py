import os, os.path, argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import wandb

import reacher_obstacles.envs
from reacher_obstacles.envs.reacher_v6 import CONFIGS
from reacher_obstacles.utils.experiments import EXPERIMENTS

os.makedirs("models", exist_ok=True)  # Create dirs
os.makedirs("log", exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--algo",
    type=str,
    default="SAC",
    help="Algorithm",
)

assert(parser.parse_known_args()[0].algo in ["SAC", "PPO"])
model_class = SAC if parser.parse_known_args()[0].algo == "SAC" else PPO

parser.add_argument(
    "expid",
    type=str,
    default="1a",
    help="Experiment id",
)

parser.add_argument(
    "--seed",
    type=int,
    default=10,
    help="Seed",
)

parser.add_argument(
    "--steps",
    type=float,
    default=None,
    help="Number of train steps",
)

args = parser.parse_args()

if __name__ == "__main__":
    expid = args.expid
    seed = args.seed

    envid = EXPERIMENTS[expid]['envid']
    train_steps = int(EXPERIMENTS[expid]['train_steps'])

    if args.steps is not None:
        train_steps = int(args.steps)

    log_name = f"{envid};{seed}"
    model_file = f"models/{envid};{seed};{args.algo}"
    envid: str = EXPERIMENTS[expid]['envid']
    config: str = envid.split('_')[1]
    target_pos = CONFIGS[config]['target']
    target_pos = np.array([*target_pos, 0.015])
    
    try:
        obstacles_pos = CONFIGS[config]['obstacles']
    except KeyError:
        obstacles_pos = []
        
    train_env = make_vec_env(envid, n_envs=8, vec_env_cls=SubprocVecEnv)
    # train_env = gym.make(envid)

    print(f"Observation: {train_env.observation_space}")
    print(f"Action: {train_env.action_space}")

    # load or create model
    if issubclass(model_class, OnPolicyAlgorithm):
        model = model_class("MlpPolicy", 
            train_env, 
            seed = seed, 
            gamma = 0.99,
            learning_rate = 1e-3,
            use_sde = False,
            verbose=1)
    elif issubclass(model_class, OffPolicyAlgorithm):
        model = model_class("MlpPolicy", 
                train_env, 
                seed = seed, 
                gamma = 0.9,
                learning_rate = 1e-3,
                use_sde = False,
                train_freq = 10,   # steps
                gradient_steps=4,
                verbose=1)
        
    wandb.init(project="reacher_obstacles", config={
        "env_id": envid,
        "train_steps": train_steps,
        "seed": seed,
        "algorithm": args.algo
    })

    class WandbCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(WandbCallback, self).__init__(verbose)

        def _on_step(self) -> bool:
            wandb.log({
                "timesteps": self.num_timesteps,
                "reward": self.locals["rewards"]
            })
            return True
        
    steps_to_train = train_steps - model.num_timesteps

    try:
        print(f"Training {envid};{seed} -> {model.num_timesteps}...{train_steps}")
        model.learn(total_timesteps = steps_to_train,
                    callback = WandbCallback(),
                    log_interval = 100,
                    reset_num_timesteps = True,
                    tb_log_name = log_name,
        )
    except KeyboardInterrupt:
        print("User QUIT")

    model.save(model_file+".pth")
    if issubclass(model_class, OffPolicyAlgorithm):
        model.save_replay_buffer(model_file+"_rb.pth")
    print(f"Saved {model_file} timesteps: {model.num_timesteps}")

    train_env.close()