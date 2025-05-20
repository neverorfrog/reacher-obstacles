import os, os.path, time, argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

import matplotlib.pyplot as plt
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

args = parser.parse_args()

expid = args.expid
seed=args.seed

envid=EXPERIMENTS[expid]['envid']
log_name = f"{envid};{seed}"
model_file = f"models/{envid};{seed};{args.algo}"
envid: str = EXPERIMENTS[expid]['envid']
config: str = envid.split('_')[1]
target_pos = CONFIGS[config]['target']
target_pos = np.array([*target_pos, 0.015])
train_env = gym.make(envid)

print(f"Observation: {train_env.observation_space}")
print(f"Action: {train_env.action_space}")
print(f"MODEL FILE: {model_file}.pth")

# load or create model
if os.path.isfile(model_file+".pth"):
    model = model_class.load(model_file+".pth", train_env)
    if issubclass(model_class, OffPolicyAlgorithm):
        model.load_replay_buffer(model_file+"_rb.pth")
    print(f"Model loaded from file timesteps: {model.num_timesteps}")
    new_model = False
else:
    raise Exception("Model not found")
     
# Play the learned policy
render_mode="human"
test_env = gym.make(envid, render_mode=render_mode)
trajectory = []
errors = []
torques = []
accelerations = []
observation, info = test_env.reset(seed=seed)

for i in range(200):
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = test_env.step(action)
    ee_pos = test_env.unwrapped.data.body("fingertip").xpos[0:test_env.unwrapped.ndim]
    ee_pos = np.array([*ee_pos, 0.015])
    trajectory.append(ee_pos)
    error = np.linalg.norm(ee_pos - target_pos)
    errors.append(error)
    torques.append(action)
    accelerations.append(test_env.unwrapped.data.qacc[:3])
    print(f"ACCELERATION: {test_env.unwrapped.data.qacc[:3]}")
    print(f"TORQUE: {action}\n") 
    
    if render_mode=="human":
        for p in trajectory:
            test_env.unwrapped.mujoco_renderer.viewer.add_marker(pos=p, size=0.005, label = "", rgba=[1, 1, 0, 1], type=2)
        test_env.render()
        time.sleep(0.1)

    if terminated or truncated:
        print(info)
        break
    
accelerations = np.array(accelerations)
torques = np.array(torques)
print(accelerations.shape)

# Plot the torques
U = np.array(torques)

plt.figure()
plt.plot(U)
plt.title("Torques over time")

plt.xlabel("Time step", fontdict={'size': 20})
plt.xticks(fontsize=15)

plt.ylabel("Torque [Nm]", fontdict={'size': 20})
plt.yticks(fontsize=15)

plt.legend([f"Joint {i+1}" for i in range(U.shape[1])], fontsize=13, loc='upper right')

plt.savefig(f"images/{expid}_rl.png", dpi=1000, bbox_inches='tight')
# plt.show()

print(f"ERRORS: {np.array(errors)}")
print(f"ACC ERROR: {np.sum(errors) / len(errors)}")
print(f"SUMMED TORQUES: {np.sum(np.sum(torques ** 2, axis=1)) * 0.01}")
test_env.close()

