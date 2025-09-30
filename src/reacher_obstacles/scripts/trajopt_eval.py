import argparse
import os
import sys
import time
import casadi
import numpy as np

from reacher_obstacles.utils import project_root, src_dir
import numpy as np
import mujoco
from reacher_obstacles.trajopt.robot_model import RobotModel
import matplotlib.pyplot as plt

from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from reacher_obstacles.dataset.experiments import EXPERIMENTS
from reacher_obstacles.envs.reacher_v6 import CONFIGS

xml_path = f"{src_dir()}/envs/assets/reacher3.xml"
parser = argparse.ArgumentParser()

parser.add_argument(
    "expid",
    type=str,
    default="1a",
    help="Experiment id",
)

parser.add_argument(
    "--nsteps",
    type=int,
    default=100,
    help="Number of steps",
)

parser.add_argument(
   "--force-training",
   action="store_true",
   help="Force training",
)

args = parser.parse_args()
expid = args.expid
nsteps = args.nsteps
envid: str = EXPERIMENTS[expid]['envid']
config: str = envid.split('_')[1]
target_pos = CONFIGS[config]['target']
target_pos = np.array([*target_pos, 0.015])

try:
    obstacles_pos = CONFIGS[config]['obstacles']
except KeyError:
    obstacles_pos = []

print(f"Experiment: {expid}")
print(f"Target position: {target_pos}")
print(f"Obstacles position: {obstacles_pos}")

robot = RobotModel(xml_path, target_pos, obstacles_pos)

if os.path.isfile(f"{project_root()}/trajectories/trajectory_{expid}.npz"):
    print("FOUND TRAJECTORY")
    T = np.load(f"{project_root()}/trajectories/trajectory_{expid}.npz")
    X = T['X']
    A = T['A']
    U = T['U']
else:
    raise FileNotFoundError("Trajectory not found")

mj_model = robot.mj_model
mj_data = robot.mj_data

# Simulate and display video.
render_mode = "human"
frames = []
mujoco.mj_resetData(mj_model, mj_data)  # Reset state and time.
trajectory = []
velocities = []
accelerations = []
torques = []
errors = []

renderer = MujocoRenderer(mj_model, mj_data)

for torque in U:
    sim_q = mj_data.qpos[:robot.nq]
    # print(f"PINOCCHIO QPOS: {robot.qpos}")
    # print(f"MUJOCO QPOS: {sim_q}\n")
    qacc, qvel, qpos = robot.apply_torque(torque)
    mj_data.qpos[:robot.nq] = robot.qpos
    mujoco.mj_step(mj_model, mj_data)
    
    ee_pos = mj_data.body("fingertip").xpos[0:2]
    ee_pos = np.array([*ee_pos, 0.015])
    trajectory.append(ee_pos)
    errors.append(np.linalg.norm(ee_pos - target_pos))
    accelerations.append(qacc)
    torques.append(torque)
    
    print(f"ACCERATION: {qacc}")
    print(f"TORQUE: {torque}\n")
    
    if render_mode == "human":
        pixels = renderer.render("human")
        time.sleep(0.1)
        for p in trajectory:
            renderer.viewer.add_marker(pos=p, size=0.005, label = "", rgba=[1, 1, 0, 1], type=2)
        frames.append(pixels)
  
  
accelerations = np.array(accelerations)
torques = np.array(torques)
print(accelerations.shape)  

# Plot the torques
plt.figure()
plt.plot(U)
plt.title("Torques over time")

plt.xlabel("Time step", fontdict={'size': 20})
plt.xticks(fontsize=15)

plt.ylabel("Torque [Nm]", fontdict={'size': 20})
plt.yticks(fontsize=15)

plt.legend([f"Joint {i+1}" for i in range(U.shape[1])], fontsize=13, loc='upper right')

plt.savefig(f"images/{expid}_trajopt.png", dpi=1000, bbox_inches='tight')
# plt.show()

print(f"ERRORS: {np.array(errors)}")
acc_error = np.sum(errors) / len(errors)
print(f"ACC ERROR: {acc_error}")
print(f"SUMMED TORQUES: {np.sum(np.sum(torques ** 2, axis=1)) * 0.01}")
renderer.close()