import argparse
import os
import numpy as np

from reacher_obstacles.utils import project_root, src_dir
import numpy as np
import mujoco
from reacher_obstacles.trajopt.robot_model import RobotModel
from reacher_obstacles.trajopt.reacher_trajopt import ReacherTrajopt

from reacher_obstacles.utils.experiments import EXPERIMENTS
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
reacher_task = ReacherTrajopt(robot, nsteps)

X, A, U = reacher_task.solve(robot.qpos)
os.makedirs(f"{project_root()}/trajectories", exist_ok=True)
np.savez(f"{project_root()}/trajectories/trajectory_{expid}.npz", X=X, A=A, U=U)