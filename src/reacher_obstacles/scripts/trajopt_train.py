import argparse
import os
import time
import numpy as np

from reacher_obstacles.utils import project_root, src_dir
import numpy as np
import mujoco
from reacher_obstacles.trajopt.robot_model import RobotModel
from reacher_obstacles.trajopt.reacher_trajopt import ReacherTrajopt

from reacher_obstacles.utils.experiments import EXPERIMENTS
from reacher_obstacles.envs.reacher_v6 import CONFIGS

from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

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

qpos = np.array([-0.5, 0.0, 0.2])
robot = RobotModel(xml_path, target_pos, obstacles_pos, q0=qpos)
reacher_task = ReacherTrajopt(robot, nsteps, expid)
X, A, U = reacher_task.solve(robot.qpos)
os.makedirs(f"{project_root()}/trajectories", exist_ok=True)
np.savez(f"{project_root()}/trajectories/trajectory_{expid}.npz", X=X, A=A, U=U)

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
trajectory = []
velocities = []
accelerations = []
torques = []
errors = []

renderer = MujocoRenderer(mj_model, mj_data)
assert np.allclose(mj_data.qpos[:robot.nq], robot.qpos)
assert np.allclose(mj_data.qvel[:robot.nq], robot.qvel)

# print("\n=== JOINT LIMITS COMPARISON ===")
# for i in range(robot.nq):
#     print(f"Joint {i}:")
#     print(f"  MuJoCo:    [{mj_model.jnt_range[i, 0]:.3f}, {mj_model.jnt_range[i, 1]:.3f}]")
#     print(f"  Pinocchio: [{robot.pin_model.lowerPositionLimit[i]:.3f}, {robot.pin_model.upperPositionLimit[i]:.3f}]")

# # Apply zero torque for 100 steps - should stay at equilibrium
# for _ in range(100):
#     # MuJoCo
#     mj_data.ctrl[:] = 0
#     mujoco.mj_step(mj_model, mj_data)
    
#     # Pinocchio
#     robot.apply_torque(np.zeros(robot.nq))
    
#     print(f"Error: {np.linalg.norm(mj_data.qpos[:robot.nq] - robot.qpos)}")
    
# # Single small torque
# tau_test = np.array([0.1, 0, 0])
# errors = []

# for i in range(100):
#     # MuJoCo version
#     mj_q_before = mj_data.qpos[:robot.nq].copy()
#     mj_data.ctrl[:] = tau_test
#     mujoco.mj_step(mj_model, mj_data)
#     mj_q_after = mj_data.qpos[:robot.nq].copy()
    
#     # Pinocchio version
#     pin_q_before = robot.qpos.copy()
#     robot.apply_torque(tau_test)
#     pin_q_after = robot.qpos.copy()
    
#     error = np.linalg.norm(mj_q_after - pin_q_after)
#     errors.append(error)
    
#     if i % 10 == 0:
#         print(f"Step {i}: error={error:.6f}, mj_q={mj_q_after}, pin_q={pin_q_after}")

for torque in U:
    print(f"PINOCCHIO QPOS: {robot.qpos}")
    print(f"MUJOCO QPOS: {mj_data.qpos[:robot.nq]}\n")
    qacc, qvel, qpos = robot.apply_torque(torque)
    # mj_data.qpos[:robot.nq] = robot.qpos
    mj_data.ctrl[:robot.nu] = torque
    mujoco.mj_step(mj_model, mj_data)
    
    ee_pos = mj_data.body("fingertip").xpos[0:2]
    ee_pos = np.array([*ee_pos, 0.015])
    trajectory.append(ee_pos)
    errors.append(np.linalg.norm(ee_pos - target_pos))
    torques.append(torque)
    
    if render_mode == "human":
        pixels = renderer.render("human")
        time.sleep(0.1)
        for p in trajectory:
            renderer.viewer.add_marker(pos=p, size=0.005, label = "", rgba=[1, 1, 0, 1], type=2)
        frames.append(pixels)
  
  
# accelerations = np.array(accelerations)
# torques = np.array(torques)
# print(accelerations.shape)  

# # Plot the torques
# plt.figure()
# plt.plot(U)
# plt.title("Torques over time")

# plt.xlabel("Time step", fontdict={'size': 20})
# plt.xticks(fontsize=15)

# plt.ylabel("Torque [Nm]", fontdict={'size': 20})
# plt.yticks(fontsize=15)

# plt.legend([f"Joint {i+1}" for i in range(U.shape[1])], fontsize=13, loc='upper right')

# # plt.savefig(f"images/{expid}_trajopt.png", dpi=1000, bbox_inches='tight')
# # plt.show()

# print(f"ERRORS: {np.array(errors)}")
# acc_error = np.sum(errors) / len(errors)
# print(f"ACC ERROR: {acc_error}")
# print(f"SUMMED TORQUES: {np.sum(np.sum(torques ** 2, axis=1)) * 0.01}")
# renderer.close()