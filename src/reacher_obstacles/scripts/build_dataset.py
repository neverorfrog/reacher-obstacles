# src/reacher_obstacles/scripts/trajopt_eval_dataset.py

import argparse
import os
import time
import numpy as np
import mujoco

from reacher_obstacles.utils import project_root, src_dir
from reacher_obstacles.trajopt.robot_model import RobotModel
from reacher_obstacles.utils.experiments import EXPERIMENTS
from reacher_obstacles.envs.reacher_v6 import CONFIGS
from reacher_obstacles.dataset.builder import TrajectoryDatasetBuilder
from reacher_obstacles.utils.experiments import get_experiment

# Parsing
parser = argparse.ArgumentParser()
parser.add_argument("expid", type=str, help="Experiment id")
args = parser.parse_args()
expid = args.expid

# Experiment config
exp = get_experiment(expid)
target_pos = np.array([*exp.task.target, 0.015])
obstacles_pos = [np.array([*obs, 0.015]) for obs in exp.task.obstacles]

# Solve trajectory optimization




# # Collect data
# qpos_traj = [mj_data.qpos[:robot.nq].copy()]
# qvel_traj = [mj_data.qvel[:robot.nq].copy()]
# qacc_traj = []
# torques_traj = []
# ee_pos_traj = [mj_data.body("fingertip").xpos.copy()]
# collision_events = []

# start_time = time.time()
# for i, torque in enumerate(U):
#     mj_data.ctrl[:] = torque
#     mujoco.mj_step(mj_model, mj_data)
    
#     qpos_traj.append(mj_data.qpos[:robot.nq].copy())
#     qvel_traj.append(mj_data.qvel[:robot.nq].copy())
#     qacc_traj.append(mj_data.qacc[:robot.nq].copy())
#     torques_traj.append(torque)
#     ee_pos_traj.append(mj_data.body("fingertip").xpos.copy())
    
#     # Check collisions
#     if mj_data.ncon > 0:
#         collision_events.append(i)

# sim_time = time.time() - start_time

# # Convert to arrays
# qpos_traj = np.array(qpos_traj)
# qvel_traj = np.array(qvel_traj)
# qacc_traj = np.array(qacc_traj)
# torques_traj = np.array(torques_traj)
# ee_pos_traj = np.array(ee_pos_traj)

# # Build dataset
# builder = TrajectoryDatasetBuilder(output_dir=f"{project_root()}/datasets")

# metadata = {
#     'experiment_id': expid,
#     'config': config,
#     'method': 'trajopt',
#     'seed': 0,
#     'algorithm_params': {
#         'solver': 'ipopt',
#         'horizon': len(U),
#     },
#     'optimization_time': None,  # Could load from trajopt_train logs
#     'converged': True,
# }

# metrics = builder.compute_metrics(
#     qpos=qpos_traj,
#     qvel=qvel_traj,
#     qacc=qacc_traj,
#     torques=torques_traj,
#     ee_positions=ee_pos_traj,
#     target_pos=target_pos,
#     obstacle_positions=obstacles_pos,
#     dt=robot.dt,
#     metadata=metadata,
#     collision_events=collision_events,
# )

# # Save
# filename = f"trajopt_{expid}_seed0"
# builder.save_trajectory(metrics, filename)

# # Print summary
# print(f"\n=== TRAJECTORY SUMMARY ===")
# print(f"Success: {metrics.success}")
# print(f"Failure reason: {metrics.failure_reason}")
# print(f"Final error: {metrics.final_error:.6f} m")
# print(f"Energy: {metrics.energy:.6f} Nm²·s")
# print(f"Min obstacle distance: {metrics.min_obstacle_distance:.6f} m")
# print(f"Execution time: {metrics.execution_time:.3f} s")