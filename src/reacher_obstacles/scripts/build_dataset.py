# src/reacher_obstacles/scripts/trajopt_eval_dataset.py

import time
import mujoco
import numpy as np
from reacher_obstacles.utils import project_root, src_dir
from reacher_obstacles.trajopt.robot_model import RobotModel
from reacher_obstacles.trajopt.reacher_trajopt import ReacherTrajopt
from reacher_obstacles.dataset.experiments import EXPERIMENTS
from reacher_obstacles.dataset.builder import TrajectoryDatasetBuilder
from reacher_obstacles.utils.plot import plot
from reacher_obstacles.dataset.inspector import DatasetInspector
from dataclasses import dataclass
import tyro

builder = TrajectoryDatasetBuilder(output_dir=f"{project_root()}/data")
all_trajectories = []


@dataclass
class Args:
    plot: bool = False
    name: str = "trajopt_dataset"


def main(args: Args):
    for exp in EXPERIMENTS.values():
        nsteps = exp.train_steps
        expid = exp.experiment_id
        print(f"\n\nGenerating dataset for experiment {expid} with {nsteps} steps")
        target_pos = np.array([*exp.task.target, 0.015])
        obstacles_pos = list(exp.task.obstacles)

        # Solve trajectory optimization
        xml_path = f"{src_dir()}/envs/assets/reacher3.xml"
        robot = RobotModel(xml_path, target_pos, obstacles_pos, q0=exp.task.init_qpos)
        reacher_task = ReacherTrajopt(robot, exp)
        X, A, U = reacher_task.solve(robot.qpos)

        # Collect data
        collision = False
        mj_model = robot.mj_model
        mj_data = robot.mj_data
        qpos_traj = [mj_data.qpos[:robot.nq].copy()]
        qvel_traj = [mj_data.qvel[:robot.nq].copy()]
        qacc_traj = []
        torques_traj = []
        ee_pos_traj = [mj_data.body("fingertip").xpos.copy()]

        for i, torque in enumerate(U):
            robot.apply_torque(torque)
            mj_data.qpos[:robot.nq] = robot.qpos
            mujoco.mj_step(mj_model, mj_data)

            qpos_traj.append(mj_data.qpos[:robot.nq].copy())
            qvel_traj.append(mj_data.qvel[:robot.nq].copy())
            qacc_traj.append(mj_data.qacc[:robot.nq].copy())
            torques_traj.append(torque)
            ee_pos_traj.append(mj_data.body("fingertip").xpos.copy())

            # Check collisions
            for i in range(mj_data.ncon):
                contact = mj_data.contact[i]
                geom1 = mj_model.geom(contact.geom1).name
                geom2 = mj_model.geom(contact.geom2).name
                if "obs" in geom1 or "obs" in geom2:
                    print(f"Collision with obstacle at step {i}")
                    collision = True
                    break
            if collision:
                break

        # Convert to arrays
        qpos_traj = np.array(qpos_traj)
        qvel_traj = np.array(qvel_traj)
        qacc_traj = np.array(qacc_traj)
        torques_traj = np.array(torques_traj)
        ee_pos_traj = np.array(ee_pos_traj)

        metadata = {
            'experiment_config': exp.to_dict(),
            'method': 'trajopt',
            'seed': 0,
            'algorithm_params': {
                'solver': 'ipopt',
                'horizon': len(U),
            },
            'converged': True,
            'training_steps': exp.train_steps
        }

        metrics = builder.compute_metrics(
            qpos=qpos_traj,
            qvel=qvel_traj,
            qacc=qacc_traj,
            torques=torques_traj,
            ee_positions=ee_pos_traj,
            target_pos=target_pos,
            obstacle_positions=obstacles_pos,
            dt=robot.dt,
            metadata=metadata,
            collision=collision
        )

        # Save
        filename = f"trajopt_{expid}"
        builder.save_trajectory(metrics, filename)

        # Print summary
        print(f"\n=== TRAJECTORY SUMMARY FOR {expid} ===")
        print(f"Success: {metrics.success}")
        print(f"Final error: {metrics.final_error:.6f} m")
        print(f"Energy: {metrics.energy:.6f} Nm²·s")
        print(f"Min obstacle distance: {metrics.min_obstacle_distance:.6f} m")
        print(f"Execution time: {metrics.execution_time:.3f} s")
        if args.plot:
            plot(robot, U, exp)

        all_trajectories.append(filename)

    print("\n\n=== DATASET SUMMARY ===")
    builder.create_dataset(all_trajectories, dataset_name=args.name)
    inspector = DatasetInspector(f"{project_root()}/data/{args.name}.h5")
    inspector.print_summary()
    inspector.check_integrity()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
