import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import mujoco
from reacher_obstacles.trajopt.robot_model import RobotModel
import time
from reacher_obstacles.utils.experiments import ExperimentConfig

def plot(robot: RobotModel, U: np.ndarray, exp: ExperimentConfig):
    
    # Reset the robot to initial state
    mj_model: mujoco.MjModel = robot.mj_model
    mj_data: mujoco.MjData = robot.mj_data
    mj_data.qpos[:robot.nq] = exp.task.init_qpos
    mj_data.qvel[:robot.nq] = np.zeros(robot.nq)
    mujoco.mj_forward(mj_model, mj_data)
    
    render_mode = "human"
    frames = []
    trajectory = []
    torques = []
    errors = []

    renderer = MujocoRenderer(mj_model, mj_data)
    # assert np.allclose(mj_data.qpos[:robot.nq], robot.qpos)
    # assert np.allclose(mj_data.qvel[:robot.nq], robot.qvel)

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
        errors.append(np.linalg.norm(ee_pos - exp.task.get_target_3d()))
        torques.append(torque)
    
        if render_mode == "human":
            pixels = renderer.render("human")
            time.sleep(0.1)
            for p in trajectory:
                renderer.viewer.add_marker(pos=p, size=0.005, label = "", rgba=[1, 1, 0, 1], type=2)
            frames.append(pixels)