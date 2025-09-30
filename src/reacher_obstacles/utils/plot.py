import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import mujoco
from reacher_obstacles.trajopt.robot_model import RobotModel
import time
from reacher_obstacles.dataset.experiments import ExperimentConfig

def plot(robot: RobotModel, U: np.ndarray, exp: ExperimentConfig):
    
    # Reset the robot to initial state
    mj_model: mujoco.MjModel = robot.mj_model
    mj_data: mujoco.MjData = robot.mj_data
    mj_data.qpos[:robot.nq] = exp.task.init_qpos
    mj_data.qvel[:robot.nq] = np.zeros(robot.nq)
    robot.qpos = mj_data.qpos[:robot.nq].copy()
    robot.qvel = mj_data.qvel[:robot.nq].copy()
    robot.qacc = np.zeros(robot.nv)
    robot.torque = np.zeros(robot.nv)
    mujoco.mj_forward(mj_model, mj_data)
    
    render_mode = "human"
    frames = []
    trajectory = []
    torques = []
    errors = []

    renderer = MujocoRenderer(mj_model, mj_data)
    collision = False
    for torque in U:
        robot.apply_torque(torque)
        mj_data.qpos[:robot.nq] = robot.qpos
        mujoco.mj_step(mj_model, mj_data)
        
        ee_pos = mj_data.body("fingertip").xpos[0:2]
        ee_pos = np.array([*ee_pos, 0.015])
        trajectory.append(ee_pos)
        errors.append(np.linalg.norm(ee_pos - exp.task.get_target_3d()))
        torques.append(torque)
        
        for i in range(mj_data.ncon):
            contact = mj_data.contact[i]
            geom1 = mj_model.geom(contact.geom1).name
            geom2 = mj_model.geom(contact.geom2).name
            if "obs" in geom1 or "obs" in geom2:
                print(f"Collision with obstacle")
                collision = True
                break
        if collision:
            break
        
    
        if render_mode == "human":
            pixels = renderer.render("human")
            time.sleep(0.1)
            frames.append(pixels)
            
    print(f"PINOCCHIO QPOS: {robot.qpos}")
    print(f"MUJOCO QPOS: {mj_data.qpos[:robot.nq]}\n")