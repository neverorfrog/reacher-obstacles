from typing import List, Tuple
import xml.etree.ElementTree as ET
import casadi
import numpy as np
import pinocchio as pin
import mujoco
import re
import copy
from numpy.linalg import norm, solve

class RobotModel:
    
    def __init__(self, xml_path: str, target_pos: np.ndarray, obstacle_pos: List[np.ndarray], final_ee_name: str = "fingertip", q0: np.ndarray = None):
        self.xml_path = xml_path
        self.target_pos = target_pos
        self.obstacle_pos = obstacle_pos
        self.final_ee_name = final_ee_name
        
        with open(xml_path, "r") as file:
            xml = file.read()
            
        # Overwrite the target and obstacle position in the XML file
        tree = copy.deepcopy(ET.ElementTree(ET.fromstring(xml)))
        root = tree.getroot()
        obstacle_pattern = re.compile(r"obstacle\d+")
        target_pattern = re.compile(r"target")

        for body in root.findall(".//body"):
            if obstacle_pattern.match(body.get("name", "")):
                obstacle_index = int(body.get("name").replace("obstacle", "")) - 1
                try:
                    new_obstacle_pos = np.array([*obstacle_pos[obstacle_index], 0.0])
                    self.obstacle_pos[obstacle_index] = new_obstacle_pos
                    body.set('pos', ' '.join(map(str, new_obstacle_pos)))
                except IndexError:
                    new_obstacle_pos = np.array([2.0, 0.0, 0.0])
                    body.set('pos', ' '.join(map(str, new_obstacle_pos)))
                
            if target_pattern.match(body.get("name", "")):
                body.set('pos', ' '.join(map(str, target_pos)))
        
        # Get the timestep
        option_element = root.find(".//option")
        self.dt = float(option_element.get("timestep"))
        
        # Extract mujoco model
        xml = ET.tostring(root, encoding="unicode")
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # Extract pinocchio model
        self.pin_robot: pin.RobotWrapper = pin.RobotWrapper()
        self.pin_robot.initFromMJCF(xml_path)
        self.pin_model: pin.Model = self.pin_robot.model
        self.pin_data: pin.Data = self.pin_robot.data
        self.nq = self.pin_model.nq
        self.nv = self.pin_model.nv
        self.nu = self.nv
        if q0 is None:
            q0 = pin.neutral(self.pin_model)
        self.pin_robot.q0 = q0
        self.pin_robot.a0 = np.zeros(self.nv)
        self.pin_robot.framesForwardKinematics(q0)
        
        # Extract end-effector frames ids
        ee_pattern = re.compile(r"body\d+")
        self.ee_id_list = []
        for body in root.findall(".//body"):
            if ee_pattern.match(body.get("name", "")):
                ee_name = body.get("name")
                ee_id = self.pin_model.getFrameId(ee_name)
                self.ee_id_list.append(ee_id)
        
        # Configuration
        self.qpos = q0
        self.qvel = np.zeros(self.nv)
        self.qacc = np.zeros(self.nv)
        
        # Frames
        self.final_ee_frame = self.pin_model.getFrameId(final_ee_name)
        joints = root.findall(".//joint")
        self.joint_names = [joint.get("name") for joint in joints if joint.get("type") == "hinge"]
        self.njoints = len(self.joint_names)
        self.joint_ids = [self.pin_model.getJointId(name) for name in self.joint_names]
        
        # Extract gear values
        self.gear_gains = np.zeros(self.njoints)
        for i, motor in enumerate(root.findall(".//motor")):
            self.gear_gains[i] = float(motor.get("gear"))
            
            
    def inverse_kinematics(self, target_pos: np.ndarray, q0: np.ndarray = None, eps: float = 1e-2, max_iter: int = 5000, damp: float = 1e-5):
        if q0 is None:
            q0 = self.qpos
        if len(target_pos) == 2:
            target_pos = np.array([*target_pos, 0.015])
            
        target_pose = pin.SE3(np.eye(3), target_pos)
            
        q = q0
        model = self.pin_model
        data = self.pin_data
        ee_id = self.final_ee_frame
        success = False
        
        for i in range(max_iter):
            self.pin_robot.forwardKinematics(q)
            iMw = data.oMi[3] # end effector frame expressed in world frame
            iMd = iMw.actInv(target_pose) # end effector frame expressed in target frame
            err = pin.log(iMd).vector # error in joint frame
            if norm(err) < eps:
                success = True
                break
            if i == max_iter - 1:
                raise ValueError("Failed to converge")
            
            J = self.pin_robot.computeJointJacobian(q, 3)
            J = -np.dot(pin.Jlog6(iMd.inverse()), J) # transformation into lie algebra (to be compliant with error)
            v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pin.integrate(model, q, v * self.dt)
                
        if success:
            print("Convergence reached!")
            print(q)
            
        return q
        
        
    def apply_torque(self, tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.qacc = pin.aba(self.pin_model, self.pin_data, self.qpos, self.qvel, np.array(tau) * self.gear_gains)  # Forward dynamics
        self.qvel += self.qacc * self.dt
        self.qpos = pin.integrate(self.pin_model, self.qpos, self.qvel * self.dt)
        self.qpos = np.clip(self.qpos, self.pin_model.lowerPositionLimit, self.pin_model.upperPositionLimit)
        self.pin_robot.framesForwardKinematics(self.qpos)
        
        return self.qacc, self.qvel, self.qpos
        
    @property
    def qpos_sup_limit(self):
        return self.pin_model.upperPositionLimit
    
    @property
    def qpos_inf_limit(self):
        return self.pin_model.lowerPositionLimit