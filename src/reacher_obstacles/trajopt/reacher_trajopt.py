import mujoco
import numpy as np
from reacher_obstacles.trajopt.robot_model import RobotModel
import casadi
from pinocchio import casadi as cpin
import pinocchio as pin
from reacher_obstacles.utils.experiments import ExperimentConfig

class ReacherTrajopt():
    
    def __init__(self, robot_model: RobotModel, exp: ExperimentConfig):
        self.robot_model = robot_model
        self.initialized = False
        self.expid = exp.experiment_id
        T = exp.train_steps
        self.T = T
        self.w_vel = exp.w_vel
        self.w_acc = exp.w_acc
        self.w_target = exp.w_target
        self.w_target_term = exp.w_target_term
        self.obs_distance_link1 = exp.obs_distance_link1
        self.obs_distance_link2 = exp.obs_distance_link2
        self.obs_distance_fingertip = exp.obs_distance_fingertip

        # Casadi Pinocchio Model
        self.cpin_model = cpin.Model(robot_model.pin_model)
        self.cpin_data = self.cpin_model.createData()
        self.nx = robot_model.nq + robot_model.nv
        
        self.damping = robot_model.damping
        self.stiffness = robot_model.stiffness
        self.qpos_spring = robot_model.qpos_spring
        self.frictionloss = robot_model.frictionloss
        
        # Casadi symbols
        cx = casadi.SX.sym("x", self.nx, 1)
        self.cq = cx[:robot_model.nq]
        self.cv = cx[robot_model.nq:]
        caq = casadi.SX.sym("a", robot_model.nv, 1)
        tauq = casadi.SX.sym("tau", robot_model.nv, 1)
        
        # Forward kinematics and dynamics
        cpin.forwardKinematics(self.cpin_model, self.cpin_data, self.cq, self.cv, caq)
        cpin.updateFramePlacements(self.cpin_model, self.cpin_data)
        
        # Compute gravity using CasADi Pinocchio
        cpin.computeGeneralizedGravity(self.cpin_model, self.cpin_data, self.cq)
        tau_gravity = self.cpin_data.g
        
        # Passive forces
        tau_passive = (
            - casadi.diag(self.damping) @ self.cv
            - casadi.diag(self.stiffness) @ (self.cq - self.qpos_spring)
            - casadi.diag(self.frictionloss) @ casadi.sign(self.cv)
        )
        
        # Total torque
        tau_total = robot_model.gear_gains * tauq + tau_gravity + tau_passive
        
        # Forward dynamics
        cpin.aba(self.cpin_model, self.cpin_data, self.cq, self.cv, tau_total)
        
        
        # Some functions needed in the casadi graph
        self.caba = casadi.Function("aba", [cx, tauq], [self.cpin_data.ddq])
        self.target_error = casadi.Function(
            "end_effector_error",
            [cx],
            [self.cpin_data.oMf[robot_model.final_ee_frame].translation - robot_model.target_pos],
        )
        
        self.cnext = casadi.Function(
            "next_state",
            [cx, caq],
            [
                casadi.vertcat(
                    casadi.fmin(
                        casadi.fmax(
                            cpin.integrate(self.cpin_model, cx[:robot_model.nq], cx[robot_model.nq:] * robot_model.dt),
                            robot_model.qpos_inf_limit
                        ),
                        robot_model.qpos_sup_limit
                    ),
                    cx[robot_model.nq:] + caq * robot_model.dt
                )
            ]
        )
                
        # Optimization problem
        self.opti = casadi.Opti()
        self.X = [self.opti.variable(self.nx) for t in range(T + 1)]
        self.Aq = [self.opti.variable(robot_model.nv) for t in range(T)]
        self.U = [self.opti.variable(robot_model.nv) for t in range(T)]
        
        print("\n=== INITIAL STATE COMPARISON ===")
        qpos_init = np.array([-0.5, 0.0, 0.2])

        # Pinocchio initial state
        robot_model.qpos = qpos_init.copy()
        robot_model.qvel = np.zeros(3)
        pin_qacc = pin.aba(robot_model.pin_model, robot_model.pin_data, robot_model.qpos, robot_model.qvel, np.zeros(3))
        print(f"Pinocchio q0: {robot_model.qpos}")
        print(f"Pinocchio qacc at rest (should be ~0 without gravity): {pin_qacc}")

        # MuJoCo initial state
        robot_model.mj_data.qpos[:robot_model.nq] = qpos_init.copy()
        robot_model.mj_data.qvel[:robot_model.nq] = np.zeros(3)
        robot_model.mj_data.ctrl[:] = np.zeros(3)
        mujoco.mj_forward(robot_model.mj_model, robot_model.mj_data)  # Compute accelerations
        print(f"MuJoCo q0: {robot_model.mj_data.qpos[:robot_model.nq]}")
        print(f"MuJoCo qacc at rest: {robot_model.mj_data.qacc[:robot_model.nq]}")

        # Check if they match
        print(f"Position difference: {np.linalg.norm(robot_model.qpos - robot_model.mj_data.qpos[:robot_model.nq])}")
        print(f"Acceleration difference: {np.linalg.norm(pin_qacc - robot_model.mj_data.qacc[:robot_model.nq])}")
                
    
    def solve(self, q0: np.ndarray):
        self._set_initial_constraint(q0)
        cost = self._set_cost()
        self._set_constraints()
        self.opti.minimize(cost)
        self.opti.solver("ipopt")
        p_opts, s_opts = {"ipopt.print_level": 4, "ipopt.tol": 1e-4, "ipopt.max_iter": 500_000, "expand": True}, {}
        self.opti.solver("ipopt", p_opts, s_opts)
        
        try:
            self.opti.solve()
            sol_X = [self.opti.value(var_x) for var_x in self.X]
            sol_Aq = [self.opti.value(var_a) for var_a in self.Aq]
            sol_U = [self.opti.value(var_u) for var_u in self.U]
            print("COST:", self.opti.value(cost))
        except:
            raise Exception("ERROR in convergence")
        
        return sol_X, sol_Aq, sol_U
        
        
    def _set_cost(self) -> float:
        cost = 0
        
        for t in range(self.T):
            cost += self.w_vel * self.robot_model.dt * casadi.sumsqr(self.X[t][self.robot_model.nq:])
            cost += self.w_acc * self.robot_model.dt * casadi.sumsqr(self.Aq[t])
            cost += self.w_target * casadi.sumsqr(self.target_error(self.X[t]))
                
        cost += self.w_target_term * casadi.sumsqr(self.target_error(self.X[self.T]))
            
        return cost
        
    def _set_constraints(self):
        for t in range(self.T):
            self.opti.subject_to(self.cnext(self.X[t], self.Aq[t]) == self.X[t + 1])
            self.opti.subject_to(self.caba(self.X[t], self.U[t]) == self.Aq[t])
            self.opti.subject_to(self.U[t] <= 1.0)
            self.opti.subject_to(self.U[t] >= -1.0)
            
            for obstacle_idx in range(len(self.robot_model.obstacle_pos)):
                self.opti.subject_to(self._ee_obstacle_distance(self.robot_model.pin_model.getFrameId("body1"), obstacle_idx)(self.X[t][:self.robot_model.nq]) >= self.obs_distance_link1)
                self.opti.subject_to(self._ee_obstacle_distance(self.robot_model.pin_model.getFrameId("body2"), obstacle_idx)(self.X[t][:self.robot_model.nq]) >= self.obs_distance_link2)
                self.opti.subject_to(self._ee_obstacle_distance(self.robot_model.pin_model.getFrameId("fingertip"), obstacle_idx)(self.X[t][:self.robot_model.nq]) >= self.obs_distance_fingertip)

    def _ee_obstacle_distance(self, ee_frame_idx: int, obstacle_idx: int):
        obstacle_pos = self.robot_model.obstacle_pos[obstacle_idx]
        
        return casadi.Function(
            "obstacle_distance",
            [self.cq],
            [casadi.norm_2(self.cpin_data.oMf[ee_frame_idx].translation[:2] - obstacle_pos[:2])],
        )
        
    def _set_initial_constraint(self, q0: np.ndarray):
        self.opti.subject_to(self.X[0][:self.robot_model.nq] == q0)
        self.opti.subject_to(self.X[0][self.robot_model.nq:] == 0)       # zero initial velocity
        self.opti.subject_to(self.X[self.T][self.robot_model.nq:] == 0)  # zero terminal velocity
        self.initialized = True
    