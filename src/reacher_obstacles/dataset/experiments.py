from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np


@dataclass(frozen=True)
class TaskConfig:
    """
    Configuration for a reaching task.
    Defines target and obstacle positions.
    """
    name: str
    target: Tuple[float, float]                 # (x, y) target position
    obstacles: List[Tuple[float, float]] = field(default_factory=list)
    init_qpos: np.ndarray = field(default_factory=lambda: np.array([-0.5, 0.0, 0.2]))
    
    @property
    def n_obstacles(self) -> int:
        return len(self.obstacles)
    
    def get_target_3d(self, z: float = 0.015) -> np.ndarray:
        """Get target as 3D array."""
        return np.array([self.target[0], self.target[1], z])
    
    def get_obstacles_3d(self, z: float = 0.015) -> List[np.ndarray]:
        """Get obstacles as list of 3D arrays."""
        return [np.array([x, y, z]) for x, y in self.obstacles]


@dataclass
class ExperimentConfig:
    """
    Complete configuration for a single Reacher3 experiment.
    """
    experiment_id: str                 # '1a', '2b', etc.
    task: TaskConfig                   # Target and obstacles
    train_steps: int                   # Number of training steps
    reward_heuristic: bool = False     # True if using reward heuristic (rhV)
    
    w_vel: float = 1e-5
    w_acc: float = 1e-5
    w_target: float = 1e-1 
    w_target_term: float = 5
    obs_distance_link1: float = 0.05
    obs_distance_link2: float = 0.05
    obs_distance_fingertip: float = 0.05
    
    @property
    def env_id(self) -> str:
        """Construct the environment ID."""
        base = f"Reacher3-v6_{self.task.name}"
        if self.reward_heuristic:
            base += "_rhV"
        return base
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'experiment_id': self.experiment_id,
            'task_name': self.task.name,
            'target': self.task.target,
            'obstacles': list(self.task.obstacles),
            'reward_heuristic': self.reward_heuristic,
            'train_steps': self.train_steps,
            'env_id': self.env_id,
            'n_obstacles': self.task.n_obstacles,
            'w_vel': self.w_vel,
            'w_acc': self.w_acc,
            'w_target': self.w_target,
            'w_target_term': self.w_target_term,
            'obs_distance_link1': self.obs_distance_link1,
            'obs_distance_link2': self.obs_distance_link2,
            'obs_distance_fingertip': self.obs_distance_fingertip,
        }
    
    def __str__(self) -> str:
        reward_str = "with heuristic" if self.reward_heuristic else "standard"
        return (
            f"Experiment {self.experiment_id}: {self.env_id}\n"
            f"  Target: {self.task.target}\n"
            f"  Obstacles: {self.task.n_obstacles}\n"
            f"  Reward: {reward_str}\n"
            f"  Training: {self.train_steps:,} steps"
        )


# =====================================================
# Task Configurations
# =====================================================

TASKS = {
   '0a': TaskConfig(
      name='0a',
      target=(-0.075, 0.185),
      obstacles=[],
   ),
   '1a': TaskConfig(
      name='1a',
      target=(-0.075, -0.185),
      obstacles=[(0.2, -0.15)],
      init_qpos=np.array([0.0, 0.0, 0.2])
   ),
   '2a': TaskConfig(
      name='2a',
      target=(-0.075, 0.185),
      obstacles=[(0.12,0.17),(0.0,-0.1)],
   ),
   '2b': TaskConfig(
        name='2b',
        target=(0.1, 0.25),
        obstacles=[(0.12,0.15),(0.12,-0.17)],
        init_qpos=np.array([0.0, 0.0, 0.2])
   ),   
   '3a': TaskConfig(
      name='3a',
      target=(-0.15, 0.17),
      obstacles=[(0.0,0.2),(-0.02,0.1),(0.12,0.17)],
   )
}


# =====================================================
# Experiment Registry
# =====================================================

EXPERIMENTS: Dict[str, ExperimentConfig] = {
    '0aa': ExperimentConfig(
        '0aa', TASKS['0a'], train_steps=150,
        w_vel=1e-5, w_acc=1e-5, w_target= 1e-1, w_target_term=5,
        obs_distance_link1=0.05, obs_distance_link2=0.05, obs_distance_fingertip=0.05
    ),
    '1aa': ExperimentConfig(
        '1aa', TASKS['1a'], train_steps=150,
        w_vel=1e-5, w_acc=1e-5, w_target= 1e-1, w_target_term=20,
        obs_distance_link1=0.05, obs_distance_link2=0.05, obs_distance_fingertip=0.05
    ),
    '2aa': ExperimentConfig(
         '2aa', TASKS['2a'], train_steps=150,
         w_vel=1e-5, w_acc=1e-5, w_target= 1e-1, w_target_term=50,
         obs_distance_link1=0.03, obs_distance_link2=0.03, obs_distance_fingertip=0.04
     ),
    '2ab': ExperimentConfig(
         '2ab', TASKS['2a'], train_steps=150,
         w_vel=1e-4, w_acc=1e-4, w_target= 1e-1, w_target_term=10,
         obs_distance_link1=0.05, obs_distance_link2=0.05, obs_distance_fingertip=0.05
     ),
    '2ba': ExperimentConfig(
         '2ba', TASKS['2b'], train_steps=150,
         w_vel=1e-5, w_acc=1e-5, w_target= 1e-1, w_target_term=50,
         obs_distance_link1=0.05, obs_distance_link2=0.05, obs_distance_fingertip=0.1
     ),
    '3aa': ExperimentConfig(
         '3aa', TASKS['3a'], train_steps=150,
         w_vel=1e-4, w_acc=1e-4, w_target= 1e-1, w_target_term=12,
         obs_distance_link1=0.05, obs_distance_link2=0.05, obs_distance_fingertip=0.05
     ),
}


def get_experiment(exp_id: str) -> ExperimentConfig:
    """Get experiment by ID."""
    if exp_id not in EXPERIMENTS:
        available = ', '.join(EXPERIMENTS.keys())
        raise ValueError(f"Unknown experiment: {exp_id}. Available: {available}")
    return EXPERIMENTS[exp_id]