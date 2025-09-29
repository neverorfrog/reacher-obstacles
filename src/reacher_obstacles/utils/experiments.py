from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np


@dataclass(frozen=True)
class TaskConfig:
    """
    Configuration for a reaching task.
    Defines target and obstacle positions.
    """
    name: str                                    # 'FT', 'FTO1', etc.
    target: Tuple[float, float]                 # (x, y) target position
    obstacles: Tuple[Tuple[float, float], ...] = field(default_factory=tuple)
    
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
    reward_heuristic: bool = False     # True if using reward heuristic (rhV)
    train_steps: int = 100_000         # Number of RL training steps
    
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
      obstacles=(),
   ),
   '1a': TaskConfig(
      name='1a',
      target=(0.15, 0.15),
      obstacles=((0.1, 0.05),),
   ),
   '1b': TaskConfig(
      name='1b',
      target=(0.15, 0.15),
      obstacles=((0.0, 0.1),),
   ),
   '1c': TaskConfig(
      name='1c',
      target=(0.15, 0.15),
      obstacles=((0.2, 0.1),),
   ),
   '2a': TaskConfig(
      name='2a',
      target=(0.2, 0.0),
      obstacles=((0.2, 0.2),(0.0, 0.2)),
   )
}


# =====================================================
# Experiment Registry
# =====================================================

EXPERIMENTS: Dict[str, ExperimentConfig] = {
   '0a': ExperimentConfig('0a', TASKS['0a']),
   '1a': ExperimentConfig('1a', TASKS['1a']),
   '1b': ExperimentConfig('1b', TASKS['1b']),
   '1c': ExperimentConfig('1c', TASKS['1c']),
   '2a': ExperimentConfig('2a', TASKS['2a']),
}


def get_experiment(exp_id: str) -> ExperimentConfig:
    """Get experiment by ID."""
    if exp_id not in EXPERIMENTS:
        available = ', '.join(EXPERIMENTS.keys())
        raise ValueError(f"Unknown experiment: {exp_id}. Available: {available}")
    return EXPERIMENTS[exp_id]