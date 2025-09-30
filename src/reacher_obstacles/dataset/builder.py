# src/reacher_obstacles/dataset/builder.py

import numpy as np
import h5py
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json

@dataclass
class TrajectoryMetrics:
    """Container for all trajectory metrics."""
    
    # Temporal data
    qpos: np.ndarray
    qvel: np.ndarray
    qacc: np.ndarray
    torques: np.ndarray
    ee_pos: np.ndarray
    ee_vel: np.ndarray
    timestamps: np.ndarray
    
    # Success
    success: bool
    goal_reached_step: int
    
    # Safety
    min_obstacle_distance: float
    mean_obstacle_distance: float
    obstacle_distances: np.ndarray
    collision: bool
    safety_margin_violations: int
    
    # Performance
    execution_time: float
    energy: float
    smoothness: float
    path_length: float
    final_error: float
    mean_error: float
    settling_time: float
    max_velocity: float
    max_acceleration: float
    control_effort: float
    control_variation: float
    
    # Metadata
    experiment_config: Dict[str, Any]
    method: str
    seed: int
    target_pos: np.ndarray
    obstacle_pos: np.ndarray
    initial_qpos: np.ndarray
    njoints: int
    dt: float
    

class TrajectoryDatasetBuilder:
    """Build trajectory datasets from simulation results."""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.trajectories_dir = self.output_dir / "trajectories"
        self.trajectories_dir.mkdir(exist_ok=True)
        
    def compute_metrics(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
        torques: np.ndarray,
        ee_positions: np.ndarray,
        target_pos: np.ndarray,
        obstacle_positions: np.ndarray,
        dt: float,
        metadata: Dict[str, Any],
        collision: bool,
        dist_tolerance: float = 0.05,
        safety_margin: float = 0.06,
    ) -> TrajectoryMetrics:
        """
        Compute all metrics from raw trajectory data.
        
        Args:
            qpos: (T+1, nq) joint positions
            qvel: (T+1, nq) joint velocities
            qacc: (T, nq) joint accelerations
            torques: (T, nq) control torques
            ee_positions: (T+1, 3) end-effector positions
            target_pos: (3,) target position
            obstacle_positions: (n_obs, 3) obstacle positions
            dt: timestep
            metadata: dict with experiment info
            collision: whether a collision occurred
            dist_tolerance: distance threshold for success
            safety_margin: minimum safe distance from obstacles
        """
        T = len(torques)
        nq = qpos.shape[1]
        
        # Timestamps
        timestamps = np.arange(T + 1) * dt
        
        # === Success Metrics ===
        errors = np.linalg.norm(ee_positions - target_pos, axis=1)
        final_error = errors[-1]
        mean_error = np.mean(errors)
        
        # Success: final error below tolerance AND no collisions
        success = (final_error < dist_tolerance) and (not collision)
        
        # Goal reached step
        goal_reached = np.where(errors < dist_tolerance)[0]
        goal_reached_step = int(goal_reached[0]) if len(goal_reached) > 0 else -1
        
        # Settling time (time to stay within tolerance)
        settling_time = -1.0
        if goal_reached_step >= 0:
            # Check if it stays within tolerance
            if np.all(errors[goal_reached_step:] < dist_tolerance):
                settling_time = timestamps[goal_reached_step]
        
        # === Safety Metrics ===
        obstacle_distances = np.zeros((T + 1, len(obstacle_positions)))
        for t in range(T + 1):
            for i, obs_pos in enumerate(obstacle_positions):
                obstacle_distances[t, i] = np.linalg.norm(ee_positions[t, :2] - obs_pos[:2])
        
        if len(obstacle_positions) > 0:
            min_obstacle_distance = np.min(obstacle_distances)
            mean_obstacle_distance = np.mean(obstacle_distances)
            safety_margin_violations = np.sum(obstacle_distances < safety_margin)
        else:
            min_obstacle_distance = np.inf
            mean_obstacle_distance = np.inf
            safety_margin_violations = 0
        
        # === Performance Metrics ===
        execution_time = timestamps[-1]
        
        # Energy: sum of squared torques
        energy = np.sum(np.sum(torques**2, axis=1)) * dt
        
        # Control effort: L1 norm
        control_effort = np.sum(np.abs(torques)) * dt
        
        # Control variation (smoothness)
        if T > 1:
            dtorques = np.diff(torques, axis=0) / dt
            control_variation = np.sum(np.sum(dtorques**2, axis=1)) * dt
        else:
            control_variation = 0.0
        
        # Smoothness: sum of squared accelerations
        smoothness = np.sum(np.sum(qacc**2, axis=1)) * dt
        
        # Path length
        path_length = np.sum(np.linalg.norm(np.diff(ee_positions, axis=0), axis=1))
        
        # Kinematics
        max_velocity = np.max(np.abs(qvel))
        max_acceleration = np.max(np.abs(qacc))
        
        # End-effector velocity
        ee_vel = np.zeros_like(ee_positions)
        ee_vel[1:] = np.diff(ee_positions, axis=0) / dt
        
        # Create metrics object
        metrics = TrajectoryMetrics(
            # Temporal
            qpos=qpos,
            qvel=qvel,
            qacc=qacc,
            torques=torques,
            ee_pos=ee_positions,
            ee_vel=ee_vel,
            timestamps=timestamps,
            
            # Success
            success=success,
            goal_reached_step=goal_reached_step,
            
            # Safety
            min_obstacle_distance=float(min_obstacle_distance),
            mean_obstacle_distance=float(mean_obstacle_distance),
            obstacle_distances=obstacle_distances,
            collision=collision,
            safety_margin_violations=int(safety_margin_violations),
            
            # Performance
            execution_time=float(execution_time),
            energy=float(energy),
            smoothness=float(smoothness),
            path_length=float(path_length),
            final_error=float(final_error),
            mean_error=float(mean_error),
            settling_time=float(settling_time),
            max_velocity=float(max_velocity),
            max_acceleration=float(max_acceleration),
            control_effort=float(control_effort),
            control_variation=float(control_variation),
            
            # Metadata
            experiment_config=metadata['experiment_config'],
            method=metadata['method'],
            seed=metadata['seed'],
            target_pos=target_pos,
            obstacle_pos=np.array(obstacle_positions) if len(obstacle_positions) > 0 else np.array([]),
            initial_qpos=qpos[0],
            njoints=nq,
            dt=dt,
        )
        
        return metrics
    
    def save_trajectory(self, metrics: TrajectoryMetrics, filename: str):
        """Save single trajectory to HDF5."""
        filepath = self.trajectories_dir / f"{filename}.h5"
        
        with h5py.File(filepath, 'w') as f:
            # Temporal data
            f.create_dataset('qpos', data=metrics.qpos, compression='gzip')
            f.create_dataset('qvel', data=metrics.qvel, compression='gzip')
            f.create_dataset('qacc', data=metrics.qacc, compression='gzip')
            f.create_dataset('torques', data=metrics.torques, compression='gzip')
            f.create_dataset('ee_pos', data=metrics.ee_pos, compression='gzip')
            f.create_dataset('ee_vel', data=metrics.ee_vel, compression='gzip')
            f.create_dataset('timestamps', data=metrics.timestamps, compression='gzip')
            f.create_dataset('obstacle_distances', data=metrics.obstacle_distances, compression='gzip')
            
            # Arrays
            f.create_dataset('target_pos', data=metrics.target_pos)
            f.create_dataset('obstacle_pos', data=metrics.obstacle_pos)
            f.create_dataset('initial_qpos', data=metrics.initial_qpos)
            
            # Scalars and metadata as attributes
            for key in ['success', 'goal_reached_step',
                       'min_obstacle_distance', 'mean_obstacle_distance', 'collision',
                       'safety_margin_violations', 'execution_time', 'energy', 'smoothness',
                       'path_length', 'final_error', 'mean_error', 'settling_time',
                       'max_velocity', 'max_acceleration', 'control_effort', 'control_variation',
                       'experiment_config', 'method', 'seed', 'njoints', 'dt']:
                value = getattr(metrics, key)
                if value is not None:
                    if key == 'experiment_config':
                        # Serialize dict to JSON string
                        f.attrs[key] = json.dumps(value)
                    else:
                        f.attrs[key] = value

        print(f"Saved trajectory to {filepath}")
    
    def create_dataset(self, trajectory_files: list, dataset_name: str):
        """Combine multiple trajectories into a single dataset."""
        output_path = self.output_dir / f"{dataset_name}.h5"
        
        with h5py.File(output_path, 'w') as f:
            # Create groups
            f.create_group('trajectories')
            f.create_group('metadata')
            
            # Load and store each trajectory
            for i, traj_file in enumerate(trajectory_files):
                traj_path = self.trajectories_dir / f"{traj_file}.h5"
                with h5py.File(traj_path, 'r') as traj_f:
                    # Copy trajectory data
                    traj_group = f['trajectories'].create_group(f'traj_{i:04d}')
                    
                    for key in traj_f.keys():
                        traj_f.copy(key, traj_group)
                    
                    # Copy attributes
                    for key in traj_f.attrs.keys():
                        traj_group.attrs[key] = traj_f.attrs[key]
            
            # Dataset-level metadata
            f.attrs['num_trajectories'] = len(trajectory_files)
            f.attrs['created_at'] = str(np.datetime64('now'))
        
        print(f"Created dataset with {len(trajectory_files)} trajectories: {output_path}")
