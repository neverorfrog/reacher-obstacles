# src/reacher_obstacles/utils/dataset_inspector.py

import h5py
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Optional


class DatasetInspector:
    """Tools to inspect and verify trajectory datasets."""
    
    def __init__(self, dataset_path: str):
        self.path = Path(dataset_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    def print_summary(self):
        """Print high-level summary of dataset."""
        with h5py.File(self.path, 'r') as f:
            print(f"\n{'='*60}")
            print(f"Dataset: {self.path.name}")
            print(f"{'='*60}")
            
            # Check if single trajectory or multi-trajectory dataset
            if 'trajectories' in f:
                # Multi-trajectory dataset
                n_traj = f.attrs['num_trajectories']
                print(f"Type: Multi-trajectory dataset")
                print(f"Number of trajectories: {n_traj}")
                print(f"Created: {f.attrs.get('created_at', 'Unknown')}")
                
                # Sample first trajectory for structure
                print(f"\nStructure (from first trajectory):")
                first_traj = f['trajectories']['traj_0000']
                self._print_datasets(first_traj)
                self._print_attributes(first_traj)
                
            else:
                # Single trajectory
                print(f"Type: Single trajectory")
                self._print_datasets(f)
                self._print_attributes(f)
    
    def _print_datasets(self, group):
        """Print datasets in a group."""
        print(f"\n  Datasets:")
        for key in group.keys():
            dset = group[key]
            print(f"    {key:25s} {str(dset.shape):20s} {dset.dtype}")
    
    def _print_attributes(self, group):
        """Print attributes."""
        print(f"\n  Attributes:")
        for key in sorted(group.attrs.keys()):
            value = group.attrs[key]
            if isinstance(value, (int, float, bool, str)):
                print(f"    {key:30s} = {value}")
            elif isinstance(value, bytes):
                print(f"    {key:30s} = {value.decode()}")
    
    def check_integrity(self) -> bool:
        """
        Check dataset integrity.
        Returns True if all checks pass.
        """
        print(f"\n{'='*60}")
        print(f"Integrity Checks")
        print(f"{'='*60}")
        
        all_checks_passed = True
        
        with h5py.File(self.path, 'r') as f:
            if 'trajectories' in f:
                n_traj = f.attrs['num_trajectories']
                print(f"Checking {n_traj} trajectories...")
                
                for i in range(n_traj):
                    traj_name = f'traj_{i:04d}'
                    traj = f['trajectories'][traj_name]
                    passed = self._check_single_trajectory(traj, traj_name)
                    all_checks_passed = all_checks_passed and passed
            else:
                all_checks_passed = self._check_single_trajectory(f, self.path.name)
        
        if all_checks_passed:
            print(f"\n✅ All checks passed!")
        else:
            print(f"\n❌ Some checks failed!")
        
        return all_checks_passed
    
    def _check_single_trajectory(self, traj, name: str) -> bool:
        """Check integrity of a single trajectory."""
        checks_passed = True
        
        try:
            # Load data
            qpos = traj['qpos'][:]
            qvel = traj['qvel'][:]
            qacc = traj['qacc'][:]
            torques = traj['torques'][:]
            ee_pos = traj['ee_pos'][:]
            timestamps = traj['timestamps'][:]
            
            T = len(torques)
            nq = qpos.shape[1]
            
            # Check 1: Shape consistency
            print(f"\n{name}:")
            print(f"  ✓ Shape check:")
            
            expected_shapes = {
                'qpos': (T + 1, nq),
                'qvel': (T + 1, nq),
                'qacc': (T, nq),
                'torques': (T, nq),
                'ee_pos': (T + 1, 3),
                'timestamps': (T + 1,),
            }
            
            for key, expected_shape in expected_shapes.items():
                actual_shape = traj[key].shape
                if actual_shape == expected_shape:
                    print(f"    ✓ {key}: {actual_shape}")
                else:
                    print(f"    ✗ {key}: expected {expected_shape}, got {actual_shape}")
                    checks_passed = False
            
            # Check 2: No NaN or Inf values
            print(f"  ✓ Value check:")
            for key in ['qpos', 'qvel', 'qacc', 'torques', 'ee_pos']:
                data = traj[key][:]
                if np.any(np.isnan(data)):
                    print(f"    ✗ {key}: contains NaN values")
                    checks_passed = False
                elif np.any(np.isinf(data)):
                    print(f"    ✗ {key}: contains Inf values")
                    checks_passed = False
                else:
                    print(f"    ✓ {key}: no NaN/Inf")
            
            # Check 3: Timestamps are monotonic increasing
            if np.all(np.diff(timestamps) > 0):
                print(f"    ✓ timestamps: monotonic increasing")
            else:
                print(f"    ✗ timestamps: not monotonic")
                checks_passed = False
            
            # Check 4: Required attributes exist
            required_attrs = ['success', 'experiment_config', 'method', 'final_error', 'energy']
            print(f"  ✓ Attribute check:")
            for attr in required_attrs:
                if attr in traj.attrs:
                    print(f"    ✓ {attr}: {traj.attrs[attr]}")
                else:
                    print(f"    ✗ {attr}: missing")
                    checks_passed = False
            
            # Check 5: Physics sanity checks
            print(f"  ✓ Physics check:")
            
            # Check torque limits (should be in [-1, 1] range)
            torque_min, torque_max = torques.min(), torques.max()
            if -1.5 <= torque_min and torque_max <= 1.5:
                print(f"    ✓ torques in reasonable range: [{torque_min:.3f}, {torque_max:.3f}]")
            else:
                print(f"    ⚠ torques outside expected range: [{torque_min:.3f}, {torque_max:.3f}]")
            
            # Check velocities are reasonable
            vel_max = np.abs(qvel).max()
            if vel_max < 50:  # rad/s
                print(f"    ✓ velocities reasonable: max={vel_max:.3f} rad/s")
            else:
                print(f"    ⚠ very high velocities: max={vel_max:.3f} rad/s")
            
            # Check accelerations are reasonable
            acc_max = np.abs(qacc).max()
            if acc_max < 500:  # rad/s²
                print(f"    ✓ accelerations reasonable: max={acc_max:.3f} rad/s²")
            else:
                print(f"    ⚠ very high accelerations: max={acc_max:.3f} rad/s²")
            
        except Exception as e:
            print(f"  ✗ Error reading trajectory: {e}")
            checks_passed = False
        
        return checks_passed
    
    def load_trajectory(self, traj_idx: Optional[int] = None):
        """
        Load a single trajectory and return as dictionary.
        
        Args:
            traj_idx: Index of trajectory (for multi-traj datasets). 
                     None for single trajectory files.
        """
        with h5py.File(self.path, 'r') as f:
            if 'trajectories' in f:
                if traj_idx is None:
                    traj_idx = 0
                traj = f['trajectories'][f'traj_{traj_idx:04d}']
            else:
                traj = f
            
            data = {
                'qpos': traj['qpos'][:],
                'qvel': traj['qvel'][:],
                'qacc': traj['qacc'][:],
                'torques': traj['torques'][:],
                'ee_pos': traj['ee_pos'][:],
                'ee_vel': traj['ee_vel'][:],
                'timestamps': traj['timestamps'][:],
                'obstacle_distances': traj['obstacle_distances'][:],
            }
            
            # Load attributes
            for key in traj.attrs.keys():
                data[key] = traj.attrs[key]
            
            return data
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert multi-trajectory dataset to pandas DataFrame.
        Each row is one trajectory with summary statistics.
        """
        rows = []
        
        with h5py.File(self.path, 'r') as f:
            if 'trajectories' not in f:
                raise ValueError("Not a multi-trajectory dataset")
            
            n_traj = f.attrs['num_trajectories']
            
            for i in range(n_traj):
                traj = f['trajectories'][f'traj_{i:04d}']
                
                row = {}
                # Copy all scalar attributes
                for key in traj.attrs.keys():
                    value = traj.attrs[key]
                    if not isinstance(value, (bytes, str)) or key != 'algorithm_params':
                        row[key] = value
                
                rows.append(row)
        
        return pd.DataFrame(rows)


# =====================================================
# Visualization Tools
# =====================================================

def plot_trajectory(data: dict, save_path: Optional[str] = None):
    """
    Plot a single trajectory with all relevant information.
    
    Args:
        data: Dictionary returned by load_trajectory()
        save_path: Path to save figure (None = show interactively)
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig)
    
    # Extract data
    timestamps = data['timestamps']
    qpos = data['qpos']
    qvel = data['qvel']
    qacc = data['qacc']
    torques = data['torques']
    ee_pos = data['ee_pos']
    target_pos = data['target_pos']
    
    nq = qpos.shape[1]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 1. Joint positions
    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(nq):
        ax1.plot(timestamps, qpos[:, i], label=f'Joint {i+1}', color=colors[i])
    ax1.set_ylabel('Position [rad]')
    ax1.set_title('Joint Positions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Joint velocities
    ax2 = fig.add_subplot(gs[1, 0])
    for i in range(nq):
        ax2.plot(timestamps, qvel[:, i], label=f'Joint {i+1}', color=colors[i])
    ax2.set_ylabel('Velocity [rad/s]')
    ax2.set_title('Joint Velocities')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Joint accelerations
    ax3 = fig.add_subplot(gs[2, 0])
    for i in range(nq):
        ax3.plot(timestamps[:-1], qacc[:, i], label=f'Joint {i+1}', color=colors[i])
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Acceleration [rad/s²]')
    ax3.set_title('Joint Accelerations')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Control torques
    ax4 = fig.add_subplot(gs[0, 1])
    for i in range(nq):
        ax4.plot(timestamps[:-1], torques[:, i], label=f'Joint {i+1}', color=colors[i])
    ax4.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Limits')
    ax4.axhline(-1.0, color='r', linestyle='--', alpha=0.5)
    ax4.set_ylabel('Torque [Nm]')
    ax4.set_title('Control Torques')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. End-effector trajectory (2D workspace)
    ax5 = fig.add_subplot(gs[1:, 1])
    
    # Plot trajectory
    ax5.plot(ee_pos[:, 0], ee_pos[:, 1], 'b-', linewidth=2, label='EE trajectory')
    ax5.plot(ee_pos[0, 0], ee_pos[0, 1], 'go', markersize=10, label='Start')
    ax5.plot(ee_pos[-1, 0], ee_pos[-1, 1], 'bo', markersize=10, label='End')
    
    # Plot target
    ax5.plot(target_pos[0], target_pos[1], 'r*', markersize=20, label='Target')
    
    # Plot obstacles
    if 'obstacle_pos' in data and len(data['obstacle_pos']) > 0:
        for obs in data['obstacle_pos']:
            circle = plt.Circle((obs[0], obs[1]), 0.02, color='gray', alpha=0.5)
            ax5.add_patch(circle)
    
    ax5.set_xlabel('X [m]')
    ax5.set_ylabel('Y [m]')
    ax5.set_title('End-Effector Trajectory')
    ax5.axis('equal')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Distance to target
    ax6 = fig.add_subplot(gs[0, 2])
    errors = np.linalg.norm(ee_pos - target_pos, axis=1)
    ax6.plot(timestamps, errors, 'r-', linewidth=2)
    ax6.axhline(0.08, color='g', linestyle='--', alpha=0.5, label='Tolerance')
    ax6.set_ylabel('Distance [m]')
    ax6.set_title('Distance to Target')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Obstacle distances (if any)
    if 'obstacle_distances' in data and data['obstacle_distances'].size > 0:
        ax7 = fig.add_subplot(gs[1, 2])
        obstacle_dists = data['obstacle_distances']
        for i in range(obstacle_dists.shape[1]):
            ax7.plot(timestamps, obstacle_dists[:, i], label=f'Obs {i+1}')
        ax7.axhline(0.05, color='r', linestyle='--', alpha=0.5, label='Safety margin')
        ax7.set_ylabel('Distance [m]')
        ax7.set_title('Obstacle Clearance')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # 8. Metrics summary
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    metrics_text = f"""
    Experiment: {data['experiment_id']}
    Method: {data['method']}
    Success: {data['success']}
    
    Final Error: {data['final_error']:.4f} m
    Mean Error: {data['mean_error']:.4f} m
    
    Energy: {data['energy']:.4f} Nm²·s
    Smoothness: {data['smoothness']:.4f}
    Exec Time: {data['execution_time']:.3f} s
    
    Min Obs Dist: {data['min_obstacle_distance']:.4f} m
    Collisions: {data['collision_count']}
    """
    
    ax8.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.suptitle(f"Trajectory Analysis: {data['experiment_id']} ({data['method']})", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()