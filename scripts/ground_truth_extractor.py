#!/usr/bin/env python3
"""
Ground Truth Transformation Extractor for Stanford Bunny Dataset
Extracts and converts transformations from bun.conf to compare with Gibbs sampler results
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler(qx, qy, qz, qw):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw)
    
    Args:
        qx, qy, qz, qw: Quaternion components
        
    Returns:
        roll, pitch, yaw in radians
    """
    # Create rotation object from quaternion
    rotation = R.from_quat([qx, qy, qz, qw])
    
    # Convert to Euler angles (ZYX convention to match our sampler)
    euler_angles = rotation.as_euler('ZYX', degrees=False)
    
    # Return as roll, pitch, yaw (reverse order from ZYX)
    yaw, pitch, roll = euler_angles
    return roll, pitch, yaw

def parse_bun_conf(conf_file_path):
    """
    Parse bun.conf file to extract transformations
    
    Args:
        conf_file_path: Path to bun.conf file
        
    Returns:
        Dictionary mapping mesh names to transformation parameters
    """
    transformations = {}
    
    with open(conf_file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('bmesh') and '.ply' in line:
            parts = line.split()
            if len(parts) >= 9:
                mesh_name = parts[1]  # e.g., "bun000.ply"
                
                # Extract transformation parameters
                tx = float(parts[2])
                ty = float(parts[3])
                tz = float(parts[4])
                qx = float(parts[5])
                qy = float(parts[6])
                qz = float(parts[7])
                qw = float(parts[8])
                
                # Convert quaternion to Euler angles
                roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)
                
                transformations[mesh_name] = {
                    'translation': [tx, ty, tz],
                    'quaternion': [qx, qy, qz, qw],
                    'euler_angles': [roll, pitch, yaw],
                    'euler_degrees': [np.degrees(roll), np.degrees(pitch), np.degrees(yaw)]
                }
    
    return transformations

def compute_relative_transformation(source_transform, target_transform):
    """
    Compute relative transformation from source to target
    
    Args:
        source_transform: Transformation dict for source mesh
        target_transform: Transformation dict for target mesh
        
    Returns:
        Relative transformation parameters (tx, ty, tz, roll, pitch, yaw)
    """
    # Get transformations
    t_source = np.array(source_transform['translation'])
    t_target = np.array(target_transform['translation'])
    
    q_source = source_transform['quaternion']
    q_target = target_transform['quaternion']
    
    # Create rotation objects
    R_source = R.from_quat(q_source)
    R_target = R.from_quat(q_target)
    
    # Compute relative transformation
    # T_relative = T_target * T_source^(-1)
    R_relative = R_target * R_source.inv()
    t_relative = t_target - R_relative.apply(t_source)
    
    # Convert relative rotation to Euler angles
    euler_relative = R_relative.as_euler('ZYX', degrees=False)
    yaw_rel, pitch_rel, roll_rel = euler_relative
    
    return t_relative[0], t_relative[1], t_relative[2], roll_rel, pitch_rel, yaw_rel

def main():
    """Extract ground truth transformations for bunny registration"""
    
    print("=" * 60)
    print("STANFORD BUNNY GROUND TRUTH EXTRACTOR")
    print("=" * 60)
    
    # Parse transformations
    conf_file = "/Users/waelbenamara/Desktop/Research/ptcloud/bunny/data/bun.conf"
    transformations = parse_bun_conf(conf_file)
    
    print(f"\nFound transformations for {len(transformations)} meshes:")
    for mesh_name, transform in transformations.items():
        print(f"\n{mesh_name}:")
        print(f"  Translation: ({transform['translation'][0]:.6f}, {transform['translation'][1]:.6f}, {transform['translation'][2]:.6f})")
        print(f"  Quaternion: ({transform['quaternion'][0]:.6f}, {transform['quaternion'][1]:.6f}, {transform['quaternion'][2]:.6f}, {transform['quaternion'][3]:.6f})")
        print(f"  Euler (rad): Roll={transform['euler_angles'][0]:.6f}, Pitch={transform['euler_angles'][1]:.6f}, Yaw={transform['euler_angles'][2]:.6f}")
        print(f"  Euler (deg): Roll={transform['euler_degrees'][0]:.2f}°, Pitch={transform['euler_degrees'][1]:.2f}°, Yaw={transform['euler_degrees'][2]:.2f}°")
    
    # Compute relative transformation from bun000 to bun045
    if 'bun000.ply' in transformations and 'bun045.ply' in transformations:
        print(f"\n" + "=" * 60)
        print("GROUND TRUTH: bun000.ply → bun045.ply TRANSFORMATION")
        print("=" * 60)
        
        source_transform = transformations['bun000.ply']
        target_transform = transformations['bun045.ply']
        
        tx_rel, ty_rel, tz_rel, roll_rel, pitch_rel, yaw_rel = compute_relative_transformation(
            source_transform, target_transform
        )
        
        print(f"\nGround Truth Relative Transformation:")
        print(f"  Translation:")
        print(f"    TX: {tx_rel:.6f}")
        print(f"    TY: {ty_rel:.6f}")
        print(f"    TZ: {tz_rel:.6f}")
        print(f"  Rotation:")
        print(f"    Roll: {roll_rel:.6f} rad ({np.degrees(roll_rel):.2f}°)")
        print(f"    Pitch: {pitch_rel:.6f} rad ({np.degrees(pitch_rel):.2f}°)")
        print(f"    Yaw: {yaw_rel:.6f} rad ({np.degrees(yaw_rel):.2f}°)")
        
        # Save ground truth for comparison
        ground_truth = {
            'tx': tx_rel,
            'ty': ty_rel,
            'tz': tz_rel,
            'roll': roll_rel,
            'pitch': pitch_rel,
            'yaw': yaw_rel
        }
        
        import json
        with open('/Users/waelbenamara/Desktop/Research/ptcloud/aux/bunny_ground_truth.json', 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        print(f"\nGround truth saved to: bunny_ground_truth.json")
        print(f"\nYou can now compare Gibbs sampler results with these ground truth values!")
        
        return ground_truth
    else:
        print("Error: Could not find bun000.ply and bun045.ply transformations")
        return None

if __name__ == "__main__":
    ground_truth = main()
