#!/usr/bin/env python3
"""
PLY to CSV Converter for Stanford Bunny Dataset
Converts PLY point cloud files to CSV format for use with Gibbs sampler
"""

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

def read_ply_file(ply_path):
    """
    Read PLY file and extract vertex coordinates
    
    Args:
        ply_path: Path to PLY file
        
    Returns:
        numpy array of shape (N, 3) with x, y, z coordinates
    """
    print(f"Reading PLY file: {ply_path}")
    
    with open(ply_path, 'r') as f:
        lines = f.readlines()
    
    # Parse header to find number of vertices
    vertex_count = 0
    header_end = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[-1])
        elif line == 'end_header':
            header_end = i + 1
            break
    
    print(f"Found {vertex_count} vertices, header ends at line {header_end}")
    
    # Read vertex data
    vertices = []
    data_lines = lines[header_end:header_end + vertex_count]
    
    for line_num, line in enumerate(data_lines):
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                vertices.append([x, y, z])
            except ValueError:
                print(f"Warning: Could not parse line {header_end + line_num + 1}: {line.strip()}")
                continue
        
        # Progress indicator for large files
        if (line_num + 1) % 10000 == 0:
            print(f"  Processed {line_num + 1}/{len(data_lines)} vertices...")
    
    vertices = np.array(vertices)
    print(f"Successfully loaded {len(vertices)} vertices")
    
    return vertices

def convert_ply_to_csv(ply_path, csv_path, subsample_factor=None):
    """
    Convert PLY file to CSV format
    
    Args:
        ply_path: Input PLY file path
        csv_path: Output CSV file path  
        subsample_factor: If provided, randomly subsample points by this factor
    """
    # Read PLY file
    vertices = read_ply_file(ply_path)
    
    # Optional subsampling for faster processing
    if subsample_factor is not None and subsample_factor > 1:
        n_original = len(vertices)
        indices = np.random.choice(n_original, n_original // subsample_factor, replace=False)
        vertices = vertices[indices]
        print(f"Subsampled from {n_original} to {len(vertices)} points (factor: {subsample_factor})")
    
    # Create DataFrame and save
    df = pd.DataFrame(vertices, columns=['x', 'y', 'z'])
    df.to_csv(csv_path, index=False)
    
    print(f"Saved {len(vertices)} vertices to {csv_path}")
    
    # Print basic statistics
    print(f"Point cloud statistics:")
    print(f"  X range: [{vertices[:, 0].min():.4f}, {vertices[:, 0].max():.4f}]")
    print(f"  Y range: [{vertices[:, 1].min():.4f}, {vertices[:, 1].max():.4f}]")
    print(f"  Z range: [{vertices[:, 2].min():.4f}, {vertices[:, 2].max():.4f}]")
    print(f"  Centroid: ({vertices[:, 0].mean():.4f}, {vertices[:, 1].mean():.4f}, {vertices[:, 2].mean():.4f})")
    
    return vertices

def main():
    """Convert Stanford Bunny PLY files to CSV for Gibbs sampler testing"""
    
    # Set random seed for reproducible subsampling
    np.random.seed(42)
    
    # Define paths
    bunny_data_dir = "/Users/waelbenamara/Desktop/Research/RandomWalk/data/bunny/data"
    output_dir = "/Users/waelbenamara/Desktop/Research/RandomWalk/data"
    
    # Check if bunny directory exists
    if not os.path.exists(bunny_data_dir):
        print(f"Error: Bunny data directory not found: {bunny_data_dir}")
        return False
    
    # Convert two bunny views for registration testing
    # Using bun000.ply (front view) and bun045.ply (45-degree rotated view)
    source_ply = os.path.join(bunny_data_dir, "bun000.ply")
    target_ply = os.path.join(bunny_data_dir, "bun315.ply")
    
    source_csv = os.path.join(output_dir, "bunny_source.csv")
    target_csv = os.path.join(output_dir, "bunny_target.csv")
    
    print("=" * 60)
    print("STANFORD BUNNY PLY TO CSV CONVERTER")
    print("=" * 60)
    
    # Check if PLY files exist
    for ply_file, name in [(source_ply, "source"), (target_ply, "target")]:
        if not os.path.exists(ply_file):
            print(f"Error: {name} PLY file not found: {ply_file}")
            return False
    
    try:
        # Convert source (bun000.ply)
        print(f"\nConverting source bunny (bun000.ply)...")
        source_vertices = convert_ply_to_csv(source_ply, source_csv, subsample_factor=4)
        
        print(f"\nConverting target bunny (bun315.ply)...")
        target_vertices = convert_ply_to_csv(target_ply, target_csv, subsample_factor=4)
        
        print(f"\n" + "=" * 60)
        print("CONVERSION COMPLETE!")
        print("=" * 60)
        print(f"Source bunny: {len(source_vertices)} points -> {source_csv}")
        print(f"Target bunny: {len(target_vertices)} points -> {target_csv}")
        print(f"\nReady for 3D Gibbs sampler registration testing!")
        
        return True
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nYou can now run the 3D Gibbs sampler with:")
        print("python sampler_3d.py")
        print("(Make sure to update the file paths in sampler_3d.py to use bunny_source.csv and bunny_target.csv)")
    else:
        print("Conversion failed!")
        sys.exit(1)
