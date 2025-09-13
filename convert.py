#!/usr/bin/env python3
"""
Script to convert checkpoint files and optionally clean up unnecessary files.
Processes all checkpoint folders in the experiments directory.

Usage:
    python process_checkpoints.py convert    # Only run conversions
    python process_checkpoints.py delete     # Only delete files (for already converted checkpoints)
    python process_checkpoints.py both       # Convert and then delete (default)
"""

import os
import subprocess
import sys
from pathlib import Path
import shutil
import argparse

def find_checkpoint_folders(experiments_dir):
    """Find all checkpoint folders in the experiments directory."""
    checkpoint_folders = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(experiments_dir):
        for dir_name in dirs:
            if dir_name.startswith('checkpoint-'):
                checkpoint_path = Path(root) / dir_name
                checkpoint_folders.append(checkpoint_path)
    
    return checkpoint_folders

def get_experiment_config(checkpoint_path):
    """Get the config file path for the experiment."""
    # Navigate up from checkpoint folder to experiment folder
    experiment_folder = checkpoint_path.parent
    experiment_name = experiment_folder.name
    config_path = experiment_folder / f"{experiment_name}.yml"
    
    # Check if config exists
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}")
        # Try to find any .yml file in the experiment folder
        yml_files = list(experiment_folder.glob("*.yml"))
        if yml_files:
            config_path = yml_files[0]
            print(f"Using config file: {config_path}")
        else:
            return None
    
    return config_path

def run_conversion(checkpoint_path, config_path):
    """Run the conversion command for a checkpoint."""
    model_path = checkpoint_path / "pytorch_model_fsdp.bin"
    save_path = checkpoint_path / "transformer_lora"
    
    # Check if model file exists
    if not model_path.exists():
        print(f"Skipping {checkpoint_path}: pytorch_model_fsdp.bin not found")
        return False
    
    # Check if already converted
    if save_path.exists():
        print(f"Already converted: {checkpoint_path} (transformer_lora exists)")
        return True
    
    # Build the command
    cmd = [
        "python", "convert_ckpt_to_hf_format.py",
        "--config_path", str(config_path),
        "--model_path", str(model_path),
        "--save_path", str(save_path)
    ]
    
    print(f"\nRunning conversion for: {checkpoint_path}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Conversion successful for {checkpoint_path}")
            return True
        else:
            print(f"✗ Conversion failed for {checkpoint_path}")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error running conversion: {e}")
        return False

def cleanup_checkpoint_files(checkpoint_path):
    """Delete the specified files from the checkpoint folder."""
    files_to_delete = [
        "pytorch_model_fsdp.bin",
        "optimizer.bin",
        "random_states_0.pkl"
    ]
    
    deleted_files = []
    for filename in files_to_delete:
        file_path = checkpoint_path / filename
        if file_path.exists():
            try:
                file_path.unlink()
                deleted_files.append(filename)
            except Exception as e:
                print(f"  Error deleting {filename}: {e}")
    
    if deleted_files:
        print(f"  Deleted files: {', '.join(deleted_files)}")
    
    return len(deleted_files) > 0

def convert_only(experiments_dir):
    """Run conversions only, without deleting files."""
    print("MODE: Convert only (no deletion)")
    
    # Check if conversion script exists
    if not Path("convert_ckpt_to_hf_format.py").exists():
        print("Error: 'convert_ckpt_to_hf_format.py' not found in current directory!")
        return
    
    # Find all checkpoint folders
    print(f"Searching for checkpoint folders in '{experiments_dir}'...")
    checkpoint_folders = find_checkpoint_folders(experiments_dir)
    
    if not checkpoint_folders:
        print("No checkpoint folders found!")
        return
    
    print(f"Found {len(checkpoint_folders)} checkpoint folders")
    
    # Process each checkpoint
    successful_conversions = 0
    failed_conversions = 0
    already_converted = 0
    
    for checkpoint_path in checkpoint_folders:
        print(f"\n{'='*60}")
        print(f"Processing: {checkpoint_path}")
        
        # Get config path
        config_path = get_experiment_config(checkpoint_path)
        if not config_path:
            print(f"Skipping: No config file found for {checkpoint_path}")
            failed_conversions += 1
            continue
        
        # Run conversion
        result = run_conversion(checkpoint_path, config_path)
        if result:
            if (checkpoint_path / "transformer_lora").exists() and not (checkpoint_path / "pytorch_model_fsdp.bin").exists():
                already_converted += 1
            else:
                successful_conversions += 1
        else:
            failed_conversions += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"CONVERSION SUMMARY:")
    print(f"  Total checkpoints found: {len(checkpoint_folders)}")
    print(f"  New conversions: {successful_conversions}")
    print(f"  Already converted: {already_converted}")
    print(f"  Failed conversions: {failed_conversions}")

def delete_only(experiments_dir):
    """Delete files only, useful for already converted checkpoints."""
    print("MODE: Delete only (no conversion)")
    
    # Find all checkpoint folders
    print(f"Searching for checkpoint folders in '{experiments_dir}'...")
    checkpoint_folders = find_checkpoint_folders(experiments_dir)
    
    if not checkpoint_folders:
        print("No checkpoint folders found!")
        return
    
    print(f"Found {len(checkpoint_folders)} checkpoint folders")
    
    # Process each checkpoint
    checkpoints_cleaned = 0
    checkpoints_skipped = 0
    
    for checkpoint_path in checkpoint_folders:
        print(f"\n{'='*60}")
        print(f"Processing: {checkpoint_path}")
        
        # Check if conversion was done (transformer_lora exists)
        if not (checkpoint_path / "transformer_lora").exists():
            print(f"Skipping: No transformer_lora folder found (not converted yet)")
            checkpoints_skipped += 1
            continue
        
        # Clean up files
        print(f"Cleaning up checkpoint files...")
        if cleanup_checkpoint_files(checkpoint_path):
            checkpoints_cleaned += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"DELETION SUMMARY:")
    print(f"  Total checkpoints found: {len(checkpoint_folders)}")
    print(f"  Checkpoints cleaned: {checkpoints_cleaned}")
    print(f"  Checkpoints skipped: {checkpoints_skipped}")

def convert_and_delete(experiments_dir):
    """Run conversions and delete files after successful conversion."""
    print("MODE: Convert and delete")
    
    # Check if conversion script exists
    if not Path("convert_ckpt_to_hf_format.py").exists():
        print("Error: 'convert_ckpt_to_hf_format.py' not found in current directory!")
        return
    
    # Find all checkpoint folders
    print(f"Searching for checkpoint folders in '{experiments_dir}'...")
    checkpoint_folders = find_checkpoint_folders(experiments_dir)
    
    if not checkpoint_folders:
        print("No checkpoint folders found!")
        return
    
    print(f"Found {len(checkpoint_folders)} checkpoint folders")
    
    # Process each checkpoint
    successful_conversions = 0
    failed_conversions = 0
    files_deleted = 0
    
    for checkpoint_path in checkpoint_folders:
        print(f"\n{'='*60}")
        print(f"Processing: {checkpoint_path}")
        
        # Get config path
        config_path = get_experiment_config(checkpoint_path)
        if not config_path:
            print(f"Skipping: No config file found for {checkpoint_path}")
            failed_conversions += 1
            continue
        
        # Run conversion
        if run_conversion(checkpoint_path, config_path):
            successful_conversions += 1
            
            # Clean up files only if conversion was successful
            print(f"Cleaning up checkpoint files...")
            if cleanup_checkpoint_files(checkpoint_path):
                files_deleted += 1
        else:
            failed_conversions += 1
            print(f"Skipping cleanup due to conversion failure")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Total checkpoints processed: {len(checkpoint_folders)}")
    print(f"  Successful conversions: {successful_conversions}")
    print(f"  Failed conversions: {failed_conversions}")
    print(f"  Checkpoints with deleted files: {files_deleted}")

def main():
    
  
    
    # Set the experiments directory
    experiments_dir = Path("experiments")
    
    if not experiments_dir.exists():
        print(f"Error: '{experiments_dir}' directory not found!")
        sys.exit(1)
    convert_and_delete(experiments_dir)
    # # Execute based on mode
    # if args.mode == 'convert':
    #     convert_only(experiments_dir)
    # elif args.mode == 'delete':
    #     delete_only(experiments_dir)
    # else:  # both
    #     convert_and_delete(experiments_dir)

if __name__ == "__main__":
    main()