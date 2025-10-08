#!/usr/bin/env python3
"""
Script for viewing a specific instance from SWE-bench by checking out the repository
and applying the model's predicted patch.

This script:
1. Loads the specified dataset
2. Loads the predictions file (JSONL format from run_agent.py)
3. Gets the patch for the specified instance_id
4. Gets the repo and base_commit from the dataset
5. Checkouts the repo in view_dir and applies the patch

Usage:
    python -m swebench.inference.view \
        --dataset_name SWE-bench/SWE-bench_Lite \
        --preds_file predictions.jsonl \
        --instance_id django__django-12345 \
        --view_dir ./view_output
"""

import json
import logging
import os
import subprocess
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Dict, Optional

from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
)
from swebench.harness.utils import (
    load_swebench_dataset,
    get_predictions_from_file,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_command(cmd: str, cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a shell command and return the result.
    
    Args:
        cmd: Command to run
        cwd: Working directory to run command in
        check: Whether to raise exception on non-zero exit code
    
    Returns:
        CompletedProcess result
    """
    logger.info(f"Running command: {cmd}")
    if cwd:
        logger.info(f"Working directory: {cwd}")
    
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False
    )
    
    if result.stdout:
        logger.info(f"STDOUT: {result.stdout}")
    if result.stderr:
        logger.info(f"STDERR: {result.stderr}")
    
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    
    return result


def main(
    dataset_name: str,
    preds_file: str,
    instance_id: str,
    view_dir: str,
    split: str = "test"
):
    """
    Main function to checkout repo and apply patch for viewing.
    
    Args:
        dataset_name: Name of the dataset or path to local dataset file
        preds_file: Path to predictions JSONL file
        instance_id: Instance ID to view
        view_dir: Directory where to checkout the repository
        split: Dataset split to use
    """
    logger.info(f"Starting view for instance {instance_id}")
    
    # 1. Load the dataset
    logger.info(f"Loading dataset {dataset_name}, split {split}")
    dataset = load_swebench_dataset(dataset_name, split)
    logger.info(f"Loaded {len(dataset)} instances from dataset")
    
    # Find the instance in the dataset
    instance = None
    for inst in dataset:
        if inst[KEY_INSTANCE_ID] == instance_id:
            instance = inst
            break
    
    if instance is None:
        raise ValueError(f"Instance {instance_id} not found in dataset")
    
    logger.info(f"Found instance {instance_id} in dataset")
    
    # 2. Load the predictions file
    logger.info(f"Loading predictions from {preds_file}")
    predictions = get_predictions_from_file(preds_file, dataset_name, split)
    
    # Convert to dict for easier lookup
    predictions_dict = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}
    
    if instance_id not in predictions_dict:
        raise ValueError(f"No prediction found for instance {instance_id}")
    
    prediction = predictions_dict[instance_id]
    logger.info(f"Found prediction for instance {instance_id}")
    
    # 3. Extract the patch and repo information
    patch = prediction[KEY_PREDICTION]
    if not patch or patch.strip() == "":
        logger.warning(f"Empty patch found for instance {instance_id}")
    
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    
    logger.info(f"Repository: {repo}")
    logger.info(f"Base commit: {base_commit}")
    logger.info(f"Patch length: {len(patch)} characters")
    
    # 4. Setup view directory
    view_path = Path(view_dir)
    if view_path.exists():
        logger.warning(f"View directory {view_path} already exists")
        user_input = input("Do you want to overwrite it? (y/N): ")
        if user_input.lower() not in ['y', 'yes']:
            logger.info("Aborted by user")
            return
        
        # Remove existing directory
        import shutil
        shutil.rmtree(view_path)
        logger.info(f"Removed existing directory {view_path}")
    
    view_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created view directory: {view_path}")
    
    # 5. Clone and setup the repository
    repo_dir = view_path / "repo"
    
    try:
        # Clone the repository
        clone_cmd = f"git clone https://github.com/{repo} {repo_dir}"
        run_command(clone_cmd)
        logger.info(f"Cloned repository to {repo_dir}")
        
        # Change to the base commit
        reset_cmd = f"git reset --hard {base_commit}"
        run_command(reset_cmd, cwd=repo_dir)
        logger.info(f"Reset to base commit {base_commit}")
        
        # Remove the remote to avoid confusion
        remove_remote_cmd = "git remote remove origin"
        run_command(remove_remote_cmd, cwd=repo_dir, check=False)  # Don't fail if remote doesn't exist
        logger.info("Removed remote origin")
        
        # 6. Apply the patch if it exists
        if patch and patch.strip():
            # Write patch to a temporary file
            patch_file = view_path / "model_patch.diff"
            patch_file.write_text(patch + "\n")
            logger.info(f"Wrote patch to {patch_file}")
            
            # Try to apply the patch using different methods (similar to run_evaluation.py)
            git_apply_commands = [
                "git apply --verbose",
                "git apply --verbose --reject",
                "patch --batch --fuzz=5 -p1 -i"
            ]
            
            patch_applied = False
            for apply_cmd in git_apply_commands:
                try:
                    # Use absolute path for patch file since we're running from repo_dir
                    full_cmd = f"{apply_cmd} {patch_file.absolute()}"
                    result = run_command(full_cmd, cwd=repo_dir, check=False)
                    
                    if result.returncode == 0:
                        logger.info(f"Successfully applied patch using: {apply_cmd}")
                        patch_applied = True
                        break
                    else:
                        logger.warning(f"Failed to apply patch with {apply_cmd}: {result.stderr}")
                
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to apply patch with {apply_cmd}: {e}")
            
            if not patch_applied:
                logger.error("Failed to apply patch with any method")
                logger.info("The repository is checked out at the base commit, but patch could not be applied")
            else:
                logger.info("Patch applied successfully")
        else:
            logger.info("No patch to apply (empty prediction)")
        
        # Show the final git status for user reference
        status_result = run_command("git status", cwd=repo_dir, check=False)
        logger.info("Final git status:")
        print(status_result.stdout)
        
        logger.info(f"Setup complete! Repository is available at: {repo_dir}")
        logger.info(f"Instance ID: {instance_id}")
        logger.info(f"Repository: {repo}")
        logger.info(f"Base commit: {base_commit}")
        
        if patch and patch.strip():
            logger.info(f"Patch file: {patch_file}")
            print(f"\nTo view the diff, run: cd {repo_dir} && git diff")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to setup repository: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset (HuggingFace dataset or local path)",
    )
    parser.add_argument(
        "--preds_file", 
        type=str,
        required=True,
        help="Path to predictions JSONL file (output from run_agent.py)",
    )
    parser.add_argument(
        "--instance_id",
        type=str,
        required=True,
        help="Instance ID to view",
    )
    parser.add_argument(
        "--view_dir",
        type=str,
        required=True,
        help="Directory to checkout repository and apply patch",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
    )
    
    args = parser.parse_args()
    
    main(
        dataset_name=args.dataset_name,
        preds_file=args.preds_file,
        instance_id=args.instance_id,
        view_dir=args.view_dir,
        split=args.split,
    )
