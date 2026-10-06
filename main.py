"""Main entry point to run the entire cybersecurity URL classification pipeline."""

import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    print(f"\n{'='*50}\n[RUNNING] {description}\n{'='*50}")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"[SUCCESS] {description} completed successfully.\n")

def main():
    parser = argparse.ArgumentParser(description="Run the end-to-end URL classification pipeline.")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature extraction if already done.")
    parser.add_argument("--classifier", default="random_forest", choices=["logreg", "random_forest"], help="Classifier to train and evaluate.")
    args = parser.parse_args()

    python_exec = sys.executable

    # Step 1: Feature Extraction
    if not args.skip_features:
        run_command(
            [python_exec, "extract_features.py", "--input-csv", "full_dataset.csv"],
            "Step 1: Extracting features from raw dataset"
        )
    else:
        print("\n[SKIPPED] Feature extraction (using existing features.csv if present).")

    # Step 2: Make Data Splits
    run_command(
        [python_exec, "make_splits.py"],
        "Step 2: Creating Train/Validation/Test splits"
    )

    # Step 3: Train Model
    run_command(
        [python_exec, "train_models.py", "--classifier", args.classifier],
        f"Step 3: Training {args.classifier} model"
    )

    # Step 4: Predict and Evaluate
    run_command(
        [python_exec, "predict.py", "--classifier", args.classifier],
        f"Step 4: Predicting test set with {args.classifier} model"
    )
    
    print("Pipeline finished successfully!")

if __name__ == "__main__":
    main()
