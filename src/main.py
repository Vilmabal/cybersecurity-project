"""Main entry point to run the entire pipeline"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    print(f"\n{'='*50}\n[RUNNING] {description}\n{'='*50}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nError: {result.returncode}")
        sys.exit(result.returncode)
    print(f"[SUCCESS] {description} completed successfully.\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", default="random_forest", choices=["logreg", "random_forest"])
    args = parser.parse_args()
    python_exec = sys.executable

    #feature extraction
    features_file = Path("data/processed/features.csv")
    if features_file.exists():
        print("\nfeatures.csv already exists.")
    else:
        run_command(
            [python_exec, "src/extract_features.py", "--input-csv", "data/raw/full_dataset.csv"],
            "Extracting features from raw dataset"
        )

    #Data splits
    run_command(
        [python_exec, "src/make_splits.py"],
        "Creating Train/Validation/Test splits"
    )

    #Training the model
    run_command(
        [python_exec, "src/train_models.py", "--classifier", args.classifier],
        f"Training {args.classifier} model"
    )

    #Predicting and evaluating
    run_command(
        [python_exec, "src/predict.py", "--classifier", args.classifier],
        f"Step 4: Predicting test set with {args.classifier} model"
    )
    
    print("Finished!")

if __name__ == "__main__":
    main()
