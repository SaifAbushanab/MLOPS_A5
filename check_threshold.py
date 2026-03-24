"""
check_threshold.py - Validate model accuracy meets deployment threshold.

Reads the MLflow Run ID from model_info.txt, queries the tracking server
for the accuracy metric, and fails (exit code 1) if accuracy < 0.85.
"""

import sys
import mlflow

THRESHOLD = 0.85


def main():
    # ------------------------------------------------------------------
    # 1. Read Run ID
    # ------------------------------------------------------------------
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()

    print(f"Checking Run ID: {run_id}")

    # ------------------------------------------------------------------
    # 2. Fetch accuracy from MLflow
    # ------------------------------------------------------------------
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")

    if accuracy is None:
        print("ERROR: 'accuracy' metric not found for this run.")
        sys.exit(1)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Threshold: {THRESHOLD}")

    # ------------------------------------------------------------------
    # 3. Gate decision
    # ------------------------------------------------------------------
    if accuracy < THRESHOLD:
        print(f"FAILED: Accuracy {accuracy:.4f} is below threshold {THRESHOLD}.")
        sys.exit(1)
    else:
        print(f"PASSED: Accuracy {accuracy:.4f} meets threshold {THRESHOLD}.")
        sys.exit(0)


if __name__ == "__main__":
    main()
