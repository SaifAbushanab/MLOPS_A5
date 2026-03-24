"""
train.py - Train a classifier and log results to MLflow.

Trains a RandomForestClassifier on the Iris dataset,
logs accuracy and model to MLflow, and writes the Run ID
to model_info.txt on success.
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    # 1. Load data (pulled via DVC)
    data_path = os.path.join("data", "iris.csv")
    df = pd.read_csv(data_path)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Train model
    model = RandomForestClassifier(n_estimators=1, max_depth=1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")

    # 3. Log to MLflow
    mlflow.set_experiment("assignment5-iris")

    with mlflow.start_run() as run:
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

    # 4. Export Run ID
    with open("model_info.txt", "w") as f:
        f.write(run_id)

    print("Run ID written to model_info.txt")


if __name__ == "__main__":
    main()
