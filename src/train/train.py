import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def train_model(data_dir="data/processed/", output_dir="outputs/"):
    """
    Train a LogisticRegression model on the processed data and save the model and metrics.

    Parameters:
    - data_dir: str, path to the directory containing processed data.
    - output_dir: str, path to the directory where model and metrics will be saved.
    """
    # Ensure output directories exist
    model_dir = os.path.join(output_dir, "models")
    metrics_dir = os.path.join(output_dir, "predictions")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Load processed data
    print("Loading processed data...")
    X_train = np.load(os.path.join(data_dir, "train_features.npz"))["arr_0"]
    y_train = np.load(os.path.join(data_dir, "train_labels.npy"))
    X_test = np.load(os.path.join(data_dir, "test_features.npz"))["arr_0"]
    y_test = np.load(os.path.join(data_dir, "test_labels.npy"))

    # Train LogisticRegression model
    print("Training LogisticRegression model...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Save the trained model
    model_path = os.path.join(model_dir, "logistic_regression_model.pkl")
    joblib.dump(clf, model_path)
    print(f"Model saved at {model_path}")

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"])

    # Save metrics
    metrics_path = os.path.join(metrics_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
    print(f"Metrics saved at {metrics_path}")

    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    train_model()