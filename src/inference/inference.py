import os
import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score


def load_model(model_path="outputs/models/logistic_regression_model.pkl"):
    """
    Load the trained LogisticRegression model from the specified path.

    Parameters:
    - model_path: str, path to the saved model file.

    Returns:
    - clf: The trained LogisticRegression model.
    """
    print("Loading the model...")
    clf = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return clf


def load_data(data_path="data/processed/"):
    """
    Load the processed data for inference (features and labels).

    Parameters:
    - data_path: str, path to the directory containing processed data.

    Returns:
    - X_test: The test features.
    - y_test: The true labels for the test data.
    """
    print("Loading the test data...")
    X_test = np.load(os.path.join(data_path, "test_features.npz"))["arr_0"]
    y_test = np.load(os.path.join(data_path, "test_labels.npy"))
    return X_test, y_test


def evaluate_model(clf, X_test, y_test):
    """
    Evaluate the trained model on the test data and print the metrics.

    Parameters:
    - clf: The trained LogisticRegression model.
    - X_test: The test features.
    - y_test: The true labels for the test data.
    """
    print("Evaluating the model...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"])

    # Print the metrics
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)


def main():
    model_path = "outputs/models/logistic_regression_model.pkl"
    data_path = "data/processed/"

    # Load the model
    clf = load_model(model_path)

    # Load test data
    X_test, y_test = load_data(data_path)

    # Evaluate the model
    evaluate_model(clf, X_test, y_test)


if __name__ == "__main__":
    main()