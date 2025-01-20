import os
import numpy as np
import joblib
from scipy.sparse import load_npz
from sklearn.metrics import classification_report, accuracy_score

def inference():
    # Define paths
    data_dir = "./data/processed"
    model_path = "./outputs/models/logistic_regression_model.pkl"
    predictions_path = "./outputs/predictions/predictions.csv"
    metrics_path = "./outputs/predictions/metrics.txt"

    # Load the model
    print("Loading the trained model...")
    model = joblib.load(model_path)

    # Load test data
    print("Loading processed test data...")
    X_test = load_npz(os.path.join(data_dir, "test_features.npz"))
    y_test = np.load(os.path.join(data_dir, "test_labels.npy"))

    # Perform predictions
    print("Performing predictions...")
    y_pred = model.predict(X_test)

    # Save predictions
    print("Saving predictions...")
    predictions = np.column_stack((y_test, y_pred))
    np.savetxt(predictions_path, predictions, fmt="%d", delimiter=",", header="Actual,Predicted", comments="")

    # Calculate metrics
    print("Calculating metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Save metrics
    print("Saving metrics...")
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print("Inference completed!")
    print(f"Predictions saved at: {predictions_path}")
    print(f"Metrics saved at: {metrics_path}")

if __name__ == "__main__":
    inference()
