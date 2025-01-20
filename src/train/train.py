from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import load_npz
import numpy as np
import os
import joblib

def train_model():
    data_dir = "/app/data/processed/"
    output_dir = "/app/outputs/models/"

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load processed data
    print("Loading processed data...")
    X_train = load_npz(os.path.join(data_dir, "train_features.npz"))
    y_train = np.load(os.path.join(data_dir, "train_labels.npy"))
    X_test = load_npz(os.path.join(data_dir, "test_features.npz"))
    y_test = np.load(os.path.join(data_dir, "test_labels.npy"))

    print("Data successfully loaded.")

    # Train the model
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("Model training completed.")

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["negative", "positive"])

    # Save metrics
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Save the model
    model_path = os.path.join(output_dir, "logistic_regression_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    train_model()