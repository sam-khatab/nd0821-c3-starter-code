# Script to train machine learning model.

import os
from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split

from starter.starter.ml.data import process_data
from starter.starter.ml.model import (
    train_rc_model, inference, compute_model_metrics
)

# Add the necessary imports for the starter code.

# Add code to load in the data.
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "census.csv"

def train_model():
    data = pd.read_csv(DATA_PATH)

    # Optional enhancement, use K-fold cross validation instead of a
    # train-test split.
    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # One-hot encoding for categorical features is handled in process_data.
    X_train, y_train, encoder, lb = process_data(
        data=train,
        categorical_features=cat_features,
        label="salary",
        encoder=None,
        lb=None
    )


    model = train_rc_model(X_train, y_train)

    X_test, y_test, encoder, lb = process_data(
        data=test,
        categorical_features=cat_features,
        label="salary",
        encoder=encoder,
        lb=lb
    )

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {fbeta:.4f}")

    os.makedirs("model", exist_ok=True)
    dump(model, "model/rf_model.joblib")
    dump(encoder, "model/encoder.joblib")
    dump(lb, "model/lb.joblib")

if __name__ == "__main__":
    train_model()
