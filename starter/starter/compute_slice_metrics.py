from ml import model
from ml.model import inference, compute_model_metrics
from ml.data import process_data
import pandas as pd
from joblib import load
from pathlib import Path

features = ['age', 'education', 'race', 'race']
values = [25, 'Bachelors', 'White', 'Black']
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "rf_model.joblib"
LB_PATH = BASE_DIR / "model" / "lb.joblib"
ENCODER_PATH = BASE_DIR / "model" / "encoder.joblib"
DATA_PATH = BASE_DIR / "data" / "census.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "slice_output.txt"

lb = load(LB_PATH) if LB_PATH.exists() else None
model = load(MODEL_PATH) if MODEL_PATH.exists() else None
encoder = load(ENCODER_PATH) if ENCODER_PATH.exists() else None
data = pd.read_csv(DATA_PATH)

try:
    assert len(features) == len(values), "Features and values lists must be of the same length."
except AssertionError as e:
    print(e)
    exit(1)

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

X, y, encoder, lb = process_data(
    data=data,
    categorical_features=cat_features,
    label="salary",
    encoder=encoder,
    lb=lb
)
    


for i in range(len(features)):
    print(f"Computing metrics for feature: {features[i]}, value: {values[i]}")
    X_slice=X[data[features[i]]==values[i]]
    y_slice=y[data[features[i]]==values[i]]
    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    # Append the results to a file
    with open(OUTPUT_PATH, "a") as fp:
        fp.write(f"Feature: {features[i]}, Value: {values[i]}\n") 
        fp.write(f"    Precision: {precision}, Recall: {recall}, F1: {fbeta}\n\n")


    


    
    