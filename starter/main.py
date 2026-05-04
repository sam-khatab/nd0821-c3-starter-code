# Put the code for your API here.
from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from joblib import load
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


from starter.ml.data import process_data
from starter.ml.model import inference


app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "rf_model.joblib"
ENCODER_PATH = BASE_DIR / "model" / "encoder.joblib"
LB_PATH = BASE_DIR / "model" / "lb.joblib"
RF_MODEL = load(MODEL_PATH) if MODEL_PATH.exists() else None
ENCODER = load(ENCODER_PATH) if ENCODER_PATH.exists() else None
LB = load(LB_PATH) if LB_PATH.exists() else None

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class CensusInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(example=39)
    workclass: str = Field(example="State-gov")
    fnlgt: int = Field(example=77516)
    education: str = Field(example="Bachelors")
    education_num: int = Field(alias="education-num", example=13)
    marital_status: str = Field(
        alias="marital-status", example="Never-married")
    occupation: str = Field(example="Adm-clerical")
    relationship: str = Field(example="Not-in-family")
    race: str = Field(example="White")
    sex: str = Field(example="Male")
    capital_gain: int = Field(alias="capital-gain", example=2174)
    capital_loss: int = Field(alias="capital-loss", example=0)
    hours_per_week: int = Field(alias="hours-per-week", example=40)
    native_country: str = Field(
        alias="native-country", example="United-States")

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "age": self.age,
                    "workclass": self.workclass,
                    "fnlgt": self.fnlgt,
                    "education": self.education,
                    "education-num": self.education_num,
                    "marital-status": self.marital_status,
                    "occupation": self.occupation,
                    "relationship": self.relationship,
                    "race": self.race,
                    "sex": self.sex,
                    "capital-gain": self.capital_gain,
                    "capital-loss": self.capital_loss,
                    "hours-per-week": self.hours_per_week,
                    "native-country": self.native_country,
                }
            ]
        )


@app.get("/")
async def get_items():
    return "Welcome to API HELL"


@app.post("/train")
async def train():
    return {"message": "Run starter/starter/train_model.py to retrain and save the model artifacts."}


@app.post("/inference")
async def inference_app(payload: CensusInput):
    if RF_MODEL is None:
        raise HTTPException(
            status_code=500,
            detail="Model file not found: model/rf_model.joblib"
        )
    if ENCODER is None:
        raise HTTPException(
            status_code=500,
            detail="Encoder file not found: model/encoder.joblib"
        )
    if LB is None:
        raise HTTPException(
            status_code=500,
            detail="Label binarizer file not found: model/lb.joblib"
        )

    processed_features, _, _, _ = process_data(
        data=payload.to_dataframe(),
        categorical_features=CATEGORICAL_FEATURES,
        label=None,
        encoder=ENCODER,
        lb=None,
    )
    preds = inference(model=RF_MODEL, X=processed_features)
    decoded_preds = LB.inverse_transform(preds)
    return {"prediction": decoded_preds[0]}
