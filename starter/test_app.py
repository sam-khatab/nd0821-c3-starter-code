import json
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome to API HELL"


def test_post_inference_low():
    payload = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    r = client.post("/inference", data=json.dumps(payload))
    print(r.json()["prediction"][0])
    assert r.status_code == 200
    assert r.json()["prediction"] == "<=50K"


def test_post_inference_high():
    payload = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States"
    }

    r = client.post("/inference", data=json.dumps(payload))
    print(r.json()["prediction"][0])
    assert r.status_code == 200
    assert r.json()["prediction"] == ">50K"
