import subprocess
import requests
from starter.main import CensusInput

feature_census_input = CensusInput(
    age=39,
    workclass="State-gov",
    fnlgt=77516,
    education="Bachelors",
    education_num=13,
    marital_status="Never-married",
    occupation="Adm-clerical",
    relationship="Not-in-family",
    race="White",
    sex="Male",
    capital_gain=2174,
    capital_loss=0,
    hours_per_week=40,
    native_country="United-States"
)

response1 = str(requests.get('https://nd0821-c3-starter-code-ux0t.onrender.com/').content)
response2 = str(requests.post('http://127.0.0.1:8000/train').content)
response3_raw = requests.post(
    'http://127.0.0.1:8000/inference',
    json=feature_census_input.model_dump(by_alias=True)
)
response3 = str(response3_raw.content)



print(response3_raw.status_code)
print(response3)