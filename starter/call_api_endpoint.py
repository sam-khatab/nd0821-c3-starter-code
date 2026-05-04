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

def call_api():
    response1 = requests.get('https://nd0821-c3-starter-code-ux0t.onrender.com/')
    response2 = requests.post('https://nd0821-c3-starter-code-ux0t.onrender.com/train')
    response3 = requests.post(
        'https://nd0821-c3-starter-code-ux0t.onrender.com/inference',
        json=feature_census_input.model_dump(by_alias=True)
    )




    print(f"Get Response Status Code: {response1.status_code}")
    print(f"Get Response: {str(response1.content)}")
    print(f"Train Response Status Code: {response2.status_code}")
    print(f"Train Response: {str(response2.content)}")
    print(f"Inference Response Status Code: {response3.status_code}")
    print(f"Inference Response: {str(response3.content)}")

if __name__ == "__main__":
    call_api()