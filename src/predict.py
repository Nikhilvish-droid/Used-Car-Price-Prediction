from pathlib import Path
import joblib
import pandas as pd
import numpy as np

model_dir = Path(__file__).parent.parent/'models'

def load_model():
    model = joblib.load(model_dir/'cat_model.pkl')
    return model

def user():
    brand = input("enter brand name : ")
    model = input("enter model name : ")
    age = int(input("for how many years do you have this car : "))
    km_driven = int(input("how much you have drive your car : "))
    transmission = int(input("is it manual : 0 or automatic : 1  : "))
    owner = int(input("are you first owner : 1 or second : 0 owner : "))
    fueltype = input("what's your car fuel type- petrol, diesel, Hybrid/CNG write same any of them : ")


    brand = brand.strip().lower()
    model_name = model.strip().lower()
    fueltype = fueltype.strip().capitalize()

    km_per_year = km_driven/max(age, 1)
    log_km = np.log1p(km_driven)
    brand_model = brand + ' ' + model_name

    user_input = pd.DataFrame([{
        'Brand': brand,
        'model': model_name,
        'Age' : age,
        'kmDriven' : km_driven,
        'Transmission': transmission,
        'Owner' : owner,
        'FuelType' : fueltype,
        'Brand_model' : brand_model,
        'km_per_year' : km_per_year,
        'log_km' : log_km,
    }])

    model = load_model()
    user_input = user_input[model.feature_names_]   

    pred_log_price = model.predict(user_input)[0]
    pred_price = np.exp(pred_log_price)
    print(f"estimated price --> {pred_price:,.0f} ")



if __name__ == "__main__":
    user()