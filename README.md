# Used Car Price Prediction

A machine learning model that predicts the asking price (in ₹) of a used car from its brand, model,
age, kilometres driven, fuel type, transmission and ownership history.

## Overview

Given the details of a used-car listing, the model estimates what price the seller is likely to ask.
The final model is a CatBoost regressor trained on the log of the price (prices are heavily
right-skewed), with predictions converted back to rupees for evaluation.

## Project structure

```
Used-Car-Price-Prediction/
├── data/
│   ├── used_cars.csv          # raw dataset (~15k car listings)
│   └── clean.csv              # cleaned dataset used for training
├── models/
│   ├── cat_model.pkl          # final CatBoost model (saved by src/train.py)
│   └── used_car_price_prediction.pkl  # earlier model from the first experiments
├── notebooks/
│   ├── EDA.ipynb              # data cleaning & exploratory analysis
│   ├── featureBuild.ipynb     # brand / model score feature experiments
│   ├── demo.ipynb             # first modelling attempt (Random Forest, XGBoost, CatBoost + grid search)
│   ├── model_train.ipynb      # model training experiments
│   ├── model_train_v2.ipynb   # final approach: CatBoost on log price + 5-fold CV
│   └── colab_modeltrain.ipynb # training run on Google Colab
├── src/
│   └── train.py               # script version of the final training pipeline
├── requirements.txt
└── README.md
```

## Dataset

Each row is one used-car listing, with columns such as:

- **Car**: `Brand`, `model`, `Year`, `Age`, `FuelType`, `Transmission`
- **Usage**: `kmDriven`, `Owner`
- **Listing**: `PostedDate`, `AdditionInfo`
- **Target**: `AskPrice` — the seller's asking price in ₹

`used_cars.csv` is the raw data. In `EDA.ipynb` the null rows and duplicates are dropped, `kmDriven`
(`"98,000 km"`) and `AskPrice` (`"₹ 1,95,000"`) are converted to integers, text columns are stripped
and lower-cased, `Transmission` and `Owner` are encoded as 0/1, and the columns that don't help
(`PostedDate`, `AdditionInfo`, `Year` — redundant with `Age`) are dropped. The result is saved as
`clean.csv`, which is what the model trains on.

## Approach

1. **EDA** (`notebooks/EDA.ipynb`) — checked nulls, duplicates and dtypes, looked at outliers in
   `Age`, `kmDriven` and `AskPrice`, and plotted price against age, mileage, fuel type, transmission
   and brand to see what drives the price.
2. **First attempt** (`notebooks/demo.ipynb`) — one-hot / target encoding with Random Forest, XGBoost
   and CatBoost tuned with `GridSearchCV`. It used `Brand_avgPrice` and `Model_avgPrice` columns
   computed from the full dataset.
3. **Removing leakage** — those two average-price columns are built from `AskPrice` itself, so they
   leak the target into the features. `train.py` drops them.
4. **Final model** (`notebooks/model_train_v2.ipynb`, mirrored in `src/train.py`):
   - engineered features: `Brand_model` (brand + model combined), `km_per_year`, `log_km`
   - target is `log(AskPrice)`, so the model isn't dominated by a few very expensive cars
   - `CatBoostRegressor` (1500 iterations, depth 7, learning rate 0.05) with `Brand`, `model`,
     `Brand_model` and `FuelType` passed as native categorical features — no manual encoding needed
   - 80/20 train/test split with `random_state=42`
5. **Evaluation** — R², MAE, MAPE, median % error and RMSE are printed after training, with
   predictions converted back from log scale to rupees first.

## Getting started

```bash
pip install pandas numpy scikit-learn catboost joblib

# retrain the model from data/clean.csv and save it to models/cat_model.pkl
python src/train.py
```

## Model evaluation

Held-out 20% test set:

- R² (rupees) --> 0.6906
- R² (log scale) --> 0.8552
- Mean absolute error --> ₹2,46,886
- Root mean squared error --> ₹8,27,781
- MAPE --> 0.395
- Median % error --> 13.6%

The typical (median) prediction is within about 14% of the asking price. The average error is much
higher because a small number of expensive and rare cars are hard to predict, which pulls up MAE,
RMSE and MAPE.

## Notes / what's next

- `requirements.txt` is currently empty — fill it with the packages above (`pip freeze` inside your
  environment will give exact versions).
- There is no `predict.py` yet; adding a script (or a Streamlit app) that takes a car's details and
  loads `models/cat_model.pkl` is the natural next step.
- Rare brands and models have very few listings, so predictions for them are less reliable.
- Ideas to try: removing price outliers, hyper-parameter tuning on the log-price model, and features
  from the `AdditionInfo` text.
