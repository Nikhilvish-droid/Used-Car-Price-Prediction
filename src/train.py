from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score

from catboost import CatBoostRegressor

data_dir = Path(__file__).parent.parent/'data'
model_dir = Path(__file__).parent.parent/'models'

df = pd.read_csv(data_dir/'clean.csv')

df = df.drop(columns=['Brand_avgPrice', 'Model_avgPrice'])
df['Brand_model'] = df['Brand'] + ' ' + df['model']
df['km_per_year'] = df['kmDriven'] / df['Age'].clip(lower=1)
df['log_km'] = np.log1p(df['kmDriven'])

cat_cols = ['Brand', 'model', 'Brand_model', 'FuelType']

X = df.drop(columns=['AskPrice'])
y = np.log(df['AskPrice'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

cat_model = CatBoostRegressor(
    iterations=1500,
    random_seed= 42,
    learning_rate = 0.05,
    cat_features= cat_cols,
    depth = 7,
    verbose = 0
)
print("loaded cat model")

cat_model.fit(X_train, y_train)
print("model train successfully")


y_pred_log = cat_model.predict(X_test)
y_true_log = y_test


y_true, y_pred = np.exp(y_true_log), np.exp(y_pred_log)   # back to rupees
print(f"R2 (rupees)        : {r2_score(y_true, y_pred):.4f}")
print(f"R2 (log scale)     : {r2_score(y_true_log, y_pred_log):.4f}")
print(f"MAE                : {mean_absolute_error(y_true, y_pred):,.0f}")
print(f"MAPE               : {mean_absolute_percentage_error(y_true, y_pred):.3f}")
print(f"accuracy       : {(1- mean_absolute_percentage_error(y_true, y_pred))*100}")
print(f"median % error     : {np.median(np.abs(y_pred - y_true) / y_true):.3f}")
print(f"rmse : {root_mean_squared_error(y_true, y_pred)}")


joblib.dump(cat_model, model_dir/'cat_model.pkl')
print("done")