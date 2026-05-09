import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "data/bahrain_2024_laps.csv"
MODEL_PATH = "models/tyre_degradation_model.pkl"

Path("models").mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

features = [
    "Driver",
    "Team",
    "Compound",
    "TyreLife",
    "Stint",
    "LapNumber",
]

target = "Degradation"

df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

categorical_features = ["Driver", "Team", "Compound"]
numeric_features = ["TyreLife", "Stint", "LapNumber"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=8,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))
print("RMSE:", mean_squared_error(y_test, preds) ** 0.5)
print("R2:", r2_score(y_test, preds))

joblib.dump(pipeline, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
