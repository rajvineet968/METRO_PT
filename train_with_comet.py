from comet_ml import Experiment
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. COMET EXPERIMENT
# -----------------------------
experiment = Experiment(
    api_key="yQNUEXDGR3j7ZETInrSTCiiCI",
    project_name="dv-pressure-project",
    workspace="vineet-raj"
)

# -----------------------------
# 2. Load data
# -----------------------------
df = pd.read_csv("data/metropt.csv")
df = df.drop(columns=["Unnamed: 0", "timestamp"], errors="ignore")

X = df.drop(columns=["DV_pressure"])
y = df["DV_pressure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Train model
# -----------------------------
model = RandomForestRegressor(n_estimators=200)
model.fit(X_train, y_train)

# -----------------------------
# 4. Evaluate
# -----------------------------
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred, squared=False)
r2 = r2_score(y_test, pred)

experiment.log_metric("mae", mae)
experiment.log_metric("rmse", rmse)
experiment.log_metric("r2", r2)

# -----------------------------
# 5. Save Model
# -----------------------------
joblib.dump(model, "rf_model.joblib")
experiment.log_model("rf_model", "rf_model.joblib")

print("Training Done. Model saved.")
