import os, time, joblib, traceback
import numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/saved_models")
def find_model(path):
    exts = (".joblib", ".pkl", ".sav")
    files = [f for f in os.listdir(path) if f.lower().endswith(exts)]
    if not files:
        raise FileNotFoundError(f"No model file found in {path}")
    files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(path,f)), reverse=True)
    return os.path.join(path, files[0])

MODEL_PATH = find_model(MODEL_DIR)
print("Loading model from:", MODEL_PATH)
model_bundle = joblib.load(MODEL_PATH)
if isinstance(model_bundle, dict) and "model" in model_bundle:
    model = model_bundle["model"]
    preprocessor = model_bundle.get("preprocessor", None)
else:
    model = model_bundle
    preprocessor = None

app = FastAPI(title="MetroPT Model API")

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTION_COUNT = Counter("predictions_total", "Total predictions")
PREDICTION_ERRORS = Counter("prediction_errors_total", "Total prediction errors")
MODEL_MAE = Gauge("model_mae", "Last MAE")
MODEL_RMSE = Gauge("model_rmse", "Last RMSE")
MODEL_R2 = Gauge("model_r2", "Last R2")

class PredictRequest(BaseModel):
    instances: list

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    endpoint = request.url.path
    method = request.method
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.time() - start
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status)).inc()

@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}

@app.post("/predict")
def predict(payload: PredictRequest):
    PREDICTION_COUNT.inc()
    try:
        instances = payload.instances
        if len(instances) == 0:
            raise HTTPException(status_code=400, detail="No instances provided.")
        X = pd.DataFrame(instances)
        if preprocessor is not None:
            X_proc = preprocessor.transform(X)
        else:
            X_proc = X.values
        preds = model.predict(X_proc)
        return {"predictions": [float(p) for p in preds]}
    except Exception as e:
        PREDICTION_ERRORS.inc()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
