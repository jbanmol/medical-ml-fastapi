# Medical ML FastAPI

## Overview
A simple end-to-end deployment of a machine-learning model using FastAPI. It classifies samples from the Wisconsin Breast Cancer Diagnostic dataset as malignant (0) or benign (1).

## Disclaimer
This project is for educational demonstration only and must not be used for medical diagnosis or clinical decision-making.

## Dataset
Uses `sklearn.datasets.load_breast_cancer`: 569 public samples and 30 original numeric features. The predetermined subset has 10 features:

mean radius, mean texture, mean perimeter, mean area, mean smoothness, mean compactness, mean concavity, mean concave points, worst radius, worst texture.

`features.py` defines the mapping from snake_case HTTP field names to the dataset columns and their order. No private patient data is used. The example below is dataset row 256, taken from the held-out test set (true label: malignant).

## Model
The saved sklearn Pipeline contains StandardScaler and LogisticRegression (`max_iter=2000`). An 80/20 stratified split with seed 42 gives 455 training and 114 test samples. Scaling is fitted only on training data. No feature selection or tuning is performed on the test set.

Held-out results; **malignant (code 0) is the positive class**:

| Metric | Value |
|---|---:|
| Accuracy | 0.947368 |
| Precision | 0.909091 |
| Recall | 0.952381 |
| F1 | 0.930233 |

Confusion matrix (rows = actual, columns = predicted, order = malignant, benign): `[[40, 2], [4, 68]]`. These are educational test results, not clinical validation. Probabilities are the model's uncalibrated estimates.

## Project Structure
- `train.py`: train, evaluate, save and reload-check the pipeline
- `features.py`: shared feature mapping and order
- `main.py`: FastAPI startup, health and prediction routes
- `model.pkl`: complete trained pipeline, including scaler
- `model_metadata.json`: dataset, class mapping, versions and measured metrics
- `example_request.json`, `example_response.json`: actual sample and API output
- `requirements.txt`, `requirements-dev.txt`: pinned dependencies
- `tests/test_api.py`: API, validation and failure tests
- `Dockerfile`, `.dockerignore`, `render.yaml`: deployment configuration

## Installation
```bash
git clone https://github.com/jbanmol/medical-ml-fastapi.git
cd medical-ml-fastapi
python3.11 -m venv .venv
```

macOS/Linux: `source .venv/bin/activate`

Windows PowerShell: `.venv\Scripts\Activate.ps1`

```bash
pip install -r requirements.txt
```

Use Python 3.11.15 and scikit-learn 1.7.2 to match the exported model. Only load trusted pickle files: deserializing a pickle can execute code.

## Train the model
```bash
python train.py
```
Creates `model.pkl`, `model_metadata.json` and a real `example_request.json`. The complete scaler/classifier pipeline is serialized with joblib and reloaded to check matching predictions. The committed pipeline is loaded during API startup; the server never trains it.

## Run locally
```bash
uvicorn main:app --reload
```

## API Documentation
Open http://127.0.0.1:8000/docs. The Swagger example contains the real sample below.

## Health Endpoint
`GET /health` returns HTTP 200 when the saved model is available:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

If loading fails, health reports HTTP 503 with `status: error` and `model_loaded: false`; prediction also returns 503. Prediction failures return 500 without internal details; diagnostic exceptions are logged on the server.

## Prediction Endpoint
`POST /predict` accepts these exact 10 numeric, finite, non-negative fields. Missing, invalid, negative or extra fields return HTTP 422.

Actual request:
```json
{
  "mean_radius": 19.55,
  "mean_texture": 28.77,
  "mean_perimeter": 133.6,
  "mean_area": 1207.0,
  "mean_smoothness": 0.0926,
  "mean_compactness": 0.2063,
  "mean_concavity": 0.1784,
  "mean_concave_points": 0.1144,
  "worst_radius": 25.05,
  "worst_texture": 36.27
}
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

Actual API response (HTTP 200):
```json
{
  "prediction": "malignant",
  "prediction_code": 0,
  "probability": 1.0,
  "class_probabilities": {
    "malignant": 1.0,
    "benign": 0.0
  },
  "disclaimer": "This project is for educational demonstration only and must not be used for medical diagnosis or clinical decision-making."
}
```

## Tests
```bash
pip install -r requirements-dev.txt
python -m pytest -v
```
14 tests cover home, health, prediction, validation, Swagger, missing/corrupt models and prediction failures.

## Docker
```bash
docker build -t medical-ml-fastapi .
docker run --rm -p 8000:8000 medical-ml-fastapi
```
The image includes the committed model and uses `PORT` (default 8000).

## Render Configuration
Native Python web service, free plan, Singapore region. Build: `pip install -r requirements.txt`. Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`. Python is fixed to 3.11.15. No database, secrets or external API credentials are required. `render.yaml` records equivalent Blueprint settings with `/health` as the health-check path. The live service settings are verified separately after creation.
