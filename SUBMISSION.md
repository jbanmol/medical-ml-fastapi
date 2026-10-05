PROJECT COMPLETE

GitHub Repository:
https://github.com/jbanmol/medical-ml-fastapi

Live API:
https://medical-ml-fastapi.onrender.com

Home:
https://medical-ml-fastapi.onrender.com/

Health Endpoint:
https://medical-ml-fastapi.onrender.com/health

Prediction Endpoint:
https://medical-ml-fastapi.onrender.com/predict

Swagger Documentation:
https://medical-ml-fastapi.onrender.com/docs

Dataset:
Wisconsin Breast Cancer Diagnostic Dataset

Model:
StandardScaler + LogisticRegression (sklearn Pipeline)

Selected Features:
mean radius, mean texture, mean perimeter, mean area, mean smoothness, mean compactness, mean concavity, mean concave points, worst radius, worst texture

Test Metrics (positive class: malignant, code 0):
Accuracy: 0.947368
Precision: 0.909091
Recall: 0.952381
F1: 0.930233

Automated Tests:
14 passed

Live Health Verification:
HTTP 200
```json
{
  "status": "ok",
  "model_loaded": true
}
```

Live Prediction Verification:
HTTP 200
Request:
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

Response:
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

Screenshot:
[screenshots/successful_prediction.jpg](screenshots/successful_prediction.jpg), 183,528 bytes, genuine live Swagger request and HTTP 200 response.

Submission Answers:
1. GitHub repo link: https://github.com/jbanmol/medical-ml-fastapi
2. Deployed API URL: https://medical-ml-fastapi.onrender.com
3. Screenshot: screenshots/successful_prediction.jpg

Files Created:
- main.py
- train.py
- features.py
- model.pkl
- model_metadata.json
- requirements.txt
- requirements-dev.txt
- README.md
- .gitignore
- .dockerignore
- .python-version
- Dockerfile
- render.yaml
- example_request.json
- example_response.json
- live_verification.json
- tests/test_api.py
- screenshots/successful_prediction.jpg
- SUBMISSION.md

Remaining Manual Action:
None

Complete Flow:
- scikit-learn provides 569 public samples. X is the 10 selected measurements; y is the malignant/benign label.
- A stratified split keeps similar class proportions across 455 training and 114 test samples.
- StandardScaler learns measurement scales from training samples only; LogisticRegression learns the classification rule.
- The saved model.pkl contains both steps. joblib.dump saves them and joblib.load restores them without training again.
- main.py loads the pipeline at startup. /health truthfully reports whether it loaded.
- FastAPI accepts POST /predict; Pydantic validates the JSON before inference.
- A named DataFrame orders the inputs correctly; the pipeline scales them, predicts a class and returns probabilities with an educational disclaimer.
- requirements.txt fixes package versions. Docker packages the same application for optional container execution.
- GitHub stores the code and trained model; Render installs dependencies and runs Uvicorn on its assigned public port.
- Local tests, actual public HTTP requests and the live Swagger screenshot provide separate evidence that the deployment works.
