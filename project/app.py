import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sklearn.ensemble import RandomForestRegressor
import shap
from mangum import Mangum  # Required for Vercel Serverless routing

app = FastAPI()

# --- VERCEL-SAFE DYNAMIC PATH ROUTING ---
# Resolves path differences between local Windows testing and Linux Vercel servers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
ARCHIVE_PATH = os.path.join(BASE_DIR, "archive")

# --- DATA PREPROCESSING & MODEL TRAINING ---
def train_model_on_cgm(filename):
    """
    Handles large Kaggle CGM files natively using cloud-safe relative paths.
    Uses Temporal Feature Engineering (Lags) instead of Heart Rate.
    """
    file_path = os.path.join(ARCHIVE_PATH, filename)
    
    print(f"Loading and Training on {file_path}...")
    df = pd.read_csv(file_path)
    
    # Standardize columns to lowercase
    df.columns = [c.lower() for c in df.columns]
    
    # Feature Engineering: Predict next glucose based on previous trends
    df['prev_1'] = df['glucose'].shift(1)
    df['prev_2'] = df['glucose'].shift(2)
    df['trend'] = df['prev_1'] - df['prev_2']
    
    df = df.dropna()
    
    # Features (X) and Target (y)
    X = df[['prev_1', 'prev_2', 'trend']]
    y = df['glucose']
    
    # Train the model
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(X, y)
    
    # Initialize SHAP for explainability
    explainer = shap.TreeExplainer(model)
    
    return model, explainer

# Initialize models
models = {
    "P-461": train_model_on_cgm("g4_Patient_461_3.csv"),
    "P-056": train_model_on_cgm("g5_Patient_56_10.csv")
}

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# FIXED: Swapped to @app.post and explicitly bound Form parameters
@app.post("/index")
async def index(
    username: str = Form(...), 
    password: str = Form(...)
):
    processed_username = username.strip().lower()
    processed_password = password.strip()
    
    if processed_username == "doctor" and processed_password == "doctor123":
        return RedirectResponse(url="/doctor", status_code=303)
    
    if processed_username == "patient" and processed_password == "patient123":
        return RedirectResponse(url="/patient/P-461", status_code=303)
    
    return HTMLResponse("Invalid Credentials. <a href='/'>Try again</a>")

@app.get("/doctor", response_class=HTMLResponse)
async def doctor_portal(request: Request):
    patients = list(models.keys())
    return templates.TemplateResponse("doctor_dashboard.html", {"request": request, "patients": patients})

@app.get("/patient/{patient_id}", response_class=HTMLResponse)
async def patient_portal(request: Request, patient_id: str):
    return templates.TemplateResponse("patient_dashboard.html", {"request": request, "patient_id": patient_id})

@app.get("/get-data/{patient_id}/{row}")
async def get_data(patient_id: str, row: int):
    model, explainer = models[patient_id]
    
    file_map = {"P-461": "g4_Patient_461_3.csv", "P-056": "g5_Patient_56_10.csv"}
    file_path = os.path.join(ARCHIVE_PATH, file_map[patient_id])
    
    df = pd.read_csv(file_path)
    df.columns = [c.lower() for c in df.columns]
    
    curr_idx = row % len(df)
    if curr_idx < 2: curr_idx = 2
    
    val_now = float(df.iloc[curr_idx]['glucose'])
    val_prev1 = float(df.iloc[curr_idx-1]['glucose'])
    val_prev2 = float(df.iloc[curr_idx-2]['glucose'])
    trend = val_prev1 - val_prev2
    
    X_input = np.array([[val_prev1, val_prev2, trend]])
    prediction = model.predict(X_input)[0]
    shap_values = explainer.shap_values(X_input)
    
    return {
        "current": round(val_now, 1),
        "prediction": round(prediction, 1),
        "explanation": {
            "immediate_history_impact": round(float(shap_values[0][0]), 2),
            "trend_impact": round(float(shap_values[0][2]), 2)
        }
    }

# --- VERCEL ASG HANDLER ENTRYPOINT ---
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
