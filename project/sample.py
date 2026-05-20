import pandas as pd
import numpy as np
import os
import shap

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# FASTAPI SETUP
# ==========================================

app = FastAPI()

templates = Jinja2Templates(
    directory="templates"
)

# ==========================================
# DATASET FOLDER
# ==========================================

ARCHIVE_PATH = r"C:\Users\Nausheen\Desktop\project\archive"

# ==========================================
# TRAINING FUNCTION
# ==========================================

def train_model_on_cgm(filename):

    """
    Train glucose prediction model
    using temporal glucose history.
    """

    # Full path
    file_path = os.path.join(
        ARCHIVE_PATH,
        filename
    )

    print(f"\nLoading dataset: {file_path}")

    # Load CSV
    df = pd.read_csv(file_path)

    # Lowercase columns
    df.columns = [
        c.lower()
        for c in df.columns
    ]

    # ==========================================
    # FEATURE ENGINEERING
    # ==========================================

    # Previous glucose values
    df['prev_1'] = df['glucose'].shift(1)
    df['prev_2'] = df['glucose'].shift(2)
    df['prev_3'] = df['glucose'].shift(3)
    df['prev_4'] = df['glucose'].shift(4)
    df['prev_5'] = df['glucose'].shift(5)

    # Trend
    df['trend'] = (
        df['prev_1'] - df['prev_2']
    )

    # Rolling mean
    df['rolling_mean'] = (
        df['glucose']
        .rolling(window=3)
        .mean()
    )

    # Predict NEXT glucose value
    df['target'] = (
        df['glucose'].shift(-1)
    )

    # Remove NaN rows
    df = df.dropna()

    # ==========================================
    # FEATURES & TARGET
    # ==========================================

    X = df[
        [
            'prev_1',
            'prev_2',
            'prev_3',
            'prev_4',
            'prev_5',
            'trend',
            'rolling_mean'
        ]
    ]

    y = df['target']

    # ==========================================
    # TIME SERIES SPLIT
    # ==========================================

    split_index = int(len(X) * 0.8)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    # ==========================================
    # RANDOM FOREST MODEL
    # ==========================================

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # ==========================================
    # PREDICTIONS
    # ==========================================

    y_pred = model.predict(X_test)

    # ==========================================
    # METRICS
    # ==========================================

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    # ==========================================
    # PRINT RESULTS
    # ==========================================

    print("\n========== MODEL PERFORMANCE ==========")

    print(f"Dataset : {filename}")

    print(f"MAE     : {mae:.2f}")

    print(f"RMSE    : {rmse:.2f}")

    print(f"R² Score: {r2:.2f}")

    print("=======================================\n")

    # ==========================================
    # SHAP EXPLAINER
    # ==========================================

    explainer = shap.TreeExplainer(
        model
    )

    return model, explainer


# ==========================================
# LOAD ALL MODELS
# ==========================================

models = {

    "P-461":
    train_model_on_cgm(
        "g4_Patient_461_3.csv"
    ),

    "P-056":
    train_model_on_cgm(
        "g5_Patient_56_10.csv"
    ),

    "P-006":
    train_model_on_cgm(
        "g4_Patient_402_7.csv"
    ),

    "P-156":
    train_model_on_cgm(
        "g5_Patient_7_3.csv"
    ),

    "P-556":
    train_model_on_cgm(
        "g5_Patient_67_5.csv"
    ),

    "P-051":
    train_model_on_cgm(
        "g5_Patient_474_7.csv"
    ),

    "P-050":
    train_model_on_cgm(
        "g5_Patient_484_2.csv"
    ),

    "P-057":
    train_model_on_cgm(
        "g5_Patient_489_1.csv"
    )
}
# ==========================================
# FILE MAP
# ==========================================

file_map = {

    "P-461":
    "g4_Patient_461_3.csv",

    "P-056":
    "g5_Patient_56_10.csv",

    "P-006":
    "archive/g4_Patient_402_7.csv",

    "P-156":
    "archive/g5_Patient_7_3.csv",

    "P-556":
    "archive/g5_Patient_67_5.csv",

    "P-051":
    "archive/g5_Patient_474_7.csv",

    "P-050":
    "archive/g5_Patient_484_2.csv",

    "P-057":
    "archive/g5_Patient_489_1.csv"
}

# ==========================================
# ROUTES
# ==========================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def login_page(request: Request):

    return templates.TemplateResponse(
        "login.html",
        {
            "request": request
        }
    )


# ==========================================
# LOGIN
# ==========================================

@app.post("/login")
async def login(request: Request):

    form_data = await request.form()

    username = str(
        form_data.get("username")
    ).strip().lower()

    password = str(
        form_data.get("password")
    ).strip()

    # Doctor login
    if (
        username == "doctor"
        and password == "doctor123"
    ):

        return RedirectResponse(
            url="/doctor",
            status_code=303
        )

    # Patient login
    if (
        username == "patient"
        and password == "patient123"
    ):

        return RedirectResponse(
            url="/patient/P-461",
            status_code=303
        )

    return HTMLResponse(
        "Invalid Credentials. "
        "<a href='/'>Try again</a>"
    )


# ==========================================
# DOCTOR DASHBOARD
# ==========================================

@app.get(
    "/doctor",
    response_class=HTMLResponse
)
async def doctor_portal(request: Request):

    patients = list(
        models.keys()
    )

    return templates.TemplateResponse(
        "doctor_dashboard.html",
        {
            "request": request,
            "patients": patients
        }
    )


# ==========================================
# PATIENT DASHBOARD
# ==========================================

@app.get(
    "/patient/{patient_id}",
    response_class=HTMLResponse
)
async def patient_portal(
    request: Request,
    patient_id: str
):

    return templates.TemplateResponse(
        "patient_dashboard.html",
        {
            "request": request,
            "patient_id": patient_id
        }
    )


# ==========================================
# REAL-TIME PREDICTION API
# ==========================================

@app.get("/get-data/{patient_id}/{row}")
async def get_data(
    patient_id: str,
    row: int
):

    # Get model
    model, explainer = models[
        patient_id
    ]

    # Get CSV file
    file_path = os.path.join(
        ARCHIVE_PATH,
        file_map[patient_id]
    )

    # Load dataset
    df = pd.read_csv(file_path)

    df.columns = [
        c.lower()
        for c in df.columns
    ]

    # Prevent invalid rows
    curr_idx = row % len(df)

    if curr_idx < 5:
        curr_idx = 5

    # ==========================================
    # CURRENT VALUES
    # ==========================================

    current_glucose = float(
        df.iloc[curr_idx]['glucose']
    )

    prev_1 = float(
        df.iloc[curr_idx - 1]['glucose']
    )

    prev_2 = float(
        df.iloc[curr_idx - 2]['glucose']
    )

    prev_3 = float(
        df.iloc[curr_idx - 3]['glucose']
    )

    prev_4 = float(
        df.iloc[curr_idx - 4]['glucose']
    )

    prev_5 = float(
        df.iloc[curr_idx - 5]['glucose']
    )

    trend = prev_1 - prev_2

    rolling_mean = np.mean(
        [
            prev_1,
            prev_2,
            prev_3
        ]
    )

    # ==========================================
    # MODEL INPUT
    # ==========================================

    X_input = np.array([
        [
            prev_1,
            prev_2,
            prev_3,
            prev_4,
            prev_5,
            trend,
            rolling_mean
        ]
    ])

    # ==========================================
    # PREDICT
    # ==========================================

    next_prediction = model.predict(
        X_input
    )[0]

    # ==========================================
    # SHAP VALUES
    # ==========================================

    shap_values = explainer.shap_values(
        X_input
    )

    # ==========================================
    # RETURN RESPONSE
    # ==========================================

    return {

        "current_glucose":
        round(current_glucose, 1),

        "next_glucose_prediction":
        round(
            float(next_prediction),
            1
        ),

        "features_used": {

            "prev_1":
            round(prev_1, 1),

            "prev_2":
            round(prev_2, 1),

            "prev_3":
            round(prev_3, 1),

            "prev_4":
            round(prev_4, 1),

            "prev_5":
            round(prev_5, 1),

            "trend":
            round(trend, 1),

            "rolling_mean":
            round(rolling_mean, 1)
        },

        "model_explanation": {

            "prev_1_impact":
            round(
                float(shap_values[0][0]),
                2
            ),

            "prev_2_impact":
            round(
                float(shap_values[0][1]),
                2
            ),

            "prev_3_impact":
            round(
                float(shap_values[0][2]),
                2
            ),

            "prev_4_impact":
            round(
                float(shap_values[0][3]),
                2
            ),

            "prev_5_impact":
            round(
                float(shap_values[0][4]),
                2
            ),

            "trend_impact":
            round(
                float(shap_values[0][5]),
                2
            ),

            "rolling_mean_impact":
            round(
                float(shap_values[0][6]),
                2
            )
        }
    }


# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )