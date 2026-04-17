from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import pickle

app = FastAPI()

# Static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Load models (dictionary)
models = pickle.load(open("breast_model.pkl1", "rb"))

# Load best algorithm
with open("best_algorithm.txt", "r") as f:
    best_algo = f.read().strip()

# Select best model
best_model = models[best_algo]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "best_result": None,
        "predictions": None
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    radius_mean: float = Form(...),
    perimeter_mean: float = Form(...),
    area_mean: float = Form(...),
    concavity_mean: float = Form(...),
    concave_points_mean: float = Form(...),
    fractal_dimension_mean: float = Form(...),
    radius_worst: float = Form(...),
    perimeter_worst: float = Form(...),
    area_worst: float = Form(...),
    concavity_worst: float = Form(...),
    fractal_dimension_worst: float = Form(...)
):

    # Input array
    user_data = np.array([[
        radius_mean,
        perimeter_mean,
        area_mean,
        concavity_mean,
        concave_points_mean,
        fractal_dimension_mean,
        radius_worst,
        perimeter_worst,
        area_worst,
        concavity_worst,
        fractal_dimension_worst
    ]])

    # ✅ BEST MODEL PREDICTION (FIXED)
    best_pred = best_model.predict(user_data)[0]

    if best_pred == 1:
        best_result = "Malignant"
    else:
        best_result = "Benign"

    print("BEST RESULT:", best_result)  # debug

    # ✅ ALL MODELS PREDICTION
    predictions = {}

    for name, model in models.items():
        pred = model.predict(user_data)[0]
        predictions[name] = "Malignant" if pred == 1 else "Benign"

    # ✅ RETURN TO HTML
    return templates.TemplateResponse("index.html", {
        "request": request,
        "best_result": best_result,
        "best_algo": best_algo,
        "predictions": predictions
    })