import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path("models/modelo_mora.pkl")
METRICS_PATH = Path("reports/metricas_modelo.json")

app = FastAPI(
    title="API de predicción de mora",
    description="API para predecir la probabilidad de falta de pago en créditos.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    features: Dict[str, float]


def cargar_modelo_y_columnas():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")

    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de métricas en {METRICS_PATH}")

    modelo = joblib.load(MODEL_PATH)

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metricas = json.load(f)

    columnas = metricas["feature_columns"]

    return modelo, columnas, metricas


@app.get("/")
def home():
    return {
        "mensaje": "API de predicción de mora operativa",
        "endpoint_prediccion": "/predict",
        "documentacion": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def features():
    try:
        _, columnas, metricas = cargar_modelo_y_columnas()
        return {
            "selected_model": metricas["selected_model"],
            "target": metricas["target"],
            "feature_columns": columnas,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        modelo, columnas, _ = cargar_modelo_y_columnas()

        datos = {col: request.features.get(col, 0) for col in columnas}
        df_prediccion = pd.DataFrame([datos], columns=columnas)

        prediccion = int(modelo.predict(df_prediccion)[0])

        if hasattr(modelo, "predict_proba"):
            probabilidad_mora = float(modelo.predict_proba(df_prediccion)[0][1])
        else:
            probabilidad_mora = None

        return {
            "prediccion": prediccion,
            "resultado": "Riesgo de mora" if prediccion == 1 else "Sin riesgo de mora",
            "probabilidad_mora": probabilidad_mora,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))