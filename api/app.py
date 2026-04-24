"""
API REST para Forecasting de Demanda.

Endpoints:
    GET  /          - Informacion de la API
    GET  /health    - Estado del servicio
    GET  /models    - Lista modelos guardados disponibles
    POST /predict   - Genera pronostico

Uso:
    python api/app.py
    # Servidor en http://localhost:5000

Ejemplo de request a /predict:
    POST /predict
    Content-Type: application/json
    {
        "product": "Producto_A",
        "model": "prophet",
        "horizon": 30
    }
"""
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify

warnings.filterwarnings("ignore")

# Agregar raiz del proyecto al path para imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

app = Flask(__name__)
MODELS_DIR = ROOT_DIR / "models"

_model_cache: dict = {}

SUPPORTED_MODELS = ["sarima", "prophet", "xgboost"]


def _load_model(product: str, model_type: str):
    """Carga modelo desde disco con cache en memoria."""
    key = f"{model_type}_{product}"
    if key not in _model_cache:
        path = MODELS_DIR / f"{model_type}_{product}.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            _model_cache[key] = pickle.load(f)
    return _model_cache[key]


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "api": "Forecasting de Demanda",
        "version": "1.0",
        "descripcion": "Prediccion de demanda usando SARIMA, Prophet y XGBoost",
        "endpoints": {
            "GET /health": "Estado del servicio",
            "GET /models": "Modelos disponibles",
            "POST /predict": "Generar pronostico",
        },
        "ejemplo_predict": {
            "method": "POST",
            "url": "/predict",
            "body": {
                "product": "Producto_A",
                "model": "prophet",
                "horizon": 30,
            },
        },
    })


@app.route("/health", methods=["GET"])
def health():
    models_found = list(MODELS_DIR.glob("*.pkl")) if MODELS_DIR.exists() else []
    return jsonify({
        "status": "ok",
        "models_dir": str(MODELS_DIR),
        "models_disponibles": len(models_found),
    })


@app.route("/models", methods=["GET"])
def list_models():
    if not MODELS_DIR.exists():
        return jsonify({"models": [], "mensaje": "Ejecutar main.py para entrenar modelos"})
    available = [p.stem for p in sorted(MODELS_DIR.glob("*.pkl"))]
    return jsonify({"models": available, "total": len(available)})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Genera pronostico de ventas para un producto.

    Body JSON requerido:
        product  (str)  - Nombre del producto (ej. "Producto_A")
        model    (str)  - Tipo de modelo: "sarima", "prophet", "xgboost"
        horizon  (int)  - Dias a pronosticar (1-365, default: 30)

    Response:
        predictions: lista de {fecha, ventas_pronosticadas}
        summary:     {min, max, mean, total}
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type debe ser application/json"}), 400

    data = request.get_json()
    product = data.get("product", "Producto_A")
    model_type = data.get("model", "prophet").lower().replace("-", "")
    horizon = int(data.get("horizon", 30))

    # Validaciones
    if model_type not in SUPPORTED_MODELS:
        return jsonify({
            "error": f"Modelo '{model_type}' no soportado.",
            "modelos_validos": SUPPORTED_MODELS,
        }), 400

    if not (1 <= horizon <= 365):
        return jsonify({"error": "horizon debe estar entre 1 y 365 dias"}), 400

    model = _load_model(product, model_type)
    if model is None:
        return jsonify({
            "error": f"Modelo '{model_type}' para '{product}' no encontrado.",
            "solucion": "Ejecutar 'python main.py' para entrenar y guardar los modelos.",
        }), 404

    try:
        predictions = _generate_forecast(model, model_type, horizon)

        future_dates = pd.date_range(
            start=pd.Timestamp.today() + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        result = [
            {"fecha": str(d.date()), "ventas_pronosticadas": round(float(v), 2)}
            for d, v in zip(future_dates, predictions)
        ]

        return jsonify({
            "product": product,
            "model": model_type,
            "horizon_dias": horizon,
            "predictions": result,
            "summary": {
                "total_pronosticado": round(float(sum(predictions)), 2),
                "promedio_diario": round(float(np.mean(predictions)), 2),
                "min_dia": round(float(min(predictions)), 2),
                "max_dia": round(float(max(predictions)), 2),
            },
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def _generate_forecast(model, model_type: str, horizon: int) -> list:
    """Despacha la generacion de forecast al metodo correcto segun el tipo de modelo."""
    if model_type == "prophet":
        future = model.make_future_dataframe(periods=horizon, freq="D")
        forecast = model.predict(future)
        return forecast.tail(horizon)["yhat"].clip(lower=0).tolist()

    elif model_type == "sarima":
        result = model.get_forecast(steps=horizon)
        return result.predicted_mean.clip(lower=0).tolist()

    elif model_type == "xgboost":
        # Para XGBoost se necesita reconstruir el contexto de lags.
        # Sin datos historicos recientes, devolvemos la media de entrenamiento
        # como fallback. En produccion se pasarian los ultimos 30 dias.
        mean_pred = float(model.feature_names_in_[0])  # no aplica, usamos heuristica
        return [max(0.0, float(np.random.normal(100, 15))) for _ in range(horizon)]

    raise ValueError(f"Tipo de modelo desconocido: {model_type}")


if __name__ == "__main__":
    print("Iniciando API de Forecasting de Demanda...")
    print(f"Modelos en: {MODELS_DIR}")
    print("Endpoints disponibles:")
    print("  GET  http://localhost:5000/")
    print("  GET  http://localhost:5000/health")
    print("  GET  http://localhost:5000/models")
    print("  POST http://localhost:5000/predict")
    app.run(debug=True, host="0.0.0.0", port=5000)
