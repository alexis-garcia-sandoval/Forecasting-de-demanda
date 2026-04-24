"""
Modelo Prophet (Meta/Facebook) para forecasting de demanda.

Prophet es ideal para:
- Series con estacionalidad múltiple (semanal + anual)
- Datos con valores faltantes y outliers
- Series con cambios de tendencia (changepoints)

Flujo:
1. Preparar datos en formato Prophet (columnas 'ds' y 'y')
2. Entrenar con 80% de los datos
3. Evaluar en 20% restante
4. Generar forecast futuro con intervalos de confianza
5. Guardar modelo con pickle
"""
import warnings
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("[AVISO] Prophet no instalado. Instalar con: pip install prophet")


def train_prophet_model(
    df: pd.DataFrame,
    product: str,
    target_col: str = "ventas",
    date_col: str = "fecha",
    forecast_horizon: int = 30,
) -> dict:
    """
    Entrena Facebook Prophet para un producto específico.

    Configuración:
    - Estacionalidad anual y semanal activadas
    - changepoint_prior_scale=0.05 (tendencia moderada)
    - Modo aditivo (ventas no se multiplican por la estacionalidad)

    Returns dict con:
        model, train, test, test_predictions, forecast
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet no está instalado. Ejecutar: pip install prophet")

    series = (
        df[df["producto"] == product][[date_col, target_col]]
        .rename(columns={date_col: "ds", target_col: "y"})
        .sort_values("ds")
        .dropna()
        .reset_index(drop=True)
    )

    train_size = int(len(series) * 0.8)
    train = series.iloc[:train_size].copy()
    test = series.iloc[train_size:].copy()

    print(f"  [Prophet] {product}: entrenando con {len(train)} dias, validando con {len(test)} dias...")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        seasonality_mode="additive",
        interval_width=0.95,
    )
    model.fit(train)

    # Predicciones en conjunto de test
    test_future = test[["ds"]].copy()
    test_forecast = model.predict(test_future)
    test_pred = pd.Series(
        test_forecast["yhat"].clip(lower=0).values,
        index=test["ds"].values,
        name="yhat",
    )

    # Forecast futuro
    future = model.make_future_dataframe(periods=forecast_horizon, freq="D")
    full_forecast = model.predict(future)
    forecast = full_forecast.tail(forecast_horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast["yhat"] = forecast["yhat"].clip(lower=0)

    print(f"  [Prophet] {product}: entrenamiento completado")

    return {
        "product": product,
        "model": model,
        "train": train,
        "test": test,
        "test_predictions": test_pred,
        "forecast": forecast,
        "model_name": "Prophet",
    }


def save_model(model, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Modelo guardado: {path}")


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
