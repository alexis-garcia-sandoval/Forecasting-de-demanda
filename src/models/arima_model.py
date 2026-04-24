"""
Modelo SARIMA para forecasting de demanda.

SARIMA(p,d,q)(P,D,Q,s) donde s=7 (estacionalidad semanal).

Flujo:
1. Test de Dickey-Fuller para determinar si la serie es estacionaria
2. Ajustar orden de diferenciación d
3. Entrenar SARIMA con 80% de los datos
4. Evaluar en 20% restante
5. Generar forecast futuro
6. Guardar modelo con pickle
"""
import warnings
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")


def check_stationarity(series: pd.Series, significance: float = 0.05) -> bool:
    """
    Prueba de Dickey-Fuller aumentada.
    Retorna True si la serie es estacionaria (rechaza hipótesis nula de raíz unitaria).
    """
    result = adfuller(series.dropna(), autolag="AIC")
    p_value = result[1]
    return p_value < significance


def fit_sarima(
    series: pd.Series,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 7),
) -> object:
    """Ajusta modelo SARIMA y retorna el resultado del fit."""
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def train_arima_model(
    df: pd.DataFrame,
    product: str,
    target_col: str = "ventas",
    date_col: str = "fecha",
    forecast_horizon: int = 30,
) -> dict:
    """
    Entrena SARIMA(1,d,1)(1,1,1,7) para un producto específico.

    Returns dict con:
        model, train, test, test_predictions, forecast, confidence_intervals
    """
    series = (
        df[df["producto"] == product]
        .set_index(date_col)[target_col]
        .sort_index()
        .asfreq("D")
        .ffill()
    )

    train_size = int(len(series) * 0.8)
    train = series.iloc[:train_size]
    test = series.iloc[train_size:]

    print(f"  [SARIMA] {product}: entrenando con {len(train)} dias, validando con {len(test)} dias...")

    is_stationary = check_stationarity(train)
    d = 0 if is_stationary else 1

    fitted = fit_sarima(train, order=(1, d, 1), seasonal_order=(1, 1, 1, 7))

    test_pred_raw = fitted.predict(start=train_size, end=len(series) - 1)
    test_pred_raw = test_pred_raw.clip(lower=0)
    test_pred = pd.Series(test_pred_raw.values, index=test.index)

    forecast_result = fitted.get_forecast(steps=forecast_horizon)
    forecast = forecast_result.predicted_mean.clip(lower=0)
    conf_int = forecast_result.conf_int()

    print(f"  [SARIMA] {product}: AIC={fitted.aic:.1f} | estacionaria={is_stationary} | d={d}")

    return {
        "product": product,
        "model": fitted,
        "train": train,
        "test": test,
        "test_predictions": test_pred,
        "forecast": forecast,
        "confidence_intervals": conf_int,
        "model_name": "SARIMA",
    }


def save_model(model, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Modelo guardado: {path}")


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
