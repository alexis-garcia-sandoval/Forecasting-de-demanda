"""
Modelo XGBoost para forecasting de demanda (enfoque supervisado).

Convierte el problema de series de tiempo en un problema supervisado:
- Variable objetivo: ventas del día t
- Features: lags (t-1, t-7, t-14, t-30), rolling stats, features temporales

Ventajas sobre ARIMA/Prophet:
- Captura relaciones no lineales
- Incorpora variables exogenas con facilidad
- Generalmente más preciso con datos suficientes

Flujo:
1. Crear lag features y rolling statistics
2. Preparar X (features) e y (target)
3. Entrenar XGBoost con 80% de los datos
4. Evaluar en 20% restante
5. Generar forecast recursivo para el horizonte futuro
6. Guardar modelo con pickle
"""
import warnings
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


FEATURE_COLS = [
    "dia_semana", "mes", "anio", "dia_anio", "semana_anio",
    "trimestre", "es_fin_semana", "es_inicio_mes", "es_fin_mes",
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_30", "rolling_max_7",
]


def create_lag_features(df: pd.DataFrame, lags: list = None) -> pd.DataFrame:
    """Crea lag features y estadísticas rolling por producto."""
    if lags is None:
        lags = [1, 7, 14, 30]

    df = df.copy().sort_values(["producto", "fecha"])

    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("producto")["ventas"].shift(lag)

    df["rolling_mean_7"] = df.groupby("producto")["ventas"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )
    df["rolling_std_7"] = df.groupby("producto")["ventas"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).std().fillna(0)
    )
    df["rolling_mean_30"] = df.groupby("producto")["ventas"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=1).mean()
    )
    df["rolling_max_7"] = df.groupby("producto")["ventas"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).max()
    )
    return df


def prepare_features(df: pd.DataFrame, product: str) -> tuple:
    """Retorna X, y, dates para el producto dado."""
    df_product = df[df["producto"] == product].copy().sort_values("fecha")
    df_product = create_lag_features(df_product)
    df_product = df_product.dropna(subset=["lag_30"]).reset_index(drop=True)

    available_cols = [c for c in FEATURE_COLS if c in df_product.columns]
    X = df_product[available_cols]
    y = df_product["ventas"]
    dates = df_product["fecha"]

    return X, y, dates


def train_xgboost_model(
    df: pd.DataFrame,
    product: str,
    forecast_horizon: int = 30,
) -> dict:
    """
    Entrena XGBRegressor para un producto específico.

    Returns dict con:
        model, X_train, X_test, y_train, y_test,
        test_predictions, forecast, feature_importance
    """
    X, y, dates = prepare_features(df, product)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    dates_test = dates.iloc[train_size:]

    print(f"  [XGBoost] {product}: entrenando con {len(X_train)} muestras, validando con {len(X_test)}...")

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    test_pred_raw = model.predict(X_test).clip(min=0)
    test_pred = pd.Series(test_pred_raw, index=dates_test.values, name="pred")

    # Forecast recursivo: usa predicciones anteriores como nuevos lags
    last_values = list(y.tail(30).values)
    future_preds = []

    df_temp = df[df["producto"] == product].copy().sort_values("fecha")
    last_date = df_temp["fecha"].max()
    last_row = df_temp.tail(1).copy()

    for i in range(forecast_horizon):
        next_date = last_date + pd.Timedelta(days=i + 1)
        row = last_row.copy()
        row["fecha"] = next_date
        row["ventas"] = last_values[-1] if last_values else y.mean()

        # Recalcular features temporales
        row["dia_semana"] = next_date.dayofweek
        row["mes"] = next_date.month
        row["anio"] = next_date.year
        row["dia_anio"] = next_date.dayofyear
        row["semana_anio"] = next_date.isocalendar()[1]
        row["trimestre"] = next_date.quarter
        row["es_fin_semana"] = int(next_date.dayofweek >= 5)
        row["es_inicio_mes"] = int(next_date.day <= 5)
        row["es_fin_mes"] = int(next_date.day >= 25)

        # Lags desde el historial acumulado
        all_vals = list(y.values) + future_preds
        row["lag_1"] = all_vals[-1] if len(all_vals) >= 1 else y.mean()
        row["lag_7"] = all_vals[-7] if len(all_vals) >= 7 else y.mean()
        row["lag_14"] = all_vals[-14] if len(all_vals) >= 14 else y.mean()
        row["lag_30"] = all_vals[-30] if len(all_vals) >= 30 else y.mean()
        row["rolling_mean_7"] = np.mean(all_vals[-7:]) if len(all_vals) >= 7 else y.mean()
        row["rolling_std_7"] = np.std(all_vals[-7:]) if len(all_vals) >= 7 else 0.0
        row["rolling_mean_30"] = np.mean(all_vals[-30:]) if len(all_vals) >= 30 else y.mean()
        row["rolling_max_7"] = np.max(all_vals[-7:]) if len(all_vals) >= 7 else y.max()

        available_cols = [c for c in FEATURE_COLS if c in row.columns]
        features = row[available_cols].values.reshape(1, -1)
        pred = float(model.predict(features)[0])
        pred = max(pred, 0)
        future_preds.append(pred)
        last_values.append(pred)

    feature_importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns,
    ).sort_values(ascending=False)

    print(f"  [XGBoost] {product}: top feature = {feature_importance.index[0]}")

    return {
        "product": product,
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "test_predictions": test_pred,
        "forecast": pd.Series(future_preds, name="forecast"),
        "feature_importance": feature_importance,
        "model_name": "XGBoost",
    }


def save_model(model, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Modelo guardado: {path}")


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
