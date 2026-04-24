"""
Módulo de preprocesamiento de datos para forecasting de demanda.

Responsabilidades:
- Cargar datos desde CSV
- Manejar valores faltantes (interpolación / forward fill)
- Detectar y tratar outliers (método IQR)
- Agregar features temporales para modelos de ML
- Agregar ventas por frecuencia semanal/mensual
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    """Carga datos desde CSV y parsea la columna fecha."""
    df = pd.read_csv(path, parse_dates=["fecha"])
    df = df.sort_values(["producto", "fecha"]).reset_index(drop=True)
    return df


def handle_missing_values(df: pd.DataFrame, method: str = "interpolate") -> pd.DataFrame:
    """
    Rellena valores faltantes en ventas.

    method:
        'interpolate' - interpolación lineal por producto (recomendado)
        'ffill'       - forward fill por producto
        'mean'        - media del producto
    """
    df = df.copy()

    if method == "interpolate":
        df["ventas"] = df.groupby("producto")["ventas"].transform(
            lambda x: x.interpolate(method="linear").ffill().bfill()
        )
    elif method == "ffill":
        df["ventas"] = df.groupby("producto")["ventas"].transform(
            lambda x: x.ffill().bfill()
        )
    elif method == "mean":
        df["ventas"] = df.groupby("producto")["ventas"].transform(
            lambda x: x.fillna(x.mean())
        )

    return df


def remove_outliers(df: pd.DataFrame, iqr_multiplier: float = 3.0) -> pd.DataFrame:
    """
    Trata outliers usando el método IQR por producto.
    Valores fuera de [Q1 - k*IQR, Q3 + k*IQR] son recortados (capping).
    """
    df = df.copy()

    def cap_outliers(series: pd.Series) -> pd.Series:
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_multiplier * IQR
        upper = Q3 + iqr_multiplier * IQR
        return series.clip(lower=lower, upper=upper)

    df["ventas"] = df.groupby("producto")["ventas"].transform(cap_outliers)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features temporales útiles para modelos supervisados (XGBoost)."""
    df = df.copy()
    df["dia_semana"] = df["fecha"].dt.dayofweek           # 0=Lunes, 6=Domingo
    df["mes"] = df["fecha"].dt.month                       # 1-12
    df["anio"] = df["fecha"].dt.year
    df["dia_anio"] = df["fecha"].dt.dayofyear              # 1-365
    df["semana_anio"] = df["fecha"].dt.isocalendar().week.astype(int)  # 1-52
    df["trimestre"] = df["fecha"].dt.quarter               # 1-4
    df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["es_inicio_mes"] = (df["fecha"].dt.day <= 5).astype(int)
    df["es_fin_mes"] = (df["fecha"].dt.day >= 25).astype(int)
    return df


def aggregate_by_product(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """
    Agrega ventas diarias a frecuencia semanal ('W') o mensual ('ME').
    Útil para SARIMA con estacionalidad semanal.
    """
    df_agg = (
        df.groupby(["producto", pd.Grouper(key="fecha", freq=freq)])["ventas"]
        .sum()
        .reset_index()
    )
    return df_agg


def preprocess_pipeline(path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Pipeline completo: carga -> valores faltantes -> outliers -> features.
    Retorna el DataFrame limpio y enriquecido listo para modelar.
    """
    df = load_data(path)
    if verbose:
        missing = df["ventas"].isna().sum()
        print(f"  Datos cargados: {len(df)} filas | {missing} valores faltantes ({missing/len(df)*100:.1f}%)")

    df = handle_missing_values(df, method="interpolate")
    if verbose:
        print(f"  Interpolación completada: {df['ventas'].isna().sum()} valores faltantes restantes")

    before_outliers = (df["ventas"] > df.groupby("producto")["ventas"].transform(lambda x: x.quantile(0.99))).sum()
    df = remove_outliers(df, iqr_multiplier=3.0)
    if verbose:
        print(f"  Outliers tratados: {before_outliers} valores ajustados por capping IQR")

    df = add_time_features(df)
    if verbose:
        print(f"  Features temporales agregadas | Shape final: {df.shape}")

    return df
