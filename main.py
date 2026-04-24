"""
Pipeline principal de Forecasting de Demanda.

Ejecucion:
    python main.py

Pasos del pipeline:
    1. Generar datos sinteticos (si no existen)
    2. Preprocesar datos (valores faltantes, outliers, features)
    3. Visualizaciones exploratorias
    4. Entrenar modelos: SARIMA, Prophet, XGBoost
    5. Evaluar y comparar modelos (MAE, RMSE, MAPE)
    6. Guardar graficas y modelos
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Agregar raiz del proyecto al path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from data.generate_data import generate_sales_data
from src.data_preprocessing import preprocess_pipeline
from src.models.arima_model import train_arima_model
from src.models.arima_model import save_model as save_arima
from src.models.prophet_model import train_prophet_model
from src.models.prophet_model import save_model as save_prophet
from src.models.xgboost_model import train_xgboost_model
from src.models.xgboost_model import save_model as save_xgb
from src.evaluation import compare_models, print_comparison
from src.visualization import (
    plot_sales_history,
    plot_forecast_comparison,
    plot_model_comparison,
    plot_feature_importance,
    plot_future_forecast,
    plot_seasonality_decomposition,
)

# ---- Configuracion ----
DATA_PATH = "data/sales_data.csv"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = "outputs"
PRODUCTS = ["Producto_A", "Producto_B", "Producto_C"]
FORECAST_HORIZON = 30


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    _header("SISTEMA DE FORECASTING DE DEMANDA")

    # ------------------------------------------------------------------
    # PASO 1: Datos
    # ------------------------------------------------------------------
    _step(1, 6, "Preparando datos")

    if not Path(DATA_PATH).exists():
        print("  Generando datos sinteticos...")
        df_raw = generate_sales_data(products=PRODUCTS)
        df_raw.to_csv(DATA_PATH, index=False)
        print(f"  {len(df_raw)} filas generadas -> {DATA_PATH}")
    else:
        print(f"  Usando datos existentes en {DATA_PATH}")

    # ------------------------------------------------------------------
    # PASO 2: Preprocesamiento
    # ------------------------------------------------------------------
    _step(2, 6, "Preprocesando datos")
    df = preprocess_pipeline(DATA_PATH, verbose=True)

    # ------------------------------------------------------------------
    # PASO 3: Visualizaciones exploratorias
    # ------------------------------------------------------------------
    _step(3, 6, "Generando visualizaciones exploratorias")
    plot_sales_history(df, output_dir=OUTPUT_DIR)
    for product in PRODUCTS:
        plot_sales_history(df, product=product, output_dir=OUTPUT_DIR)
        try:
            plot_seasonality_decomposition(df, product=product, output_dir=OUTPUT_DIR)
        except Exception as e:
            print(f"  [AVISO] Descomposicion {product}: {e}")

    # ------------------------------------------------------------------
    # PASO 4: Entrenar modelos
    # ------------------------------------------------------------------
    _step(4, 6, "Entrenando modelos")
    all_results = []

    for product in PRODUCTS:
        print(f"\n  === {product} ===")

        # SARIMA
        try:
            res = train_arima_model(df, product, forecast_horizon=FORECAST_HORIZON)
            save_arima(res["model"], str(MODELS_DIR / f"sarima_{product}.pkl"))
            all_results.append(res)
        except Exception as e:
            print(f"  [SARIMA] ERROR: {e}")

        # Prophet
        try:
            res = train_prophet_model(df, product, forecast_horizon=FORECAST_HORIZON)
            save_prophet(res["model"], str(MODELS_DIR / f"prophet_{product}.pkl"))
            all_results.append(res)
        except Exception as e:
            print(f"  [Prophet] ERROR: {e}")

        # XGBoost
        try:
            res = train_xgboost_model(df, product, forecast_horizon=FORECAST_HORIZON)
            save_xgb(res["model"], str(MODELS_DIR / f"xgboost_{product}.pkl"))
            all_results.append(res)
        except Exception as e:
            print(f"  [XGBoost] ERROR: {e}")

    # ------------------------------------------------------------------
    # PASO 5: Evaluacion y comparacion
    # ------------------------------------------------------------------
    _step(5, 6, "Evaluando y comparando modelos")

    if not all_results:
        print("  ERROR: No se entrenaron modelos. Revisar logs anteriores.")
        return

    comparison_df = compare_models(all_results)
    print_comparison(comparison_df)
    comparison_path = f"{OUTPUT_DIR}/comparacion_modelos.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"  Tabla de comparacion guardada: {comparison_path}")

    # ------------------------------------------------------------------
    # PASO 6: Graficas de resultados
    # ------------------------------------------------------------------
    _step(6, 6, "Generando graficas de resultados")
    plot_model_comparison(comparison_df, output_dir=OUTPUT_DIR)

    for product in PRODUCTS:
        product_results = [r for r in all_results if r["product"] == product]
        if product_results:
            plot_forecast_comparison(product_results, product, output_dir=OUTPUT_DIR)
            plot_future_forecast(product_results, product, output_dir=OUTPUT_DIR)

        xgb = [r for r in all_results if r["product"] == product and r["model_name"] == "XGBoost"]
        if xgb:
            plot_feature_importance(xgb[0], output_dir=OUTPUT_DIR)

    # ------------------------------------------------------------------
    # Resumen final
    # ------------------------------------------------------------------
    _header("PIPELINE COMPLETADO")
    print(f"  Modelos guardados : {MODELS_DIR}/")
    print(f"  Graficas          : {OUTPUT_DIR}/")
    print(f"  Comparacion CSV   : {comparison_path}")
    print()
    print("  Para iniciar la API REST:")
    print("    python api/app.py")
    print()
    best = comparison_df[comparison_df["rank"] == 1][["product", "model", "RMSE"]]
    print("  Mejor modelo por producto:")
    for _, row in best.iterrows():
        print(f"    {row['product']}: {row['model']} (RMSE={row['RMSE']})")
    print()


def _header(text: str):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def _step(n: int, total: int, text: str):
    print(f"\n[{n}/{total}] {text}...")


if __name__ == "__main__":
    main()
