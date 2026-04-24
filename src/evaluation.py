"""
Módulo de evaluación y comparación de modelos de forecasting.

Métricas implementadas:
- MAE  (Mean Absolute Error): promedio de errores absolutos
                               Interpreta como: "en promedio me equivoco X unidades"
- RMSE (Root Mean Squared Error): penaliza errores grandes más que MAE
                                   Interpreta como: error típico en la misma unidad que ventas
- MAPE (Mean Absolute Percentage Error): error porcentual
                                          Útil para comparar entre productos de escalas distintas
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Error Absoluto Medio."""
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Raíz del Error Cuadrático Medio."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Error Porcentual Absoluto Medio (excluye ceros en y_true)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def align_predictions(result: dict) -> tuple:
    """
    Extrae y_true e y_pred alineados para cualquier tipo de modelo.
    Maneja las diferencias de formato entre SARIMA, Prophet y XGBoost.
    """
    model_name = result["model_name"]

    if model_name == "Prophet":
        y_true = result["test"]["y"].values
        y_pred = result["test_predictions"].values
    else:
        # SARIMA y XGBoost retornan pd.Series
        y_true = np.asarray(result["test"])
        y_pred = np.asarray(result["test_predictions"])

    min_len = min(len(y_true), len(y_pred))
    return y_true[:min_len], y_pred[:min_len]


def evaluate_model(result: dict) -> dict:
    """Calcula MAE, RMSE y MAPE para los resultados de un modelo."""
    y_true, y_pred = align_predictions(result)

    return {
        "model": result["model_name"],
        "product": result["product"],
        "MAE": round(mae(y_true, y_pred), 2),
        "RMSE": round(rmse(y_true, y_pred), 2),
        "MAPE (%)": round(mape(y_true, y_pred), 2),
        "n_test": len(y_true),
    }


def compare_models(results: list) -> pd.DataFrame:
    """
    Compara todos los modelos y retorna DataFrame ordenado por RMSE.
    Agrega columna 'rank' dentro de cada producto (1 = mejor modelo).
    """
    evaluations = [evaluate_model(r) for r in results]
    df = pd.DataFrame(evaluations)
    df = df.sort_values(["product", "RMSE"]).reset_index(drop=True)
    df["rank"] = df.groupby("product")["RMSE"].rank(method="dense").astype(int)
    return df


def print_comparison(comparison_df: pd.DataFrame):
    """Imprime tabla de comparación formateada en consola."""
    print("\n" + "=" * 70)
    print("  COMPARACION DE MODELOS - METRICAS DE ERROR")
    print("=" * 70)

    for product in sorted(comparison_df["product"].unique()):
        print(f"\n  Producto: {product}")
        subset = comparison_df[comparison_df["product"] == product][
            ["rank", "model", "MAE", "RMSE", "MAPE (%)", "n_test"]
        ].sort_values("rank")
        print(subset.to_string(index=False))

    best_per_product = comparison_df[comparison_df["rank"] == 1][["product", "model", "RMSE"]]
    print("\n  --- Mejor modelo por producto ---")
    print(best_per_product.to_string(index=False))
    print("\n" + "=" * 70)
    print("  MAE  = Error Absoluto Medio (unidades)")
    print("  RMSE = Raiz del Error Cuadratico Medio (penaliza errores grandes)")
    print("  MAPE = Error Porcentual Absoluto Medio (%)")
    print("=" * 70 + "\n")
