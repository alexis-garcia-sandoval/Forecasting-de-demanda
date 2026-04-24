"""
Módulo de visualización para forecasting de demanda.

Gráficas disponibles:
1. plot_sales_history       - Serie temporal de ventas históricas
2. plot_forecast_comparison - Real vs Predicción por modelo
3. plot_model_comparison    - Barras comparando MAE y RMSE entre modelos
4. plot_feature_importance  - Importancia de variables (XGBoost)
5. plot_future_forecast     - Forecast futuro de todos los modelos
6. plot_seasonality         - Descomposición estacional
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "axes.titlesize": 13})


def _save(fig: plt.Figure, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {path}")


def plot_sales_history(df: pd.DataFrame, product: str = None, output_dir: str = "outputs"):
    """Serie temporal de ventas. Si product=None grafica el total."""
    if product:
        data = df[df["producto"] == product].copy()
        title = f"Historico de Ventas - {product}"
        fname = f"historico_{product}.png"
    else:
        data = df.groupby("fecha")["ventas"].sum().reset_index()
        title = "Historico de Ventas - Todos los Productos"
        fname = "historico_total.png"

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [3, 1]})

    # Panel superior: serie de ventas
    ax = axes[0]
    ax.plot(data["fecha"], data["ventas"], linewidth=0.8, color="steelblue", alpha=0.8)
    ax.fill_between(data["fecha"], data["ventas"], alpha=0.15, color="steelblue")
    ma = data["ventas"].rolling(7, center=True).mean()
    ax.plot(data["fecha"], ma, linewidth=1.8, color="darkred", label="Media movil 7 dias")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Unidades vendidas")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Panel inferior: barras mensuales
    ax2 = axes[1]
    monthly = data.set_index("fecha")["ventas"].resample("ME").sum()
    ax2.bar(monthly.index, monthly.values, width=20, color="steelblue", alpha=0.7, edgecolor="white")
    ax2.set_ylabel("Ventas/mes")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    fig.tight_layout()
    _save(fig, f"{output_dir}/{fname}")


def plot_forecast_comparison(
    results: list,
    product: str,
    output_dir: str = "outputs",
):
    """Real vs Predicción en el conjunto de test para cada modelo."""
    product_results = [r for r in results if r["product"] == product]
    if not product_results:
        return

    n = len(product_results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4.5 * n), squeeze=False)

    for i, result in enumerate(product_results):
        ax = axes[i][0]
        model_name = result["model_name"]

        if model_name == "Prophet":
            dates = pd.to_datetime(result["test"]["ds"].values)
            y_true = result["test"]["y"].values
            y_pred = result["test_predictions"].values
        else:
            y_true = np.asarray(result["test"])
            y_pred = np.asarray(result["test_predictions"])
            dates = (
                result["test"].index
                if hasattr(result["test"], "index")
                else np.arange(len(y_true))
            )

        min_len = min(len(y_true), len(y_pred))
        ax.plot(dates[:min_len], y_true[:min_len],
                label="Real", color="#2c3e50", linewidth=1.4)
        ax.plot(dates[:min_len], y_pred[:min_len],
                label="Prediccion", color="#e74c3c", linewidth=1.4, linestyle="--")

        from src.evaluation import mae, rmse
        _mae = round(mae(y_true[:min_len], y_pred[:min_len]), 1)
        _rmse = round(rmse(y_true[:min_len], y_pred[:min_len]), 1)
        ax.set_title(f"{model_name} - {product}  |  MAE={_mae}  RMSE={_rmse}",
                     fontweight="bold")
        ax.set_ylabel("Ventas")
        ax.legend()
        if hasattr(dates[0], "strftime"):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    fig.tight_layout()
    _save(fig, f"{output_dir}/forecast_test_{product}.png")


def plot_model_comparison(comparison_df: pd.DataFrame, output_dir: str = "outputs"):
    """Barras agrupadas comparando MAE y RMSE de todos los modelos por producto."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        pivot = comparison_df.pivot(index="product", columns="model", values=metric)
        pivot.plot(
            kind="bar", ax=ax,
            colormap="Set2", edgecolor="grey", linewidth=0.5, width=0.7,
        )
        ax.set_title(f"Comparacion de Modelos - {metric}", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.legend(title="Modelo", loc="upper right")
        ax.tick_params(axis="x", rotation=30)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f", padding=2, fontsize=8)

    fig.suptitle("Comparacion de Errores entre Modelos de Forecasting", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"{output_dir}/comparacion_modelos.png")


def plot_feature_importance(result: dict, output_dir: str = "outputs"):
    """Importancia de variables del modelo XGBoost."""
    if "feature_importance" not in result:
        return

    fi = result["feature_importance"].head(14).sort_values()
    product = result["product"]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#2ecc71" if fi.index[i].startswith("lag") else
              "#3498db" if fi.index[i].startswith("rolling") else
              "#e67e22"
              for i in range(len(fi))]
    fi.plot(kind="barh", ax=ax, color=colors, edgecolor="grey", linewidth=0.4)
    ax.set_title(f"Importancia de Variables - XGBoost ({product})", fontweight="bold")
    ax.set_xlabel("Importancia (ganancia)")

    from matplotlib.patches import Patch
    legend = [
        Patch(color="#2ecc71", label="Lag features"),
        Patch(color="#3498db", label="Rolling stats"),
        Patch(color="#e67e22", label="Features temporales"),
    ]
    ax.legend(handles=legend, loc="lower right")
    fig.tight_layout()
    _save(fig, f"{output_dir}/feature_importance_{product}.png")


def plot_future_forecast(results: list, product: str, output_dir: str = "outputs"):
    """Forecast futuro de todos los modelos para el producto dado."""
    product_results = [r for r in results if r["product"] == product]
    if not product_results:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["steelblue", "tomato", "seagreen", "purple"]
    horizon = None

    for i, result in enumerate(product_results):
        model_name = result["model_name"]
        forecast = result["forecast"]

        if model_name == "Prophet":
            values = forecast["yhat"].values
        else:
            values = np.asarray(forecast)

        horizon = len(values)
        ax.plot(range(1, horizon + 1), values,
                label=model_name, color=colors[i % len(colors)],
                linewidth=2.2, marker="o", markersize=3)

        if model_name == "Prophet" and "yhat_lower" in forecast.columns:
            ax.fill_between(
                range(1, horizon + 1),
                forecast["yhat_lower"].values,
                forecast["yhat_upper"].values,
                color=colors[i % len(colors)], alpha=0.12,
            )

    ax.set_title(f"Pronostico Futuro - {product} (proximos {horizon} dias)",
                 fontweight="bold")
    ax.set_xlabel("Dias en el futuro")
    ax.set_ylabel("Ventas pronosticadas")
    ax.legend(title="Modelo")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle=":")
    fig.tight_layout()
    _save(fig, f"{output_dir}/forecast_futuro_{product}.png")


def plot_seasonality_decomposition(
    df: pd.DataFrame, product: str, output_dir: str = "outputs"
):
    """Descomposición de la serie en tendencia, estacionalidad y residuo."""
    from statsmodels.tsa.seasonal import seasonal_decompose

    series = (
        df[df["producto"] == product]
        .set_index("fecha")["ventas"]
        .sort_index()
        .asfreq("D")
        .ffill()
    )

    decomposition = seasonal_decompose(series, model="additive", period=7)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    components = [
        (series, "Serie original", "steelblue"),
        (decomposition.trend, "Tendencia", "darkorange"),
        (decomposition.seasonal, "Estacionalidad", "seagreen"),
        (decomposition.resid, "Residuo", "gray"),
    ]

    for ax, (data, label, color) in zip(axes, components):
        ax.plot(data.index, data.values, color=color, linewidth=0.9)
        ax.set_ylabel(label)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    axes[0].set_title(f"Descomposicion Estacional - {product} (periodo=7 dias)",
                      fontweight="bold")
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
    fig.tight_layout()
    _save(fig, f"{output_dir}/descomposicion_{product}.png")
