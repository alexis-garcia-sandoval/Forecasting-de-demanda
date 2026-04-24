"""
Generador de datos sintéticos de ventas para el proyecto de forecasting.

Genera datos con:
- Tendencia creciente
- Estacionalidad semanal (picos en fin de semana)
- Estacionalidad anual (picos en Nov-Dic)
- Ruido gaussiano
- ~2% valores faltantes
- ~1% outliers
"""
import pandas as pd
import numpy as np
from pathlib import Path


def generate_sales_data(
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    products: list = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Genera datos diarios de ventas con tendencia y estacionalidad."""
    np.random.seed(seed)

    if products is None:
        products = ["Producto_A", "Producto_B", "Producto_C"]

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(date_range)
    records = []

    for product in products:
        base_sales = np.random.randint(50, 200)
        trend = np.linspace(0, np.random.uniform(20, 50), n)

        # Estacionalidad semanal: picos viernes-sabado
        weekly = 20 * np.sin(2 * np.pi * np.arange(n) / 7)

        # Estacionalidad anual: picos noviembre-diciembre
        yearly = 30 * np.sin(2 * np.pi * np.arange(n) / 365 - np.pi / 2)

        noise = np.random.normal(0, 10, n)
        sales = base_sales + trend + weekly + yearly + noise
        sales = np.maximum(sales, 0).round().astype(float)

        df_product = pd.DataFrame({
            "fecha": date_range,
            "producto": product,
            "ventas": sales,
            "precio": round(np.random.uniform(10, 100), 2),
            "categoria": np.random.choice(["Electrónica", "Ropa", "Hogar", "Alimentos"], size=n),
        })
        records.append(df_product)

    df = pd.concat(records, ignore_index=True)

    # Introducir ~2% valores faltantes en ventas
    missing_idx = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
    df.loc[missing_idx, "ventas"] = np.nan

    # Introducir ~1% outliers (ventas x3-5)
    outlier_idx = np.random.choice(df.index, size=int(len(df) * 0.01), replace=False)
    df.loc[outlier_idx, "ventas"] = (
        df.loc[outlier_idx, "ventas"] * np.random.uniform(3, 5, size=len(outlier_idx))
    )

    return df


if __name__ == "__main__":
    df = generate_sales_data()
    output_path = Path(__file__).parent / "sales_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Datos generados: {len(df)} filas -> {output_path}")
    print(f"\nPrimeras filas:\n{df.head(10)}")
    print(f"\nEstadísticas de ventas:\n{df['ventas'].describe()}")
    print(f"\nValores faltantes: {df['ventas'].isna().sum()} ({df['ventas'].isna().mean()*100:.1f}%)")
