# Forecasting de Demanda

Sistema de predicción de demanda para productos de retail/e-commerce utilizando múltiples modelos de series de tiempo y machine learning.

## Descripción

Este proyecto implementa un pipeline completo de forecasting de demanda que incluye:

- **Generación de datos sintéticos** con tendencia, estacionalidad semanal/anual y ruido
- **Limpieza de datos**: manejo de valores faltantes y outliers
- **Tres modelos predictivos** con comparación objetiva:
  - ARIMA / SARIMA (statsmodels)
  - Prophet (Facebook/Meta)
  - XGBoost (modelo supervisado con features de tiempo)
- **Métricas de evaluación**: MAE, RMSE, MAPE
- **Visualizaciones** con matplotlib/seaborn
- **API REST** con Flask (endpoint `/predict`)
- **Consultas SQL** para análisis de datos

## Estructura del Proyecto

```
forecasting-de-demanda/
├── data/
│   └── generate_data.py        # Generación de datos sintéticos
├── src/
│   ├── data_preprocessing.py   # Limpieza y feature engineering
│   ├── evaluation.py           # Métricas MAE, RMSE, MAPE
│   ├── visualization.py        # Gráficas con matplotlib/seaborn
│   └── models/
│       ├── arima_model.py       # SARIMA
│       ├── prophet_model.py     # Facebook Prophet
│       └── xgboost_model.py     # XGBoost supervisado
├── sql/
│   └── queries.sql             # Consultas SQL de análisis
├── api/
│   └── app.py                  # Flask API con /predict
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── models/                     # Modelos guardados (pickle)
├── outputs/                    # Gráficas generadas
├── main.py                     # Pipeline principal
└── requirements.txt
```

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/alexis-garcia-sandoval/forecasting-de-demanda.git
cd forecasting-de-demanda

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Uso Rápido

### Ejecutar el pipeline completo

```bash
python main.py
```

Esto genera datos sintéticos, entrena los tres modelos, compara resultados y guarda gráficas en `outputs/`.

### Iniciar la API

```bash
python api/app.py
```

### Hacer una predicción via API

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "product": "Producto_A",
    "model": "prophet",
    "horizon": 30
  }'
```

## Métricas de Error

| Métrica | Descripción |
|---------|-------------|
| **MAE** | Error Absoluto Medio - interpreta como unidades de error promedio |
| **RMSE** | Raíz del Error Cuadrático Medio - penaliza errores grandes |
| **MAPE** | Error Porcentual Absoluto Medio - útil para comparar entre productos |

## Modelos Implementados

### SARIMA
- Modelo clásico de series de tiempo
- Captura tendencia, estacionalidad y autocorrelación
- Ideal para series con patrones estacionales claros

### Prophet
- Desarrollado por Meta/Facebook
- Maneja bien estacionalidades múltiples y datos faltantes
- Robusto ante outliers

### XGBoost
- Modelo de gradient boosting tratado como problema supervisado
- Usa features: lags, rolling stats, día de semana, mes, etc.
- Generalmente el más preciso con datos suficientes

## Casos de Uso

- **Planner de demanda**: Planificar inventario para los próximos 30 días
- **Supply chain analyst**: Anticipar necesidades de abastecimiento
- **Retail / E-commerce**: Optimizar stock y evitar quiebres o sobrestock
- **FMCG**: Productos de consumo masivo con patrones estacionales

## Empresas que usan esto

Walmart, Amazon, Mercado Libre, FEMSA, Bimbo, y cualquier empresa con inventario.
