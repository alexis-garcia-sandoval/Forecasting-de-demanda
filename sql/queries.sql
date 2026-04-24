-- ============================================================
-- SQL Queries - Forecasting de Demanda
-- ============================================================
-- Estas consultas asumen una tabla: sales_data(fecha, producto, ventas, precio, categoria)
-- Compatible con MySQL / PostgreSQL / BigQuery (con ajustes menores)
-- ============================================================


-- ------------------------------------------------------------
-- 1. Ventas totales por fecha (vista de calendario)
-- ------------------------------------------------------------
SELECT
    fecha,
    SUM(ventas)          AS ventas_totales,
    COUNT(DISTINCT producto) AS num_productos
FROM sales_data
GROUP BY fecha
ORDER BY fecha;


-- ------------------------------------------------------------
-- 2. Ventas por producto y fecha
-- ------------------------------------------------------------
SELECT
    fecha,
    producto,
    SUM(ventas) AS ventas
FROM sales_data
GROUP BY fecha, producto
ORDER BY producto, fecha;


-- ------------------------------------------------------------
-- 3. Ventas mensuales con variación mes a mes (MoM)
-- ------------------------------------------------------------
WITH ventas_mensuales AS (
    SELECT
        producto,
        DATE_FORMAT(fecha, '%Y-%m')  AS mes,
        SUM(ventas)                  AS ventas_mes
    FROM sales_data
    GROUP BY producto, DATE_FORMAT(fecha, '%Y-%m')
)
SELECT
    producto,
    mes,
    ventas_mes,
    LAG(ventas_mes) OVER (PARTITION BY producto ORDER BY mes) AS ventas_mes_anterior,
    ROUND(
        (ventas_mes - LAG(ventas_mes) OVER (PARTITION BY producto ORDER BY mes))
        / NULLIF(LAG(ventas_mes) OVER (PARTITION BY producto ORDER BY mes), 0) * 100,
        2
    ) AS variacion_mom_pct
FROM ventas_mensuales
ORDER BY producto, mes;


-- ------------------------------------------------------------
-- 4. Promedio de ventas por día de la semana (patrones semanales)
-- ------------------------------------------------------------
SELECT
    producto,
    DAYOFWEEK(fecha)    AS num_dia,
    DAYNAME(fecha)      AS dia_semana,
    ROUND(AVG(ventas), 2) AS promedio_ventas,
    ROUND(STDDEV(ventas), 2) AS desv_estandar
FROM sales_data
GROUP BY producto, DAYOFWEEK(fecha), DAYNAME(fecha)
ORDER BY producto, num_dia;


-- ------------------------------------------------------------
-- 5. Ventas por trimestre
-- ------------------------------------------------------------
SELECT
    producto,
    YEAR(fecha)          AS anio,
    QUARTER(fecha)       AS trimestre,
    SUM(ventas)          AS ventas_trimestre,
    ROUND(AVG(ventas), 2) AS promedio_diario,
    MIN(ventas)          AS min_ventas,
    MAX(ventas)          AS max_ventas
FROM sales_data
GROUP BY producto, YEAR(fecha), QUARTER(fecha)
ORDER BY producto, anio, trimestre;


-- ------------------------------------------------------------
-- 6. Comparación año a año (YoY) por mes
-- ------------------------------------------------------------
SELECT
    producto,
    MONTH(fecha)           AS mes,
    MONTHNAME(fecha)       AS nombre_mes,
    SUM(CASE WHEN YEAR(fecha) = 2022 THEN ventas ELSE 0 END) AS ventas_2022,
    SUM(CASE WHEN YEAR(fecha) = 2023 THEN ventas ELSE 0 END) AS ventas_2023,
    ROUND(
        (SUM(CASE WHEN YEAR(fecha) = 2023 THEN ventas ELSE 0 END)
        - SUM(CASE WHEN YEAR(fecha) = 2022 THEN ventas ELSE 0 END))
        / NULLIF(SUM(CASE WHEN YEAR(fecha) = 2022 THEN ventas ELSE 0 END), 0) * 100,
        2
    ) AS variacion_yoy_pct
FROM sales_data
GROUP BY producto, MONTH(fecha), MONTHNAME(fecha)
ORDER BY producto, mes;


-- ------------------------------------------------------------
-- 7. Deteccion de outliers (z-score > 3)
-- ------------------------------------------------------------
SELECT
    s.fecha,
    s.producto,
    s.ventas,
    ROUND(stats.media, 2)   AS media_producto,
    ROUND(stats.desv_std, 2) AS desv_std_producto,
    ROUND((s.ventas - stats.media) / NULLIF(stats.desv_std, 0), 2) AS z_score
FROM sales_data s
JOIN (
    SELECT
        producto,
        AVG(ventas)    AS media,
        STDDEV(ventas) AS desv_std
    FROM sales_data
    GROUP BY producto
) stats ON s.producto = stats.producto
WHERE ABS((s.ventas - stats.media) / NULLIF(stats.desv_std, 0)) > 3
ORDER BY ABS((s.ventas - stats.media) / NULLIF(stats.desv_std, 0)) DESC;


-- ------------------------------------------------------------
-- 8. Ventas acumuladas (running total) por producto
-- ------------------------------------------------------------
SELECT
    fecha,
    producto,
    ventas,
    SUM(ventas) OVER (
        PARTITION BY producto
        ORDER BY fecha
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS ventas_acumuladas
FROM sales_data
ORDER BY producto, fecha;


-- ------------------------------------------------------------
-- 9. Media movil de 7 dias por producto
-- ------------------------------------------------------------
SELECT
    fecha,
    producto,
    ventas,
    ROUND(
        AVG(ventas) OVER (
            PARTITION BY producto
            ORDER BY fecha
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ),
        2
    ) AS media_movil_7d
FROM sales_data
ORDER BY producto, fecha;


-- ------------------------------------------------------------
-- 10. Top 10 dias con mas ventas por producto
-- ------------------------------------------------------------
SELECT
    producto,
    fecha,
    ventas,
    RANK() OVER (PARTITION BY producto ORDER BY ventas DESC) AS rank_dia
FROM sales_data
HAVING rank_dia <= 10
ORDER BY producto, rank_dia;


-- ------------------------------------------------------------
-- 11. Ventas promedio por categoria y mes
-- ------------------------------------------------------------
SELECT
    categoria,
    DATE_FORMAT(fecha, '%Y-%m') AS mes,
    ROUND(AVG(ventas), 2)       AS promedio_ventas,
    SUM(ventas)                 AS ventas_totales,
    COUNT(*)                    AS num_registros
FROM sales_data
GROUP BY categoria, DATE_FORMAT(fecha, '%Y-%m')
ORDER BY categoria, mes;


-- ------------------------------------------------------------
-- 12. Forecast vs Real (para evaluar el modelo despues)
--     Requiere tabla: forecasts(fecha, producto, modelo, ventas_pronosticadas)
-- ------------------------------------------------------------
SELECT
    f.fecha,
    f.producto,
    f.modelo,
    f.ventas_pronosticadas,
    s.ventas                                              AS ventas_reales,
    ABS(s.ventas - f.ventas_pronosticadas)                AS error_absoluto,
    ROUND(
        ABS(s.ventas - f.ventas_pronosticadas)
        / NULLIF(s.ventas, 0) * 100,
        2
    )                                                     AS error_porcentual
FROM forecasts f
JOIN sales_data s
    ON f.fecha = s.fecha AND f.producto = s.producto
ORDER BY f.modelo, f.producto, f.fecha;
