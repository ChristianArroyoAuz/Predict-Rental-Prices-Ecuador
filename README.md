# Ecuador Real Estate Price Prediction

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar un modelo de Machine Learning de regresión para predecir los precios de arriendo de propiedades en Ecuador, con un enfoque principal en la provincia de **Pichincha (Quito)**, dado que la mayoría de los datos del dataset provienen de esa región.

Utilizando un dataset real con títulos, precios, ubicación y características como número de habitaciones, baños y área, se realiza un proceso completo de análisis exploratorio de datos (EDA), limpieza y preprocesamiento, y finalmente, la implementación y evaluación de diferentes modelos de regresión.

## Dataset

El dataset utilizado (`real_state_ecuador_dataset.csv`) contiene información de más de 400 propiedades en alquiler. Las principales variables son:

*   `Titulo`: Descripción textual de la propiedad.
*   `Precio`: Precio de alquiler (variable objetivo).
*   `Provincia`: Provincia donde se ubica la propiedad.
*   `Lugar`: Dirección detallada.
*   `Num. dormitorios`: Número de habitaciones.
*   `Num. banos`: Número de baños.
*   `Area`: Área de la propiedad en metros cuadrados.
*   `Num. garages`: Número de plazas de garaje.

## Metodología

1.  **Análisis Exploratorio de Datos (EDA):** Análisis de distribuciones, detección de valores atípicos (outliers) y identificación de patrones en los datos.
2.  **Limpieza y Preprocesamiento:**
    *   Manejo de valores faltantes (NaN) en columnas como `Num. banos`, `Area`, `Num. garages`.
    *   Extracción de características a partir de la variable `Lugar` (ej. sector/barrio).
    *   Codificación de variables categóricas.
    *   Escalado de características numéricas.
3.  **Modelado:**
    *   Implementación de varios modelos de regresión como Regresión Lineal, Árboles de Decisión, Random Forest, y Gradient Boosting (XGBoost, LightGBM).
    *   Uso de técnicas de validación cruzada para la evaluación y el ajuste de hiperparámetros.
4.  **Evaluación:** Comparación del rendimiento de los modelos usando métricas como Error Absoluto Medio (MAE), Error Cuadrático Medio (RMSE) y R².

## Tecnologías Utilizadas

*   **Lenguaje:** Python 3.x
*   **Librerías Principales:**
    *   `Pandas` - Para manipulación y análisis de datos.
    *   `NumPy` - Para computación numérica.
    *   `Matplotlib` / `Seaborn` - Para visualización de datos.
    *   `Scikit-learn` - Para implementar los modelos de machine learning y preprocesamiento.
    *   `Jupyter Notebooks` - Para el desarrollo y experimentación.|

## Instalación y Uso

1.  Clona el repositorio:
    ```bash
    git clone https://github.com/tu-usuario/ecuador-real-estate-price-prediction.git
