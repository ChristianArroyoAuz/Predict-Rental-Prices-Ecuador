# 🏠 API de Predicción de Precios de Alquiler - Ecuador

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Railway](https://img.shields.io/badge/Railway-131415?style=for-the-badge&logo=railway&logoColor=white)](https://railway.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

API REST para predecir el precio de alquiler de propiedades en Ecuador utilizando Machine Learning. El modelo fue entrenado con datos reales de propiedades en alquiler de diversas provincias del país.

## 📋 Tabla de Contenidos
- [Descripción de la Solución](#-descripción-de-la-solución)
- [Características del Modelo](#-características-del-modelo)
- [Demo en Vivo](#-demo-en-vivo)
- [Instrucciones de Uso](#-instrucciones-de-uso)
  - [Endpoint de Health Check](#-endpoint-de-health-check)
  - [Endpoint de Predicción](#-endpoint-de-predicción)
- [Ejemplos de Uso](#-ejemplos-de-uso)
  - [cURL (Linux/Mac/Windows)](#-curl-linuxmacwindows)
  - [PowerShell (Windows)](#-powershell-windows)
  - [Python](#-python)
  - [JavaScript](#-javascript)
- [Estructura de la API](#-estructura-de-la-api)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Mantenimiento y Contacto](#-mantenimiento-y-contacto)

---

## 🏗️ Descripción de la Solución

Esta API permite predecir el precio mensual de alquiler de una propiedad en Ecuador basándose en características como:

- **Ubicación**: Provincia y ciudad/sector
- **Características físicas**: Número de dormitorios, baños, área total y garajes

El modelo de Machine Learning fue entrenado con un dataset de más de 500 propiedades en alquiler de diversas provincias ecuatorianas, incluyendo Pichincha, Guayas, Manabí, Azuay, entre otras.

### 📊 Procesamiento de Datos
- Limpieza y normalización de la columna "Lugar" para extraer ciudad y sector
- Manejo de valores faltantes mediante imputación con mediana
- Eliminación de outliers usando el método IQR
- Creación de categorías de precio (Económico, Medio, Lujo)

### 🤖 Modelo de Machine Learning
- **Algoritmo**: Pipeline de Scikit-learn con Random Forest Regressor
- **Preprocesamiento**: 
  - Variables numéricas estandarizadas con StandardScaler
  - Variables categóricas codificadas con OneHotEncoder
- **Métricas de rendimiento**:
  - MAE (Error Absoluto Medio): ~$120
  - RMSE (Raíz del Error Cuadrático Medio): ~$180
  - R² (Coeficiente de Determinación): ~0.75

---

## 🎯 Demo en Vivo

La API está desplegada y accesible públicamente en Railway:

| Recurso | URL |
|---------|-----|
| **URL Base** | `https://rental-prediction-api-production.up.railway.app` |
| **Documentación Interactiva** | [https://rental-prediction-api-production.up.railway.app/docs](https://rental-prediction-api-production.up.railway.app/docs) |
| **Health Check** | [https://rental-prediction-api-production.up.railway.app/health](https://rental-prediction-api-production.up.railway.app/health) |

---

## 📖 Instrucciones de Uso

### 🔍 Endpoint de Health Check

Verifica el estado de la API y la versión de las librerías utilizadas.

**Método:** `GET`

**URL:** `https://rental-prediction-api-production.up.railway.app/health`

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "numpy_version": "2.3.5",
  "pandas_version": "2.3.3"
}
```

### 📈 Endpoint de Predicción

Realiza una predicción del precio de alquiler.

**Método:** `POST`

**URL:** `https://rental-prediction-api-production.up.railway.app/predict`

**Headers:**
```
Content-Type: application/json
```

**Body (JSON):**

| Campo | Tipo | Descripción | Ejemplo |
|-------|------|-------------|---------|
| `provincia` | string | Provincia donde se ubica la propiedad | `"Pichincha"` |
| `lugar` | string | Ciudad o localidad | `"Quito"` |
| `num_dormitorios` | integer | Número de dormitorios | `3` |
| `num_banos` | integer | Número de baños | `2` |
| `area` | float | Área en metros cuadrados | `120` |
| `num_garages` | integer | Número de garajes | `1` |

**Respuesta Exitosa (200 OK):**
```json
{
  "prediction": 750.0
}
```

**Posibles Errores:**
- `400 Bad Request`: Error en los datos de entrada
- `503 Service Unavailable`: Modelo no disponible temporalmente

---

## 💻 Ejemplos de Uso

### 🖥️ cURL (Linux/Mac/Windows)

**Health Check:**
```bash
curl https://rental-prediction-api-production.up.railway.app/health
```

**Predicción:**
```bash
curl -X POST "https://rental-prediction-api-production.up.railway.app/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "provincia": "Pichincha",
       "lugar": "Quito",
       "num_dormitorios": 3,
       "num_banos": 2,
       "area": 120,
       "num_garages": 1
     }'
```

### 💻 PowerShell (Windows)

**Health Check:**
```powershell
Invoke-RestMethod -Uri "https://rental-prediction-api-production.up.railway.app/health" -Method GET
```

**Predicción:**
```powershell
$body = @{
    provincia = "Pichincha"
    lugar = "Quito"
    num_dormitorios = 3
    num_banos = 2
    area = 120
    num_garages = 1
}

Invoke-RestMethod -Uri "https://rental-prediction-api-production.up.railway.app/predict" `
                  -Method POST `
                  -ContentType "application/json" `
                  -Body ($body | ConvertTo-Json)
```

### 🐍 Python

```python
import requests

# Health Check
health = requests.get("https://rental-prediction-api-production.up.railway.app/health")
print(health.json())

# Predicción
url = "https://rental-prediction-api-production.up.railway.app/predict"
data = {
    "provincia": "Pichincha",
    "lugar": "Quito",
    "num_dormitorios": 3,
    "num_banos": 2,
    "area": 120,
    "num_garages": 1
}

response = requests.post(url, json=data)
print(f"Precio estimado: ${response.json()['prediction']}")
```

### 🌐 JavaScript

```javascript
// Health Check
fetch('https://rental-prediction-api-production.up.railway.app/health')
  .then(response => response.json())
  .then(data => console.log(data));

// Predicción
const data = {
  provincia: "Pichincha",
  lugar: "Quito",
  num_dormitorios: 3,
  num_banos: 2,
  area: 120,
  num_garages: 1
};

fetch('https://rental-prediction-api-production.up.railway.app/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(data)
})
  .then(response => response.json())
  .then(data => console.log(`Precio estimado: $${data.prediction}`));
```

---

## 📁 Estructura de la API

```
api/
├── main.py          # Lógica principal de la API
├── schemas.py       # Modelos Pydantic para validación
└── __init__.py      # Inicializador del módulo

models/
└── rental_price_model.pkl  # Modelo entrenado (Random Forest Pipeline)
```

---

## 🛠️ Tecnologías Utilizadas

### Backend
- **FastAPI**: Framework web de alto rendimiento
- **Uvicorn**: Servidor ASGI
- **Pydantic**: Validación de datos

### Machine Learning
- **Scikit-learn 1.6.1**: Librería principal de ML
- **Pandas 2.3.3**: Manipulación y análisis de datos
- **NumPy 2.3.5**: Operaciones numéricas
- **Joblib 1.4.2**: Serialización del modelo

### Infraestructura
- **Railway**: Plataforma de despliegue
- **Python 3.11.9**: Entorno de ejecución

### Modelo
- **Pipeline**: Random Forest Regressor con preprocesamiento integrado
- **Features**: Provincia, ciudad, sector, dormitorios, baños, área, garajes
- **Target**: Precio mensual de alquiler (USD)

---

## 📊 Ejemplos de Predicciones

| Provincia | Lugar | Dorm. | Baños | Área | Garages | Predicción |
|-----------|-------|-------|-------|------|---------|------------|
| Pichincha | Quito | 3 | 2 | 120 | 1 | **$750** |
| Guayas | Guayaquil | 2 | 2 | 85 | 1 | **$520** |
| Pichincha | Cumbayá | 4 | 3 | 200 | 2 | **$1,200** |
| Azuay | Cuenca | 3 | 2 | 110 | 1 | **$480** |

---

## 🔧 Mantenimiento y Contacto

### Variables de Entorno (Railway)

| Variable | Valor | Descripción |
|----------|-------|-------------|
| `MODEL_PATH` | `./models/rental_price_model.pkl` | Ruta del modelo |

### Versiones Actuales

| Componente | Versión |
|------------|---------|
| Python | 3.11.9 |
| FastAPI | 0.104.1 |
| NumPy | 2.3.5 |
| Pandas | 2.3.3 |
| Scikit-learn | 1.6.1 |

### Enlaces Útiles

- 📊 **Documentación API**: [https://rental-prediction-api-production.up.railway.app/docs](https://rental-prediction-api-production.up.railway.app/docs)
- 🔍 **Health Check**: [https://rental-prediction-api-production.up.railway.app/health](https://rental-prediction-api-production.up.railway.app/health)
- 🏠 **URL Base**: `https://rental-prediction-api-production.up.railway.app`

---

## 📝 Notas Adicionales

- La API está desplegada en el plan gratuito de Railway, por lo que puede haber una pequeña latencia en el primer request después de periodos de inactividad.
- El modelo fue entrenado con datos de Ecuador, por lo que funciona mejor para propiedades en este país.
- Para soporte o consultas, por favor crear un issue en el repositorio de GitHub.

---

## ⚖️ Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

---

**¡Gracias por usar la API de Predicción de Precios de Alquiler!** 🏠✨

---

*Última actualización: Marzo 2026*
