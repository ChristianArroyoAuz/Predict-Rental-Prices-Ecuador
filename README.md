# 🏠 API de Predicción de Precios de Alquiler - Ecuador

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Railway](https://img.shields.io/badge/Railway-131415?style=for-the-badge&logo=railway&logoColor=white)](https://railway.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

API REST para predecir el precio de alquiler de propiedades en Ecuador utilizando Machine Learning.

---

## 📋 Tabla de Contenidos
- [Descripción de la Solución](#-descripción-de-la-solución)
- [Demo en Vivo](#-demo-en-vivo)
- [Instrucciones de Uso](#-instrucciones-de-uso)
- [Ejemplos de Uso](#-ejemplos-de-uso)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Mantenimiento](#-mantenimiento)

---

## 🏗️ Descripción de la Solución

Esta API permite predecir el precio mensual de alquiler de una propiedad en Ecuador basándose en características como ubicación (provincia y lugar), número de dormitorios, baños, área total y garajes.

El modelo de Machine Learning fue entrenado con un dataset de más de 500 propiedades en alquiler de diversas provincias ecuatorianas.

### 📊 Características del Modelo
- **Algoritmo**: Random Forest Regressor con pipeline de preprocesamiento
- **Features**: Provincia, lugar, dormitorios, baños, área, garajes
- **Target**: Precio mensual de alquiler en USD

---

## 🎯 Demo en Vivo

La API está desplegada y accesible públicamente en Railway:

| Recurso | URL |
|---------|-----|
| **URL Base** | `https://rental-prediction-api-production.up.railway.app` |
| **Documentación** | [https://rental-prediction-api-production.up.railway.app/docs](https://rental-prediction-api-production.up.railway.app/docs) |
| **Health Check** | [https://rental-prediction-api-production.up.railway.app/health](https://rental-prediction-api-production.up.railway.app/health) |

---

## 📖 Instrucciones de Uso

### 🔍 Health Check

Verifica el estado de la API y las versiones de las librerías.

**Endpoint:** `GET /health`

**Ejemplo:**
```bash
curl https://rental-prediction-api-production.up.railway.app/health
```

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "numpy_version": "2.3.5",
  "pandas_version": "2.3.3"
}
```

### 📈 Predicción

Realiza una predicción del precio de alquiler.

**Endpoint:** `POST /predict`

**Headers:**
```
Content-Type: application/json
```

**Body:**

| Campo | Tipo | Descripción | Ejemplo |
|-------|------|-------------|---------|
| `provincia` | string | Provincia de la propiedad | `"Pichincha"` |
| `lugar` | string | Ciudad o localidad | `"Quito"` |
| `num_dormitorios` | integer | Número de dormitorios | `3` |
| `num_banos` | integer | Número de baños | `2` |
| `area` | float | Área en metros cuadrados | `120` |
| `num_garages` | integer | Número de garajes | `1` |

**Respuesta Exitosa:**
```json
{
  "prediction": 532.07
}
```

---

## 💻 Ejemplos de Uso

### 📌 cURL (Linux/Mac/Windows)

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

### 📌 PowerShell (Windows)

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

### 📌 Python

```python
import requests

# Health Check
response = requests.get("https://rental-prediction-api-production.up.railway.app/health")
print(response.json())

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

---

## 📊 Ejemplo de Predicción

**Request:**
```json
{
  "provincia": "Pichincha",
  "lugar": "Quito",
  "num_dormitorios": 3,
  "num_banos": 2,
  "area": 120,
  "num_garages": 1
}
```

**Response:**
```json
{
  "prediction": 532.07
}
```

---

## 🛠️ Tecnologías Utilizadas

### Backend
- **FastAPI** - Framework web
- **Uvicorn** - Servidor ASGI
- **Pydantic** - Validación de datos

### Machine Learning
- **Scikit-learn 1.6.1** - Librería de ML
- **Pandas 2.3.3** - Manipulación de datos
- **NumPy 2.3.5** - Operaciones numéricas
- **Joblib 1.4.2** - Serialización del modelo

### Infraestructura
- **Railway** - Plataforma de despliegue
- **Python 3.11.9** - Entorno de ejecución

---

## 🔧 Mantenimiento

### Variables de Entorno (Railway)

| Variable | Valor |
|----------|-------|
| `MODEL_PATH` | `./models/rental_price_model.pkl` |

### Versiones Actuales

| Componente | Versión |
|------------|---------|
| Python | 3.11.9 |
| FastAPI | 0.104.1 |
| NumPy | 2.3.5 |
| Pandas | 2.3.3 |
| Scikit-learn | 1.6.1 |

### Enlaces Útiles

| Recurso | URL |
|---------|-----|
| **URL Base** | `https://rental-prediction-api-production.up.railway.app` |
| **Documentación** | [https://rental-prediction-api-production.up.railway.app/docs](https://rental-prediction-api-production.up.railway.app/docs) |
| **Health Check** | [https://rental-prediction-api-production.up.railway.app/health](https://rental-prediction-api-production.up.railway.app/health) |

---

## 📝 Notas

- La API está desplegada en el plan gratuito de Railway
- El modelo fue entrenado exclusivamente con datos de Ecuador
- Para soporte, crear un issue en el repositorio de GitHub

---

**¡Gracias por usar la API de Predicción de Precios de Alquiler!** 🏠✨

---

*Última actualización: Marzo 2026*
