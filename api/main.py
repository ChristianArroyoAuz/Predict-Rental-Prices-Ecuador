# Librerías a instalar
# !pip install "uvicorn[standard]"
# !pip install fastapi
# !pip install pandas
# !pip install joblib
# !pip install numpy

# Importar Librerías

from fastapi.middleware.cors import CORSMiddleware   # Middleware para permitir peticiones desde diferentes orígenes (Cross-Origin Resource Sharing)
from fastapi import FastAPI, HTTPException           # FastAPI para crear la aplicación web y HTTPException para manejar errores HTTP
from pathlib import Path                             # Manejo de rutas de archivos de forma multiplataforma
import pandas as pd                                  # Biblioteca para Ciencia de Datos
import numpy as np                                   # Librerías para operaciones numéricas
import joblib                                        # Biblioteca para guardar y cargar modelos de Machine Learning
import sys                                           # Acceso a variables y funciones del sistema

# Agregar ruta padre para importar módulos
sys.path.append(str(Path(__file__).parent.parent))                       # Permite importar módulos desde el directorio padre del proyecto
from api.schemas import PropertyInput, PropertyOutput, ErrorResponse     # Importa los modelos Pydantic definidos en schemas.py para validación de datos

# Inicializar FastAPI
# Crea la aplicación FastAPI con metadatos para documentación
app = FastAPI(
    title="API de Predicción de Alquileres",
    description="API para predecir el precio de alquiler de propiedades en Ecuador",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Permite peticiones desde cualquier origen
    allow_credentials=True,         # Permite enviar cookies/credenciales
    allow_methods=["*"],            # Permite todos los métodos HTTP
    allow_headers=["*"],            # Permite todos los headers
)

# Cargar modelo al iniciar
# Define la ruta donde se encuentra el modelo entrenado
MODEL_PATH = Path(__file__).parent.parent / "models" / "rental_price_model.pkl"

# Intenta cargar el modelo al iniciar la aplicación
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    model = None

@app.get("/")
async def root():
    # Endpoint raíz que muestra información básica de la API
    return {
        "message": "API de Predicción de Alquileres",
        "docs": "/docs",                                     # Ruta a la documentación automática
        "status": "active" if model else "model_not_loaded"  # Estado del modelo
    }

@app.get("/health")
async def health_check():
    # Endpoint para verificar el estado de salud de la API
    return {
        "status": "healthy",
        "model_loaded": model is not None                    # Indica si el modelo está cargado
    }

@app.post("/predict", 
          response_model=PropertyOutput,                    # Define el modelo de respuesta
          responses={                                       # Documenta posibles respuestas de error
              400: {"model": ErrorResponse},
              500: {"model": ErrorResponse}
          })
async def predict(property_data: PropertyInput):           # Valida entrada con PropertyInput
    """
    Predice el precio de alquiler de una propiedad
    
    - **provincia**: Provincia donde se ubica la propiedad
    - **lugar**: Ciudad o localidad
    - **num_dormitorios**: Número de dormitorios
    - **num_banos**: Número de baños
    - **area**: Área en metros cuadrados
    - **num_garages**: Número de garajes
    """
    if model is None:
        # Si el modelo no está cargado, retorna error 500
        raise HTTPException(
            status_code=500,
            detail="El modelo no está disponible"
        )

    # Convierte los datos validados del schema al formato que espera el modelo ML
    try:
        # Convertir entrada a DataFrame
        input_data = pd.DataFrame([{
            'Provincia': property_data.provincia,
            'ciudad': property_data.lugar,                   # Mapea 'lugar' del schema a 'ciudad' que espera el modelo
            'sector': property_data.lugar,                   # Usa lugar como sector también
            'Num. dormitorios': property_data.num_dormitorios,
            'Num. banos': property_data.num_banos,
            'Area': property_data.area,
            'Num. garages': property_data.num_garages
        }])
        
        # Hacer predicción
        prediction = model.predict(input_data)[0]  # Obtiene el primer (y único) valor predicho
        
        # Redondear a 2 decimales
        prediction = round(float(prediction), 2)  # Formatea la predicción
        
        return PropertyOutput(prediction=prediction)  # Retorna usando el schema de salida
        
    except Exception as e:
        # Captura cualquier error durante la predicción
        raise HTTPException(
            status_code=400,
            detail=f"Error en la predicción: {str(e)}"
        )

# Ejecuta la aplicación con uvicorn si se llama directamente
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)