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
import os
import warnings
warnings.filterwarnings("ignore")                    # Ignorar warnings de versiones

# Agregar ruta padre para importar módulos
sys.path.append(str(Path(__file__).parent.parent))   # Permite importar módulos desde el directorio padre del proyecto
from api.schemas import PropertyInput, PropertyOutput, ErrorResponse  # Importa los modelos Pydantic definidos en schemas.py para validación de datos

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

def load_model():
    """
    Carga el modelo con manejo de errores mejorado y detección del entorno Railway
    """
    # Detectar si estamos en Railway
    is_railway = os.environ.get('RAILWAY_ENVIRONMENT') is not None
    
    # Obtener ruta del modelo desde variable de entorno
    model_path_env = os.environ.get('MODEL_PATH', './models/rental_price_model.pkl')
    
    print(f"🔍 Railway environment: {is_railway}")
    print(f"🔍 MODEL_PATH env: {model_path_env}")
    print(f"🔍 Current working directory: {Path.cwd()}")
    
    # Construir ruta absoluta según el entorno
    if model_path_env.startswith('./'):
        if is_railway:
            # En Railway, el repositorio se clona en /main/
            # Verificar si /main existe
            main_dir = Path('/main')
            if main_dir.exists():
                base_path = main_dir
                print(f"✅ Usando /main/ como base path (Railway)")
            else:
                # Fallback a directorio actual si /main no existe
                base_path = Path.cwd()
                print(f"⚠️ /main/ no existe, usando {base_path} como base path")
        else:
            # Localmente, usar ruta relativa al archivo
            base_path = Path(__file__).parent.parent
            print(f"✅ Usando {base_path} como base path (local)")
        
        MODEL_PATH = base_path / model_path_env[2:]
    else:
        MODEL_PATH = Path(model_path_env)
    
    print(f"📁 Buscando modelo en: {MODEL_PATH}")
    print(f"📁 El archivo existe: {MODEL_PATH.exists()}")
    
    # Debug: Explorar sistema de archivos para diagnóstico
    print("\n📂 Explorando sistema de archivos:")
    
    # Listar directorios relevantes
    dirs_to_check = ['/', '/main', '/app', Path.cwd()]
    for dir_path in dirs_to_check:
        path = Path(dir_path) if isinstance(dir_path, str) else dir_path
        if path.exists():
            print(f"\n📂 Contenido de {path}:")
            try:
                for item in sorted(path.glob("*"))[:10]:  # Limitar a 10 items
                    if item.is_dir():
                        print(f"  📁 {item.name}/")
                    else:
                        size = item.stat().st_size
                        if size > 1024*1024:
                            print(f"  📄 {item.name} ({size/(1024*1024):.2f} MB)")
                        elif size > 1024:
                            print(f"  📄 {item.name} ({size/1024:.2f} KB)")
                        else:
                            print(f"  📄 {item.name} ({size} bytes)")
            except Exception as e:
                print(f"  Error listando {path}: {e}")
    
    # Verificar directorio de models específicamente
    models_dir = MODEL_PATH.parent
    if models_dir.exists():
        print(f"\n📂 Contenido de {models_dir}:")
        for file in models_dir.glob("*"):
            if file.is_file():
                size = file.stat().st_size / (1024*1024)  # tamaño en MB
                print(f"  📄 {file.name} ({size:.2f} MB)")
    else:
        print(f"\n❌ El directorio {models_dir} NO existe")
        
        # Buscar archivos .pkl en todo el sistema
        print("\n🔍 Buscando archivos .pkl:")
        for root in ['/main', '/app', Path.cwd()]:
            root_path = Path(root) if isinstance(root, str) else root
            if root_path.exists():
                for pkl_file in root_path.rglob("*.pkl"):
                    print(f"  📄 Encontrado: {pkl_file}")
    
    if not MODEL_PATH.exists():
        print("❌ No se puede cargar el modelo: archivo no encontrado")
        return None
    
    # Intentar cargar el modelo
    try:
        print(f"\n🔄 Cargando modelo desde {MODEL_PATH}...")
        print(f"📊 Versión de NumPy: {np.__version__}")
        
        # Intentar cargar con diferentes métodos si es necesario
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            if "BitGenerator" in str(e):
                print("⚠️ Error de BitGenerator detectado, intentando parche...")
                # Parche para compatibilidad con NumPy 2.x
                if not hasattr(np.random, 'MT19937'):
                    import numpy.random._mt19937 as mt19937
                    np.random.MT19937 = mt19937.MT19937
                model = joblib.load(MODEL_PATH)
            else:
                raise e
        
        print(f"✅ Modelo cargado exitosamente")
        print(f"📊 Tipo de modelo: {type(model).__name__}")
        
        # Si es un pipeline, mostrar los pasos
        if hasattr(model, 'steps'):
            print(f"📊 Pasos del pipeline: {[step[0] for step in model.steps]}")
        elif hasattr(model, 'named_steps'):
            print(f"📊 Pasos del pipeline: {list(model.named_steps.keys())}")
        
        return model
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("💡 Sugerencia: Verifica que todas las dependencias estén instaladas con las versiones correctas")
        return None
    except Exception as e:
        print(f"❌ Error inesperado cargando modelo: {e}")
        return None

# Cargar modelo al iniciar
print("\n🚀 Iniciando API...")
print(f"🐍 Python version: {sys.version}")
print(f"📦 NumPy version: {np.__version__}")
print(f"📦 Pandas version: {pd.__version__}")
print(f"📦 Joblib version: {joblib.__version__}")

model = load_model()

@app.get("/model-info")
async def model_info():
    """Endpoint para ver información del modelo cargado"""
    if model is None:
        # Información de debug cuando el modelo no está cargado
        model_path_env = os.environ.get('MODEL_PATH', './models/rental_price_model.pkl')
        is_railway = os.environ.get('RAILWAY_ENVIRONMENT') is not None
        
        # Construir posibles rutas para debug
        possible_paths = {
            "env_path": model_path_env,
            "cwd_models": str(Path.cwd() / "models" / "rental_price_model.pkl"),
            "main_models": str(Path('/main/models/rental_price_model.pkl')),
            "app_models": str(Path('/app/models/rental_price_model.pkl')),
            "parent_models": str(Path(__file__).parent.parent / "models" / "rental_price_model.pkl")
        }
        
        exists_info = {path_name: Path(path_str).exists() 
                      for path_name, path_str in possible_paths.items() 
                      if path_str != str(Path(__file__).parent.parent / "models" / "rental_price_model.pkl")}
        
        return {
            "status": "no_model",
            "railway_environment": is_railway,
            "model_path_env": model_path_env,
            "possible_paths": possible_paths,
            "exists": exists_info,
            "cwd": str(Path.cwd())
        }
    
    # Obtener información básica del modelo
    info = {
        "status": "loaded",
        "model_path": str(MODEL_PATH) if 'MODEL_PATH' in dir() else "unknown",
        "model_type": type(model).__name__
    }
    
    # Si es un pipeline, obtener más detalles
    if hasattr(model, 'steps'):
        info["steps"] = [{"name": step[0], "type": type(step[1]).__name__} for step in model.steps]
    elif hasattr(model, 'named_steps'):
        info["steps"] = [{"name": name, "type": type(step).__name__} 
                        for name, step in model.named_steps.items()]
    
    return info

@app.get("/")
async def root():
    # Endpoint raíz que muestra información básica de la API
    return {
        "message": "API de Predicción de Alquileres",
        "docs": "/docs",                                     # Ruta a la documentación automática
        "status": "active" if model else "model_not_loaded", # Estado del modelo
        "environment": os.environ.get('RAILWAY_ENVIRONMENT', 'local'),
        "numpy_version": np.__version__
    }

@app.get("/health")
async def health_check():
    # Endpoint para verificar el estado de salud de la API
    model_path_env = os.environ.get('MODEL_PATH', './models/rental_price_model.pkl')
    is_railway = os.environ.get('RAILWAY_ENVIRONMENT') is not None
    
    # Determinar ruta potencial para debug
    if is_railway:
        potential_path = Path('/main/models/rental_price_model.pkl')
    else:
        potential_path = Path(__file__).parent.parent / "models" / "rental_price_model.pkl"
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "railway_environment": is_railway,
        "model_path_env": model_path_env,
        "potential_model_exists": potential_path.exists(),
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0]
    }

@app.get("/debug/paths")
async def debug_paths():
    """Endpoint para debug de rutas"""
    is_railway = os.environ.get('RAILWAY_ENVIRONMENT') is not None
    
    paths = {
        "current_dir": str(Path.cwd()),
        "file_dir": str(Path(__file__).parent),
        "parent_dir": str(Path(__file__).parent.parent),
        "railway_environment": is_railway,
        "model_path_env": os.environ.get('MODEL_PATH', './models/rental_price_model.pkl'),
    }
    
    # Verificar rutas específicas
    paths_to_check = [
        "/main/models/rental_price_model.pkl",
        "/app/models/rental_price_model.pkl",
        str(Path.cwd() / "models" / "rental_price_model.pkl"),
        str(Path(__file__).parent.parent / "models" / "rental_price_model.pkl")
    ]
    
    for path_str in paths_to_check:
        path = Path(path_str)
        paths[f"exists_{path_str}"] = path.exists()
        if path.exists():
            paths[f"size_{path_str}"] = f"{path.stat().st_size / (1024*1024):.2f} MB"
    
    return paths

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
        # Si el modelo no está cargado, retorna error 503 (Service Unavailable)
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Por favor intenta más tarde."
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
        
        print(f"📥 Input recibido: {property_data}")
        print(f"📊 DataFrame creado: {input_data.to_dict()}")
        
        # Hacer predicción
        prediction = model.predict(input_data)[0]  # Obtiene el primer (y único) valor predicho
        
        # Redondear a 2 decimales
        prediction = round(float(prediction), 2)  # Formatea la predicción
        
        print(f"📤 Predicción: {prediction}")
        
        return PropertyOutput(prediction=prediction)  # Retorna usando el schema de salida
        
    except Exception as e:
        # Captura cualquier error durante la predicción
        print(f"❌ Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error en la predicción: {str(e)}"
        )

# Ejecuta la aplicación con uvicorn si se llama directamente
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
