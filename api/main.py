# Librerías a instalar
# !pip install "uvicorn[standard]"
# !pip install fastapi
# !pip install pandas
# !pip install joblib
# !pip install numpy

# Importar Librerías
from fastapi.middleware.cors import CORSMiddleware   # Middleware para permitir peticiones desde diferentes orígenes
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
from api.schemas import PropertyInput, PropertyOutput, ErrorResponse  # Importa los modelos Pydantic

# Inicializar FastAPI
app = FastAPI(
    title="API de Predicción de Alquileres",
    description="API para predecir el precio de alquiler de propiedades en Ecuador",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fix_numpy_compatibility():
    """
    Aplica parches de compatibilidad para NumPy
    Soluciona el error: <class 'numpy.random._mt19937.MT19937'> is not a known BitGenerator module
    """
    try:
        # Método 1: Parche directo para MT19937
        if not hasattr(np.random, 'MT19937'):
            try:
                from numpy.random import MT19937
                np.random.MT19937 = MT19937
                print("✅ Parche MT19937 aplicado (método 1)")
            except ImportError:
                try:
                    # Método 2: Importar desde el módulo interno
                    import numpy.random._mt19937 as mt19937
                    np.random.MT19937 = mt19937.MT19937
                    print("✅ Parche MT19937 aplicado (método 2)")
                except ImportError:
                    # Método 3: Crear clase dummy si todo falla
                    class DummyMT19937:
                        def __init__(self, *args, **kwargs):
                            pass
                    np.random.MT19937 = DummyMT19937
                    print("⚠️ Parche MT19937 dummy aplicado")
        
        # Método 4: Parche para el módulo _mt19937 completo
        if 'numpy.random._mt19937' not in sys.modules:
            import types
            mock_module = types.ModuleType('numpy.random._mt19937')
            mock_module.MT19937 = getattr(np.random, 'MT19937', type('MT19937', (), {}))
            sys.modules['numpy.random._mt19937'] = mock_module
            print("✅ Módulo _mt19937 mock creado")
            
    except Exception as e:
        print(f"⚠️ Error aplicando parches: {e}")

def load_model():
    """
    Carga el modelo con manejo de errores mejorado
    """
    # Aplicar parches de compatibilidad ANTES de cargar el modelo
    fix_numpy_compatibility()
    
    # Detectar si estamos en Railway
    is_railway = os.environ.get('RAILWAY_ENVIRONMENT') is not None
    
    # Obtener ruta del modelo desde variable de entorno
    model_path_env = os.environ.get('MODEL_PATH', './models/rental_price_model.pkl')
    
    print(f"🔍 Railway environment: {is_railway}")
    print(f"🔍 MODEL_PATH env: {model_path_env}")
    print(f"🔍 Current working directory: {Path.cwd()}")
    
    # Construir ruta absoluta según el entorno
    if model_path_env.startswith('./'):
        # En Railway, usar /app como base (según logs)
        base_path = Path('/app')
        MODEL_PATH = base_path / model_path_env[2:]
        print(f"✅ Usando {base_path} como base path")
    else:
        MODEL_PATH = Path(model_path_env)
    
    print(f"📁 Buscando modelo en: {MODEL_PATH}")
    print(f"📁 El archivo existe: {MODEL_PATH.exists()}")
    
    # Verificar directorio de models
    models_dir = MODEL_PATH.parent
    if models_dir.exists():
        print(f"\n📂 Contenido de {models_dir}:")
        for file in models_dir.glob("*"):
            if file.is_file():
                size = file.stat().st_size / (1024*1024)  # tamaño en MB
                print(f"  📄 {file.name} ({size:.2f} MB)")
    else:
        print(f"\n❌ El directorio {models_dir} NO existe")
        return None
    
    if not MODEL_PATH.exists():
        print("❌ No se puede cargar el modelo: archivo no encontrado")
        return None
    
    # Intentar cargar el modelo con diferentes estrategias
    try:
        print(f"\n🔄 Cargando modelo desde {MODEL_PATH}...")
        print(f"📊 Versión de NumPy: {np.__version__}")
        
        # Estrategia 1: Carga normal
        try:
            model = joblib.load(MODEL_PATH)
            print("✅ Modelo cargado exitosamente (estrategia 1)")
        except Exception as e:
            if "BitGenerator" in str(e):
                print("⚠️ Error de BitGenerator detectado, intentando estrategia 2...")
                
                # Estrategia 2: Carga con mmap_mode
                try:
                    model = joblib.load(MODEL_PATH, mmap_mode='r')
                    print("✅ Modelo cargado exitosamente (estrategia 2 - mmap_mode)")
                except Exception as e2:
                    print(f"⚠️ Estrategia 2 falló: {e2}")
                    
                    # Estrategia 3: Carga en entorno limpio
                    import pickle
                    try:
                        with open(MODEL_PATH, 'rb') as f:
                            model = pickle.load(f)
                        print("✅ Modelo cargado exitosamente (estrategia 3 - pickle)")
                    except Exception as e3:
                        print(f"⚠️ Estrategia 3 falló: {e3}")
                        
                        # Estrategia 4: Usar parche adicional
                        import numpy.random._mt19937 as mt19937
                        np.random.MT19937 = mt19937.MT19937
                        sys.modules['numpy.random._mt19937'] = mt19937
                        
                        model = joblib.load(MODEL_PATH)
                        print("✅ Modelo cargado exitosamente (estrategia 4 - parche forzado)")
            else:
                raise e
        
        # Verificar que el modelo tiene método predict
        if not hasattr(model, 'predict'):
            print("❌ El objeto cargado no tiene método predict")
            return None
        
        print(f"✅ Modelo cargado exitosamente")
        print(f"📊 Tipo de modelo: {type(model).__name__}")
        
        # Mostrar información del pipeline
        if hasattr(model, 'steps'):
            print(f"📊 Pasos del pipeline: {[step[0] for step in model.steps]}")
        elif hasattr(model, 'named_steps'):
            print(f"📊 Pasos del pipeline: {list(model.named_steps.keys())}")
        
        return model
        
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
        return {
            "status": "no_model",
            "message": "Modelo no cargado",
            "numpy_version": np.__version__,
            "has_mt19937": hasattr(np.random, 'MT19937')
        }
    
    info = {
        "status": "loaded",
        "model_path": str(MODEL_PATH) if 'MODEL_PATH' in dir() else "unknown",
        "model_type": type(model).__name__,
        "numpy_version": np.__version__,
        "has_mt19937": hasattr(np.random, 'MT19937')
    }
    
    if hasattr(model, 'steps'):
        info["steps"] = [{"name": step[0], "type": type(step[1]).__name__} for step in model.steps]
    elif hasattr(model, 'named_steps'):
        info["steps"] = [{"name": name, "type": type(step).__name__} 
                        for name, step in model.named_steps.items()]
    
    return info

@app.get("/")
async def root():
    return {
        "message": "API de Predicción de Alquileres",
        "docs": "/docs",
        "status": "active" if model else "model_not_loaded",
        "numpy_version": np.__version__,
        "has_mt19937": hasattr(np.random, 'MT19937')
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "numpy_version": np.__version__,
        "has_mt19937": hasattr(np.random, 'MT19937'),
        "python_version": sys.version.split()[0]
    }

@app.get("/debug/random")
async def debug_random():
    """Endpoint para debug del sistema aleatorio"""
    import numpy.random
    return {
        "has_mt19937": hasattr(np.random, 'MT19937'),
        "mt19937_type": str(type(np.random.MT19937)) if hasattr(np.random, 'MT19937') else None,
        "random_dir": dir(np.random)[:20],
        "has_module": 'numpy.random._mt19937' in sys.modules
    }

@app.post("/predict", 
          response_model=PropertyOutput,
          responses={
              400: {"model": ErrorResponse},
              500: {"model": ErrorResponse}
          })
async def predict(property_data: PropertyInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Por favor intenta más tarde."
        )

    try:
        # Convertir entrada a DataFrame
        input_data = pd.DataFrame([{
            'Provincia': property_data.provincia,
            'ciudad': property_data.lugar,
            'sector': property_data.lugar,
            'Num. dormitorios': property_data.num_dormitorios,
            'Num. banos': property_data.num_banos,
            'Area': property_data.area,
            'Num. garages': property_data.num_garages
        }])
        
        print(f"📥 Input recibido: {property_data}")
        print(f"📊 DataFrame creado: {input_data.to_dict()}")
        
        # Hacer predicción
        prediction = model.predict(input_data)[0]
        prediction = round(float(prediction), 2)
        
        print(f"📤 Predicción: {prediction}")
        
        return PropertyOutput(prediction=prediction)
        
    except Exception as e:
        print(f"❌ Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error en la predicción: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get('PORT', 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
