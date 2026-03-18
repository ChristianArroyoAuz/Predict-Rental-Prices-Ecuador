# Librerías a instalar
# !pip install "uvicorn[standard]"
# !pip install fastapi
# !pip install pandas
# !pip install joblib
# !pip install numpy

# Importar Librerías
from pydantic import BaseModel        # Importa BaseModel de Pydantic para crear modelos de datos con validación
from typing import Optional           # Importa Optional para permitir que un campo pueda ser None
from pydantic import Field            # Importa Field para agregar validaciones y metadatos adicionales a los campos

# Define el modelo de datos para la entrada de propiedades inmobiliarias
# Hereda de BaseModel para obtener funcionalidades de validación
class PropertyInput(BaseModel):

    # Campo obligatorio de tipo string (los ... indican requerido)
    # example proporciona un valor de ejemplo para la documentación
    provincia: str = Field(..., example="Pichincha")

    # Campo obligatorio que representa la ciudad o localidad
    lugar: str = Field(..., example="Quito")

    # Campo obligatorio de tipo entero con validación ge=0 (mayor o igual a 0)
    num_dormitorios: int = Field(..., ge=0, example=3)

    # Campo obligatorio de tipo entero para número de baños
    num_banos: int = Field(..., ge=0, example=2)

    # Campo obligatorio de tipo float con validación gt=0 (estrictamente mayor a 0)
    area: float = Field(..., gt=0, example=120)

    # Campo obligatorio de tipo entero para número de garajes
    num_garages: int = Field(..., ge=0, example=1)
    
    class Config:
        # Clase de configuración interna para el modelo
        schema_extra = {
            # Proporciona un ejemplo completo para la documentación OpenAPI
            "example": {
                "provincia": "Pichincha",
                "lugar": "Quito",
                "num_dormitorios": 3,
                "num_banos": 2,
                "area": 120,
                "num_garages": 1
            }
        }
# Define el modelo de datos para la salida de predicciones
class PropertyOutput(BaseModel):
    # Campo de tipo float que contendrá el valor de la predicción
    prediction: float
    
    class Config:
        # Configuración para el modelo de salida
        schema_extra = {
            # Ejemplo de respuesta para la documentación
            "example": {
                "prediction": 750.0
            }
        }
        
# Define el modelo de datos para respuestas de error
class ErrorResponse(BaseModel):
    # Campo obligatorio con el mensaje de error
    error: str

    # Campo opcional para detalles adicionales del error
    # Por defecto es None si no se proporciona
    detail: Optional[str] = None