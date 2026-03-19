"""
Módulo API - Servicio REST para predicción de alquileres.
"""

from api.main import app
from api.schemas import PropertyInput, PropertyOutput, ErrorResponse

__all__ = [
    'app',
    'PropertyInput',
    'PropertyOutput',
    'ErrorResponse'
]

__version__ = '1.0.0'