"""
Módulo src - Procesamiento de datos y modelado para predicción de alquileres.
"""

from src.data_processing import DataProcessor
from src.model import ModelTrainer
from src.utils import setup_logging, save_artifacts, load_artifacts

__all__ = [
    'DataProcessor',
    'ModelTrainer',
    'setup_logging',
    'save_artifacts',
    'load_artifacts'
]

__version__ = '1.0.0'