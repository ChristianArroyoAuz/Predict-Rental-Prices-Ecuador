"""
Módulo de utilidades para el proyecto.
"""

import logging
import json
import yaml
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de logging
def setup_logging(log_level: str = 'INFO', 
                  log_file: Optional[Union[str, Path]] = None) -> logging.Logger:
    """
    Configura el sistema de logging.
    
    Args:
        log_level: Nivel de logging ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Archivo para guardar logs (opcional)
        
    Returns:
        Logger configurado
    """
    # Crear logger
    logger = logging.getLogger('rental_prediction')
    logger.setLevel(getattr(logging, log_level))
    
    # Formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo (opcional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_artifacts(artifacts: Dict[str, Any], 
                   base_path: Union[str, Path],
                   subdirs: Optional[Dict[str, str]] = None) -> None:
    """
    Guarda múltiples artefactos del modelo.
    
    Args:
        artifacts: Diccionario {nombre: objeto} a guardar
        base_path: Ruta base para guardar
        subdirs: Diccionario {nombre: subdirectorio} opcional
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for name, obj in artifacts.items():
        # Determinar subdirectorio
        if subdirs and name in subdirs:
            save_dir = base_path / subdirs[name]
        else:
            save_dir = base_path
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Nombre del archivo
        if hasattr(obj, 'predict'):  # Es un modelo
            filename = save_dir / f"{name}_{timestamp}.pkl"
            joblib.dump(obj, filename)
        elif isinstance(obj, pd.DataFrame):
            filename = save_dir / f"{name}_{timestamp}.csv"
            obj.to_csv(filename, index=False)
        elif isinstance(obj, (dict, list)):
            filename = save_dir / f"{name}_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
        elif isinstance(obj, plt.Figure):
            filename = save_dir / f"{name}_{timestamp}.png"
            obj.savefig(filename, dpi=100, bbox_inches='tight')
            plt.close(obj)
        else:
            # Guardar como joblib por defecto
            filename = save_dir / f"{name}_{timestamp}.pkl"
            joblib.dump(obj, filename)
        
        print(f"Guardado: {filename}")


def load_artifacts(filepath: Union[str, Path]) -> Any:
    """
    Carga artefactos guardados.
    
    Args:
        filepath: Ruta del archivo a cargar
        
    Returns:
        Objeto cargado
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
    
    # Determinar tipo por extensión
    if filepath.suffix == '.pkl':
        return joblib.load(filepath)
    elif filepath.suffix == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif filepath.suffix == '.csv':
        return pd.read_csv(filepath)
    elif filepath.suffix == '.yaml' or filepath.suffix == '.yml':
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Extensión no soportada: {filepath.suffix}")


def save_config(config: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Guarda configuración en formato YAML.
    
    Args:
        config: Diccionario de configuración
        filepath: Ruta donde guardar
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Configuración guardada en {filepath}")


def load_config(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Carga configuración desde YAML.
    
    Args:
        filepath: Ruta del archivo de configuración
        
    Returns:
        Diccionario de configuración
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def create_submission(df: pd.DataFrame, predictions: np.ndarray, 
                      id_column: str = 'index',
                      output_path: Union[str, Path] = 'submission.csv') -> None:
    """
    Crea archivo de submission para competencias.
    
    Args:
        df: DataFrame original
        predictions: Predicciones del modelo
        id_column: Columna con IDs
        output_path: Ruta de salida
    """
    submission = pd.DataFrame({
        id_column: df.index if id_column == 'index' else df[id_column],
        'predicted_price': predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Submission guardado en {output_path}")
    print(f"Shape: {submission.shape}")


def print_metrics(metrics: Dict[str, float], title: str = "Métricas del Modelo") -> None:
    """
    Imprime métricas de forma formateada.
    
    Args:
        metrics: Diccionario con métricas
        title: Título del reporte
    """
    print("=" * 50)
    print(f"{title:^50}")
    print("=" * 50)
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            if 'percentage' in metric.lower() or 'mape' in metric.lower():
                print(f"{metric:20}: {value:10.2f}%")
            else:
                print(f"{metric:20}: {value:10.2f}")
        else:
            print(f"{metric:20}: {value}")
    
    print("=" * 50)


def validate_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida y limpia datos de entrada para predicción.
    
    Args:
        data: Diccionario con datos de entrada
        
    Returns:
        Diccionario validado
    """
    validated = {}
    
    # Mapeo de campos esperados
    field_mapping = {
        'provincia': ['provincia', 'Provincia', 'province'],
        'lugar': ['lugar', 'Lugar', 'ciudad', 'Ciudad', 'city'],
        'num_dormitorios': ['num_dormitorios', 'Num. dormitorios', 'dormitorios', 'bedrooms'],
        'num_banos': ['num_banos', 'Num. banos', 'banos', 'bathrooms'],
        'area': ['area', 'Area', 'Área', 'surface', 'm2'],
        'num_garages': ['num_garages', 'Num. garages', 'garages', 'parking']
    }
    
    # Validar cada campo
    for std_name, possible_names in field_mapping.items():
        value = None
        for name in possible_names:
            if name in data:
                value = data[name]
                break
        
        if value is None:
            raise ValueError(f"Campo requerido no encontrado: {std_name}")
        
        # Convertir tipos según sea necesario
        if std_name in ['num_dormitorios', 'num_banos', 'num_garages']:
            validated[std_name] = int(float(value))
        elif std_name == 'area':
            validated[std_name] = float(value)
        else:
            validated[std_name] = str(value).strip()
    
    return validated


def create_sample_data(n_samples: int = 10) -> pd.DataFrame:
    """
    Crea datos de muestra para pruebas.
    
    Args:
        n_samples: Número de muestras
        
    Returns:
            DataFrame con datos de muestra
    """
    np.random.seed(42)
    
    provincias = ['Pichincha', 'Guayas', 'Azuay', 'Manabí', 'Imbabura']
    ciudades = ['Quito', 'Guayaquil', 'Cuenca', 'Manta', 'Ibarra']
    sectores = ['Norte', 'Sur', 'Centro', 'Valle', 'Costero']
    
    data = {
        'Provincia': np.random.choice(provincias, n_samples),
        'ciudad': np.random.choice(ciudades, n_samples),
        'sector': np.random.choice(sectores, n_samples),
        'Num. dormitorios': np.random.randint(1, 5, n_samples),
        'Num. banos': np.random.randint(1, 4, n_samples),
        'Area': np.random.randint(50, 300, n_samples),
        'Num. garages': np.random.randint(0, 3, n_samples),
        'Precio': np.random.randint(200, 2000, n_samples)
    }
    
    return pd.DataFrame(data)


# Decorador para medir tiempo de ejecución
def timer_decorator(func):
    """Decorador para medir tiempo de ejecución de funciones."""
    from functools import wraps
    import time
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} ejecutado en {end_time - start_time:.2f} segundos")
        return result
    return wrapper


# Clase para manejar memoria en datasets grandes
class MemoryEfficientDataFrame:
    """Clase para optimizar uso de memoria en DataFrames."""
    
    @staticmethod
    def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce el uso de memoria de un DataFrame.
        
        Args:
            df: DataFrame a optimizar
            
        Returns:
            DataFrame optimizado
        """
        start_mem = df.memory_usage().sum() / 1024**2
        print(f'Uso de memoria inicial: {start_mem:.2f} MB')
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        end_mem = df.memory_usage().sum() / 1024**2
        print(f'Uso de memoria final: {end_mem:.2f} MB')
        print(f'Reducción: {100 * (start_mem - end_mem) / start_mem:.1f}%')
        
        return df