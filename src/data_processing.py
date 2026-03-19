"""
Módulo para procesamiento y limpieza de datos.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Tuple, Dict, List, Optional, Union
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Clase para procesar y limpiar datos de propiedades en alquiler.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Inicializa el procesador de datos.
        
        Args:
            random_state: Semilla para reproducibilidad
        """
        self.random_state = random_state
        self.ciudades_conocidas = [
            'Quito', 'Guayaquil', 'Cuenca', 'Sangolquí', 'Cumbayá', 
            'Tumbaco', 'Machala', 'Manta', 'Babahoyo', 'Latacunga',
            'Esmeraldas', 'Samborondón', 'Durán', 'Loja', 'Ambato',
            'Santo Domingo', 'Quevedo', 'Milagro', 'Ibarra', 'Riobamba'
        ]
        self.preprocessor = None
        self.feature_names = None
        
    def load_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Carga el dataset desde un archivo CSV.
        
        Args:
            filepath: Ruta al archivo CSV
            
        Returns:
            DataFrame con los datos cargados
        """
        logger.info(f"Cargando datos desde {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    
    def extract_location_info(self, lugar_str: str) -> pd.Series:
        """
        Extrae ciudad y sector de la columna Lugar.
        
        Args:
            lugar_str: String con la información de ubicación
            
        Returns:
            Series con ciudad y sector
        """
        if pd.isna(lugar_str):
            return pd.Series({'ciudad': None, 'sector': None})
        
        # Limpiar el string
        lugar_str = str(lugar_str).strip()
        partes = [p.strip() for p in lugar_str.split(',')]
        
        ciudad_encontrada = None
        sector = None
        
        # Buscar ciudad conocida
        for parte in partes:
            for ciudad in self.ciudades_conocidas:
                if ciudad.lower() in parte.lower():
                    ciudad_encontrada = ciudad
                    break
            if ciudad_encontrada:
                break
        
        # Si no se encuentra ciudad, buscar patrones comunes
        if not ciudad_encontrada:
            for parte in partes:
                if 'quito' in parte.lower():
                    ciudad_encontrada = 'Quito'
                    break
                elif 'guayaquil' in parte.lower():
                    ciudad_encontrada = 'Guayaquil'
                    break
                elif 'cuenca' in parte.lower():
                    ciudad_encontrada = 'Cuenca'
                    break
        
        # Determinar sector (parte anterior a la ciudad)
        if ciudad_encontrada:
            idx_ciudad = None
            for i, parte in enumerate(partes):
                if ciudad_encontrada.lower() in parte.lower():
                    idx_ciudad = i
                    break
            
            if idx_ciudad and idx_ciudad > 0:
                sector = partes[idx_ciudad - 1]
                
                # Limpiar sector de palabras comunes
                palabras_limpiar = ['sector', 'urbanización', 'ciudadela', 'conjunto', 'barrio']
                for palabra in palabras_limpiar:
                    if sector and palabra in sector.lower():
                        sector = sector.replace(palabra, '').strip()
        
        return pd.Series({'ciudad': ciudad_encontrada, 'sector': sector})
    
    def infer_property_type(self, titulo: str) -> str:
        """
        Infiere el tipo de propiedad desde el título.
        
        Args:
            titulo: Título de la publicación
            
        Returns:
            Tipo de propiedad
        """
        if pd.isna(titulo):
            return 'Desconocido'
        
        titulo_lower = str(titulo).lower()
        
        # Mapeo de palabras clave a tipos
        type_mapping = {
            'casa': 'Casa',
            'departamento': 'Departamento',
            'depto': 'Departamento',
            'suite': 'Suite',
            'local': 'Local Comercial',
            'oficina': 'Oficina',
            'consulta': 'Consultorio',
            'bodega': 'Bodega',
            'galpón': 'Galpón',
            'terreno': 'Terreno',
            'lote': 'Terreno',
            'edificio': 'Edificio',
            'habitación': 'Habitación',
            'cuarto': 'Habitación',
            'penthouse': 'Penthouse',
            'loft': 'Loft',
            'duplex': 'Dúplex',
            'quinta': 'Quinta',
            'hostal': 'Hostal',
            'hotel': 'Hotel'
        }
        
        for keyword, prop_type in type_mapping.items():
            if keyword in titulo_lower:
                return prop_type
        
        return 'Otro'
    
    def clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y convierte columnas numéricas.
        
        Args:
            df: DataFrame con datos
            
        Returns:
            DataFrame con columnas numéricas limpias
        """
        df_clean = df.copy()
        
        # Columnas numéricas a limpiar
        numeric_cols = ['Precio', 'Num. dormitorios', 'Num. banos', 'Area', 'Num. garages']
        
        for col in numeric_cols:
            if col in df_clean.columns:
                # Convertir a string y limpiar
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].str.replace(',', '').str.strip()
                df_clean[col] = df_clean[col].str.extract(r'(\d+\.?\d*)')[0]
                
                # Convertir a numérico
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Validaciones específicas
                if col == 'Precio':
                    df_clean.loc[df_clean[col] < 10, col] = np.nan  # Precios muy bajos
                    df_clean.loc[df_clean[col] > 100000, col] = np.nan  # Precios muy altos
                elif col == 'Area':
                    df_clean.loc[df_clean[col] < 5, col] = np.nan  # Áreas muy pequeñas
                    df_clean.loc[df_clean[col] > 10000, col] = np.nan  # Áreas muy grandes
                elif col in ['Num. dormitorios', 'Num. banos', 'Num. garages']:
                    df_clean.loc[df_clean[col] < 0, col] = np.nan
                    df_clean.loc[df_clean[col] > 20, col] = np.nan
        
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maneja valores faltantes en el DataFrame.
        
        Args:
            df: DataFrame con datos
            
        Returns:
            DataFrame sin valores faltantes
        """
        df_clean = df.copy()
        
        # Columnas numéricas - imputar con mediana
        numeric_cols = ['Num. dormitorios', 'Num. banos', 'Area', 'Num. garages']
        for col in numeric_cols:
            if col in df_clean.columns:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                logger.info(f"Columna {col}: imputados {df[col].isna().sum()} valores con mediana {median_val}")
        
        # Columnas categóricas - imputar con 'Desconocido'
        categorical_cols = ['Provincia', 'ciudad', 'sector']
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col].fillna('Desconocido', inplace=True)
                logger.info(f"Columna {col}: imputados {df[col].isna().sum()} valores con 'Desconocido'")
        
        return df_clean
    
    def remove_outliers(self, df: pd.DataFrame, column: str = 'Precio', 
                        method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Elimina outliers de una columna.
        
        Args:
            df: DataFrame con datos
            column: Columna para detectar outliers
            method: Método de detección ('iqr' o 'zscore')
            threshold: Umbral para considerar outlier
            
        Returns:
            DataFrame sin outliers
        """
        df_clean = df.copy()
        
        if method == 'iqr':
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            mask = (df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)
            n_removed = (~mask).sum()
            
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df_clean[column].dropna()))
            mask = z_scores < threshold
            n_removed = len(df_clean) - len(mask)
            
        else:
            raise ValueError(f"Método {method} no soportado")
        
        df_clean = df_clean[mask].copy()
        logger.info(f"Eliminados {n_removed} outliers de {column}")
        
        return df_clean
    
    def create_price_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea categoría de precio basada en cuartiles por ciudad.
        
        Args:
            df: DataFrame con datos
            
        Returns:
            DataFrame con columna 'tipo_precio'
        """
        df_clean = df.copy()
        
        # Cuartiles generales como fallback
        q1_general = df_clean['Precio'].quantile(0.25)
        q3_general = df_clean['Precio'].quantile(0.75)
        
        def categorizar_por_ciudad(grupo):
            if len(grupo) >= 5:
                q1 = grupo['Precio'].quantile(0.25)
                q3 = grupo['Precio'].quantile(0.75)
                
                def categorizar(precio):
                    if precio < q1:
                        return 'Económico'
                    elif precio > q3:
                        return 'Lujo'
                    else:
                        return 'Medio'
                
                return grupo['Precio'].apply(categorizar)
            else:
                # Usar cuartiles generales
                def categorizar(precio):
                    if precio < q1_general:
                        return 'Económico'
                    elif precio > q3_general:
                        return 'Lujo'
                    else:
                        return 'Medio'
                
                return grupo['Precio'].apply(categorizar)
        
        # Aplicar por ciudad
        df_clean['tipo_precio'] = df_clean.groupby('ciudad', group_keys=False).apply(categorizar_por_ciudad)
        
        logger.info(f"Categorías creadas: {df_clean['tipo_precio'].value_counts().to_dict()}")
        
        return df_clean
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara características para modelado.
        
        Args:
            df: DataFrame procesado
            
        Returns:
            Tupla (X, y) con características y target
        """
        # Seleccionar columnas relevantes
        feature_cols = ['Provincia', 'ciudad', 'sector', 'Num. dormitorios', 
                        'Num. banos', 'Area', 'Num. garages']
        target_col = 'Precio'
        
        # Verificar que todas las columnas existen
        available_features = [col for col in feature_cols if col in df.columns]
        missing = set(feature_cols) - set(available_features)
        if missing:
            logger.warning(f"Columnas faltantes: {missing}")
        
        X = df[available_features].copy()
        y = df[target_col].copy() if target_col in df.columns else None
        
        logger.info(f"Features preparadas: {X.shape}")
        
        return X, y
    
    def create_preprocessing_pipeline(self, categorical_cols: List[str], 
                                       numeric_cols: List[str]) -> ColumnTransformer:
        """
        Crea pipeline de preprocesamiento.
        
        Args:
            categorical_cols: Lista de columnas categóricas
            numeric_cols: Lista de columnas numéricas
            
        Returns:
            ColumnTransformer configurado
        """
        # Transformador para categóricas
        categorical_transformer = OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        )
        
        # Transformador para numéricas
        numeric_transformer = StandardScaler()
        
        # Combinar transformadores
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, 
                   val_size: Optional[float] = None) -> Tuple:
        """
        Divide datos en entrenamiento, validación y prueba.
        
        Args:
            X: Características
            y: Target
            test_size: Tamaño del conjunto de prueba
            val_size: Tamaño del conjunto de validación (opcional)
            
        Returns:
            Tupla con splits (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if val_size:
            # Primero separar prueba
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
            
            # Luego separar validación del resto
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=self.random_state
            )
            
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
            return X_train, X_test, y_train, y_test
    
    def process_complete(self, filepath: Union[str, Path], 
                         remove_outliers: bool = True) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo de procesamiento.
        
        Args:
            filepath: Ruta al archivo CSV
            remove_outliers: Si se deben eliminar outliers
            
        Returns:
            DataFrame procesado
        """
        # Cargar datos
        df = self.load_data(filepath)
        
        # Limpiar columnas numéricas
        df = self.clean_numeric_columns(df)
        
        # Extraer información de ubicación
        location_info = df['Lugar'].apply(self.extract_location_info)
        df['ciudad'] = location_info['ciudad']
        df['sector'] = location_info['sector']
        
        # Inferir tipo de propiedad
        df['tipo_propiedad'] = df['Titulo'].apply(self.infer_property_type)
        
        # Normalizar nombres de provincia
        df['Provincia'] = df['Provincia'].str.strip().str.title()
        
        # Manejar valores faltantes
        df = self.handle_missing_values(df)
        
        # Eliminar outliers si se solicita
        if remove_outliers:
            df = self.remove_outliers(df, column='Precio')
        
        # Crear categoría de precio
        df = self.create_price_category(df)
        
        logger.info(f"Procesamiento completado: {df.shape[0]} filas finales")
        
        return df


# Funciones de utilidad para mantener compatibilidad con notebooks
def load_and_process_data(filepath: str) -> pd.DataFrame:
    """Función helper para cargar y procesar datos."""
    processor = DataProcessor()
    return processor.process_complete(filepath)

def extract_lugar_info(lugar_str: str) -> pd.Series:
    """Función helper para extraer info de lugar."""
    processor = DataProcessor()
    return processor.extract_location_info(lugar_str)