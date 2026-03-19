"""
Módulo para entrenamiento y evaluación de modelos.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Clase para entrenar y evaluar modelos de regresión.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Inicializa el entrenador de modelos.
        
        Args:
            random_state: Semilla para reproducibilidad
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = []
        self.feature_importance = None
        
        # Definir modelos base
        self.base_models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(random_state=random_state),
            'Lasso': Lasso(random_state=random_state),
            'Decision Tree': DecisionTreeRegressor(random_state=random_state),
            'Random Forest': RandomForestRegressor(random_state=random_state, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(random_state=random_state),
            'KNN': KNeighborsRegressor(n_jobs=-1),
            'SVR': SVR()
        }
        
        # Grid de parámetros para optimización
        self.param_grids = {
            'Ridge': {
                'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'Lasso': {
                'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            },
            'Random Forest': {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__max_depth': [10, 20, None],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4],
                'regressor__max_features': ['sqrt', 'log2']
            },
            'Gradient Boosting': {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__max_depth': [3, 5, 7],
                'regressor__learning_rate': [0.01, 0.1, 0.2],
                'regressor__min_samples_split': [2, 5],
                'regressor__subsample': [0.8, 1.0]
            },
            'Decision Tree': {
                'regressor__max_depth': [5, 10, 15, 20, None],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4]
            },
            'KNN': {
                'regressor__n_neighbors': [3, 5, 7, 9, 11],
                'regressor__weights': ['uniform', 'distance'],
                'regressor__p': [1, 2]
            },
            'SVR': {
                'regressor__C': [0.1, 1.0, 10.0],
                'regressor__gamma': ['scale', 'auto', 0.1, 0.01],
                'regressor__kernel': ['rbf', 'linear']
            }
        }
    
    def train_baseline_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame, y_test: pd.Series,
                              preprocessor: Optional[Any] = None) -> pd.DataFrame:
        """
        Entrena modelos baseline y compara su rendimiento.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_test: Features de prueba
            y_test: Target de prueba
            preprocessor: Pipeline de preprocesamiento (opcional)
            
        Returns:
            DataFrame con resultados
        """
        self.results = []
        
        for name, model in self.base_models.items():
            try:
                logger.info(f"Entrenando modelo: {name}")
                
                # Crear pipeline
                if preprocessor:
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', model)
                    ])
                else:
                    pipeline = model
                
                # Entrenar
                pipeline.fit(X_train, y_train)
                self.models[name] = pipeline
                
                # Predecir
                y_pred = pipeline.predict(X_test)
                
                # Métricas
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # Validación cruzada
                cv_scores = cross_val_score(pipeline, X_train, y_train, 
                                           cv=5, scoring='r2')
                
                self.results.append({
                    'Modelo': name,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R²': r2,
                    'CV R² (mean)': cv_scores.mean(),
                    'CV R² (std)': cv_scores.std()
                })
                
                logger.info(f"{name} - R²: {r2:.4f}, MAE: {mae:.2f}")
                
            except Exception as e:
                logger.error(f"Error entrenando {name}: {str(e)}")
        
        # Crear DataFrame de resultados
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('R²', ascending=False)
        
        # Guardar mejor modelo
        if len(results_df) > 0:
            self.best_model_name = results_df.iloc[0]['Modelo']
            self.best_model = self.models[self.best_model_name]
            logger.info(f"Mejor modelo: {self.best_model_name} con R²={results_df.iloc[0]['R²']:.4f}")
        
        return results_df
    
    def optimize_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                       preprocessor: Optional[Any] = None,
                       search_method: str = 'grid',
                       n_iter: int = 20,
                       cv: int = 5) -> Dict[str, Any]:
        """
        Optimiza hiperparámetros de un modelo.
        
        Args:
            model_name: Nombre del modelo a optimizar
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            preprocessor: Pipeline de preprocesamiento
            search_method: 'grid' o 'random'
            n_iter: Número de iteraciones para random search
            cv: Número de folds para validación cruzada
            
        Returns:
            Diccionario con mejor modelo y parámetros
        """
        if model_name not in self.base_models:
            raise ValueError(f"Modelo {model_name} no encontrado")
        
        if model_name not in self.param_grids:
            logger.warning(f"No hay grid definido para {model_name}, usando grid por defecto")
            param_grid = {}
        else:
            param_grid = self.param_grids[model_name]
        
        # Crear pipeline
        if preprocessor:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', self.base_models[model_name])
            ])
        else:
            pipeline = self.base_models[model_name]
        
        # Seleccionar método de búsqueda
        if search_method == 'grid':
            search = GridSearchCV(
                pipeline, param_grid, cv=cv, 
                scoring='r2', n_jobs=-1, verbose=1
            )
        else:
            search = RandomizedSearchCV(
                pipeline, param_grid, n_iter=n_iter, cv=cv,
                scoring='r2', n_jobs=-1, verbose=1, random_state=self.random_state
            )
        
        # Ejecutar búsqueda
        logger.info(f"Optimizando {model_name} con {search_method} search...")
        search.fit(X_train, y_train)
        
        logger.info(f"Mejores parámetros: {search.best_params_}")
        logger.info(f"Mejor puntuación CV: {search.best_score_:.4f}")
        
        # Guardar modelo optimizado
        self.models[f"{model_name}_optimized"] = search.best_estimator_
        
        return {
            'best_estimator': search.best_estimator_,
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, 
                       y_test: pd.Series) -> Dict[str, float]:
        """
        Evalúa un modelo con múltiples métricas.
        
        Args:
            model: Modelo entrenado
            X_test: Features de prueba
            y_test: Target de prueba
            
        Returns:
            Diccionario con métricas
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred),
            'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        # Mediana de error absoluto
        metrics['MedAE'] = np.median(np.abs(y_test - y_pred))
        
        # Error porcentual absoluto mediano
        metrics['MdAPE'] = np.median(np.abs((y_test - y_pred) / y_test)) * 100
        
        return metrics
    
    def plot_results(self, y_test: pd.Series, y_pred: np.ndarray, 
                     model_name: str = 'Modelo') -> plt.Figure:
        """
        Genera gráficos de evaluación del modelo.
        
        Args:
            y_test: Valores reales
            y_pred: Valores predichos
            model_name: Nombre del modelo
            
        Returns:
            Figura de matplotlib
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Predicho vs Real
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
        axes[0, 0].plot([y_test.min(), y_test.max()], 
                       [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Valor Real')
        axes[0, 0].set_ylabel('Valor Predicho')
        axes[0, 0].set_title(f'{model_name}: Predicho vs Real')
        
        # 2. Distribución de errores
        errors = y_test - y_pred
        axes[0, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Error de Predicción')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].set_title('Distribución de Errores')
        
        # 3. Error vs Valor Real
        axes[1, 0].scatter(y_test, errors, alpha=0.5)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Valor Real')
        axes[1, 0].set_ylabel('Error')
        axes[1, 0].set_title('Error vs Valor Real')
        
        # 4. Q-Q plot de errores
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot de Errores')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, model: Any, feature_names: List[str], 
                                 top_n: int = 20) -> plt.Figure:
        """
        Visualiza importancia de características.
        
        Args:
            model: Modelo entrenado (debe tener feature_importances_)
            feature_names: Nombres de las características
            top_n: Número de características a mostrar
            
        Returns:
            Figura de matplotlib
        """
        # Extraer el regressor del pipeline si es necesario
        if hasattr(model, 'named_steps'):
            regressor = model.named_steps['regressor']
        else:
            regressor = model
        
        # Verificar si el modelo tiene importancia de características
        if hasattr(regressor, 'feature_importances_'):
            importances = regressor.feature_importances_
        elif hasattr(regressor, 'coef_'):
            importances = np.abs(regressor.coef_)
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'Este modelo no soporta\nimportancia de características',
                   ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig
        
        # Crear DataFrame y ordenar
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=True).tail(top_n)
        
        self.feature_importance = importance_df
        
        # Graficar
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(importance_df['feature'], importance_df['importance'])
        ax.set_xlabel('Importancia')
        ax.set_title(f'Top {top_n} Características más Importantes')
        plt.tight_layout()
        
        return fig
    
    def save_model(self, model: Any, filepath: Union[str, Path]) -> None:
        """
        Guarda modelo entrenado.
        
        Args:
            model: Modelo a guardar
            filepath: Ruta donde guardar
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, filepath)
        logger.info(f"Modelo guardado en {filepath}")
    
    def load_model(self, filepath: Union[str, Path]) -> Any:
        """
        Carga modelo guardado.
        
        Args:
            filepath: Ruta del modelo
            
        Returns:
            Modelo cargado
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {filepath}")
        
        model = joblib.load(filepath)
        logger.info(f"Modelo cargado desde {filepath}")
        return model
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Obtiene resumen de todos los modelos entrenados.
        
        Returns:
            DataFrame con resumen
        """
        if not self.results:
            return pd.DataFrame()
        
        summary = pd.DataFrame(self.results)
        return summary.sort_values('R²', ascending=False)


# Función helper para entrenamiento rápido
def train_quick_model(X_train, y_train, X_test, y_test, 
                       model_type: str = 'random_forest') -> Tuple[Any, Dict]:
    """
    Entrena un modelo rápidamente con configuración por defecto.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_test: Features de prueba
        y_test: Target de prueba
        model_type: Tipo de modelo ('random_forest', 'gradient_boosting', 'ridge')
        
    Returns:
        Tupla (modelo, métricas)
    """
    trainer = ModelTrainer()
    
    # Seleccionar modelo
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
    else:
        raise ValueError(f"Tipo de modelo {model_type} no soportado")
    
    # Entrenar
    model.fit(X_train, y_train)
    
    # Evaluar
    metrics = trainer.evaluate_model(model, X_test, y_test)
    
    return model, metrics