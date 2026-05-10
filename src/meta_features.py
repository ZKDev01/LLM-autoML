import pandas as pd

from typing import Optional, Dict, Any

def _get_column_types(X: pd.DataFrame, target_col: str) -> Dict[str, str]:
  column_types = {}
  for col in X.columns:
    if col == target_col:
      continue  # La target se maneja aparte

    dtype = X[col].dtype

    if pd.api.types.is_integer_dtype(dtype):
      column_types[col] = 'integer'
    elif pd.api.types.is_float_dtype(dtype):
      column_types[col] = 'float'
    elif pd.api.types.is_bool_dtype(dtype):
      column_types[col] = 'boolean'
    elif pd.api.types.is_datetime64_any_dtype(dtype):
      column_types[col] = 'datetime'
    elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
      # Para object, verificamos si podría ser categórico por cardinalidad baja
      unique_ratio = X[col].nunique() / len(X)
      if unique_ratio < 0.05:  # umbral ejemplo
        column_types[col] = 'categorical'
      else:
        column_types[col] = 'text'  # o 'mixed'
    else:
      column_types[col] = 'other'
  return column_types

def compute_meta_features(X: pd.DataFrame, y: str, y_data_type: str = 'nominal', include_landmarkers: bool = False) -> Dict[str, Any]:
  """Calcular las Meta-Features del dataset proporcionado

  Args:
    X (pd.DataFrame): dataset proporcionado
    y (str): nombre de la columna objetivo
    y_data_type (str): tipo de dato de la columna objetivo
  """

  # Calcular Meta-Feature: Número de instancias
  num_instances = X.shape[0]

  # Calcular Meta-Feature: Número de características
  num_features = X.shape[1] - 1  # Restar la columna objetivo

  # Calcular Meta-Feature: Número de clases
  if y_data_type == 'nominal':
    num_classes = X[y].nunique()
  else:
    num_classes = None

  # Calcular Meta-Feature: Proporción de clases
  if y_data_type == 'nominal':
    class_proportions = X[y].value_counts(normalize=True).to_dict()
  else:
    class_proportions = {}

  # Calcular Meta-Feature: Número de valores faltantes
  num_missing_values = X.isnull().sum().sum()

  # Calcular Meta-Feature: Número de outliers
  # - Usar el rango intercuartil (IQR) con factor 1.5
  numeric_cols = X.select_dtypes(include=['number']).columns
  # - Excluir la columna objetivo si es numérica (para no considerarla como predictor)
  if y in numeric_cols:
    numeric_cols = numeric_cols.drop(y)
  outliers = 0
  for col in numeric_cols:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers += ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()

  column_types = _get_column_types(X, y)

  # TODO: tipo de datos por columna
  # TODO: en correspondencia al tipo de dato, extraer de cada columna independiente sus meta-features (example: mean, std, min, max, unique, top_values, porcentage)
  # TODO: en función de si es aplicable a una, dos o más columnas, o todo el dataset, se deben aplicar meta-features: skewness, kurtosis, correlación entre variables, covarianza, concentración, dispersión, gravity, ANOVA p-value, coeficiente de variación, varianza explicada por PC1, asimetría de PCA, 95% de PCA, probabilidad de clase
  # TODO: añadir landmarkers a través de un parámetro (si el parámetro es True entonces aplicar landmarkers, por defecto = False) (se define también un pipeline genérico)

  return {
      "Number of Instances": num_instances,
      "Number of Features": num_features,
      "Number of Classes": num_classes,
      "Class Proportions": class_proportions,
      "Number of Missing Values": num_missing_values,
      "Number of Outliers": outliers,
      # "Column Types": column_types
  }
