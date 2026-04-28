import pandas as pd

from typing import Optional, Dict, Any

def compute_meta_features(X: pd.DataFrame, y: str, y_data_type: str = 'nominal') -> Dict[str, Any]:
  """Calcular las Meta-Features del dataset proporcionado

  Args:
    X (pd.DataFrame): dataset proporcionado
    y (str): nombre de la columna objetivo
    y_data_type (str): tipo de dato de la columna objetivo
  """

  # Calcular Meta-Feature: Número de instancias
  num_instances = X.shape[0]

  # Calcular Meta-Feature: Número de características
  num_instances = X.shape[1] - 1  # Restar la columna objetivo

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

  # TODO: información del dataset (si no está en meta-features)
  # TODO: - stats por columnas: mean, std, min, max, unique, top_values, porc
  # TODO: -

  return {
      "Number of Instances": num_instances,
      "Number of Features": num_instances,
      "Number of Classes": num_classes,
      "Class Proportions": class_proportions,
      "Number of Missing Values": num_missing_values,
      "Number of Outliers": outliers
  }
