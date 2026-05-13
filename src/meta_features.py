from sklearn.feature_selection import f_classif
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional

import sys
from pathlib import Path
sys.path.append('..')


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

  # column_types = _get_column_types(X, y)

  return {
      "Number of Instances": num_instances,
      "Number of Features": num_features,
      "Number of Classes": num_classes,
      "Class Proportions": class_proportions,
      "Number of Missing Values": num_missing_values,
      "Number of Outliers": outliers,
      # "Column Types": column_types
  }

def compute_all_meta_feats(X: pd.DataFrame, target_col: str, y_data_type: str = 'nominal', include_landmarkers: bool = True) -> Dict[str, Any]:
  """Calcula meta-features para un dataset, organizadas por categorías:
  - Simples
  - Estadísticas
  - Basadas en teoría de la información
  - Basadas en modelos
  - Landmarkers (opcional)

  Args:
    X (pd.DataFrame): dataframe completo (X|y)
    target_col (str): nombre de la columna de la variable objetivo
    y_data_type (str, optional): tipo de dato de la variable objetivo. Defaults to 'nominal'.
    include_landmarkers (bool, optional): activador para incluir los landmarkers dentro de las meta-features de salida. Defaults to True.
  """

  base = compute_meta_features(X, target_col, y_data_type)

  meta_feats = {
      "simple": {
          "n_instances": base["Number of Instances"],
          "n_features": base["Number of Features"],
          "n_classes": base["Number of Classes"],
          "class_proportions": list(base["Class Proportions"].values()),
          "n_missing_values": int(base["Number of Missing Values"]),
          "n_outliers": int(base["Number of Outliers"]),
      }
  }

  # Se separan las características y target
  y = X[target_col]
  X_feat = X.drop(columns=[target_col])
  numeric_cols = X_feat.select_dtypes(include=[np.number]).columns.tolist()

  # Meta-Features Estadísticas
  stats = {}
  if len(numeric_cols) > 0:
    num_data = X_feat[numeric_cols]
    stats['skewness_mean'] = float(num_data.skew().mean())
    stats['skewness_std'] = float(num_data.skew().std())
    stats["kurtosis_mean"] = float(num_data.kurtosis().mean())
    stats["kurtosis_std"] = float(num_data.kurtosis().std())

    # Correlación media entre pares de features (valor absoluto)
    corr_matrix = num_data.corr().abs()
    # Eliminar diagonal
    corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]
    stats["mean_abs_corr"] = float(np.mean(corr_values)) if len(corr_values) > 0 else 0.0
    stats["max_abs_corr"] = float(np.max(corr_values)) if len(corr_values) > 0 else 0.0

    # Varianza explicada por el primer componente principal
    if len(numeric_cols) > 1:
      pca = PCA(n_components=1)
      pca.fit(num_data.fillna(0))
      stats["pca_var_ratio_pc1"] = float(pca.explained_variance_ratio_[0])
    else:
      stats["pca_var_ratio_pc1"] = 1.0
  else:
    stats = {
        "skewness_mean": None,
        "skewness_std": None,
        "kurtosis_mean": None,
        "kurtosis_std": None,
        "mean_abs_corr": None,
        "max_abs_corr": None,
        "pca_var_ratio_pc1": None
    }

  meta_feats['stats'] = stats

  # Meta-Features basadas en Teoría de la Información
  info = {}
  # Entropía de la clase
  class_counts = y.value_counts(normalize=True)
  class_entropy = -np.sum(class_counts * np.log2(class_counts))
  info["class_entropy"] = float(class_entropy)

  # Información mutua y ANOVA
  if len(numeric_cols) > 0:
    # ANOVA F-value promedio
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    f_vals, p_vals = f_classif(X_feat[numeric_cols].fillna(0), y_enc)
    info["mean_anova_f"] = float(np.mean(f_vals))
    info["mean_anova_p"] = float(np.mean(p_vals))
    # Relación señal-ruido simple (F medio / número de features)
    info["signal_to_noise"] = float(np.mean(f_vals) / len(numeric_cols)) if len(numeric_cols) > 0 else None
  else:
    info["mean_anova_f"] = None
    info["mean_anova_p"] = None
    info["signal_to_noise"] = None

  meta_feats["info"] = info

  return meta_feats

def format_meta_feats_text(meta: Dict[str, Any]) -> str:
  "Convierte un diccionario de meta-features (organizado por categorías) en una cadena de texto"
  lines = []
  lines.append("Meta-Features del Dataset")

  # Simple
  simple = meta.get("simple", {})
  if simple:
    lines.append("\n[Características Simples]")
    items = [
        ("Número de instancias", simple.get("n_instances")),
        ("Número de features", simple.get("n_features")),
        ("Número de clases", simple.get("n_classes")),
        ("Proporciones de clases", simple.get("class_proportions")),
        ("Valores faltantes totales", simple.get("n_missing_values")),
        ("Número de outliers (IQR)", simple.get("n_outliers")),
    ]
    for label, value in items:
      if value is not None:
        lines.append(f"  {label}: {value}")

  # Estadísticas
  stats = meta.get("stats", {})
  if stats:
    lines.append("\n[Meta-features Estadísticas]")
    stat_labels = {
        "skewness_mean": "Asimetría media (skewness)",
        "skewness_std": "Desviación estándar de asimetría",
        "kurtosis_mean": "Curtosis media",
        "kurtosis_std": "Desviación estándar de curtosis",
        "mean_abs_corr": "Correlación absoluta media entre features",
        "max_abs_corr": "Correlación absoluta máxima",
        "pca_var_ratio_pc1": "Varianza explicada por el primer componente principal (PC1)",
    }
    for key, label in stat_labels.items():
      val = stats.get(key)
      if val is not None:
        lines.append(f"  {label}: {val:.4f}" if isinstance(val, float) else f"  {label}: {val}")

  # Información
  info = meta.get("info", {})
  if info:
    lines.append("\n[Meta-features basadas en Teoría de la Información]")
    info_labels = {
        "class_entropy": "Entropía de la clase",
        "mean_anova_f": "F-valor ANOVA medio",
        "mean_anova_p": "P-valor ANOVA medio",
        "signal_to_noise": "Relación señal-ruido (F-medio / n_features)",
    }
    for key, label in info_labels.items():
      val = info.get(key)
      if val is not None:
        lines.append(f"  {label}: {val:.4f}" if isinstance(val, float) else f"  {label}: {val}")

  return "\n".join(lines)
