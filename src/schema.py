from __future__ import annotations

import re
import json
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import importlib

import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from src.file_handling import load_sklearn_map

# Constants
JSON_EXAMPLE = """
{
  "steps": [
    {
      "name": "scaler",
      "component": "StandardScaler",
      "hyperparameters": {"with_mean": true, "with_std": true}
    },
    {
      "name": "feature_sel",
      "component": "SelectKBest",
      "hyperparameters": {"k": 10}
    },
    {
      "name": "clf",
      "component": "RandomForestClassifier",
      "hyperparameters": {"n_estimators": 100, "max_depth": 5, "min_samples_split": 4}
    }
  ]
}
"""


def _load_components_from_json() -> Dict[str, Dict]:
  "Construye ALLOWED_COMPONENTS a partir del archivo JSON"
  data = load_sklearn_map()

  allowed = {}

  # Mapeo de tipos de step a stage interno
  stage_map = {
      "imputer": "preprocessor",
      "scaler": "preprocessor",
      "encoder": "preprocessor",
      "transformer": "preprocessor",
      "feature_selection": "feature_selection",
      "dimensionality_reduction": "feature_selection",
      "column_transformer": "preprocessor",
  }

  # Procesar preprocessing_steps
  for step_def in data.get("preprocessing_steps", []):
    step_type = step_def["step"]
    if step_type == "column_transformer":
      # Ya no se omite, se procesa normalmente
      pass
    name = step_def["name"]
    module_path = step_def["module"]
    try:
      module = importlib.import_module(".".join(module_path.split(".")[:-1]))
      cls = getattr(module, module_path.split(".")[-1])
    except (ImportError, AttributeError) as e:
      raise RuntimeError(f"No se pudo importar {module_path}: {e}")

    hyperparams = {}
    for hp_name, hp_def in step_def.get("hyperparameters", {}).items():
      hp_type = hp_def["type"]
      search_space = hp_def.get("search_space", [])
      nullable = None in search_space if search_space else False

      if hp_type == "categorical":
        # Detectar si todos los valores no nulos son booleanos
        non_null = [v for v in search_space if v is not None]
        if all(isinstance(v, bool) for v in non_null):
          param_type = bool
          allowed_vals = [v for v in search_space if v is not None]
        else:
          param_type = object
          allowed_vals = search_space
        hyperparams[hp_name] = {
            "type": param_type,
            "allowed": allowed_vals,
            "nullable": nullable,
        }
      elif hp_type == "integer":
        rng = search_space  # se espera [min, max]
        hyperparams[hp_name] = {
            "type": int,
            "range": {"min": rng[0], "max": rng[1]},
            "nullable": nullable,
        }
      elif hp_type == "continuous":
        rng = search_space
        hyperparams[hp_name] = {
            "type": float,
            "range": {"min": rng[0], "max": rng[1]},
            "nullable": nullable,
        }
      elif hp_type == "list":
        # No se valida estructura interna; se pasa como object
        hyperparams[hp_name] = {
            "type": object,
            "nullable": nullable,
        }
      else:
        # Desconocido: se pasa como object sin validación
        hyperparams[hp_name] = {"type": object, "nullable": nullable}

    allowed[name] = {
        "class": cls,
        "stage": stage_map.get(step_type, "preprocessor"),
        "hyperparameters": hyperparams,
    }

  # Procesar clasificadores
  for clf_def in data.get("sklearn_classification_algorithms", []):
    name = clf_def["name"]
    module_path = clf_def["module"]
    try:
      module = importlib.import_module(".".join(module_path.split(".")[:-1]))
      cls = getattr(module, module_path.split(".")[-1])
    except (ImportError, AttributeError) as e:
      raise RuntimeError(f"No se pudo importar {module_path}: {e}")

    hyperparams = {}
    for hp_name, hp_def in clf_def.get("hyperparameters", {}).items():
      hp_type = hp_def["type"]
      search_space = hp_def.get("search_space", [])
      nullable = None in search_space if search_space else False

      if hp_type == "categorical":
        non_null = [v for v in search_space if v is not None]
        if all(isinstance(v, bool) for v in non_null):
          param_type = bool
          allowed_vals = [v for v in search_space if v is not None]
        else:
          param_type = object
          allowed_vals = search_space
        hyperparams[hp_name] = {
            "type": param_type,
            "allowed": allowed_vals,
            "nullable": nullable,
        }
      elif hp_type == "integer":
        rng = search_space
        hyperparams[hp_name] = {
            "type": int,
            "range": {"min": rng[0], "max": rng[1]},
            "nullable": nullable,
        }
      elif hp_type == "continuous":
        rng = search_space
        hyperparams[hp_name] = {
            "type": float,
            "range": {"min": rng[0], "max": rng[1]},
            "nullable": nullable,
        }
      else:
        hyperparams[hp_name] = {"type": object, "nullable": nullable}

    allowed[name] = {
        "class": cls,
        "stage": "classifier",
        "hyperparameters": hyperparams,
    }

  # Parámetros fijos que no están en el JSON (score_func para SelectKBest/SelectPercentile)
  for name in ["SelectKBest", "SelectPercentile"]:
    if name in allowed:
      allowed[name]["fixed_params"] = {"score_func": f_classif}

  return allowed

# Cargar componentes (falla si no existe el archivo o hay errores de importación)
ALLOWED_COMPONENTS = _load_components_from_json()

@dataclass
class ParseResult:
  success: bool
  pipeline: Pipeline | None = None
  errors: list[str] = field(default_factory=list)
  warnings: list[str] = field(default_factory=list)

  def to_feedback(self, add_warning: bool = True) -> str:
    if self.success:
      return "[SUCCESS] Pipeline parseado correctamente."
    lines = ["[ERROR] Errores al parsear el pipeline:"]
    for e in self.errors:
      lines.append(f"- {e}")
    if self.warnings and add_warning:
      lines.append("[WARNING] Advertencias:")
      for w in self.warnings:
        lines.append(f"- {w}")
    lines.append("\nFormato esperado (JSON):\n" + JSON_EXAMPLE)
    return "\n".join(lines)

@dataclass
class EvaluationResult:
  success: bool
  metrics: dict[str, float] = field(default_factory=dict)
  errors: list[str] = field(default_factory=list)
  warnings: list[str] = field(default_factory=list)

  def to_feedback(self, add_warning: bool = True) -> str:
    if self.success:
      lines = ["[SUCCESS] Pipeline evaluado correctamente."]
      lines.append("[RESULTS] Métricas obtenidas:")
      for k, v in self.metrics.items():
        lines.append(f"- {k}: {v:.4f}")
      return "\n".join(lines)
    lines = ["[ERROR] Errores al evaluar el pipeline:"]
    for e in self.errors:
      lines.append(f"- {e}")
    if self.warnings and add_warning:
      lines.append("[WARNING] Advertencias:")
      for w in self.warnings:
        lines.append(f"- {w}")
    return "\n".join(lines)

def _extract_json_from_text(text: str) -> dict | None:
  md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
  if md_match:
    candidate = md_match.group(1)
    try:
      return json.loads(candidate)
    except json.JSONDecodeError:
      pass

  brace_start = text.find("{")
  if brace_start != -1:
    depth = 0
    for i, ch in enumerate(text[brace_start:], start=brace_start):
      if ch == "{":
        depth += 1
      elif ch == "}":
        depth -= 1
        if depth == 0:
          candidate = text[brace_start: i + 1]
          try:
            return json.loads(candidate)
          except json.JSONDecodeError:
            break
  return None

def _validate_column_transformer_transformers(component_name: str, param_value: Any, param_path: str) -> list[str]:
  "Valida la lista 'transformers' de ColumnTransformer."
  errors = []
  if not isinstance(param_value, list):
    errors.append(f"'{param_path}': debe ser una lista de transformadores.")
    return errors

  for idx, tf_dict in enumerate(param_value):
    prefix = f"{param_path}[{idx}]"
    if not isinstance(tf_dict, dict):
      errors.append(f"{prefix}: cada transformador debe ser un objeto (dict).")
      continue

    # Verificar claves obligatorias
    required = {"name", "transformer", "columns"}
    missing = required - set(tf_dict.keys())
    if missing:
      errors.append(f"{prefix}: faltan claves obligatorias: {missing}.")

    # Validar 'name'
    name = tf_dict.get("name")
    if name is not None and not isinstance(name, str):
      errors.append(f"{prefix}.name: debe ser string.")

    # Validar 'transformer'
    transformer_name = tf_dict.get("transformer")
    if transformer_name is None:
      errors.append(f"{prefix}.transformer: es obligatorio.")
    elif transformer_name not in ALLOWED_COMPONENTS:
      errors.append(f"{prefix}.transformer: '{transformer_name}' no es un componente permitido.\nPermitidos: {list(ALLOWED_COMPONENTS.keys())}.")
    else:
      # Validar hiperparámetros del sub‑transformer si existen
      sub_hp = tf_dict.get("transformer_hyperparameters", {})
      sub_schema = ALLOWED_COMPONENTS[transformer_name]
      allowed_hp = set(sub_schema["hyperparameters"].keys())
      given_hp = set(sub_hp.keys())
      unknown = given_hp - allowed_hp
      if unknown:
        errors.append(f"{prefix}.transformer '{transformer_name}':\nHiperparámetros desconocidos: {unknown}.\nVálidos: {allowed_hp}.")
      for hp_name, hp_value in sub_hp.items():
        if hp_name not in sub_schema["hyperparameters"]:
          continue
        param_errors = _validate_hyperparameter(transformer_name, hp_name, hp_value, sub_schema["hyperparameters"][hp_name])
        errors.extend(param_errors)

    # Validar 'columns' (debe ser lista de enteros o strings)
    cols = tf_dict.get("columns")
    if cols is None:
      errors.append(f"{prefix}.columns: es obligatorio.")
    elif not isinstance(cols, list):
      errors.append(f"{prefix}.columns: debe ser una lista de índices o nombres.")
    else:
      for col in cols:
        if not isinstance(col, (int, str)):
          errors.append(f"{prefix}.columns: cada elemento debe ser int o str, encontrado {type(col)}.")

  return errors

def _validate_hyperparameter(component_name: str, param_name: str, param_value: Any, param_schema: dict) -> list[str]:
  errors = []

  # Validación especial para ColumnTransformer.transformers
  if component_name == "ColumnTransformer" and param_name == "transformers":
    return _validate_column_transformer_transformers(component_name, param_value, f"{component_name}.{param_name}")

  expected_type = param_schema["type"]
  nullable = param_schema.get("nullable", False)

  if param_value is None:
    if not nullable:
      errors.append(f"'{component_name}.{param_name}': no acepta None.")
    return errors

  # Conversión de tipos tolerante
  coerced = param_value
  if expected_type == float and isinstance(param_value, int):
    coerced = float(param_value)
  elif expected_type == int and isinstance(param_value, float) and param_value.is_integer():
    coerced = int(param_value)
  elif expected_type == bool and isinstance(param_value, str):
    coerced = param_value.lower() == "true" if param_value.lower() in ["true", "false"] else param_value
  elif expected_type == tuple and isinstance(param_value, list):
    coerced = tuple(param_value)

  if not isinstance(coerced, expected_type):
    errors.append(f"'{component_name}.{param_name}': tipo incorrecto.\nSe esperaba {expected_type.__name__}, se recibió {type(param_value).__name__}.")
    return errors

  if "allowed" in param_schema:
    if coerced not in param_schema["allowed"]:
      errors.append(f"'{component_name}.{param_name}': valor '{coerced}' no está permitido.\nValores válidos: {param_schema['allowed']}.")

  if "range" in param_schema:
    rng = param_schema["range"]
    if not (rng["min"] <= coerced <= rng["max"]):
      errors.append(f"'{component_name}.{param_name}': valor {coerced} fuera de rango.\n[{rng['min']}, {rng['max']}].")

  return errors

def _build_sklearn_step(step_def: dict) -> tuple[str, Any]:
  component_name = step_def["component"]
  schema = ALLOWED_COMPONENTS[component_name]
  raw_params: dict = step_def.get("hyperparameters", {})

  # Tratamiento especial para ColumnTransformer
  if component_name == "ColumnTransformer":
    transformers_list = raw_params.get("transformers", [])
    built_transformers = []
    for tf_dict in transformers_list:
      name = tf_dict["name"]
      sub_comp = tf_dict["transformer"]
      if sub_comp in ("drop", "passthrough"):
        transformer_obj = sub_comp
      else:
        sub_hp = tf_dict.get("transformer_hyperparameters", {})
        sub_step_def = {
            "name": name,
            "component": sub_comp,
            "hyperparameters": sub_hp,
        }
        _, transformer_obj = _build_sklearn_step(sub_step_def)
      columns = tf_dict["columns"]
      built_transformers.append((name, transformer_obj, columns))
    raw_params["transformers"] = built_transformers

  # Coerción de tipos normal
  coerced_params = {}
  for k, v in raw_params.items():
    if k == "transformers" and component_name == "ColumnTransformer":
      coerced_params[k] = v
      continue
    # Si el parámetro no está en el esquema, lo pasamos tal cual (podría ser un hiperparámetro no declarado, pero ya validamos antes)
    if k not in schema["hyperparameters"]:
      coerced_params[k] = v
      continue
    expected_type = schema["hyperparameters"][k]["type"]
    if expected_type == float and isinstance(v, int):
      coerced_params[k] = float(v)
    elif expected_type == int and isinstance(v, float) and v == int(v):
      coerced_params[k] = int(v)
    elif expected_type == bool and isinstance(v, str):
      coerced_params[k] = v.lower() == "true"
    elif expected_type == tuple and isinstance(v, list):
      coerced_params[k] = tuple(v)
    else:
      coerced_params[k] = v

  # Parche específico para MinMaxScaler
  if component_name == "MinMaxScaler" and "feature_range" in coerced_params:
    fr = coerced_params["feature_range"]
    if isinstance(fr, list) and len(fr) == 2:
      coerced_params["feature_range"] = tuple(fr)

  fixed = schema.get("fixed_params", {})
  coerced_params.update(fixed)

  estimator = schema["class"](**coerced_params)
  return (step_def["name"], estimator)

def parse_llm_response_to_pipeline(llm_response: str) -> ParseResult:
  errors = []
  warnings = []

  raw_json = _extract_json_from_text(llm_response)
  if raw_json is None:
    errors.append("No se encontró ningún bloque JSON válido en la respuesta. Asegúrate de incluir el pipeline en formato JSON.")
    return ParseResult(success=False, errors=errors, warnings=warnings)

  if "steps" not in raw_json or not isinstance(raw_json["steps"], list):
    errors.append("El JSON debe contener una clave 'steps' que sea una lista de componentes.")
    return ParseResult(success=False, errors=errors, warnings=warnings)

  steps_raw = raw_json["steps"]
  if len(steps_raw) == 0:
    errors.append("La lista 'steps' está vacía; el pipeline debe tener al menos un componente.")
    return ParseResult(success=False, errors=errors, warnings=warnings)

  steps_meta = []
  for idx, step in enumerate(steps_raw):
    prefix = f"Step {idx + 1}"

    for field_name in ("name", "component"):
      if field_name not in step:
        errors.append(f"{prefix}: falta el campo '{field_name}'.")

    if errors:
      continue

    component_name = step["component"]
    if component_name not in ALLOWED_COMPONENTS:
      errors.append(f"{prefix}: componente '{component_name}' no está permitido.\nComponentes válidos: {list(ALLOWED_COMPONENTS.keys())}.")
      continue

    schema = ALLOWED_COMPONENTS[component_name]
    hyperparams = step.get("hyperparameters", {})

    allowed_hp = set(schema["hyperparameters"].keys())
    given_hp = set(hyperparams.keys())
    unknown = given_hp - allowed_hp
    if unknown:
      errors.append(f"{prefix} '{component_name}': hiperparámetros no reconocidos: {unknown}.\nHiperparámetros válidos: {allowed_hp}.")

    for param_name, param_value in hyperparams.items():
      if param_name not in schema["hyperparameters"]:
        continue
      param_errors = _validate_hyperparameter(component_name, param_name, param_value, schema["hyperparameters"][param_name])
      errors.extend(param_errors)

    missing_hp = allowed_hp - given_hp
    if missing_hp:
      warnings.append(f"'{component_name}': hiperparámetros no especificados (se usarán defaults): {missing_hp}.")

    steps_meta.append({"name": step["name"], "component": component_name, "stage": schema["stage"]})

  if errors:
    return ParseResult(success=False, errors=errors, warnings=warnings)

  sklearn_steps = []
  for step in steps_raw:
    try:
      sklearn_steps.append(_build_sklearn_step(step))
    except Exception as exc:
      errors.append(f"Error al instanciar '{step['component']}': {exc}")

  if errors:
    return ParseResult(success=False, errors=errors, warnings=warnings)

  pipeline = Pipeline(sklearn_steps)
  return ParseResult(success=True, pipeline=pipeline, warnings=warnings)

def evaluate_pipeline(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, *, cv: int = 5, scoring: list[str] | None = None) -> EvaluationResult:
  errors = []
  warnings = []
  metrics = {}

  if scoring is None:
    scoring = ["accuracy", "f1_weighted"]

  VALID_SCORING = {"accuracy", "f1_weighted", "roc_auc"}

  if not isinstance(pipeline, Pipeline):
    errors.append("El argumento 'pipeline' no es un sklearn Pipeline.")
    return EvaluationResult(success=False, errors=errors)

  if X is None or y is None:
    errors.append("X e y no pueden ser None.")
    return EvaluationResult(success=False, errors=errors)

  try:
    X = np.asarray(X)
    y = np.asarray(y)
  except Exception as exc:
    errors.append(f"No se pudo convertir X o y a numpy array: {exc}")
    return EvaluationResult(success=False, errors=errors)

  if X.ndim != 2:
    errors.append(f"X debe ser 2D (encontrado: {X.ndim}D).")
  if y.ndim != 1:
    errors.append(f"y debe ser 1D (encontrado: {y.ndim}D).")
  if X.shape[0] != y.shape[0]:
    errors.append(f"X e y tienen distinto número de muestras: {X.shape[0]} vs {y.shape[0]}.")
  n_samples = X.shape[0]
  if n_samples < cv * 2:
    errors.append(
        f"Muy pocas muestras ({n_samples}) para {cv} folds. "
        f"Necesitas al menos {cv * 2} muestras."
    )
  unknown_scoring = set(scoring) - VALID_SCORING
  if unknown_scoring:
    errors.append(
        f"Métricas no reconocidas: {unknown_scoring}. "
        f"Métricas válidas: {VALID_SCORING}."
    )

  if errors:
    return EvaluationResult(success=False, errors=errors)

  # Ajustes automáticos para SelectKBest y PCA
  from sklearn.feature_selection import SelectKBest, SelectPercentile
  from sklearn.decomposition import PCA

  for step_name, estimator in pipeline.steps:
    if isinstance(estimator, (SelectKBest, SelectPercentile)):
      if hasattr(estimator, "k"):
        k = estimator.k
        if isinstance(k, int) and k > X.shape[1]:
          warnings.append(
              f"{estimator.__class__.__name__}(k={k}) pero X solo tiene {X.shape[1]} features. "
              f"Se ajustará k automáticamente."
          )
          estimator.k = X.shape[1]
    if isinstance(estimator, PCA):
      n = estimator.n_components
      if isinstance(n, int) and n > min(X.shape):
        warnings.append(
            f"PCA(n_components={n}) pero min(n_samples, n_features)={min(X.shape)}. "
            f"Se ajustará n_components automáticamente."
        )
        estimator.n_components = min(X.shape) - 1

  n_classes = len(np.unique(y))
  for metric in scoring:
    sklearn_metric = metric
    if metric == "roc_auc" and n_classes > 2:
      sklearn_metric = "roc_auc_ovr"
      warnings.append("roc_auc con más de 2 clases: se usa 'roc_auc_ovr' (One-vs-Rest).")

    try:
      scores = cross_val_score(pipeline, X, y, cv=cv, scoring=sklearn_metric)
      metrics[metric + "_mean"] = float(np.mean(scores))
      metrics[metric + "_std"] = float(np.std(scores))
    except Exception as exc:
      tb = traceback.format_exc()
      errors.append(f"Error al calcular '{metric}' con cross_val_score: {exc}\n{tb}")

  if errors:
    return EvaluationResult(success=False, metrics=metrics, errors=errors, warnings=warnings)

  try:
    pipeline.fit(X, y)
  except Exception as exc:
    tb = traceback.format_exc()
    errors.append(f"Error al hacer fit del pipeline en el dataset completo: {exc}\n{tb}")
    return EvaluationResult(success=False, metrics=metrics, errors=errors, warnings=warnings)

  try:
    y_pred = pipeline.predict(X)
    metrics["train_accuracy"] = float(accuracy_score(y, y_pred))
  except Exception as exc:
    warnings.append(f"No se pudieron calcular métricas de train: {exc}")

  return EvaluationResult(success=True, metrics=metrics, errors=[], warnings=warnings)

class MLPipelineGenerator:
  def __init__(self) -> None:
    pass

  def generate_prompt(self) -> str:
    components_doc = json.dumps(
        {
            name: {
                "stage": info["stage"],
                "hyperparameters": {
                    hp: {k: v for k, v in schema.items() if k != "type"}
                    for hp, schema in info["hyperparameters"].items()
                },
            }
            for name, info in ALLOWED_COMPONENTS.items()
        },
        indent=2,
        default=str,
    )
    return f"""Eres un experto en Machine Learning. Tu tarea es diseñar un Pipeline de sklearn para tareas de clasificación.

REGLAS ESTRICTAS:
1. Solo puedes usar los componentes listados abajo.
2. Los hiperparámetros deben estar dentro de los rangos o valores permitidos.
3. El pipeline debe tener exactamente 1 clasificador (al final).
4. El orden debe ser: [preprocessor*] → [feature_selection?] → classifier
5. Responde ÚNICAMENTE con un bloque JSON válido con la clave "steps".

COMPONENTES DISPONIBLES Y SUS RESTRICCIONES:
{components_doc}

EJEMPLO DE RESPUESTA:
{JSON_EXAMPLE}
"""

  @staticmethod
  def parse_response(llm_response: str) -> ParseResult:
    return parse_llm_response_to_pipeline(llm_response)

  @staticmethod
  def evaluate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: list[str] | None = None) -> EvaluationResult:
    return evaluate_pipeline(pipeline, X, y, cv=cv, scoring=scoring)

  def run(self, llm_response: str, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Tuple[ParseResult, EvaluationResult | None]:
    parse_result = parse_llm_response_to_pipeline(llm_response)
    if not parse_result.success:
      return parse_result, None
    eval_result = evaluate_pipeline(parse_result.pipeline, X, y, cv=cv)
    return parse_result, eval_result

def test_pipeline_generation(n_samples: int = 300, n_features: int = 20) -> None:
  from sklearn.datasets import make_classification

  X, y = make_classification(n_samples=n_samples, n_features=n_features)
  generator = MLPipelineGenerator()

  # Simulación de respuesta del LLM
  response = """
  {
    "steps": [
      {
        "name": "scaler",
        "component": "StandardScaler",
        "hyperparameters": {"with_mean": true, "with_std": true}
      },
      {
        "name": "feature_sel",
        "component": "SelectKBest",
        "hyperparameters": {"k": 10}
      },
      {
        "name": "clf",
        "component": "RandomForestClassifier",
        "hyperparameters": {"n_estimators": 100, "max_depth": 5, "min_samples_split": 4}
      }
    ]
  }
  """
  print(f"[RESPONSE] {response}")
  parse_response, eval_response = generator.run(response, X, y)
  print(parse_response.to_feedback(add_warning=False))
  if eval_response:
    print(eval_response.to_feedback(add_warning=False))

  print("=" * 60)
  response = """
  {
    "steps": [
      {
        "name": "scaler",
        "component": "StandardScaler",
        "hyperparameters": {"with_mean": true, "with_std": true}
      },
      {
        "name": "clf",
        "component": "LogisticRegression",
        "hyperparameters": {"C": 2000, "max_iter": 100}
      },
      {
        "name": "feature_sel",
        "component": "SelectKBest",
        "hyperparameters": {"k": 10}
      }
    ]
  }
  """
  print(f"[RESPONSE] {response}")
  parse_response, eval_response = generator.run(response, X, y)
  print(parse_response.to_feedback(add_warning=False))
  if eval_response:
    print(eval_response.to_feedback(add_warning=False))

  print("=" * 60)
  response = """
  {
    "steps": [
      {
        "name": "col_trans",
        "component": "ColumnTransformer",
        "hyperparameters": {
          "transformers": [
            {
              "name": "num_scaler",
              "transformer": "StandardScaler",
              "columns": [0, 1, 2],
              "transformer_hyperparameters": {"with_mean": true}
            }
          ],
        "remainder": "drop"
        }
      },
      {
        "name": "clf",
        "component": "RandomForestClassifier",
        "hyperparameters": {"n_estimators": 10}
      }
    ]
  }
  """
  print(f"[RESPONSE] {response}")
  parse_response, eval_response = generator.run(response, X, y)
  print(parse_response.to_feedback(add_warning=False))
  if eval_response:
    print(eval_response.to_feedback(add_warning=False))

  print("=" * 60)
  response = """
  {
    "steps": [
      {
        "name": "column_transformer",
        "component": "ColumnTransformer",
        "hyperparameters": {
          "transformers": [
            {
              "name": "num_std",
              "transformer": "StandardScaler",
              "columns": [0, 2, 4],
              "transformer_hyperparameters": {"with_mean": true, "with_std": true}
            },
            {
              "name": "num_minmax",
              "transformer": "MinMaxScaler",
              "columns": [1, 3, 5],
              "transformer_hyperparameters": {"feature_range": [0, 1]}
            },
            {
              "name": "cat_onehot",
              "transformer": "OneHotEncoder",
              "columns": [6, 7],
              "transformer_hyperparameters": {"handle_unknown": "ignore", "sparse_output": false}
            }
          ],
          "remainder": "passthrough"
        }
      },
      {
        "name": "feature_selector",
        "component": "SelectKBest",
        "hyperparameters": {"k": 8}
      },
      {
        "name": "classifier",
        "component": "SVC",
        "hyperparameters": {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
      }
    ]
  }
  """
  print(f"[RESPONSE] {response}")
  parse_response, eval_response = generator.run(response, X, y)
  print(parse_response.to_feedback(add_warning=False))
  if eval_response:
    print(eval_response.to_feedback(add_warning=False))
