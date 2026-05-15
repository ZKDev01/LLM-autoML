import re
import json
import warnings
import traceback
import importlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import f_classif

from pydantic import BaseModel, Field, model_validator, ValidationError

# reducir ruido en las pruebas
warnings.filterwarnings("ignore")

# Carga de sklearn_map.json y construcción de espacio de búsqueda de componentes permitidos
PATH = "./src/sklearn_map.json"

def load_sklearn_map(path: str = "sklearn_map.json") -> Dict:
  "Carga el archivo JSON que define los componentes e hiperparámetros permitidos."
  with open(PATH, "r", encoding="utf-8") as f:
    return json.load(f)
  raise Exception(f"Ha ocurrido un problema al cargar el archivo JSON: {path}")

def build_allowed_components(sklearn_map: Dict) -> Dict[str, Dict]:
  "Construye el diccionario ALLOWED_COMPONENTS a partir del mapa cargado."
  ALLOWED = {}

  stage_map = {
      "imputer": "preprocessor",
      "scaler": "preprocessor",
      "encoder": "preprocessor",
      "transformer": "preprocessor",
      "feature_selection": "feature_selection",
      "dimensionality_reduction": "feature_selection",
      "column_transformer": "preprocessor",
  }

  # Preprocesamientos
  for step_def in sklearn_map.get("preprocessing_steps", []):
    name = step_def["name"]
    module_path = step_def["module"]
    # Importar la clase
    mod_name, cls_name = module_path.rsplit(".", 1)
    try:
      mod = importlib.import_module(mod_name)
      cls = getattr(mod, cls_name)
    except (ImportError, AttributeError) as e:
      raise RuntimeError(f"No se pudo importar {module_path}: {e}")

    hyperparams = {}
    for hp_name, hp_def in step_def.get("hyperparameters", {}).items():
      hp_type = hp_def["type"]
      search_space = hp_def.get("search_space", [])
      nullable = None in search_space if search_space else False

      if hp_type == "categorical":
        # Si todos los no nulos son bool, consideramos booleano
        non_null = [v for v in search_space if v is not None]
        if all(isinstance(v, bool) for v in non_null):
          param_type = bool
          allowed_vals = [v for v in search_space if v is not None]
        else:
          param_type = object  # acepta cualquier valor dentro de search_space
          allowed_vals = search_space
        hyperparams[hp_name] = {
            "type": param_type,
            "allowed": allowed_vals,
            "nullable": nullable,
        }
      elif hp_type in ("integer", "continuous"):
        py_type = int if hp_type == "integer" else float
        rng = search_space  # [min, max]
        hyperparams[hp_name] = {
            "type": py_type,
            "range": {"min": rng[0], "max": rng[1]},
            "nullable": nullable,
        }
      elif hp_type == "list":
        # No validamos estructura interna; aceptamos cualquier objeto
        hyperparams[hp_name] = {"type": object, "nullable": nullable}
      else:
        hyperparams[hp_name] = {"type": object, "nullable": nullable}

    stage = stage_map.get(step_def.get("step", ""), "preprocessor")
    ALLOWED[name] = {
        "class": cls,
        "stage": stage,
        "hyperparameters": hyperparams,
    }

  # Clasificadores
  for clf_def in sklearn_map.get("sklearn_classification_algorithms", []):
    name = clf_def["name"]
    module_path = clf_def["module"]
    mod_name, cls_name = module_path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)

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
      elif hp_type in ("integer", "continuous"):
        py_type = int if hp_type == "integer" else float
        rng = search_space
        hyperparams[hp_name] = {
            "type": py_type,
            "range": {"min": rng[0], "max": rng[1]},
            "nullable": nullable,
        }
      else:
        hyperparams[hp_name] = {"type": object, "nullable": nullable}

    ALLOWED[name] = {
        "class": cls,
        "stage": "classifier",
        "hyperparameters": hyperparams,
    }

  # Parches fijos: score_func para SelectKBest/SelectPercentile
  for name in ["SelectKBest", "SelectPercentile"]:
    if name in ALLOWED:
      ALLOWED[name]["fixed_params"] = {"score_func": f_classif}

  return ALLOWED


# Carga única del mapa
sklearn_map = load_sklearn_map()
ALLOWED_COMPONENTS = build_allowed_components(sklearn_map)

# Definición de modelos Pydantic para la especificación del Pipeline
# Auxiliar Functions
def _validate_column_transformer_transformers(transformers_value: List) -> List[str]:
  "Valida la estructura de la lista 'transformers' de ColumnTransformer. Retornando una lista de mensajes de error (vacía si todo correcto)"
  errors = []
  if not isinstance(transformers_value, list):
    return ["'transformers' debe ser una lista."]

  # Claves permitidas en cada objeto transformador
  ALLOWED_KEYS = {"name", "transformer", "columns", "transformer_hyperparameters"}

  for idx, item in enumerate(transformers_value):
    prefix = f"[{idx}]"

    # El elemento debe ser un diccionario, no una lista
    if not isinstance(item, dict):
      errors.append(f"{prefix}: se esperaba un objeto (dict), pero se recibió {type(item).__name__}.")
      continue

    # Detectar claves no permitidas y sugerir el nombre correcto si ponen 'hyperparameters'
    extra_keys = set(item.keys()) - ALLOWED_KEYS
    if extra_keys:
      if "hyperparameters" in extra_keys:
        errors.append(f"{prefix}: clave no permitida 'hyperparameters'. Use 'transformer_hyperparameters' para los parámetros del sub-transformador.")
        extra_keys.discard("hyperparameters")
      if extra_keys:
        errors.append(f"{prefix}: claves no reconocidas: {extra_keys}. Claves permitidas: {ALLOWED_KEYS}.")
      # Si hay claves extra, continuamos para poder reportar todos los errores,
      # pero no seguimos validando el contenido de las claves obligatorias que puedan faltar.

    # Claves obligatorias
    missing = []
    for key in ("name", "transformer", "columns"):
      if key not in item:
        missing.append(key)
    if missing:
      errors.append(f"{prefix}: faltan las claves obligatorias: {missing}.")
      continue

    # Validar 'name'
    name = item.get("name")
    if not isinstance(name, str):
      errors.append(f"{prefix}.name: debe ser una cadena de texto (str).")

    # Validar 'transformer'
    transformer = item.get("transformer")
    if transformer is None:
      errors.append(f"{prefix}.transformer: es obligatorio y no puede ser None.")
    elif not isinstance(transformer, str):
      errors.append(f"{prefix}.transformer: debe ser un nombre de componente (str), no {type(transformer).__name__}.")
    elif transformer not in ALLOWED_COMPONENTS:
      errors.append(f"{prefix}.transformer '{transformer}': no es un componente permitido. Componentes válidos: {list(ALLOWED_COMPONENTS.keys())}.")
    else:
      # Validar hiperparámetros del sub-transformador, si existen
      sub_hp = item.get("transformer_hyperparameters", {})
      if not isinstance(sub_hp, dict):
        errors.append(f"{prefix}.transformer_hyperparameters: debe ser un objeto (dict).")
      else:
        sub_schema = ALLOWED_COMPONENTS[transformer]["hyperparameters"]
        allowed_hp = set(sub_schema.keys())
        given_hp = set(sub_hp.keys())
        unknown = given_hp - allowed_hp
        if unknown:
          errors.append(f"{prefix} '{transformer}': hiperparámetros desconocidos {unknown}. Válidos: {allowed_hp if allowed_hp else '(ninguno)'}.")
        # Validación individual de cada hiperparámetro (tipo, rango, etc.)
        for hp_name, hp_value in sub_hp.items():
          if hp_name not in sub_schema:
            continue
          hp_schema = sub_schema[hp_name]
          expected_type = hp_schema["type"]
          nullable = hp_schema.get("nullable", False)
          if hp_value is None:
            if not nullable:
              errors.append(f"{prefix}.{transformer}.{hp_name}: no acepta None.")
            continue
          # Coerción simple
          coerced = hp_value
          if expected_type == float and isinstance(hp_value, int):
            coerced = float(hp_value)
          elif expected_type == int and isinstance(hp_value, float) and hp_value.is_integer():
            coerced = int(hp_value)
          elif expected_type == bool and isinstance(hp_value, str):
            if hp_value.lower() in ("true", "false"):
              coerced = hp_value.lower() == "true"
            else:
              errors.append(f"{prefix}.{transformer}.{hp_name}: se esperaba bool, pero la cadena '{hp_value}' no es válida.")
              continue
          if not isinstance(coerced, expected_type):
            errors.append(f"{prefix}.{transformer}.{hp_name}: tipo incorrecto. Se esperaba {expected_type.__name__}, se recibió {type(hp_value).__name__}.")
            continue
          if "allowed" in hp_schema and coerced not in hp_schema["allowed"]:
            errors.append(f"{prefix}.{transformer}.{hp_name}: valor '{coerced}' no permitido. Valores válidos: {hp_schema['allowed']}.")
          if "range" in hp_schema:
            rng = hp_schema["range"]
            if not (rng["min"] <= coerced <= rng["max"]):
              errors.append(f"{prefix}.{transformer}.{hp_name}: valor {coerced} fuera de rango. Rango permitido: [{rng['min']}, {rng['max']}].")

    # Validar 'columns'
    columns = item.get("columns")
    if not isinstance(columns, list):
      errors.append(f"{prefix}.columns: debe ser una lista de índices o nombres, no {type(columns).__name__}.")
    else:
      for col_idx, col in enumerate(columns):
        if not isinstance(col, (int, str)):
          errors.append(f"{prefix}.columns[{col_idx}]: debe ser int o str, no {type(col).__name__}.")

  return errors

class StepSpec(BaseModel):
  name: str
  component: str
  hyperparameters: Dict[str, Any] = Field(default_factory=dict)

  @model_validator(mode="after")
  def validate_component_and_hyperparams(self):
    comp = self.component
    if comp not in ALLOWED_COMPONENTS:
      allowed = list(ALLOWED_COMPONENTS.keys())
      raise ValueError(f"Componente '{comp}' no permitido. Permitidos: {allowed}")

    schema = ALLOWED_COMPONENTS[comp]
    allowed_hp = set(schema["hyperparameters"].keys())
    given_hp = set(self.hyperparameters.keys())
    extra = given_hp - allowed_hp
    if extra:
      raise ValueError(f"Hiperparámetros desconocidos para '{comp}': {extra}. Válidos: {allowed_hp if allowed_hp else '(ninguno)'}")

    if self.component == "ColumnTransformer" and "transformers" in self.hyperparameters:
      transformers = self.hyperparameters["transformers"]
      ct_errors = _validate_column_transformer_transformers(transformers)
      if ct_errors:
        raise ValueError(f"La lista 'transformers' de ColumnTransformer no es válida: {'; '.join(ct_errors)}")

    # Validación individual
    for hp_name, hp_val in self.hyperparameters.items():
      if hp_name not in schema["hyperparameters"]:
        continue  # ya se reportó como extra antes
      hp_schema = schema["hyperparameters"][hp_name]
      nullable = hp_schema.get("nullable", False)

      if hp_val is None:
        if not nullable:
          raise ValueError(f"'{comp}.{hp_name}' no puede ser None.")
        continue

      expected_type = hp_schema["type"]
      # Coerción sencilla para ayudar
      coerced = hp_val
      if expected_type == float and isinstance(hp_val, int):
        coerced = float(hp_val)
      elif expected_type == int and isinstance(hp_val, float) and hp_val.is_integer():
        coerced = int(hp_val)
      elif expected_type == bool and isinstance(hp_val, str):
        if hp_val.lower() in ("true", "false"):
          coerced = hp_val.lower() == "true"
        else:
          raise ValueError(f"'{comp}.{hp_name}': se esperaba bool, pero la cadena '{hp_val}' no es 'true' ni 'false'.")
      elif expected_type == tuple and isinstance(hp_val, list):
        coerced = tuple(hp_val)
      elif expected_type == object:
        # no coercion, aceptamos cualquier cosa (ej. listas)
        pass
      else:
        if not isinstance(coerced, expected_type):
          raise ValueError(f"'{comp}.{hp_name}': tipo inválido. Se esperaba {expected_type.__name__}, se encontró {type(hp_val).__name__}.")

      # Validación de rango/allowed
      if "allowed" in hp_schema:
        if coerced not in hp_schema["allowed"]:
          raise ValueError(f"'{comp}.{hp_name}': valor '{coerced}' no permitido. Valores válidos: {hp_schema['allowed']}")
      if "range" in hp_schema:
        rng = hp_schema["range"]
        if not (rng["min"] <= coerced <= rng["max"]):
          raise ValueError(f"'{comp}.{hp_name}': valor {coerced} fuera de rango. Rango permitido: [{rng['min']}, {rng['max']}].")
    return self

class PipelineSpec(BaseModel):
  steps: List[StepSpec]

  @model_validator(mode="after")
  def check_at_least_one_step(self):
    if len(self.steps) == 0:
      raise ValueError("El pipeline debe tener al menos un paso.")
    return self

# Extracción de candidatos JSON desde una respuesta de texto
def extract_json_candidates(text: str) -> List[dict]:
  "Extrae todos los objetos JSON (diccionarios) completos del texto. Soporta bloques markdown ```json ... ``` y JSON suelto, incluso si hay fragmentos malformados intercalados"
  candidates = []
  # 1. Extraer de bloques markdown (```json ... ```)
  md_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
  for match in md_pattern.finditer(text):
    try:
      obj = json.loads(match.group(1))
      candidates.append(obj)
    except json.JSONDecodeError:
      pass

  # 2. Extraer objetos JSON fuera de bloques markdown, usando raw_decode
  #    que maneja correctamente fragmentos no válidos sin afectar a los posteriores.
  #    Primero eliminamos del texto las partes que ya capturamos con markdown
  #    para no duplicar.
  cleaned_text = text
  for match in md_pattern.finditer(text):
    cleaned_text = cleaned_text.replace(match.group(0), " ")  # reemplazar por espacios

  decoder = json.JSONDecoder()
  idx = 0
  while idx < len(cleaned_text):
    # Buscar el siguiente '{'
    brace_idx = cleaned_text.find('{', idx)
    if brace_idx == -1:
      break
    try:
      # Intentar decodificar un objeto JSON completo desde esa posición
      obj, end_idx = decoder.raw_decode(cleaned_text, brace_idx)
      candidates.append(obj)
      idx = end_idx  # avanzar justo después del objeto extraído
    except (json.JSONDecodeError, ValueError) as e:
      # No se pudo decodificar un JSON válido desde aquí.
      # Saltamos el carácter problemático y seguimos buscando.
      idx = brace_idx + 1

  return candidates

# Funciones de Construcción y Evaluación del Pipeline de Scikit-Learn
def _build_sklearn_step(step_spec: StepSpec) -> Tuple[str, Any]:
  "Construye una tupla (nombre, estimador) para un paso individual."
  comp_name = step_spec.component
  schema = ALLOWED_COMPONENTS[comp_name]
  raw_params = step_spec.hyperparameters.copy()

  # construcción para el caso especial de ColumnTransformer
  if comp_name == "ColumnTransformer":
    transformers_list = raw_params.get("transformers", [])
    built_transformers = []
    for tf_dict in transformers_list:
      name = tf_dict["name"]
      sub_comp = tf_dict["transformer"]
      if sub_comp in ("drop", "passthrough"):
        transformer_obj = sub_comp
      else:
        sub_hp = tf_dict.get("transformer_hyperparameters", {})
        sub_step_spec = StepSpec(name=name, component=sub_comp, hyperparameters=sub_hp)
        _, transformer_obj = _build_sklearn_step(sub_step_spec)
      columns = tf_dict["columns"]
      built_transformers.append((name, transformer_obj, columns))
    raw_params["transformers"] = built_transformers

  # Coerción de tipos
  coerced = {}
  for k, v in raw_params.items():
    if k == "transformers" and comp_name == "ColumnTransformer":
      coerced[k] = v
      continue
    # Si el hiperparámetro no está en el esquema, lo pasamos tal cual (no debería ocurrir si se validó)
    if k not in schema["hyperparameters"]:
      coerced[k] = v
      continue
    expected = schema["hyperparameters"][k]["type"]
    # Conversiones análogas a las usadas en validación
    if expected == float and isinstance(v, int):
      coerced[k] = float(v)
    elif expected == int and isinstance(v, float) and v == int(v):
      coerced[k] = int(v)
    elif expected == bool and isinstance(v, str):
      coerced[k] = v.lower() == "true"
    elif expected == tuple and isinstance(v, list):
      coerced[k] = tuple(v)
    else:
      coerced[k] = v

  # Parche específico para MinMaxScaler.feature_range
  if comp_name == "MinMaxScaler" and "feature_range" in coerced:
    fr = coerced["feature_range"]
    if isinstance(fr, list) and len(fr) == 2:
      coerced["feature_range"] = tuple(fr)

  # Parámetros fijos como score_func
  fixed = schema.get("fixed_params", {})
  coerced.update(fixed)

  estimator = schema["class"](**coerced)
  return step_spec.name, estimator

def build_pipeline(spec: PipelineSpec) -> Pipeline:
  "Construye un sklearn Pipeline a partir de una especificación válidada. Lanza excepciones propias de scikit-learn o de Python si los parámetros no son aceptados por el constructor"
  steps = []
  for step_spec in spec.steps:
    name, est = _build_sklearn_step(step_spec)
    steps.append((name, est))
  return Pipeline(steps)

def _extract_core_error(exc: Exception) -> str:
  "Reduce un traceback complejo a solo la línea más informativa."
  msg = str(exc)
  # Caso típico: "All the X fits failed." contiene líneas con la causa real
  if "All the" in msg and "fits failed" in msg:
    lines = msg.splitlines()
    for line in lines:
      for err_type in ("ValueError:", "TypeError:", "KeyError:", "IndexError:"):
        if err_type in line:
          return line.strip()
    # Si no encontramos, devolvemos la última línea que suele contener el motivo
    return lines[-1].strip() if lines else msg[:500]
  # Otras excepciones: limitamos longitud
  if len(msg) > 500:
    return msg[:250] + "..." + msg[-250:]
  return msg


@dataclass
class EvaluationResult:
  success: bool
  metrics: Dict[str, float] = field(default_factory=dict)
  errors: List[str] = field(default_factory=list)
  warnings: List[str] = field(default_factory=list)

  def to_feedback(self, add_warning: bool = False) -> str:
    if self.success:
      lines = ["[SUCCESS] Evaluación exitosa."]
      lines.append("[SUCCESS] Métricas:")
      for k, v in self.metrics.items():
        lines.append(f"- {k}: {v:.4f}")
      return "\n".join(lines)

    lines = ["[ERROR] Errores durante la evaluación:"]
    for err in self.errors:
      lines.append(f"- {err}")
    if add_warning and self.warnings:
      lines.append("[WARNING] Advertencias:")
      for w in self.warnings:
        lines.append(f"- {w}")
    return "\n".join(lines)

def evaluate_pipeline(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, *, cv: int = 5, scoring: List[str] | None = None) -> EvaluationResult:
  "Evalúa el pipeline con validación cruzada y devuelve métricas."
  errors: List[str] = []
  warnings_list: List[str] = []
  metrics: Dict[str, float] = {}

  if scoring is None:
    scoring = ["accuracy", "f1_weighted", "roc_auc"]
  VALID_SCORING = {"accuracy", "f1_weighted", "roc_auc"}

  # Validaciones iniciales
  if not isinstance(pipeline, Pipeline):
    errors.append("El argumento 'pipeline' no es un sklearn Pipeline.")
    return EvaluationResult(success=False, errors=errors)
  if X is None or y is None:
    errors.append("X e y no pueden ser None.")
    return EvaluationResult(success=False, errors=errors)

  try:
    X = np.asarray(X)
    y = np.asarray(y)
  except Exception as e:
    errors.append(f"No se pudo convertir X o y a numpy array: {e}")
    return EvaluationResult(success=False, errors=errors)

  if X.ndim != 2:
    errors.append(f"X debe ser 2D, se encontró {X.ndim}D.")
  if y.ndim != 1:
    errors.append(f"y debe ser 1D, se encontró {y.ndim}D.")
  if X.shape[0] != y.shape[0]:
    errors.append(f"X e y tienen distinto número de muestras: {X.shape[0]} vs {y.shape[0]}.")
  if X.shape[0] < cv * 2:
    errors.append(f"Pocas muestras ({X.shape[0]}) para {cv} folds. Se necesitan al menos {cv * 2}.")
  unknown_scoring = set(scoring) - VALID_SCORING
  if unknown_scoring:
    errors.append(f"Métricas no reconocidas: {unknown_scoring}. Válidas: {VALID_SCORING}.")

  if errors:
    return EvaluationResult(success=False, errors=errors)

  # Validación particular para ColumnTransformer
  from sklearn.compose import ColumnTransformer
  for step_name, est in pipeline.steps:
    if isinstance(est, ColumnTransformer):
      for name, trans, columns in est.transformers:
        if isinstance(columns, list) and all(isinstance(c, int) for c in columns):
          max_idx = max(columns) if columns else -1
          if max_idx >= X.shape[1]:
            errors.append(f"ColumnTransformer '{step_name}' tiene índices de columna fuera de rango. El índice máximo es {max_idx}, pero X solo tiene {X.shape[1]} columnas.")
  if errors:
    return EvaluationResult(success=False, errors=errors)

  # Ajustes automáticos de parámetros que dependen de X
  from sklearn.feature_selection import SelectKBest, SelectPercentile
  from sklearn.decomposition import PCA

  for step_name, est in pipeline.steps:
    if isinstance(est, SelectKBest):  # SelectPercentile
      k = est.k
      if isinstance(k, int) and k > X.shape[1]:
        warnings_list.append(f"{type(est).__name__}(k={k}) pero X solo tiene {X.shape[1]} características. Se ajusta k a {X.shape[1]}.")
        est.k = X.shape[1]
    if isinstance(est, PCA):
      n = est.n_components
      if isinstance(n, int) and n > min(X.shape):
        warnings_list.append(f"PCA(n_components={n}) excede min(n_samples, n_features)={min(X.shape)}. Se ajusta a {min(X.shape) - 1}.")
        est.n_components = min(X.shape) - 1

  # Cross‑validation
  n_classes = len(np.unique(y))
  for metric in scoring:
    sklearn_metric = metric
    if metric == "roc_auc" and n_classes > 2:
      sklearn_metric = "roc_auc_ovr"
      warnings_list.append("roc_auc con >2 clases: se utiliza 'roc_auc_ovr'.")
    try:
      scores = cross_val_score(pipeline, X, y, cv=cv, scoring=sklearn_metric)
      metrics[f"{metric}_mean"] = float(np.mean(scores))
      metrics[f"{metric}_std"] = float(np.std(scores))
    except Exception as exc:
      core = _extract_core_error(exc)
      # Mejora: añadir aclaración si el error es por strings en columnas
      if "Specifying the columns using strings" in str(exc):
        core += " (Nota: los datos se pasan como array NumPy; utiliza índices enteros en el campo 'columns'.)"
      errors.append(f"Error en '{metric}': {core}")

  if errors:
    return EvaluationResult(success=False, errors=errors, warnings=warnings_list)

  # Fit completo
  try:
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    metrics["train_accuracy"] = float(accuracy_score(y, y_pred))
  except Exception as exc:
    core = _extract_core_error(exc)
    errors.append(f"Error en fit/predict final: {core}")
    return EvaluationResult(success=False, metrics=metrics, errors=errors, warnings=warnings_list)

  return EvaluationResult(success=True, metrics=metrics, errors=[], warnings=warnings_list)

# Orquestacion Principal: Parsear -> Construir -> Evaluar
@dataclass
class ParseResult:
  "Resultado del parsing de una respuesta del LLM."
  success: bool
  spec: Optional[PipelineSpec] = None
  pipeline: Optional[Pipeline] = None
  errors: List[str] = field(default_factory=list)
  warnings: List[str] = field(default_factory=list)

  def to_feedback(self) -> str:
    if self.success:
      return "[SUCCESS] Pipeline parseado y construido correctamente."
    lines = ["[ERROR] Falló el parsing/construcción del pipeline."]
    for e in self.errors:
      lines.append(f"- {e}")
    if self.warnings:
      lines.append("[WARNING] Advertencias:")
      for w in self.warnings:
        lines.append(f"- {w}")
    return "\n".join(lines)


def parse_and_build(llm_response: str) -> ParseResult:
  "Extrae todos los objetos JSON, intenta validarlos como PipelineSpec, construye el primer pipeline exitoso. Devuelve errores acumulados"
  candidates = extract_json_candidates(llm_response)
  if not candidates:
    return ParseResult(success=False, errors=["No se encontró ningún bloque JSON válido en la respuesta."])

  all_errors: List[str] = []
  warnings_list: List[str] = []

  for idx, candidate in enumerate(candidates, start=1):
    # Verificar que contenga clave "steps"
    if not isinstance(candidate, dict) or "steps" not in candidate:
      all_errors.append(f"Candidato {idx}: falta la clave 'steps' o no es un objeto.")
      continue

    try:
      # Validación con Pydantic
      spec = PipelineSpec.model_validate(candidate)
    except ValidationError as ve:
      # Formatear los errores de Pydantic de manera legible
      error_messages = []
      for err in ve.errors():
        loc = " → ".join(str(x) for x in err["loc"])
        msg = err["msg"]
        error_messages.append(f"  en {loc}: {msg}")
      all_errors.append(f"Candidato {idx} no válido:\n" + "\n".join(error_messages))
      continue

    # Intentar construir el pipeline de sklearn
    try:
      sk_pipeline = build_pipeline(spec)
    except Exception as e:
      all_errors.append(f"Candidato {idx}: la especificación es válida pero falló la construcción: {e}")
      continue

    # Éxito
    return ParseResult(
        success=True,
        spec=spec,
        pipeline=sk_pipeline,
        errors=[],
        warnings=[],
    )

  # Si llegamos aquí, todos los intentos fallaron.
  # Agregamos un mensaje resumen con los fallos acumulados
  feedback = "Ningún candidato pudo convertirse en un pipeline válido.\n"
  feedback += "Fallos detectados:\n" + "\n".join(all_errors)
  return ParseResult(success=False, errors=[feedback], warnings=warnings_list)


def _build_step_simple(step_dict: dict) -> Tuple[str, Any]:
  "Construye un paso de pipeline a partir de un diccionario, sin validar hiperparámetro ni componentes. Solo falla si la clase no existe o la instanciación lanza una excepción."
  name = step_dict.get("name")
  if not name or not isinstance(name, str):
    raise ValueError("Cada paso debe tener una clave 'name' (string).")

  component_name = step_dict.get("component")
  if not component_name or not isinstance(component_name, str):
    raise ValueError(f"Paso '{name}': falta la clave 'component' (string).")

  hyperparams = step_dict.get("hyperparameters", {})
  if not isinstance(hyperparams, dict):
    raise ValueError(f"Paso '{name}': 'hyperparameters' debe ser un objeto.")

  cls = None
  if component_name in ALLOWED_COMPONENTS:
    cls = ALLOWED_COMPONENTS[component_name]["class"]
  else:
    raise ValueError(f"Componente '{component_name}' no soportado en modo simple.")

  if component_name == "ColumnTransformer" and "transformers" in hyperparams:
    transformers_raw = hyperparams["transformers"]
    built_transformers = []
    for tf_dict in transformers_raw:
      sub_name = tf_dict["name"]
      sub_comp = tf_dict["transformer"]
      if sub_comp in ("drop", "passthrough"):
        transformer_obj = sub_comp
      else:
        _, transformer_obj = _build_step_simple({
            "name": sub_name,
            "component": sub_comp,
            "hyperparameters": tf_dict.get("transformer_hyperparameters", {})
        })
      columns = tf_dict["columns"]
      built_transformers.append((sub_name, transformer_obj, columns))
    hyperparams["transformers"] = built_transformers

  try:
    estimator = cls(**hyperparams)
  except Exception as e:
    raise type(e)(f"Error al instanciar '{component_name}': {e}")

  return name, estimator

def parse_and_build_simple(llm_response: str) -> ParseResult:
  "Extrae JSON candidatos y construye el primer pipeline que sea instanciable, sin validar contra ALLOWED_COMPONENTS más allá de la existencia de la clase. Devuelve errores mínimos."
  candidates = extract_json_candidates(llm_response)
  if not candidates:
    return ParseResult(success=False, errors=["No se encontró JSON válido en la respuesta."])

  all_errors = []
  for idx, cand in enumerate(candidates, 1):
    if not isinstance(cand, dict) or "steps" not in cand:
      all_errors.append(f"Candidato {idx}: falta la clave 'steps'.")
      continue

    steps_data = cand["steps"]
    if not isinstance(steps_data, list) or len(steps_data) == 0:
      all_errors.append(f"Candidato {idx}: 'steps' debe ser una lista no vacía.")
      continue

    steps = []
    failed = False
    for j, step_dict in enumerate(steps_data):
      try:
        name, est = _build_step_simple(step_dict)
        steps.append((name, est))
      except Exception as e:
        all_errors.append(f"Candidato {idx}, paso {j + 1}: {e}")
        failed = True
        break

    if failed:
      continue

    try:
      pipeline = Pipeline(steps)
    except Exception as e:
      all_errors.append(f"Candidato {idx}: error al construir Pipeline: {e}")
      continue

    return ParseResult(success=True, pipeline=pipeline, errors=[], warnings=[])

  return ParseResult(success=False, errors=all_errors)
