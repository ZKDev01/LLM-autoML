import re
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.terminal_tools import *
from src.chatbot import Ollama_ChatBot
from src.meta_features import compute_meta_features
from src.openml_manager import OpenMLManager
from src.schema import MLPipelineGenerator, parse_llm_response_to_pipeline, evaluate_pipeline, _extract_json_from_text

class AutoML_Bot(Ollama_ChatBot):
  "AutoML using an LLM to generate pipelines"

  def __init__(self, model: str | None = None, host: str | None = None, stream: bool = True, task_description: str = "classification", cv_folds: int = 5, verbose: bool = False) -> None:
    self.task_description = task_description
    self.cv_folds = cv_folds
    self.verbose = verbose

    # Dataset attributes
    self.dataset: Optional[pd.DataFrame] = None
    self.dataset_info: Optional[Dict[str, Any]] = None
    self.target_column: Optional[str] = None
    self.columns_info: Optional[Dict[str, Any]] = None
    self.dataset_info_text: Optional[str] = None

    # Build the system prompt using the new generator
    self.generator = MLPipelineGenerator()
    system_prompt = self.generator.generate_prompt()

    # Initialise the underlying chatbot
    super().__init__(model=model, host=host, system_prompt=system_prompt, stream=stream)

  def load_dataset_from_openml(self, dataset_id: int) -> Tuple[bool, str]:
    manager = OpenMLManager(path="./datasets")
    if self.verbose:
      header(f" [CARGA DATASET] OpenML Dataset #{dataset_id}")

    self.dataset = manager.get_dataset(dataset_id=dataset_id)
    self.dataset_info = manager.get_dataset_info(dataset_id=dataset_id)
    self.target_column = self.dataset_info['target']
    return True, "Dataset loaded successfully"

  def anonymize_columns(self) -> Tuple[bool, str]:
    if self.dataset is None:
      fail("Dataset not loaded")
      return False, "Dataset not loaded"

    if self.verbose:
      header(" [ANONYMIZE COLUMNS]")

    try:
      rename_map = {}
      col_mapping = {}
      col_idx = 0
      for col in self.dataset.columns:
        alias = 'target' if col == self.target_column else f"column_{col_idx}"
        if col != self.target_column:
          col_idx += 1
        rename_map[col] = alias
        col_mapping[alias] = col

      self.dataset = self.dataset.rename(columns=rename_map)
      self.columns_info = col_mapping
      old_target = self.target_column
      self.target_column = rename_map[old_target]

      if self.verbose:
        ok(f"Anonymized {len(rename_map)} columns")
        for alias, original in col_mapping.items():
          print(f"      {alias:15s} <- {original}")
      return True, "Columns anonymized"
    except Exception as e:
      fail(f"Error anonymizing columns: {e}")
      return False, str(e)

  def build_dataset_info(self, k_examples: int = 0, include_anonymize_columns: bool = True, include_meta_features: bool = True) -> Tuple[bool, str]:
    "Construct a natural-language description of the dataset for the LLM."
    if include_anonymize_columns:
      result, _ = self.anonymize_columns()
      if not result:
        return False, "Failed to anonymize columns"

    col_list = [col for col in self.dataset.columns]
    columns_text = ', '.join(col_list)

    if include_meta_features:
      meta = compute_meta_features(self.dataset, self.target_column)
      self.dataset_info['meta-features'] = meta
    else:
      meta = self.dataset_info.get('meta-features', {})

    dataset_info_text = (f"Información del dataset\nColumns: {columns_text}\nMeta-Features:\n{str(meta)}")

    if k_examples > 0:
      dataset_info_text += f"\nPrimeros {k_examples} datos:\n{self.dataset.head(k_examples)}"

    if self.verbose:
      ok(f"Dataset info text:\n{dataset_info_text}")

    self.dataset_info_text = dataset_info_text
    return True, dataset_info_text

  def prepare_for_llm(self, k_examples: int = 0, include_anonymize_columns: bool = True) -> Tuple[bool, str]:
    "Final preparation before calling the LLM. The system prompt (with allowed components) is already in place. Here we build the user-visible dataset description."

    success, msg = self.build_dataset_info(k_examples=k_examples, include_anonymize_columns=include_anonymize_columns, include_meta_features=True)
    if not success:
      return False, f"build_dataset_info failed: {msg}"
    return True, "Ready for LLM"

  def _user_prompt(self, error_msg: str = "") -> str:
    "Base user prompt with the dataset description and optional error feedback."
    base = f"{self.dataset_info_text}\n\nGenera una propuesta de pipeline de clasificación para este dataset."
    if error_msg:
      base += f"\n\nEl pipeline anterior falló con los siguientes errores:\n{error_msg}\nCorrígelos y responde con un JSON válido."
    return base

  def generate_pipelines(self, k_repair: int = 3, add_reasoning: bool = True, save_log_path: Optional[str] = None, print_chat: bool = False) -> Tuple[Pipeline, str, Dict[str, float], Dict]:
    "Generate a single pipeline using K attempts"
    log = {
        "algorithm": "single_generation",
        "dataset_id": self.dataset_info.get("dataset_id") if self.dataset_info else None,
        "k_repair": k_repair,
        "attempts": [],
        "success": False,
    }

    if self.dataset is None:
      raise RuntimeError("Dataset not loaded")

    X = self.dataset.drop(columns=[self.target_column]).to_numpy()
    y = self.dataset[self.target_column].to_numpy()

    prompt = self._user_prompt()

    for attempt in range(1, k_repair + 1):
      header(f"    [ATTEMPT {attempt}]")
      attempt_log = {"attempt": attempt}

      response = self.chat(prompt)           # user message added automatically
      raw_text = response.message.content

      if print_chat:
        print(f"\n[LLM Response]:\n{raw_text}\n")

      # 1. Parse & Build
      parse_result = parse_llm_response_to_pipeline(raw_text)
      attempt_log["parse_success"] = parse_result.success
      attempt_log["parse_errors"] = parse_result.errors
      attempt_log["parse_warnings"] = parse_result.warnings

      if not parse_result.success:
        error_msg = parse_result.to_feedback()
        fail(f"  [PARSE ERROR] {error_msg}")
        prompt = self._user_prompt(error_msg)  # re‑prompt with error
        log["attempts"].append(attempt_log)
        continue

      pipeline = parse_result.pipeline

      # 2. Evaluate
      eval_result = evaluate_pipeline(pipeline, X, y, cv=self.cv_folds, scoring=["accuracy"])
      attempt_log["eval_success"] = eval_result.success
      attempt_log["eval_errors"] = eval_result.errors
      attempt_log["eval_warnings"] = eval_result.warnings
      attempt_log["metrics"] = eval_result.metrics

      if not eval_result.success:
        error_msg = eval_result.to_feedback()
        fail(f"  [EVAL ERROR] {error_msg}")
        prompt = self._user_prompt(error_msg)
        log["attempts"].append(attempt_log)
        continue

      # 3. Success – store config (the original JSON is best)
      config_dict = self._extract_steps_json(raw_text)
      attempt_log["config"] = config_dict

      # 4. Ask for reasoning (OPTIONAL)
      reasoning = ""
      if add_reasoning:
        reasoning = self._generate_reasoning(config_dict, eval_result.metrics)
        attempt_log["reasoning"] = reasoning
        log["final_reasoning"] = reasoning

      # Finish
      log["success"] = True
      log["final_config"] = config_dict
      log["final_metrics"] = eval_result.metrics
      log["attempts"].append(attempt_log)

      if save_log_path:
        self._save_execution_log(log, save_log_path)

      return pipeline, reasoning, eval_result.metrics, config_dict

    # All attempts exhausted
    if save_log_path:
      self._save_execution_log(log, save_log_path)
    raise RuntimeError("No se pudo generar un pipeline funcional")

  def generate_pipelines_with_optimization(self, target_metric: str = 'accuracy_mean', add_reasoning: bool = True, max_iterations: int = 10, max_history_size: int = 5, k_repair: int = 3, save_log_path: Optional[str] = None, print_chat: bool = False) -> Tuple[Pipeline, str, Dict[str, float]]:
    "Iteratively improve a pipeline using the LLM as optimiser"
    log = {
        "algorithm": "optimization",
        "dataset_id": self.dataset_info.get("dataset_id") if self.dataset_info else None,
        "target_metric": target_metric,
        "max_iterations": max_iterations,
        "max_history_size": max_history_size,
        "k_repair": k_repair,
        "iterations": [],
        "success": False,
    }

    # 1. Generate a starting pipeline
    try:
      _, _, _, initial_config = self.generate_pipelines(k_repair=k_repair, add_reasoning=add_reasoning, save_log_path=None, print_chat=print_chat)
    except Exception as e:
      log["error_initial"] = str(e)
      if save_log_path:
        self._save_execution_log(log, save_log_path)
      raise RuntimeError(f"Initial pipeline generation failed: {e}")

    # Evaluate initial to get metrics
    X = self.dataset.drop(columns=[self.target_column]).to_numpy()
    y = self.dataset[self.target_column].to_numpy()
    pipeline_initial = self._build_from_config(initial_config)   # helper
    eval_init = evaluate_pipeline(pipeline_initial, X, y, cv=self.cv_folds)
    if not eval_init.success:
      raise RuntimeError("Initial pipeline evaluation failed")

    best_score = eval_init.metrics[target_metric]
    best_pipeline = pipeline_initial
    best_metrics = eval_init.metrics

    reasoning_init = ""
    if add_reasoning:
      reasoning_init = self._generate_reasoning(initial_config, best_metrics)

    history = [{
        "config": initial_config,
        "metrics": best_metrics,
        "reasoning": reasoning_init,
    }]

    # 2. Iterative improvement loop
    for iteration in range(1, max_iterations + 1):
      header(f" \n[OPTIMIZATION ITERATION {iteration}]")
      iter_log = {"iteration": iteration, "attempts": [], "improved": False}

      # Build history text
      sorted_history = sorted(history, key=lambda h: h["metrics"][target_metric])
      history_text = self._format_history_for_prompt(sorted_history)

      meta_prompt = (
          f"Historial de pipelines (peor -> mejor) según {target_metric}:\n"
          f"{history_text}\n\n"
          f"Mejor {target_metric} actual: {best_score:.4f}\n"
          f"Genera un NUEVO pipeline que supere esta puntuación. "
          f"Puedes cambiar componentes o ajustar hiperparámetros.\n"
          f"Responde SÓLO con el JSON válido."
      )

      # Use the base dataset info as context
      full_prompt = f"{self.dataset_info_text}\n\n{meta_prompt}"

      # Repair loop inside each iteration
      success_iter = False
      for attempt in range(1, k_repair + 1):
        header(f"    [ATTEMPT {attempt}]")
        attempt_log = {"attempt": attempt}
        response = self.chat(full_prompt)
        raw_text = response.message.content

        parse_result = parse_llm_response_to_pipeline(raw_text)
        if not parse_result.success:
          full_prompt = self._user_prompt(parse_result.to_feedback())
          iter_log["attempts"].append(attempt_log)
          continue

        pipeline = parse_result.pipeline
        eval_result = evaluate_pipeline(pipeline, X, y, cv=self.cv_folds)
        if not eval_result.success:
          full_prompt = self._user_prompt(eval_result.to_feedback())
          iter_log["attempts"].append(attempt_log)
          continue

        metrics = eval_result.metrics
        config_dict = self._extract_steps_json(raw_text)

        attempt_log["config"] = config_dict
        attempt_log["metrics"] = metrics

        reasoning = ""
        if add_reasoning:
          reasoning = self._generate_reasoning(config_dict, metrics)
          attempt_log["reasoning"] = reasoning

        iter_log["attempts"].append(attempt_log)
        success_iter = True
        break

      if not success_iter:
        warn(f"  Iteration {iteration}: no valid pipeline produced")
        log["iterations"].append(iter_log)
        continue

      # Add to history
      history.append({
          "config": config_dict,
          "metrics": metrics,
          "reasoning": reasoning,
      })
      history.sort(key=lambda h: h["metrics"][target_metric], reverse=True)
      if len(history) > max_history_size:
        history = history[:max_history_size]

      new_score = metrics[target_metric]
      if new_score > best_score:
        best_score = new_score
        best_pipeline = pipeline
        best_metrics = metrics
        best_reasoning = reasoning
        iter_log["improved"] = True
        ok(f"  New best {target_metric} = {best_score:.4f}")
      else:
        iter_log["improved"] = False

      log["iterations"].append(iter_log)

      if best_score >= 0.99:
        break

    # 3. Final reasoning across all iterations
    final_reasoning = ""
    if add_reasoning:
      final_reasoning = self._generate_final_reasoning(history, best_metrics, target_metric)
    log["success"] = True
    log["final_best_metrics"] = best_metrics
    log["final_best_reasoning"] = final_reasoning

    if save_log_path:
      self._save_execution_log(log, save_log_path)

    return best_pipeline, final_reasoning, best_metrics

  def _extract_steps_json(self, llm_text: str) -> dict:
    "Extract the original JSON config from the LLM response (same as schema does)."
    data = _extract_json_from_text(llm_text)
    if data is None:
      raise ValueError("No JSON found in LLM response")
    return data

  def _build_from_config(self, config: dict) -> Pipeline:
    "Re-build a Pipeline from a config dict (needed for the initial pipeline in optimisation). We simply re-parse the config as if it were an LLM response."
    # Simulate a response by serialising the dict back to a string
    raw = json.dumps(config)
    parse_result = parse_llm_response_to_pipeline(raw)
    if not parse_result.success:
      raise ValueError(f"Cannot rebuild pipeline: {parse_result.errors}")
    return parse_result.pipeline

  def _generate_reasoning(self, config: dict, metrics: Dict[str, float]) -> str:
    "Ask the LLM to explain why this pipeline works."
    prompt = (
        f"Pipeline: {json.dumps(config, indent=2)}\n"
        f"Métricas: {json.dumps(metrics, indent=2)}\n\n"
        "Explica brevemente por qué este pipeline es adecuado para la tarea de clasificación "
        "y qué características del dataset lo hacen funcionar bien."
    )
    response = self.chat(prompt)
    return response.message.content

  def _generate_final_reasoning(self, history: list, best_metrics: dict, metric: str) -> str:
    "Summarise the whole optimisation journey."
    lines = [f"{h['metrics'][metric]:.4f}" for h in history]
    prompt = (
        f"Evolución de {metric}: {lines}\n"
        f"Mejor pipeline final: {json.dumps(best_metrics, indent=2)}\n\n"
        "Explica por qué el pipeline final es bueno y cómo las mejoras sucesivas llevaron a él."
    )
    response = self.chat(prompt)
    return response.message.content

  def _format_history_for_prompt(self, sorted_history: list) -> str:
    "Format the history list (worst -> best) into a concise string."
    parts = []
    for idx, entry in enumerate(sorted_history, 1):
      acc = entry['metrics'].get('accuracy_mean', 'N/A')
      reasoning = entry['reasoning'][:200]
      config_snippet = json.dumps(entry['config'], indent=2)[:500]
      parts.append(f"{idx}. Accuracy: {acc:.4f}\n   Reason: {reasoning}\n   Config snippet: {config_snippet}")
    return "\n\n".join(parts)

  def _sanitize_for_json(self, obj):
    """Convierte objetos no serializables a JSON de forma segura."""
    import types
    import inspect

    if isinstance(obj, dict):
      return {k: self._sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
      return [self._sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.generic, np.integer, np.floating)):
      return obj.item()
    if isinstance(obj, type) and issubclass(obj, np.generic):
      return str(obj)
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
      return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16)):
      return float(obj)
    if isinstance(obj, np.ndarray):
      return self._sanitize_for_json(obj.tolist())
    if inspect.isclass(obj) or isinstance(obj, (types.MethodType, types.FunctionType)):
      return str(obj)
    if hasattr(obj, 'item') and callable(obj.item) and not inspect.isclass(obj):
      try:
        return obj.item()
      except Exception:
        return str(obj)
    return obj

  def _save_execution_log(self, log_data: dict, filename: Optional[str] = None) -> str:
    sanitized = self._sanitize_for_json(log_data)

    from datetime import datetime

    if filename is None:
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      filename = f"automl_log_{timestamp}.json"
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
      json.dump(sanitized, f, indent=2, ensure_ascii=False)
    ok(f"Log saved to {filepath}")
    return str(filepath)
