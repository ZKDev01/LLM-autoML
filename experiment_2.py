#!/usr/bin/env python3
"""
Experiment script to run multiple independent optimisation trials for a given dataset.

Usage:
    python experiment_2.py <dataset_id> [max_iterations] [k_repair]

Example:
    python experiment_2.py 31 10 3

This will run 30 independent optimisation runs (each using `generate_pipelines_with_optimization`
with the specified parameters) and collect results in a JSON file.
If a trial fails (RuntimeError), it will be retried up to 3 times before marking as failed.
"""

import sys
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.autoML_bot import AutoML_Bot
from src.terminal_tools import header, ok, fail, warn

class OptimExperimentBot(AutoML_Bot):
  """
  Subclass of AutoML_Bot that returns the full optimisation log from
  generate_pipelines_with_optimization.
  """

  def generate_pipelines_with_optimization(
      self,
      target_metric: str = 'accuracy_mean',
      add_reasoning: bool = True,
      max_iterations: int = 10,
      max_history_size: int = 5,
      k_repair: int = 3,
      save_log_path: Optional[str] = None,
      print_chat: bool = False,
      save_result_path: Optional[str] = None,
      auto_generate_filename: bool = True,
  ) -> Tuple[Pipeline, str, Dict[str, float], Dict]:
    """
    Same as parent, but returns (pipeline, reasoning, metrics, log_dict)
    where log_dict contains the full optimisation log.
    """
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

    # 1. Generate a starting pipeline using the single-generation method
    try:
      _, _, _, initial_config = self.generate_pipelines(
          k_repair=k_repair,
          add_reasoning=add_reasoning,
          save_log_path=None,
          print_chat=print_chat,
          save_result_path=None,
          auto_generate_filename=False,
      )
    except Exception as e:
      log["error_initial"] = str(e)
      raise RuntimeError(f"Initial pipeline generation failed: {e}")

    # Evaluate initial to get metrics
    X = self.dataset.drop(columns=[self.target_column]).to_numpy()
    y = self.dataset[self.target_column].to_numpy()
    pipeline_initial = self._build_from_config(initial_config)
    eval_init = self.generator.evaluate(pipeline_initial, X, y, cv=self.cv_folds)
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

    log["iterations"].append({
        "iteration": 0,
        "type": "initial",
        "success": True,
        "improved": False,
        "best_score": best_score,
        "config": initial_config,
        "metrics": best_metrics,
        "reasoning": reasoning_init,
        "attempts": [],
    })

    # 2. Iterative improvement loop
    for iteration in range(1, max_iterations + 1):
      header(f" \n[OPTIMIZATION ITERATION {iteration}]")
      iter_log = {
          "iteration": iteration,
          "attempts": [],
          "improved": False,
          "best_score": best_score,
      }

      # Build history text (worst -> best)
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
      full_prompt = f"{self.dataset_info_text}\n\n{meta_prompt}"

      success_iter = False
      for attempt in range(1, k_repair + 1):
        header(f"    [ATTEMPT {attempt}]")
        attempt_log = {"attempt": attempt}
        response = self.chat(full_prompt)
        raw_text = response.message.content

        parse_result = self.generator.parse_response(raw_text)
        if not parse_result.success:
          full_prompt = self._user_prompt(parse_result.to_feedback())
          iter_log["attempts"].append(attempt_log)
          continue

        pipeline = parse_result.pipeline
        eval_result = self.generator.evaluate(pipeline, X, y, cv=self.cv_folds)
        if not eval_result.success:
          full_prompt = self._user_prompt(eval_result.to_feedback())
          iter_log["attempts"].append(attempt_log)
          continue

        # Success – extract config and reasoning
        config_dict = self._extract_steps_json(raw_text)
        metrics = eval_result.metrics

        reasoning = ""
        if add_reasoning:
          reasoning = self._generate_reasoning(config_dict, metrics)
        attempt_log["config"] = config_dict
        attempt_log["metrics"] = metrics
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

      iter_log["best_score"] = best_score
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
    log["final_best_config"] = history[0]["config"] if history else None
    log["total_iterations"] = len(log["iterations"]) - 1  # exclude initial

    # Save result if requested (similar to parent)
    if save_result_path is not None or auto_generate_filename:
      extra = {
          "target_metric": target_metric,
          "max_iterations": max_iterations,
          "max_history_size": max_history_size,
          "k_repair": k_repair,
          "iterations": log["iterations"],
      }
      result_dict = self._create_result_dict(
          algorithm="optimization",
          final_config=log["final_best_config"],
          final_metrics=best_metrics,
          final_reasoning=final_reasoning,
          attempts_log=[],
          extra=extra,
      )
      output_path = save_result_path if save_result_path else None
      if auto_generate_filename and output_path is None:
        output_path = None
      self._save_result(result_dict, output_path)

    return best_pipeline, final_reasoning, best_metrics, log


# ============================================================================
#  Experiment runner
# ============================================================================
def run_single_trial(
    dataset_id: int,
    max_iterations: int,
    k_repair: int,
    trial_index: int,
    max_retries: int = 3,
    verbose: bool = False,
) -> Dict[str, Any]:
  """
  Run a single optimisation trial with retries.
  Returns a dictionary with trial results and the optimisation log.
  """
  for retry in range(1, max_retries + 1):
    start_time = time.time()
    bot = None
    try:
      bot = OptimExperimentBot(
          task_description="classification",
          cv_folds=5,
          verbose=verbose,
          model="gpt-oss:120b",   # default, can be changed
          host=None,
          stream=False,
      )

      # Load dataset
      success, msg = bot.load_dataset_from_openml(dataset_id=dataset_id)
      if not success:
        raise RuntimeError(f"Dataset load failed: {msg}")

      # Prepare for LLM
      success, msg = bot.prepare_for_llm(k_examples=0, include_anonymize_columns=True)
      if not success:
        raise RuntimeError(f"Preparation failed: {msg}")

      # Run optimisation
      pipeline, reasoning, metrics, optim_log = bot.generate_pipelines_with_optimization(
          target_metric='accuracy_mean',
          add_reasoning=False,          # save time
          max_iterations=max_iterations,
          max_history_size=5,
          k_repair=k_repair,
          print_chat=False,
          save_result_path=None,
          auto_generate_filename=False,
      )

      elapsed = time.time() - start_time

      result = {
          "trial": trial_index,
          "success": True,
          "retry": retry,
          "elapsed_seconds": round(elapsed, 2),
          "final_accuracy": metrics.get("accuracy_mean", None),
          "final_config": optim_log.get("final_best_config"),
          "final_metrics": metrics,
          "optimisation_log": optim_log,   # full log including iterations
          "pipeline_str": str(pipeline),
          "error": None,
      }
      return result

    except Exception as e:
      elapsed = time.time() - start_time
      error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
      if retry < max_retries:
        warn(f"Trial {trial_index}, retry {retry} failed: {e}. Retrying...")
        continue
      else:
        result = {
            "trial": trial_index,
            "success": False,
            "retry": retry,
            "elapsed_seconds": round(elapsed, 2),
            "final_accuracy": None,
            "final_config": None,
            "final_metrics": None,
            "optimisation_log": None,
            "pipeline_str": None,
            "error": error_msg,
        }
        return result
    finally:
      if bot and bot.dataset is not None:
        del bot

  # Should never reach here
  return {
      "trial": trial_index,
      "success": False,
      "error": "Max retries exceeded without success",
  }


def main():
  # Parse command line arguments
  if len(sys.argv) < 2:
    print("Usage: python experiment_2.py <dataset_id> [max_iterations] [k_repair]")
    sys.exit(1)

  try:
    dataset_id = int(sys.argv[1])
  except ValueError:
    print("Error: dataset_id must be an integer")
    sys.exit(1)

  max_iterations = 5       # default
  if len(sys.argv) >= 3:
    try:
      max_iterations = int(sys.argv[2])
    except ValueError:
      print("Error: max_iterations must be an integer")
      sys.exit(1)

  k_repair = 3
  if len(sys.argv) >= 4:
    try:
      k_repair = int(sys.argv[3])
    except ValueError:
      print("Error: k_repair must be an integer")
      sys.exit(1)

  N_TRIALS = 30
  MAX_RETRIES_PER_TRIAL = 3

  print(f"\n{'=' * 70}")
  print(f"EXPERIMENT 2: {N_TRIALS} optimisation trials on dataset {dataset_id}")
  print(f"max_iterations = {max_iterations}, k_repair = {k_repair}")
  print(f"max retries per trial = {MAX_RETRIES_PER_TRIAL}")
  print(f"{'=' * 70}\n")

  results = []
  start_time_total = time.time()

  for trial_idx in range(1, N_TRIALS + 1):
    header(f"Trial {trial_idx}/{N_TRIALS}")
    trial_result = run_single_trial(
        dataset_id=dataset_id,
        max_iterations=max_iterations,
        k_repair=k_repair,
        trial_index=trial_idx,
        max_retries=MAX_RETRIES_PER_TRIAL,
        verbose=False,
    )
    results.append(trial_result)

    # Print short summary
    if trial_result["success"]:
      acc = trial_result["final_accuracy"]
      ok(f"Trial {trial_idx} OK   final accuracy = {acc:.4f}  (elapsed {trial_result['elapsed_seconds']}s)")
    else:
      fail(f"Trial {trial_idx} FAILED after {trial_result['retry']} retries")

    # Save partial results after each trial
    partial_file = f"experiment2_partial_{dataset_id}.json"
    with open(partial_file, "w", encoding="utf-8") as f:
      json.dump(
          {
              "dataset_id": dataset_id,
              "max_iterations": max_iterations,
              "k_repair": k_repair,
              "total_trials": N_TRIALS,
              "completed_trials": trial_idx,
              "results": results,
          },
          f,
          indent=2,
          ensure_ascii=False,
      )

  total_elapsed = time.time() - start_time_total
  success_count = sum(1 for r in results if r["success"])
  accuracies = [r["final_accuracy"] for r in results if r["success"]]

  print(f"\n{'=' * 70}")
  print(f"EXPERIMENT FINISHED")
  print(f"Successful trials: {success_count}/{N_TRIALS}")
  if accuracies:
    print(f"Mean final accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Min final accuracy: {np.min(accuracies):.4f}")
    print(f"Max final accuracy: {np.max(accuracies):.4f}")
  print(f"Total time: {total_elapsed:.2f} seconds")
  print(f"{'=' * 70}\n")

  # Save final results
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  final_output = f"experiment2_results_{dataset_id}_{timestamp}.json"
  output_data = {
      "experiment_config": {
          "dataset_id": dataset_id,
          "algorithm": "optimization",
          "max_iterations": max_iterations,
          "k_repair": k_repair,
          "n_trials": N_TRIALS,
          "max_retries_per_trial": MAX_RETRIES_PER_TRIAL,
          "total_elapsed_seconds": round(total_elapsed, 2),
      },
      "summary": {
          "successful_trials": success_count,
          "failed_trials": N_TRIALS - success_count,
          "success_rate": success_count / N_TRIALS,
          "final_accuracy_mean": float(np.mean(accuracies)) if accuracies else None,
          "final_accuracy_std": float(np.std(accuracies)) if accuracies else None,
      },
      "results": results,
  }

  with open(final_output, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

  print(f"Final results saved to: {final_output}")


if __name__ == "__main__":
  main()
