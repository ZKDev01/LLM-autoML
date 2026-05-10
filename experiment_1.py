#!/usr/bin/env python3
"""
Experiment script to run multiple independent pipeline generations for a given dataset.

Usage:
    python experiment.py <dataset_id> [k_repair]

Example:
    python experiment.py 31 5

This will run 30 independent trials (each using `generate_pipelines` with the specified
number of repair attempts) and collect results in a JSON file.
If a trial fails (RuntimeError), it will be retried up to 3 times before marking as failed.
"""

import sys
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.autoML_bot import AutoML_Bot
from src.terminal_tools import header, ok, fail, warn

class ExperimentBot(AutoML_Bot):
  """
  Subclass of AutoML_Bot that returns the full attempt log from generate_pipelines.
  """

  def generate_pipelines(
      self,
      k_repair: int = 3,
      add_reasoning: bool = True,
      save_log_path: Optional[str] = None,
      print_chat: bool = False,
      save_result_path: Optional[str] = None,
      auto_generate_filename: bool = True,
  ) -> Tuple[Pipeline, str, Dict[str, float], Dict, List[Dict]]:
    """
    Same as parent, but returns (pipeline, reasoning, metrics, config_dict, attempts_log).
    """
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

      response = self.chat(prompt)
      raw_text = response.message.content

      if print_chat:
        print(f"\n[LLM Response]:\n{raw_text}\n")

      parse_result = self.generator.parse_response(raw_text)
      attempt_log["parse_success"] = parse_result.success
      attempt_log["parse_errors"] = parse_result.errors
      attempt_log["parse_warnings"] = parse_result.warnings

      if not parse_result.success:
        error_msg = parse_result.to_feedback()
        fail(f"  [PARSE ERROR] {error_msg}")
        prompt = self._user_prompt(error_msg)
        log["attempts"].append(attempt_log)
        continue

      pipeline = parse_result.pipeline

      eval_result = self.generator.evaluate(pipeline, X, y, cv=self.cv_folds, scoring=["accuracy"])
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

      # Success – extract config
      config_dict = self._extract_steps_json(raw_text)
      attempt_log["config"] = config_dict

      reasoning = ""
      if add_reasoning:
        reasoning = self._generate_reasoning(config_dict, eval_result.metrics)
        attempt_log["reasoning"] = reasoning
        log["final_reasoning"] = reasoning

      log["success"] = True
      log["final_config"] = config_dict
      log["final_metrics"] = eval_result.metrics
      log["attempts"].append(attempt_log)

      if save_result_path is not None or auto_generate_filename:
        extra = {"k_repair": k_repair, "add_reasoning": add_reasoning}
        result_dict = self._create_result_dict(
            algorithm="single_generation",
            final_config=config_dict,
            final_metrics=eval_result.metrics,
            final_reasoning=reasoning,
            attempts_log=log["attempts"],
            extra=extra,
        )
        output_path = save_result_path if save_result_path else None
        if auto_generate_filename and output_path is None:
          output_path = None
        self._save_result(result_dict, output_path)

      return pipeline, reasoning, eval_result.metrics, config_dict, log["attempts"]

    # All attempts exhausted
    if save_log_path:
      self._save_execution_log(log, save_log_path)
    raise RuntimeError("No se pudo generar un pipeline funcional")


# ============================================================================
#  Experiment runner
# ============================================================================
def run_single_trial(
    dataset_id: int,
    k_repair: int,
    trial_index: int,
    max_retries: int = 3,
    verbose: bool = False,
) -> Dict[str, Any]:
  """
  Run a single trial (one call to generate_pipelines) with retries.
  Returns a dictionary with trial results.
  """
  for retry in range(1, max_retries + 1):
    start_time = time.time()
    bot = None
    try:
      bot = ExperimentBot(
          task_description="classification",
          cv_folds=5,
          verbose=verbose,
          model="gpt-oss:120b",  # default model, can be changed
          host=None,
          stream=False,  # set to False for cleaner logging
      )

      # Load dataset
      success, msg = bot.load_dataset_from_openml(dataset_id=dataset_id)
      if not success:
        raise RuntimeError(f"Dataset load failed: {msg}")

      # Prepare for LLM (anonymize columns, compute meta-features)
      success, msg = bot.prepare_for_llm(k_examples=0, include_anonymize_columns=True)
      if not success:
        raise RuntimeError(f"Preparation failed: {msg}")

      # Generate pipeline
      pipeline, reasoning, metrics, config, attempts_log = bot.generate_pipelines(
          k_repair=k_repair,
          add_reasoning=False,      # we can skip reasoning to save time
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
          "config": config,
          "metrics": metrics,
          "attempts_log": attempts_log,
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
        # Final failure
        result = {
            "trial": trial_index,
            "success": False,
            "retry": retry,
            "elapsed_seconds": round(elapsed, 2),
            "config": None,
            "metrics": None,
            "attempts_log": None,
            "pipeline_str": None,
            "error": error_msg,
        }
        return result
    finally:
      # Clean up if needed
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
    print("Usage: python experiment.py <dataset_id> [k_repair]")
    sys.exit(1)

  try:
    dataset_id = int(sys.argv[1])
  except ValueError:
    print("Error: dataset_id must be an integer")
    sys.exit(1)

  k_repair = 5
  if len(sys.argv) >= 3:
    try:
      k_repair = int(sys.argv[2])
    except ValueError:
      print("Error: k_repair must be an integer")
      sys.exit(1)

  N_TRIALS = 30
  MAX_RETRIES_PER_TRIAL = 3

  print(f"\n{'=' * 70}")
  print(f"EXPERIMENT: {N_TRIALS} trials on dataset {dataset_id}")
  print(f"k_repair = {k_repair}, max retries per trial = {MAX_RETRIES_PER_TRIAL}")
  print(f"{'=' * 70}\n")

  results = []
  start_time_total = time.time()

  for trial_idx in range(1, N_TRIALS + 1):
    header(f"Trial {trial_idx}/{N_TRIALS}")
    trial_result = run_single_trial(
        dataset_id=dataset_id,
        k_repair=k_repair,
        trial_index=trial_idx,
        max_retries=MAX_RETRIES_PER_TRIAL,
        verbose=False,
    )
    results.append(trial_result)

    # Print short summary
    if trial_result["success"]:
      acc = trial_result["metrics"].get("accuracy_mean", 0.0)
      ok(f"Trial {trial_idx} OK   accuracy_mean = {acc:.4f}  (elapsed {trial_result['elapsed_seconds']}s)")
    else:
      fail(f"Trial {trial_idx} FAILED after {trial_result['retry']} retries")

    # Save partial results after each trial (optional)
    partial_file = f"experiment_partial_{dataset_id}.json"
    with open(partial_file, "w", encoding="utf-8") as f:
      json.dump(
          {
              "dataset_id": dataset_id,
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
  print(f"\n{'=' * 70}")
  print(f"EXPERIMENT FINISHED")
  print(f"Successful trials: {success_count}/{N_TRIALS}")
  print(f"Total time: {total_elapsed:.2f} seconds")
  print(f"{'=' * 70}\n")

  # Save final results
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  final_output = f"experiment_results_{dataset_id}_{timestamp}.json"
  output_data = {
      "experiment_config": {
          "dataset_id": dataset_id,
          "k_repair": k_repair,
          "n_trials": N_TRIALS,
          "max_retries_per_trial": MAX_RETRIES_PER_TRIAL,
          "total_elapsed_seconds": round(total_elapsed, 2),
      },
      "summary": {
          "successful_trials": success_count,
          "failed_trials": N_TRIALS - success_count,
          "success_rate": success_count / N_TRIALS,
      },
      "results": results,
  }

  with open(final_output, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

  print(f"Final results saved to: {final_output}")


if __name__ == "__main__":
  main()
