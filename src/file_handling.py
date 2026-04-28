import os
import json
from typing import Dict
from pathlib import Path

# * Sklearn Map Loader
JSON_PATH = Path(__file__).parent / "sklearn_map.json"

def load_sklearn_map() -> Dict:
  "Carga el archivo sklearn_map.json"
  if not JSON_PATH.exists():
    raise FileNotFoundError(f"No se encontró '{JSON_PATH}'")

  try:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
      return json.load(f)
  except json.JSONDecodeError as e:
    raise ValueError(f"Error al parsear '{JSON_PATH}': {str(e)}")
  except Exception as e:
    raise RuntimeError(f"Error al cargar '{JSON_PATH}': {str(e)}")

# * Test Pipelines Loader
TEST_PIPELINES_PATH = Path(__file__).parent / "pipeline_examples.json"

def load_test_pipelines() -> Dict:
  "Carga el archivo pipeline_examples.json con los pipelines de prueba"
  if not TEST_PIPELINES_PATH.exists():
    raise FileNotFoundError(f"No se encontró '{TEST_PIPELINES_PATH}'")

  try:
    with open(TEST_PIPELINES_PATH, "r", encoding="utf-8") as f:
      return json.load(f)
  except json.JSONDecodeError as e:
    raise ValueError(f"Error al parsear '{TEST_PIPELINES_PATH}': {str(e)}")
  except Exception as e:
    raise RuntimeError(f"Error al cargar '{TEST_PIPELINES_PATH}': {str(e)}")

# * Dataset Registry (para tracking de datasets y sus metadatos)
REGISTRY_PATH = Path("data/registry.json")

def load_registry() -> Dict:
  if not REGISTRY_PATH.exists():
    return {}
  with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
    return json.load(f)

def save_registry(registry) -> None:
  os.makedirs(REGISTRY_PATH.parent, exist_ok=True)
  with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
    json.dump(registry, f, indent=4)

def update_dataset_entry(dataset_id, data) -> None:
  registry = load_registry()
  dataset_id = str(dataset_id)
  if dataset_id not in registry:
    registry[dataset_id] = {}
  registry[dataset_id].update(data)
  save_registry(registry)
