from openml import OpenMLDataset
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Union

import openml
import pandas as pd
import numpy as np

from src.terminal_tools import *

_REGISTRY_FILE = "registry.json"

class OpenMLManager:
  "Gestiona la descarga, almacenamiento y consulta de datasets de OpenML con extracción de meta-features."

  def __init__(self, path: Union[str, Path]) -> None:
    self.root = Path(path)
    self.root.mkdir(parents=True, exist_ok=True)
    self._registry_path = self.root / _REGISTRY_FILE
    self._registry: Dict = self._load_registry()

  def download_dataset(self, dataset_id: int, suite_id: Optional[int] = None, suite_name: Optional[str] = None, skip_existing: bool = True, verbose: bool = False) -> Path:
    csv_path = self._csv_path(dataset_id)
    if csv_path.exists() and skip_existing:
      print(f"[INFO] Dataset {dataset_id} ya existe, omitiendo descarga.")
      if suite_id is not None:
        self._ensure_suite_link(dataset_id, suite_id, suite_name)
      return csv_path

    dataset: OpenMLDataset = openml.datasets.get_dataset(dataset_id, download_data=True, download_qualities=False, download_features_meta_data=True)
    target_col = dataset.default_target_attribute
    X, _, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe")
    df = X.copy()

    # Determinar tipo de variable objetivo
    target_feature = None
    if target_col in X.columns:
      target_index = X.columns.get_loc(target_col)
      target_feature = dataset.features[target_index].data_type

    if verbose:
      print(X.head())
      print(X[target_col])
      print(f"  Target column: {target_col}, type: {target_feature}")

    df.to_csv(csv_path, index=False)

    # TODO: Extraer meta-features (usando df y target_col)
    from src.meta_features import compute_meta_features
    meta_features = compute_meta_features(df, target_col, y_data_type=target_feature)

    evaluations = self._fetch_evaluations(dataset_id)

    entry = {
        "dataset_id": dataset_id,
        "name": dataset.name,
        "description": (dataset.description or "")[:500],
        "csv_path": str(csv_path),
        "n_features": len(attribute_names) if attribute_names else None,
        "n_rows": len(df),
        "target": target_col,
        "target_type": dataset.features[target_col].data_type if target_col in dataset.features else None,
        "suites": {},
        "meta-features": meta_features,
        "evaluations": evaluations,
    }
    if suite_id is not None:
      entry["suites"][str(suite_id)] = suite_name or f"suite_{suite_id}"

    self._registry[str(dataset_id)] = entry
    self._save_registry()
    ok(f"[OK] Dataset {dataset_id} guardado en {csv_path}")
    return csv_path

  def get_dataset(self, dataset_id: int) -> pd.DataFrame:
    csv_path = self._csv_path(dataset_id)
    if csv_path.exists():
      print(f"[INFO] Dataset {dataset_id} encontrado en caché ({csv_path}).")
    else:
      print(f"[INFO] Dataset {dataset_id} no encontrado localmente. Descargando…")
      self.download_dataset(dataset_id, skip_existing=False)
    return pd.read_csv(csv_path)

  def download_suite(self, suite: Union[int, str], skip_existing: bool = True) -> List[int]:
    open_ml_suite = self._resolve_suite(suite)
    suite_id = open_ml_suite["suite_id"]
    suite_name = open_ml_suite["name"]
    dataset_ids = open_ml_suite["data"]
    print(f"[INFO] Suite '{suite_name}' (ID {suite_id}) — {len(dataset_ids)} datasets.")
    downloaded = []
    for did in dataset_ids:
      try:
        self.download_dataset(did, suite_id=suite_id, suite_name=suite_name, skip_existing=skip_existing)
        downloaded.append(did)
      except Exception as exc:
        warn(f"[ERROR] Dataset {did}: {exc}")
    print(f"[OK] Suite '{suite_name}': {len(downloaded)} nuevos datasets descargados.")
    return downloaded

  def info(self, suite: Optional[Union[int, str]] = None) -> None:
    if not self._registry:
      print("[INFO] El registro está vacío.")
      return
    if suite is not None:
      self._info_suite(suite)
    else:
      self._info_global()

  def lookup_suites(self, dataset_id: int) -> Dict[str, str]:
    print(f"[INFO] Buscando suites para dataset {dataset_id}…")
    suites = self._fetch_suites_for_dataset(dataset_id)
    if suites:
      print(f"[OK] Dataset {dataset_id} pertenece a {len(suites)} suite(s):")
      for sid, sname in suites.items():
        print(f"     • {sname} (ID {sid})")
    else:
      print(f"[INFO] No pertenece a ninguna suite registrada.")
    key = str(dataset_id)
    if key in self._registry:
      self._registry[key]["suites"].update(suites)
      self._save_registry()
    return suites

  @property
  def registry(self) -> dict:
    return self._registry

  @property
  def downloaded_ids(self) -> List[int]:
    return [int(k) for k in self._registry.keys()]

  def __repr__(self) -> str:
    return f"OpenMLManager(path='{self.root}', datasets_registrados={len(self._registry)})"

  def _csv_path(self, dataset_id: int) -> Path:
    return self.root / f"dataset_{dataset_id}.csv"

  def _load_registry(self) -> dict:
    if self._registry_path.exists():
      with open(self._registry_path, "r", encoding="utf-8") as f:
        return json.load(f)
    return {}

  def _save_registry(self) -> None:
    clean = self._sanitize_for_json(self._registry)
    with open(self._registry_path, "w", encoding="utf-8") as f:
      json.dump(clean, f, indent=2, ensure_ascii=False)

  @staticmethod
  def _sanitize_for_json(obj):
    import math
    if isinstance(obj, dict):
      return {k: OpenMLManager._sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
      return [OpenMLManager._sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
      if math.isnan(obj) or math.isinf(obj):
        return None
      return obj
    if isinstance(obj, (np.integer, np.int64)):
      return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
      v = float(obj)
      return None if (math.isnan(v) or math.isinf(v)) else v
    return obj

  def _ensure_suite_link(self, dataset_id: int, suite_id: int, suite_name: Optional[str]) -> None:
    key = str(dataset_id)
    if key not in self._registry:
      return
    suites = self._registry[key].setdefault("suites", {})
    sid_str = str(suite_id)
    if sid_str not in suites:
      suites[sid_str] = suite_name or f"suite_{suite_id}"
      self._save_registry()

  def _resolve_suite(self, suite: Union[int, str]) -> dict:
    known_aliases = {"OpenML-CC18": 99, "CC18": 99}
    if isinstance(suite, int):
      return self._get_suite_info(suite)
    if suite.isdigit():
      return self._get_suite_info(int(suite))
    if suite in known_aliases:
      return self._get_suite_info(known_aliases[suite])

    suites_df = openml.study.list_suites(output_format="dataframe")
    if suites_df.empty:
      raise ValueError("No se encontró ninguna suite.")
    id_col = "suite_id" if "suite_id" in suites_df.columns else "id"
    name_col = "name"
    match = suites_df[suites_df[name_col] == suite]
    if match.empty:
      match = suites_df[suites_df[name_col].str.lower() == suite.lower()]
    if match.empty:
      match = suites_df[suites_df[name_col].str.lower().str.startswith(suite.lower())]
    if match.empty:
      match = suites_df[suites_df[name_col].str.lower().str.contains(suite.lower())]
    if match.empty:
      disponibles = suites_df[name_col].head(10).tolist()
      raise ValueError(f"No se encontró suite '{suite}'. Ejemplos: {disponibles}...")
    suite_id = int(match.iloc[0][id_col])
    return self._get_suite_info(suite_id)

  def _get_suite_info(self, suite_id: int) -> dict:
    suite_obj = openml.study.get_suite(suite_id)
    return {"suite_id": suite_id, "name": suite_obj.name, "data": suite_obj.data}

  def _fetch_suites_for_dataset(self, dataset_id: int) -> Dict[str, str]:
    try:
      suites_df = openml.study.list_suites(output_format="dataframe")
      id_col = "suite_id" if "suite_id" in suites_df.columns else "id"
      found = {}
      for _, row in suites_df.iterrows():
        sid = int(row[id_col])
        try:
          suite_obj = openml.study.get_suite(sid)
          if dataset_id in suite_obj.data:
            found[str(sid)] = suite_obj.name
        except Exception:
          continue
      return found
    except Exception as exc:
      warn(f"[WARN] No se pudieron buscar suites: {exc}")
      return {}

  def _fetch_evaluations(self, dataset_id: int) -> List[dict]:
    import warnings

    try:
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tasks_df = openml.tasks.list_tasks(data_id=dataset_id, output_format="dataframe")
      if tasks_df.empty:
        return []
      if "tid" in tasks_df.columns:
        task_ids = tasks_df["tid"].tolist()
      else:
        task_ids = tasks_df.index.tolist()
      if not task_ids:
        return []
      evals_df = openml.evaluations.list_evaluations(
          function="predictive_accuracy",
          tasks=task_ids,
          output_format="dataframe",
          per_fold=False,
      )
      if evals_df.empty:
        return []
      evals_df = evals_df.sort_values("value", ascending=False).head(20)
      results = []
      for _, row in evals_df.iterrows():
        results.append({
            "flow_name": str(row.get("flow_name", "unknown")),
            "flow_id": int(row.get("flow_id", -1)),
            "setup_id": int(row.get("setup_id", -1)),
            "task_id": int(row.get("task_id", -1)),
            "metric": "predictive_accuracy",
            "value": round(float(row.get("value", 0.0)), 6),
        })
      return results
    except Exception as exc:
      warn(f"[WARN] Evaluaciones: {exc}")
      return []

  def _info_global(self) -> None:
    total = len(self._registry)
    print(f"\n{'═' * 60}\n  REGISTRO OPENML — {total} dataset(s)\n  Raíz: {self.root}\n{'═' * 60}")
    for did, entry in self._registry.items():
      print(f"\n▶ {entry.get('name', did)} [ID {did}]")
      print(f"  Filas: {entry.get('n_rows', '?')}, Columnas: {entry.get('n_features', '?')}, Target: {entry.get('target', '?')}")
      mf = entry.get('meta-features', {})
      if mf:
        s = mf.get('simple', {})
        print(f"  Meta: instancias={s.get('n_instances')}, features={s.get('n_features')}, clases={s.get('n_classes')}, missing={s.get('n_missing_values')}")
        lm = mf.get('landmarkers', {})
        if '1NN_accuracy' in lm:
          print(f"  Landmarkers: 1NN={lm.get('1NN_accuracy', 0):.3f}, Tree={lm.get('tree_accuracy', 0):.3f}")
      evals = entry.get('evaluations', [])
      if evals:
        print("  Evaluaciones top:")
        self._print_evaluations(evals[:3], indent=4)
    print(f"\n{'═' * 60}\n")

  def _info_suite(self, suite: Union[int, str]) -> None:
    oml = self._resolve_suite(suite)
    suite_id, suite_name = str(oml["suite_id"]), oml["name"]
    all_ids = set(str(d) for d in oml["data"])
    downloaded_ids = set(self._registry.keys())
    in_reg = all_ids & downloaded_ids
    pending = all_ids - downloaded_ids
    print(f"\n{'═' * 60}\n  SUITE: {suite_name} (ID {suite_id})\n  Total: {len(all_ids)} | Descargados: {len(in_reg)} | Pendientes: {len(pending)}\n{'═' * 60}")
    if in_reg:
      print("\n  ✅ DESCARGADOS")
      for did in sorted(in_reg, key=int):
        entry = self._registry[did]
        print(f"    • {entry.get('name', did)} [{did}]")
    if pending:
      print("\n  ⏳ PENDIENTES")
      for did in sorted(pending, key=int):
        print(f"    • Dataset ID {did}")
    print(f"\n{'═' * 60}\n")

  @staticmethod
  def _print_evaluations(evals: List[dict], indent: int = 0) -> None:
    pad = " " * indent
    for e in evals:
      print(f"{pad}{e.get('flow_name', '?')[:40]:<40} acc: {e.get('value', 0):.4f}")

  def get_dataset_info(self, dataset_id: int, download_if_missing: bool = True) -> Dict:
    "Devuelve todas las características de un dataset desde el registro"

    key = str(dataset_id)
    if key not in self._registry:
      if download_if_missing:
        print(f"[INFO] Dataset {dataset_id} no está en el registro. Descargando ...")
        self.download_dataset(dataset_id)
      else:
        raise KeyError(f"Dataset {dataset_id} no encontrado en registry.json")
    return self._registry[key].copy()
