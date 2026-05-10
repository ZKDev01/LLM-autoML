import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

def generate_synthetic_meta_features(n_datasets: int = 150, seed: int = 42) -> Tuple[List[int], np.ndarray]:
  """
  Crea vectores de meta‑features realistas para 'n_datasets'.
  Devuelve una lista de IDs y una matriz (n_datasets, n_features)
  con los siguientes atributos (por columna):
      0: log10(n_instances)
      1: n_features
      2: n_classes
      3: missing_rate
      4: pct_numeric
      5: imbalance_ratio (mayoría / minoría)
      6: mean_correlation
  """
  rng = np.random.default_rng(seed)

  # Distribuciones con rangos típicos de OpenML
  n_instances = np.round(10 ** rng.uniform(1.5, 5.5, n_datasets)).astype(int)  # 30 .. 300k
  n_features = rng.choice([5, 10, 20, 50, 100, 200, 500, 1000], size=n_datasets)
  n_classes = rng.choice([2, 3, 4, 5, 6, 8, 10, 20, 50], size=n_datasets)
  missing_rate = rng.beta(1.5, 10, n_datasets) * 0.5           # mayormente bajo
  pct_numeric = rng.beta(5, 2, n_datasets)                    # sesgado hacia numérico
  imbalance_ratio = rng.exponential(5, n_datasets) + 1          # 1 .. ~30
  mean_correlation = rng.normal(0.2, 0.15, n_datasets).clip(-0.5, 0.9)

  # Asegurar que n_features < n_instances (razonable)
  for i in range(n_datasets):
    if n_features[i] >= n_instances[i]:
      n_features[i] = max(2, n_instances[i] // 2)

  features = np.column_stack([
      np.log10(n_instances),   # mejor distribución normalizada
      n_features,
      n_classes,
      missing_rate,
      pct_numeric,
      imbalance_ratio,
      mean_correlation
  ])

  ids = list(range(n_datasets))
  return ids, features

# ----------------------------------------------------------------------
# 2. Clase para manejar el dataset de meta‑features y sus particiones
# ----------------------------------------------------------------------
class MetaFeatureDataset:
  def __init__(self, ids: List[int], features: np.ndarray, scale: bool = True):
    """
    Args:
        ids: identificadores de cada dataset
        features: matriz (n, d) de meta‑features
        scale: si se deben estandarizar (recomendado)
    """
    self.ids = np.array(ids)
    self.raw_features = features.copy()
    if scale:
      self.scaler = StandardScaler()
      self.features = self.scaler.fit_transform(features)
    else:
      self.features = features
    self.n = len(ids)

  def random_split(self, test_size: float = 0.3,
                   stratify_by: str = 'n_classes',
                   seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Divide los IDs en train (base de conocimiento) y test de forma
    estratificada según una meta‑feature categórica (binned).
    """
    rng = np.random.default_rng(seed)

    # Crear estratos según la variable elegida
    if stratify_by == 'n_classes':
      # Bins: 2, 3-5, 6-10, >10
      n_classes = self.raw_features[:, 2]
      bins = [0, 2, 5, 10, np.inf]
      labels = [0, 1, 2, 3]
      strata = np.digitize(n_classes, bins) - 1
    elif stratify_by == 'n_features':
      n_feats = self.raw_features[:, 1]
      bins = [0, 10, 50, 200, np.inf]
      labels = [0, 1, 2, 3]
      strata = np.digitize(n_feats, bins) - 1
    elif stratify_by == 'n_instances':
      n_inst = self.raw_features[:, 0]   # log10
      bins = [-np.inf, 2, 3, 4, np.inf]  # <100, 100-1k, 1k-10k, >10k
      strata = np.digitize(n_inst, bins) - 1
    else:
      strata = np.zeros(self.n, dtype=int)  # sin estratificar

    train_ids, test_ids = [], []
    for s in np.unique(strata):
      idx = np.where(strata == s)[0]
      n_test = max(1, int(len(idx) * test_size))
      test_idx = rng.choice(idx, size=n_test, replace=False)
      train_idx = np.setdiff1d(idx, test_idx)
      train_ids.extend(self.ids[train_idx])
      test_ids.extend(self.ids[test_idx])

    rng.shuffle(train_ids)
    rng.shuffle(test_ids)
    return np.array(train_ids), np.array(test_ids)

# ----------------------------------------------------------------------
# 3. Métricas de calidad de una partición
# ----------------------------------------------------------------------
def evaluate_split_quality(dataset: MetaFeatureDataset,
                           train_ids: np.ndarray,
                           test_ids: np.ndarray,
                           k: int = 5) -> Dict[str, float]:
  """
  Calcula varias métricas de calidad para un split dado.
  train_ids / test_ids son los identificadores de la base de conocimiento y test.
  """
  train_mask = np.isin(dataset.ids, train_ids)
  test_mask = np.isin(dataset.ids, test_ids)
  train_feats = dataset.features[train_mask]
  test_feats = dataset.features[test_mask]

  results = {}

  # 1. Distancia media al vecino más cercano en train (para cada test)
  if len(train_feats) > 0 and len(test_feats) > 0:
    dist_matrix = cdist(test_feats, train_feats, metric='euclidean')
    min_dists = dist_matrix.min(axis=1)
    results['mean_min_dist'] = float(np.mean(min_dists))
    results['max_min_dist'] = float(np.max(min_dists))
    results['std_min_dist'] = float(np.std(min_dists))

    # 2. Cobertura: fracción de test con distancia menor que cierto umbral
    #    (umbral = percentil 75 de las distancias intra-train)
    if len(train_feats) > 1:
      intra_dists = cdist(train_feats, train_feats, metric='euclidean')
      threshold = np.percentile(intra_dists, 75)
      covered = min_dists <= threshold
      results['coverage_75p'] = float(np.mean(covered))
  else:
    results['mean_min_dist'] = float('nan')
    results['coverage_75p'] = float('nan')

  # 3. Similitud de distribuciones (KS test en cada dimensión)
  ks_stats = []
  for j in range(dataset.raw_features.shape[1]):
    train_col = dataset.raw_features[train_mask, j]
    test_col = dataset.raw_features[test_mask, j]
    if len(train_col) > 0 and len(test_col) > 0:
      stat, _ = ks_2samp(train_col, test_col)
      ks_stats.append(stat)
  results['mean_ks_stat'] = np.mean(ks_stats) if ks_stats else float('nan')

  # 4. Distancia de Wasserstein (usando la primera dimensión como ejemplo)
  #    Podemos calcular la media de las distancias de Wasserstein normalizadas
  ws_dists = []
  for j in range(dataset.raw_features.shape[1]):
    train_col = dataset.raw_features[train_mask, j]
    test_col = dataset.raw_features[test_mask, j]
    if len(train_col) > 0 and len(test_col) > 0:
      # Normalizar el rango común para hacerlo comparable
      col_all = np.concatenate([train_col, test_col])
      col_range = np.max(col_all) - np.min(col_all) if np.max(col_all) > np.min(col_all) else 1.0
      ws = wasserstein_distance(train_col, test_col) / col_range
      ws_dists.append(ws)
  results['mean_wasserstein'] = np.mean(ws_dists) if ws_dists else float('nan')

  # 5. Score combinado (menor es mejor -> minimiza distancia y KS, maximiza cobertura)
  #    Invertimos cobertura para que sea un costo
  coverage_cost = 1 - results.get('coverage_75p', 0)
  results['combined_score'] = (
      0.4 * results['mean_min_dist'] +
      0.3 * results['mean_ks_stat'] +
      0.3 * coverage_cost
  )
  return results

# ----------------------------------------------------------------------
# 4. Selección del mejor split entre varios
# ----------------------------------------------------------------------
def select_best_split(dataset: MetaFeatureDataset, test_size: float = 0.3,
                      stratify_by: str = 'n_classes',
                      n_trials: int = 50,
                      metric: str = 'combined_score',
                      seed: int = None) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
  """
  Genera 'n_trials' particiones y devuelve los IDs y las métricas
  del split que optimiza la métrica escogida (minimizando 'combined_score').
  """
  best_train, best_test = None, None
  best_score = float('inf')
  all_results = []
  rng = np.random.default_rng(seed)

  for trial in range(n_trials):
    train_ids, test_ids = dataset.random_split(test_size=test_size,
                                               stratify_by=stratify_by,
                                               seed=rng.integers(0, 1e9))
    metrics = evaluate_split_quality(dataset, train_ids, test_ids)
    metrics['trial'] = trial
    all_results.append(metrics)

    if metrics[metric] < best_score:
      best_score = metrics[metric]
      best_train = train_ids
      best_test = test_ids

  return best_train, best_test, all_results

# ----------------------------------------------------------------------
# 5. Ejemplo de uso
# ----------------------------------------------------------------------
if __name__ == "__main__":
  # Generar datos sintéticos
  ids, raw_features = generate_synthetic_meta_features(n_datasets=150, seed=0)

  # Construir el objeto con escalado
  dataset = MetaFeatureDataset(ids, raw_features, scale=True)

  # Mejor split según combined_score (minimizando)
  best_train, best_test, results_list = select_best_split(
      dataset, test_size=0.3, stratify_by='n_classes', n_trials=50,
      metric='combined_score', seed=42
  )

  # Mostrar métricas del split ganador
  best_metrics = evaluate_split_quality(dataset, best_train, best_test)
  print("Mejor split encontrado:")
  for k, v in best_metrics.items():
    print(f"  {k:>20s}: {v:.4f}")

  # También puedes inspeccionar todos los intentos
  print(f"\nSe evaluaron {len(results_list)} splits.")
  print("Ejemplo de primeras 3 métricas de distintos splits:")
  for i, m in enumerate(results_list[:3]):
    print(f"  Split {m['trial']}: mean_min_dist={m['mean_min_dist']:.3f}, "
          f"coverage={m['coverage_75p']:.3f}, combined={m['combined_score']:.3f}")
