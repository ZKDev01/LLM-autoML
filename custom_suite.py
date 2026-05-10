import openml
from collections import defaultdict

# Diccionario de suites a analizar
SUITES = {
    "OpenML100": "14",
    "amlb-classification-all": "amlb-classification-all",  # se puede usar el alias
    "OpenML-CC18": "99",
    "Suite 271": "271",
}

def get_suite_info(suite_id):
  """Obtiene nombre, ID y lista de tareas de una suite."""
  suite = openml.study.get_suite(suite_id)
  return {"name": suite.name, "id": suite.id, "tasks": suite.tasks}

# Obtener info de todas las suites
suites_info = {}
for name, sid in SUITES.items():
  try:
    suites_info[name] = get_suite_info(sid)
    print(f"Suite '{name}': {len(suites_info[name]['tasks'])} tareas")
  except Exception as e:
    print(f"Error con suite '{name}': {e}")

# Para cada suite, extraer los dataset_id únicos y analizarlos
dataset_ids = set()
for suite_name, info in suites_info.items():
  print(f"\n===== Analizando {suite_name} =====")
  for task_id in info['tasks']:
    try:
      task = openml.tasks.get_task(task_id)
      dataset_ids.add(task.dataset_id)
    except Exception as e:
      print(f"Error en tarea {task_id}: {e}")
  # Aquí puedes continuar con el análisis de meta-features
print(f"Datasets únicos: {len(dataset_ids)}")
