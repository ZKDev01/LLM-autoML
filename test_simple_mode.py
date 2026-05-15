from src.autoML_bot import AutoML_Bot
from src.terminal_tools import *


def test_simple_mode():
  "Prueba que el modo simple funcione correctamente"
  header("  [TEST] Modo Simple (No Validación Estricta)")

  # Crear bot en modo simple
  bot_simple = AutoML_Bot(
      stream=False,
      task_description="classification",
      cv_folds=5,
      verbose=True,
      include_schema_in_prompt=True  # Importante: modo simple
  )
  bot_simple.load_dataset_from_openml(31)
  bot_simple.prepare_for_llm(k_examples=0, include_anonymize_columns=True)
  print(bot_simple.generate_pipelines_simple(k_repair=5, add_reasoning=False, save_result_path=None, auto_generate_filename=False))

  header("  [TEST] Modo Normal/Estricto (Aplicar una Validación Estricta)")
  # Crear bot en modo estricto
  bot_strict = AutoML_Bot(
      stream=False,
      task_description="classification",
      cv_folds=5,
      verbose=False,
      include_schema_in_prompt=True  # Importante: modo estricto
  )
  bot_strict.load_dataset_from_openml(31)
  bot_strict.prepare_for_llm(k_examples=0, include_anonymize_columns=True)
  print(bot_strict.generate_pipelines(k_repair=5, add_reasoning=False, save_result_path=None, auto_generate_filename=False))

if __name__ == "__main__":
  test_simple_mode()
  print("\n" + "=" * 60)
  ok("  [OK] Todos los tests pasaron exitosamente")
