import sys
import json
from src.terminal_tools import *

AVAILABLE_MODELS = {
    "1": ("gemini-3-flash-preview", "Gemini 3 Flash Preview"),
    "2": ("gpt-oss:120b", "GPT-OSS 120B"),
    "3": ("gpt-oss:20b", "GPT-OSS 20B"),
    "4": ("kimi-k2-thinking", "Kimi K2 Thinking"),
    "5": ("mistral-large-3:675b", "Mistral Large 3"),
    "6": ("qwen3-coder-next", "Qwen3 Coder Next"),
    "7": ("qwen3-coder:480b", "Qwen3 Coder 480B"),
}

def test_generate_pipelines() -> None:
  from src.autoML_bot import AutoML_Bot

  bot = AutoML_Bot(
      task_description="classification",
      cv_folds=5,
      verbose=True
  )
  # Cargar dataset desde OpenML
  success, msg = bot.load_dataset_from_openml(dataset_id=31)
  if not success:
    fail(f"[ERROR] Error al cargar dataset: {msg}")
    return

  # Preparar para el LLM (anonimiza columnas, meta-features)
  success, msg = bot.prepare_for_llm(K=0, include_anonymize_columns=True)
  if not success:
    fail(f"[ERROR] Error al preparar para LLM: {msg}")
    return

  # Generar Pipeline (hasta k_repair intentos de reparación)
  try:
    pipeline, reasoning, metrics, config = bot.generate_pipelines(K=3, print_chat=True)
  except Exception as e:
    fail(f"[ERROR] No se pudo generar ningún pipeline: {e}")
    return

  # Mostrar resultados
  print("=" * 20 + " [Configuración JSON] " + "=" * 20)
  print(json.dumps(config, indent=2, ensure_ascii=False))
  print("=" * 20 + " [Pipeline scikit-learn] " + "=" * 20)
  print(pipeline)
  print("=" * 20 + " [Métricas] " + "=" * 20)
  for k, v in metrics.items():
    print(f"   {k}: {v:.4f}")

def select_model() -> str:
  "Permite seleccionar un modelo de forma dinámica"
  header("Selección de Modelo Ollama Cloud")

  print(f"\n{CYAN}Modelos disponibles:{RESET}\n")
  for key, (model_id, model_name) in AVAILABLE_MODELS.items():
    print(f"  {CYAN}{key}.{RESET} {model_name}")

  print(f"\n  {CYAN}0.{RESET} Usar defecto (gpt-oss:120b)")

  while True:
    choice = input(f"\n{CYAN}Selecciona modelo (número): {RESET}").strip()

    if choice == "0":
      model_id = "gpt-oss:120b"
      ok(f"Usando modelo por defecto: {model_id}")
      return model_id

    if choice in AVAILABLE_MODELS:
      model_id = AVAILABLE_MODELS[choice][0]
      ok(f"Modelo seleccionado: {model_id}")
      return model_id

    warn("Opción inválida. Intenta de nuevo.")

def show_menu() -> None:
  print(f"\n{BOLD}{'═' * 70}{RESET}")
  print(f"{BOLD}  LLM-autoML - Panel de Control{RESET}")
  print(f"{BOLD}{'═' * 70}{RESET}\n")

  print("Selecciona un test para ejecutar:\n")
  print(f"  {CYAN}1.{RESET} Generación de Pipelines de Sklearn vía LLM")
  print(f"\n{BOLD}{'─' * 70}{RESET}")


def main() -> None:
  # Loop principal del CLI
  while True:
    show_menu()
    try:
      option = input(f"{CYAN}Option: {RESET}").strip()

      match option:
        case "1":
          test_generate_pipelines()

    except KeyboardInterrupt:
      print(f"\n{RED}Exiting the CLI{RESET}")
      break
    except Exception as e:
      print(f"{RED}An error occurred: {e}{RESET}")

if __name__ == "__main__":
  main()
