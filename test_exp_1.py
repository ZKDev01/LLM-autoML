from src.autoML_bot import AutoML_Bot

# 1. Crear una instancia del bot (elige el modelo y host de Ollama)
bot = AutoML_Bot(
    # model="deepseek-r1:1.5b",
    stream=True,
    task_description="classification",
    cv_folds=5,
    verbose=True
)

# 2. Cargar un dataset desde OpenML (por ejemplo, el dataset iris, ID 61)
bot.load_dataset_from_openml(dataset_id=61)

# 3. Preparar la información del dataset para el LLM (anonimiza columnas, genera meta‑features)
bot.prepare_for_llm(k_examples=0, include_anonymize_columns=True)

# 4. Generar un pipeline (máximo 3 intentos de reparación si falla)
pipeline, razonamiento, metricas, configuracion = bot.generate_pipelines(
    k_repair=3,
    add_reasoning=True,          # pedir al LLM que explique su elección
    print_chat=True,             # ver la respuesta cruda del LLM
    save_result_path="test_1.json"  # guardar resultados en archivo
)

# 5. Mostrar resultados
print("Pipeline final:", pipeline)
print("Métricas:", metricas)
print("Razonamiento:", razonamiento[:500] + "...")
