from src.autoML_bot import AutoML_Bot

# 1. Configurar el bot (ajusta modelo y host según tu Ollama)
bot = AutoML_Bot(
    stream=True,
    task_description="classification",
    cv_folds=5,
    verbose=True
)

# 2. Cargar un dataset de prueba (iris, ID 61 en OpenML)
bot.load_dataset_from_openml(dataset_id=61)

# 3. Preparar la descripción para el LLM
bot.prepare_for_llm(k_examples=5, include_anonymize_columns=True)

# 4. Ejecutar optimización iterativa
best_pipeline, final_reasoning, best_metrics = bot.generate_pipelines_with_optimization(
    target_metric='accuracy_mean',   # métrica a optimizar
    max_iterations=10,               # cuántas iteraciones de mejora
    max_history_size=5,              # cuántos pipelines pasados recordar
    k_repair=3,                      # intentos de reparación por iteración
    add_reasoning=True,              # que explique sus decisiones
    print_chat=True,                 # muestra la respuesta cruda del LLM
    save_result_path="result_reasoning.json"
)

# 5. Mostrar resultados finales
print("\n===== MEJOR PIPELINE =====")
print(best_pipeline)
print("\n===== MÉTRICAS =====")
for k, v in best_metrics.items():
  print(f"{k}: {v:.4f}")
print("\n===== RAZONAMIENTO FINAL =====")
print(final_reasoning[:1000])  # primeros 1000 caracteres
