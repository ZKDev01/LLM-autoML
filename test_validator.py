from sklearn.datasets import make_classification
from src.pipeline_generator import parse_and_build, evaluate_pipeline

# Generar datos de prueba
X, y = make_classification(n_samples=300, n_features=20)
print(f"Datos: X={X.shape}, y={y.shape}")

# Lista de respuestas simuladas (como las que podría devolver un LLM)
test_responses = [
    # 1. Respuesta perfecta
    """
    {
      "steps": [
        {
          "name": "scaler",
          "component": "StandardScaler",
          "hyperparameters": {"with_mean": true, "with_std": true}
        },
        {
          "name": "fs",
          "component": "SelectKBest",
          "hyperparameters": {"k": 10}
        },
        {
          "name": "clf",
          "component": "RandomForestClassifier",
          "hyperparameters": {"n_estimators": 100, "max_depth": 5}
        }
      ]
    }
    """,
    # 2. Hiperparámetro fuera de rango (C de LogisticRegression)
    """
    {
      "steps": [
        {"name": "scaler", "component": "StandardScaler", "hyperparameters": {}},
        {
          "name": "clf",
          "component": "LogisticRegression",
          "hyperparameters": {"C": 2000, "max_iter": 100}
        }
      ]
    }
    """,
    # 3. Componente no permitido
    """
    {
      "steps": [
        {"name": "bad", "component": "XGBClassifier", "hyperparameters": {}}
      ]
    }
    """,
    # 4. ColumnTransformer con sub‑transformador válido
    """
    {
      "steps": [
        {
          "name": "col_trans",
          "component": "ColumnTransformer",
          "hyperparameters": {
            "transformers": [
              {
                "name": "num",
                "transformer": "StandardScaler",
                "columns": [0, 1, 2],
                "transformer_hyperparameters": {"with_mean": true}
              }
            ],
            "remainder": "drop"
          }
        },
        {
          "name": "clf",
          "component": "KNeighborsClassifier",
          "hyperparameters": {"n_neighbors": 5}
        }
      ]
    }
    """,
    # 5. Varios pipelines en un mismo texto (debe extraer el primero bueno)
    """
    Aquí hay un pipeline inicial que falla por un componente incorrecto:
    {"steps":[{"name":"a","component":"FakeComponent","hyperparameters":{}}]}
    Pero luego hay uno correcto:
    ```json
    {
      "steps": [
        {
          "name": "scaler",
          "component": "StandardScaler",
          "hyperparameters": {"with_mean": true, "with_std": false}
        },
        {
          "name": "clf",
          "component": "DecisionTreeClassifier",
          "hyperparameters": {"max_depth": 3}
        }
      ]
    }
    ```
    """,
    # 6. JSON malformado (cierre incorrecto) y luego uno bueno en mismo texto
    """
    Primer intento (roto):
    {"steps": [{"name": "x", "component": "PCA", "hyperparameters": {"n_components": 5}]
    Pero el siguiente sí es correcto:
    {
      "steps": [
        {"name": "clf", "component": "GaussianNB", "hyperparameters": {}}
      ]
    }
    """,
    # 7. Hiperparámetro con tipo incorrecto (string en lugar de int)
    """
    {
      "steps": [
        {
          "name": "fs",
          "component": "SelectKBest",
          "hyperparameters": {"k": "muchos"}
        }
      ]
    }
    """,
    # 8. Pipeline vacío (sin steps)
    """
    {"steps": []}
    """,
    # 9. Texto sin ningún JSON
    "No se me ocurre ningún pipeline hoy, disculpa.",
    # 10. Múltiples bloques de JSON malformados (falta una comilla, etc.) y ninguno válido
    """
    Intento 1: {"steps": [{"name": "sc", "component": "StandardScaler", "hyperparameters": {"with_mean": true, "with_std": true}}]
    Intento 2: {"steps": [{"name": "clf", "component": "LogisticRegression", "hyperparameters": {"C": 0.1 "max_iter": 100}}]}
    """,
    # 11. JSON correcto pero dentro de un comentario más grande y con caracteres extra que lo invalidan
    """
    Aquí tienes un pipeline: steps: [{ "name": "scaler", "component": "StandardScaler" }]. Pero no está completo
    """,
    # 12. Especificación válida según Pydantic, pero falla al construir porque se pasa un valor no soportado por el constructor de sklearn
    """
    {
      "steps": [
        {
          "name": "clf",
          "component": "LogisticRegression",
          "hyperparameters": {"C": 1.0, "solver": "invalid_solver", "max_iter": 100}
        }
      ]
    }
    """,
    # 13. ColumnTransform con columnas que no existen
    """
    {
      "steps": [
        {
          "name": "col_trans",
          "component": "ColumnTransformer",
          "hyperparameters": {
            "transformers": [
              {
                "name": "scaler",
                "transformer": "StandardScaler",
                "columns": [50, 51, 52],   # índices fuera de rango (X tiene 20 columnas)
                "transformer_hyperparameters": {}
              }
            ],
            "remainder": "drop"
          }
        },
        {
          "name": "clf",
          "component": "LogisticRegression",
          "hyperparameters": {}
        }
      ]
    }
    """,
    # Error: cross_val_score (sklearn internamente espera un clasificador)
    """
    {
      "steps": [
        {
          "name": "scaler",
          "component": "StandardScaler",
          "hyperparameters": {"with_mean": true}
        },
        {
          "name": "fs",
          "component": "SelectKBest",
          "hyperparameters": {"k": 10}
        }
      ]
    }
    """,
    # Pipeline con un clasificador que requiere que todas las muestras tengan pesos positivos. Error en fit.
    """
    {
      "steps": [
        {
          "name": "scaler",
          "component": "StandardScaler",
          "hyperparameters": {}
        },
        {
          "name": "clf",
          "component": "MultinomialNB",
          "hyperparameters": {}
        }
      ]
    }
    """,
    #
    """
    {
      "steps": [
        {
          "name": "scaler",
          "component": "StandardScaler",
          "hyperparameters": {}
        },
        {
          "name": "clf",
          "component": "SVC",
          "hyperparameters": {"kernel": "rbf"}
        }
      ]
    }
    """,
    #
    """
    {
      "steps": [
        {
          "name": "imp",
          "component": "SimpleImputer",
          "hyperparameters": {"strategy": "constant"}
        },
        {
          "name": "clf",
          "component": "LogisticRegression",
          "hyperparameters": {}
        }
      ]
    }
    """
]

# Ejecutamos el flujo completo para cada respuesta
for i, response in enumerate(test_responses, start=1):
  print(f"\n{'=' * 60}")
  print(f"📨 Test {i}:")
  print(f"{response}")
  print("-" * 60)
  parse_result = parse_and_build(response)
  print(parse_result.to_feedback())
  if parse_result.success and parse_result.pipeline is not None:
    eval_result = evaluate_pipeline(parse_result.pipeline, X, y)
    print(eval_result.to_feedback(add_warning=False))
  else:
    print("No se pudo construir el pipeline; se omite la evaluación.")
