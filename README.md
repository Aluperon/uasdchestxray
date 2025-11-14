<<<<<<< HEAD
# INF-8239: Proyecto Final - Clasificación de Neumonía en Radiografías de Tórax

## Maestría de Ciencia de Datos y Inteligencia Artificial - UASD

| Atributo | Detalle |
| :--- | :--- |
| **Asignatura** | Ciencia de Datos II (INF-8239) |
| **Estudiante** | Alvin Luperon |
| **Dataset** | Chest X-ray Images (Hospital Infantil de Guangzhou) |
| **Finalidad** | Clasificación binaria (Normal vs. Neumonía) en radiografías, utilizando Deep Learning y Transfer Learning. |
| **Modelo Base** | **DenseNet121** (Utilizado en ambos escenarios) |
| **Tecnologías** | PyTorch, Torchvision, Matplotlib, Seaborn, Scikit-learn, Pandas. |

---

## 💡 Objetivo del Proyecto

Demostrar la mejora en el desempeño de modelos de Deep Learning a través de la optimización de hiperparámetros y la aplicación de técnicas avanzadas de Data Augmentation y Fine-Tuning, en comparación con un escenario de entrenamiento básico, **utilizando la misma arquitectura DenseNet121**.

## 🚀 Escenarios de Entrenamiento (Ambos con DenseNet121)

El proyecto compara dos estrategias principales que utilizan la arquitectura DenseNet121 pre-entrenada:

1.  **Escenario Básico (Fast Feature Extractor)**:
    * **Modelo:** DenseNet121.
    * **Estrategia:** Capas pre-entrenadas **Congeladas** (solo se entrena la capa clasificadora final).
    * **Data Augmentation:** Básico (Volteo, Rotación $10^\circ$).
    * **Hiperparámetros:** Tasa de Aprendizaje (LR) alta (`0.001`), `Epochs = 3`.

2.  **Escenario Optimizado (Fine-Tuning Avanzado)**:
    * **Modelo:** DenseNet121.
    * **Estrategia:** **Fine-Tuning** (Todas las capas son ajustadas con LR muy baja).
    * **Data Augmentation:** **Avanzado** (Incluye `ColorJitter` y `RandomAffine`).
    * **Hiperparámetros:** Tasa de Aprendizaje (LR) muy baja (`0.0001`), `Epochs = 25`.

## ⚙️ Estructura del Proyecto y Modularidad

El proyecto está diseñado con un enfoque modular, con código bien organizado en los siguientes archivos:

| Archivo | Contenido |
| :--- | :--- |
| **`main.ipynb`** | Notebook principal de orquestación, configuración y ejecución de los experimentos. |
| **`data_loader.py`** | Lógica de carga de datos, transformaciones y Data Augmentation Básico/Avanzado. |
| **`model_builder.py`** | Lógica para construir y modificar la arquitectura DenseNet121 (congelación/descongelación). |
| **`train.py`** | Bucle de entrenamiento, validación y registro de métricas en CSV. |
| **`evaluation.py`** | Evaluación final, Matriz de Confusión y Reporte de Clasificación. |
| **`utils.py`** | Funciones auxiliares para la comparación visual de curvas de entrenamiento. |
| **`requirements.txt`** | Listado de todas las librerías de Python necesarias para la ejecución. |

## 💻 Cómo Usar

1.  **Instalar Dependencias:** Instale todas las librerías necesarias utilizando el archivo de requerimientos:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configuración de Rutas:** Modifique las variables de ruta (`train_dir`, `val_dir`, `test_dir`, `base_path`) en la primera celda de `main.ipynb` para que apunten a la ubicación de su dataset.
3.  **Ejecución:** Ejecute las celdas de `main.ipynb` en orden, asegurándose de tener una **GPU** activada.
4.  **Selección de Modo:** Al llegar a la celda de orquestación, elija **E** (Entrenar) o **C** (Cargar modelos).
=======
# uasdchestxray
Ciencia de Datos II
>>>>>>> a4e1465851ec8f71a9f4ce35ea2d61204af3615f
