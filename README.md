# Tarea1IA

# Comparación de Modelos de Aprendizaje Supervisado

Este proyecto implementa y compara tres algoritmos de clasificación diferentes para la detección de cáncer de mama utilizando el conjunto de datos Wisconsin Breast Cancer incluido en scikit-learn.

## Descripción

El proyecto consiste en tres modelos de machine learning aplicados al mismo conjunto de datos de cáncer de mama:

1. 

SVM_cancermama.py

 - Implementa un clasificador de Máquinas de Vectores de Soporte (SVM)
2. 

RLOG_cancermama.py

 - Implementa un clasificador de Regresión Logística
3. 

RFOR_cancermama.py

 - Implementa un clasificador de Random Forest

## Estructura del Proyecto

Cada archivo implementa un flujo de trabajo similar:

1. Carga del conjunto de datos de cáncer de mama
2. Preprocesamiento:
   - División en conjuntos de entrenamiento (70%) y prueba (30%)
   - Normalización de características con StandardScaler
3. Entrenamiento del modelo específico
4. Evaluación:
   - Cálculo de precisión
   - Generación de informe de clasificación
   - Matriz de confusión
   - Visualización gráfica de resultados

## Modelos Implementados

### SVM (Support Vector Machine)
- Utiliza un kernel lineal para clasificar los datos
- Apropiado para problemas con alta dimensionalidad

### Regresión Logística
- Implementa un clasificador probabilístico lineal
- Configurado con un mayor número de iteraciones (1000)

### Random Forest
- Utiliza un conjunto de árboles de decisión
- Ofrece buenos resultados sin necesidad de ajustar muchos hiperparámetros

## Métricas Analizadas

Todos los modelos son evaluados usando:
- Precisión (accuracy)
- Informe de clasificación (precision, recall, f1-score)
- Matrices de confusión visualizadas mediante mapas de calor

## Requisitos

```
numpy
pandas
scikit-learn
matplotlib
seaborn
```

## Ejecución

Para ejecutar cualquiera de los modelos:

```sh
python SVM_cancermama.py
python RLOG_cancermama.py
python RFOR_cancermama.py
```