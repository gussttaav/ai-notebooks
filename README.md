# 🧠 Notebooks de Inteligencia Artificial

---

## 📑 Índice

### 🔹 Modelado Clásico en Machine Learning

1. [Guía de Clasificación: Evaluación de Riesgo Crediticio](#guía-de-clasificación-evaluación-de-riesgo-crediticio)

2. [Guía de Regresión: Predicción de Precios de Propiedades](#guía-de-regresión-predicción-de-precios-de-propiedades)

3. [Guía de Clustering: Segmentación de Clientes Bancarios](#guía-de-clustering-segmentación-de-clientes-bancarios)

### 🔹 Fundamentos Matemáticos de Redes Neuronales

4. [MLP explicado como un Proceso de Ajuste de Curvas](#mlp-explicado-como-un-proceso-de-ajuste-de-curvas)

### 🔹 Procesamiento de Lenguaje Natural

5. [Fine-tuning de BETO para sentimiento en español](#fine-tuning-de-beto-para-sentimiento-en-español)

---

## 📂 Contenido

### Modelado Clásico en Machine Learning

#### Guía de Clasificación: Evaluación de Riesgo Crediticio
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gussttaav/ai-notebooks/blob/main/notebooks/ml-clasification-project-example-guide.ipynb)
[![Ver Notebook](https://img.shields.io/badge/📖-View-blue)](https://github.com/gussttaav/ai-notebooks/blob/main/notebooks/ml-clasification-project-example-guide.ipynb)

Este notebook desarrolla un caso de **clasificación de riesgo crediticio** a partir de un dataset sintético. Se muestra el flujo completo de trabajo en Machine Learning: desde el análisis exploratorio de datos hasta el preprocesamiento con `ColumnTransformer` y `Pipeline`, pasando por la comparación entre diferentes algoritmos de clasificación como Random Forest, Gradient Boosting y SVM. Además, se incluye la optimización de hiperparámetros mediante `GridSearchCV` y una evaluación detallada de métricas para seleccionar el modelo con mejor rendimiento.  

---

#### Guía de Regresión: Predicción de Precios de Propiedades
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gussttaav/ai-notebooks/blob/main/notebooks/ml-regression-project-example-guide.ipynb)
[![Ver Notebook](https://img.shields.io/badge/📖-View-blue)](https://github.com/gussttaav/ai-notebooks/blob/main/notebooks/ml-regression-project-example-guide.ipynb)


En este notebook se construye un sistema de **predicción de precios de propiedades** empleando técnicas de regresión. Se realiza un análisis exploratorio de variables numéricas y temporales, acompañado de un proceso de ingeniería de características adaptado a datos inmobiliarios. Posteriormente, se comparan seis modelos distintos —incluyendo regresión lineal, Ridge, Lasso, Random Forest, Gradient Boosting y SVR—, evaluando su rendimiento a través del análisis de residuos y la medición de errores en distintos segmentos de datos.  

---

#### Guía de Clustering: Segmentación de Clientes Bancarios
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gussttaav/ai-notebooks/blob/main/notebooks/ml-clustering-project-example-guide.ipynb)
[![Ver Notebook](https://img.shields.io/badge/📖-View-blue)](https://github.com/gussttaav/ai-notebooks/blob/main/notebooks/ml-clustering-project-example-guide.ipynb)


Este notebook aborda la **segmentación de clientes bancarios** utilizando el dataset real de marketing bancario. Se experimenta con diferentes algoritmos de clustering, como K-Means, Agglomerative Clustering y DBSCAN, comparando sus resultados y discutiendo las ventajas de cada enfoque. Además, se emplean técnicas como el método del codo y el silhouette score para determinar el número óptimo de clusters. Finalmente, se realiza un análisis interpretativo de los segmentos obtenidos, resaltando patrones relevantes en el comportamiento de los clientes.  

---

### Fundamentos Matemáticos de Redes Neuronales

#### MLP explicado como un Proceso de Ajuste de Curvas
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gussttaav/ai-notebooks/blob/main/notebooks/mlp-function-fitting-perspective.ipynb)
[![Ver Notebook](https://img.shields.io/badge/📖-View-blue)](https://github.com/gussttaav/ai-notebooks/blob/main/notebooks/mlp-function-fitting-perspective.ipynb)


Este notebook presenta una introducción detallada al **perceptrón multicapa (MLP)** desde un enfoque matemático. Se desarrolla la formulación completa del forward y backward propagation y se implementa el algoritmo de backpropagation desde cero en NumPy. Como caso práctico, se trabaja con la aproximación de la función \(z = \sin(x)\cos(y)\), utilizando visualizaciones 3D para mostrar cómo la red aprende la superficie. El notebook concluye con una comparación entre la implementación manual y una equivalente realizada en Keras.  

---

### Procesamiento de Lenguaje Natural

#### Fine-tuning de BETO para sentimiento en español
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gussttaav/ai-notebooks/blob/main/notebooks/fine-tuning-beto-sentiment-analysis.ipynb)
[![Ver Notebook](https://img.shields.io/badge/📖-View-blue)](https://github.com/gussttaav/ai-notebooks/blob/main/notebooks/fine-tuning-beto-sentiment-analysis.ipynb)

Este notebook realiza **fine-tuning sobre el modelo BETO** (versión en español de BERT) para **clasificación de sentimientos** en reseñas de productos de Amazon. Para ello, carga el modelo preentrenado y su tokenizador, procesa un conjunto de reseñas en español (etiquetadas con 0 para negativas y 1 para positivas), las tokeniza y trunca a 128 posiciones, y luego entrena el modelo durante solo **3 épocas** con una tasa de aprendizaje baja (2e-5) sobre **3000 ejemplos de entrenamiento**. El bucle de entrenamiento se implementa manualmente usando PyTorch. Práctica del curso [Deep Learning para NLP: del Perceptrón al Transformer](gustavoai.dev/cursos/dl-nlp).