# Clustering de Tensores

Este documento describe la metodología propuesta para identificar patrones de compra similares en Big Data, utilizando un enfoque híbrido de Deep Learning (CNN 2D) para la extracción de características y Machine Learning distribuido (Dask-ML) para el clustering.

## 1. Preparación y Transformación de Datos (Dask-ML)

La fase inicial se enfoca en adaptar el tensor 3D por cliente a una matriz 2D que maximice la detección de localidad por la CNN.

* **Carga Distribuida:** Utilizar **`dask.array`** o **`dask.dataframe`** para cargar y gestionar eficientemente los montos de compra tensoriales a escala de Big Data.
* **Aplanamiento Inteligente (a 2D):** Se transforma el tensor original ($4 \times 7 \times 16$) en una matriz 2D ($7 \times 64$).
    * **Eje Vertical (Filas):** Días de la Semana (7).
    * **Eje Horizontal (Columnas):** Combinación de Turno $\times$ MCC (4 $\times$ 16 = 64).
    * **Justificación:** Esta reorganización impone una **estructura temporal (días adyacentes)** y una **estructura funcional (combinaciones de compra)** que la CNN 2D puede explotar eficazmente.
* **Forma de Entrada para CNN:** Reorganizar a la forma final $\text{(Número de Clientes, 7, 64, 1)}$.

---

## 2. Extracción de Características (CNN 2D)

El objetivo es generar un **Embedding** denso y significativo que capture las relaciones no lineales entre Turno, Día y MCC.

* **Diseño y Aplicación de la CNN 2D:** Se utiliza una Red Neuronal Convolucional 2D con kernels que se deslizan a través del eje temporal (Días) y el eje funcional (Turno-MCC).
* **Generación del Embedding:** La salida aplanada (`Flatten`) de la última capa convolucional constituye el **vector de características ($\mathbf{X}_{\text{CNN}}$)**.

$$\mathbf{X}_{\text{CNN}} = \text{Matriz de Embeddings } (\text{Clientes} \times \text{Características Aprendidas})$$

**Justificación:**

* **Detección de Patrones No Lineales:** Las CNNs son superiores a los métodos lineales para aprender características jerárquicas y no lineales en datos estructurados (LeCun et al., 1989).
* **Representación Semántica:** El *embedding* resultante transforma el espacio ruidoso de la entrada a un espacio de baja dimensionalidad donde la distancia euclidiana es más indicativa de la **similitud semántica** de los patrones de compra (Mikolov et al., 2013).

---

## 3. Reducción de Dimensionalidad y Normalización (Dask-ML)

Se refinan los *embeddings* para optimizar el rendimiento de K-Means.

### 3.1. Normalización Z-Score

* **Acción:** Aplicar **`dask_ml.preprocessing.StandardScaler`** a la matriz $\mathbf{X}_{\text{CNN}}$.
* **Justificación:** **K-Means** se basa en la **distancia euclidiana**, que es sensible a la escala. La Estandarización fuerza $\mu=0$ y $\sigma=1$ por característica, asegurando que ninguna característica (dimensión del embedding) domine el cálculo de la distancia, lo cual es crucial para la robustez del clustering (Jain & Dubes, 1988).

### 3.2. PCA (Análisis de Componentes Principales)

* **Acción:** Aplicar **`dask_ml.decomposition.PCA`** sobre los *embeddings* normalizados, seleccionando los $K_{pca}$ componentes que retengan la varianza deseada ($\geq 90\%$).
* **Justificación:**
    * **Descorrelación:** PCA garantiza que las características finales sean **ortogonales** (descorrelacionadas), mejorando la calidad de los clusters.
    * **Mitigación de la Maldición de la Dimensionalidad:** El PCA reduce las dimensiones a un subespacio donde las distancias son más significativas (Bellman, 1961; Jolliffe, 2002).

$$\mathbf{X}_{\text{PCA}} = \text{Matriz de Embeddings Final } (\text{Clientes} \times K_{pca})$$

---

## 4. Clustering (Dask-ML)

Se realiza la agrupación de clientes aprovechando la escalabilidad de Dask.

* **Optimización de $K_{cluster}$:** Determinar el número óptimo de clusters $K$ mediante el **Método del Codo** o el **Coeficiente de Silueta**.
* **Aplicación de K-Means:** Entrenar el modelo **`dask_ml.cluster.KMeans`** con el $K$ óptimo sobre la matriz $\mathbf{X}_{\text{PCA}}$.
* **Justificación:** **K-Means** es elegido por su inherente **paralelizabilidad** y escalabilidad en entornos distribuidos como Dask. Los datos preprocesados y reducidos lo hacen un algoritmo altamente eficiente y efectivo para esta tarea (MacQueen, 1967).

---

## 5. Interpretación y Uso

* **Análisis del Centroide:** Interpretar los centroides de los clusters en el espacio PCA para definir las características distintivas de cada grupo de patrones de compra.
* **Segmentación:** Utilizar las etiquetas de cluster para la segmentación estratégica de clientes.

---

## bib

* **Bellman, R. E. (1961).** *Adaptive control processes: a guided tour*. Princeton University Press. (Introducción de la Maldición de la Dimensionalidad).
* **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press. (Capítulo 9: Fundamentos de CNN).
* **Jain, A. K., & Dubes, R. C. (1988).** *Algorithms for Clustering Data*. Prentice Hall. (Discusión sobre la preparación de datos para clustering).
* **Jolliffe, I. T. (2002).** *Principal Component Analysis* (2nd ed.). Springer. (Referencia fundamental sobre PCA).
* **LeCun, Y., Boser, B., Denker, J. S., et al. (1989).** Backpropagation applied to handwritten recognition. *Neural Computation*, 1(4), 541–551. (Introducción de las CNNs).
* **MacQueen, J. (1967).** Some methods for classification and analysis of multivariate observations. *Proceedings of the fifth Berkeley symposium on mathematical statistics and probability*, 1(281-297), 14. (Artículo fundacional de K-Means).
* **Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013).** Efficient estimation of word representations in vector space. *ICLR*. (Concepto de Embedding como vector de significado).
* **Pedregosa, F., et al. (2011).** Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830. (scikit-learn y su influencia en Dask-ML para escalabilidad).