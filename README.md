##Clasificación de Emociones en Textos en Español
Este proyecto implementa un sistema de clasificación de emociones basado en aprendizaje automático. Utiliza un modelo SVM con kernel RBF entrenado sobre representaciones vectoriales (embeddings) obtenidas mediante el modelo RoBERTuito, combinando técnicas de procesamiento de lenguaje natural con modelos modernos de lenguaje preentrenados.


##Lista e instalación de dependencias
**Instrucciones:**

-Abrir Visual Studio Code
-Abrir la pestaña terminal de Visual Studio Code en la parte superior del programa.
-Seleccionar "New Terminal".
-Escribir la entrada textual de "Instalación" del apartado siguiente en la terminal (Nota: Una por una).
-Ejemplo: Escribir en la terminal (pip install pandas) y dar Enter. 
   
Módulo               | Uso principal                                  | Instalación
---------------------|------------------------------------------------|-------------------------------
pandas               | Lectura y manipulación de datos                | pip install pandas
numpy                | Cálculos numéricos y vectores                  | pip install numpy
matplotlib           | Gráficas básicas                               | pip install matplotlib
seaborn              | Visualización estadística                      | pip install seaborn
scikit-learn         | Modelado ML, métricas y preprocesamiento       | pip install scikit-learn
nltk                 | Tokenización y stopwords                       | pip install nltk
transformers         | Uso de modelos tipo RoBERTuito                 | pip install transformers
torch                | Backend para transformers                      | pip install torch
sentence-transformers| Generación de embeddings                       | pip install sentence-transformers
joblib               | Guardar/cargar modelos                         | pip install joblib
tqdm                 | Barras de progreso                             | pip install tqdm
langdetect           | Detección de idioma                            | pip install langdetect

Para la instalación de todas las dependencias ingresar en la terminal: pip install pandas numpy matplotlib seaborn scikit-learn nltk transformers torch sentence-transformers joblib tqdm langdetect


##Ejecución del modelo y reproducción de experimentos

**Sigue los siguientes pasos en orden para reproducir todo el pipeline:** 

1) Instalación de dependencias
	```Código de ejecución en terminal de Visual Studio:
	pip install pandas numpy matplotlib seaborn scikit-learn nltk transformers torch sentence-transformers joblib tqdm langdetect

2) Reestructuración del dataset
	-Ejecuta 1.Reestrucutración.py para transformar el CSV original en un DataFrame limpio con columnas respuesta, emoción y pregunta.
	```Código de ejecución en terminal de Visual Studio:
		"python 1.Reestrucutración.py"
	-Salida: dataset_emociones_transformado.csv

3) Análisis exploratorio de datos
	-Ejecuta 2.EDA.py para generar estadísticas y gráficas (distribución, longitud, nubes de palabras).
	```Código de ejecución en terminal de Visual Studio:
		"python 2.EDA.py"
	-Salidas: Gráficos en pantalla y resultados de análisis en terminal.

4) Preprocesamiento y guardado
	-Ejecuta 3.Preprocesamiento.py para tokenizar, limpiar, eliminar stopwords y lematizar.
	```Código de ejecución en terminal de Visual Studio:
		"python  3.Preprocesamiento.py"
	-Salida: dataset_emociones_preprocesado.csv
	
5) Vectorización TF-IDF
	-Descomentar al final de 3.Preprocesamiento.py el código "CODIGO DE VECTORIZACIÓN TF-IDF" y comentar los demás códigos del archivo.
	-Ejecuta 3.Preprocesamiento.py para ajustar el TfidfVectorizer y guardar el modelo.
	```Código de ejecución en terminal de Visual Studio:
		"python 3.Preprocesamiento.py"
	-Salida: tfidf_vectorizer.pkl

6) Generación de embeddings con RoBERTuito
	-Ejecuta 4.Embeddings.py para producir un archivo NumPy con embeddings.
	```Código de ejecución en terminal de Visual Studio:
		"python Embeddings"
	-Salidas: embeddings.npy, labels.csv

7) División del dataset
	-Ejecuta 5.División Dataset.py para crear particiones de entrenamiento y prueba 80/20 (estratificado).
	```Código de ejecución en terminal de Visual Studio:
		"python 5.División Dataset.py"
	-Salidas: X_train.npy, X_test.npy, y_train.csv, y_test.csv, comparacion_modelos_completa.png

8) Entrenamiento
	-Ejecuta 6.Algortimos_PY.py para entrenar KNN, NB, árboles, SVM y RF.
	```Código de ejecución en terminal de Visual Studio:
		"6.Algortimos_PY.py"
	-Salidas: Matriz de confusión en formato .png y gráficos de barras.

9) Visualización 
	-Ejecuta 6.1 Visualización ROC_AUC.py para crear una tabla con los valores AUC-ROC por clase para cada modelo.
	```Código de ejecución en terminal de Visual Studio:
		"python 6.1 Visualización ROC_AUC.py"
	-Salidas: auc_roc_por_clase.csv, modelos de curvas .png

10) Validación de modelos
	-Ejecuta 7.Validacion_de_modelos.py para evalúar la robustez del modelo SVM (kernel RBF) usando validación estratificada en k-folds (5 y 10 pliegues).
	```Código de ejecución en terminal de Visual Studio:
		"python 7.Validacion_de_modelos.py"
	-Salidas: validacion_cruzada_resultados_paso_7.csv, tabla con Precision, Recall y F1-Score para 5-fold y 10-fold CV.

11) Ejemplos de predicciones
	-Ejecuta 8.Ejemplos de predicciones.py para entrenar SVM final y probar oraciones de ejemplo o modo interactivo (para el modo interactivo descomente su sección).
	```Código de ejecución en terminal de Visual Studio:
		"8.Ejemplos de predicciones.py"
	-Salida: ejemplos_predicciones_svm.csv, resultados de predicciones en la consola.

	








