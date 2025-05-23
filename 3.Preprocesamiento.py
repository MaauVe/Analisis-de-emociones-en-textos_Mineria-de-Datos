# import pandas as pd
# import re
# from nltk.corpus import stopwords
# from collections import Counter
# import nltk


## ---------------REVISION DE STOPWORDS------------------#

# nltk.download('stopwords')

# # Cargar dataset
# df = pd.read_csv('dataset_emociones_transformado.csv')

# # Limpieza de texto
# def limpiar_texto(texto):
#     if not isinstance(texto, str):
#         texto = ""
#     texto = texto.lower()
#     texto = re.sub(r'[^a-záéíóúñü\s]', '', texto)
#     return texto

# df['texto_limpio'] = df['respuesta'].apply(limpiar_texto)

# # Crear lista de todas las palabras
# tokens = []
# for texto in df['texto_limpio']:
#     tokens.extend(texto.split())  # usamos split() en lugar de word_tokenize()

# # Contar frecuencia de stopwords
# stop_words = set(stopwords.words('spanish'))
# stop_freq = Counter([word for word in tokens if word in stop_words])

# # Mostrar las 100 stopwords más frecuentes
# print("Top 100 stopwords más comunes en el dataset:\n")
# for palabra, freq in stop_freq.most_common(100):
#     print(f"{palabra:<10} {freq}")


#---------------------CODIGO DE ELIMINACIÓN DE STOPWORDS PRECISO----------------#
#------------------------STOPWORDS----------------------#
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import spacy

# 1. Configuración inicial
nltk.download('stopwords')

# 2. Cargar el dataset transformado
df = pd.read_csv('dataset_emociones_transformado.csv', encoding='utf-8')

# 3. Definir stopwords
stop_nltk = set(stopwords.words('spanish'))
keep = {'me', 'mi', 'yo', 'te', 'nos', 'no', 'nada', 'sin', 'muy', 'poco', 'mucho', 'porque', 'cuando'}
stop_final = stop_nltk - keep

# 4. Función de limpieza y tokenización
def preprocesar(texto):
    # Asegurar string
    if not isinstance(texto, str):
        texto = ""
    # Minúsculas y eliminar caracteres no alfabéticos
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñü\s]', '', texto)
    
    # Tokenizar por espacios
    tokens = texto.split()
    
    # Eliminar stopwords finales y palabras muy cortas
    tokens = [t for t in tokens if t not in stop_final and len(t) > 2]
    
    return tokens

# 5. Aplicar preprocesamiento
df['tokens'] = df['respuesta'].apply(preprocesar)

# 6. Lematización con spaCy (opcional, pero recomendable)

nlp = spacy.load('es_core_news_sm')  # asegurarte de haber instalado: python -m spacy download es_core_news_sm

def lematizar(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]

df['lemmas'] = df['tokens'].apply(lematizar)

# 7. Texto final para vectorización
df['texto_final'] = df['lemmas'].apply(lambda lst: " ".join(lst))

# 8. Vista de resultados para verifificar
print(df[['respuesta', 'tokens', 'texto_final']].head())
df.to_csv('dataset_emociones_preprocesado.csv', index=False, encoding='utf-8')


# ##--------------CODIGO DE VECTORIZACION TF-IDF------------------------------------#
# ##--------------------------------------vECTORIZACIÓN------------------------------#
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# import pickle

# # 1. Cargar el dataset preprocesado
# df = pd.read_csv('dataset_emociones_preprocesado.csv', encoding='utf-8')

# #Verificacion de columnas#
# # print("Columnas disponibles:", df.columns.tolist())
# # print("Ejemplo de etiquetas:", df['emocion'].unique())


# # 2. Asegurarnos de que no haya NaN en 'texto_final'
# df['texto_final'] = df['texto_final'].fillna('')

# # 3. Separar X e y
# X_text = df['texto_final']
# y = df['emocion']

# # 4. Configurar TF‑IDF
# tfidf = TfidfVectorizer(
#     max_features=5000,
#     ngram_range=(1,2),
#     min_df=5,
#     max_df=0.95,
#     norm='l2',
#     use_idf=True
# )

# # 5. Ajustar el vectorizador y transformar
# X_tfidf = tfidf.fit_transform(X_text)
# print(f"Matriz TF-IDF: {X_tfidf.shape[0]} documentos × {X_tfidf.shape[1]} términos")

# # 6. Guardar el vectorizador
# with open('tfidf_vectorizer.pkl', 'wb') as f:
#     pickle.dump(tfidf, f)
# print("Vectorizador guardado en 'tfidf_vectorizer.pkl'")
