import os
import numpy as np
import pandas as pd
import pickle
import re
import spacy
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Configuración de paths
project_path = os.path.dirname(os.path.abspath(__file__))

# Cargar el modelo preentrenado y datos necesarios
print("Cargando modelo y datos...")

# Cargar datos de entrenamiento para reentrenar el modelo
X_train = np.load(os.path.join(project_path, "X_train.npy"))
y_train = pd.read_csv(os.path.join(project_path, "y_train.csv")).squeeze()

# Configurar el codificador de etiquetas
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)

# Entrenar el mejor modelo (SVM con kernel RBF)
print("Entrenando modelo SVM con kernel RBF...")
modelo_final = SVC(kernel="rbf", probability=True, random_state=42)
modelo_final.fit(X_train, y_train_enc)

# Configurar herramientas de preprocesamiento
print("Configurando herramientas de preprocesamiento...")

# Cargar modelo de spaCy para lematización
nlp = spacy.load('es_core_news_sm')

# Configurar RoBERTuito para embeddings
MODEL_NAME = 'pysentimiento/robertuito-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_embeddings = AutoModel.from_pretrained(MODEL_NAME)

# Definir stopwords (mismas que se usaron en preprocesamiento)
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)
stop_nltk = set(stopwords.words('spanish'))
keep = {'me', 'mi', 'yo', 'te', 'nos', 'no', 'nada', 'sin', 'muy', 'poco', 'mucho', 'porque', 'cuando'}
stop_final = stop_nltk - keep

def preprocesar_texto(texto):
    """Función para preprocesar texto nuevo (igual que en el entrenamiento)"""
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

def lematizar_tokens(tokens):
    """Lematización con spaCy"""
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]

def mean_pooling(token_embeddings, attention_mask):
    """Mean pooling para obtener embeddings de texto"""
    mask_exp = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_exp, 1)
    counts = torch.clamp(mask_exp.sum(1), min=1e-9)
    return summed / counts

def obtener_embedding(texto):
    """Obtener embedding de un texto nuevo"""
    # Preprocesar
    tokens = preprocesar_texto(texto)
    lemmas = lematizar_tokens(tokens)
    texto_final = " ".join(lemmas)
    
    # Si el texto queda vacío después del preprocesamiento
    if not texto_final.strip():
        texto_final = "texto vacio"
    
    # Obtener embedding
    model_embeddings.eval()
    with torch.no_grad():
        enc = tokenizer([texto_final], padding=True, truncation=True,
                       max_length=128, return_tensors='pt')
        out = model_embeddings(**enc)
        emb = mean_pooling(out.last_hidden_state, enc['attention_mask'])
        return emb.cpu().numpy()[0]

def predecir_emocion(texto):
    """Predecir emoción de un texto nuevo"""
    # Obtener embedding
    embedding = obtener_embedding(texto)
    
    # Hacer predicción
    prediccion = modelo_final.predict([embedding])[0]
    probabilidades = modelo_final.predict_proba([embedding])[0]
    
    # Decodificar la predicción
    emocion_predicha = label_encoder.inverse_transform([prediccion])[0]
    
    # Crear diccionario de probabilidades por clase
    prob_dict = {}
    for i, clase in enumerate(label_encoder.classes_):
        prob_dict[clase] = probabilidades[i]
    
    return emocion_predicha, prob_dict

# Ejemplos de textos para demostrar el modelo
print("\n" + "="*80)
print("EJEMPLOS DE PREDICCIONES DEL MODELO SVM CON KERNEL RBF")
print("="*80)

ejemplos_textos = [
    # Ejemplos de Felicidad
    ("Me siento muy contento porque aprobé mi examen final con excelente calificación", "Felicidad"),
    ("Hoy es un día maravilloso, estoy lleno de alegría y energía positiva", "Felicidad"),
    
    # Ejemplos de Tristeza
    ("Me siento muy triste porque perdí a mi mascota que era muy importante para mí", "Tristeza"),
    ("Estoy pasando por un momento difícil y me siento desanimado con todo", "Tristeza"),
    
    # Ejemplos de Ira
    ("Estoy muy enojado porque me trataron de manera injusta en el trabajo", "Ira"),
    ("Me da mucha rabia cuando las personas no respetan las reglas básicas de convivencia", "Ira"),
    
    # Ejemplos de Miedo
    ("Tengo mucho miedo de hablar en público frente a tantas personas", "Miedo"),
    ("Me da pánico pensar en el futuro y todas las incertidumbres que trae", "Miedo"),
    
    # Ejemplos de Disgusto
    ("Me da asco la comida que sirven en esa cafetería, está horrible", "Disgusto"),
    ("Siento repulsión hacia las personas que no respetan el medio ambiente", "Disgusto"),
    
    # Ejemplos de Sorpresa
    ("No puedo creer lo que acaba de pasar, me quedé completamente sorprendido", "Sorpresa"),
    ("¡Qué impresionante! Nunca esperé que algo así pudiera suceder", "Sorpresa"),
    
    # Ejemplos ambiguos o complejos
    ("A veces me siento confundido entre la alegría y la nostalgia del pasado", "Ambiguo"),
    ("No sé si estoy emocionado o nervioso por este nuevo desafío en mi vida", "Ambiguo")
]

# Procesar cada ejemplo
resultados_ejemplos = []
aciertos = 0
total_con_etiqueta = 0

for i, (texto, etiqueta_esperada) in enumerate(ejemplos_textos, 1):
    print(f"\nEJEMPLO {i}:")
    print(f"Texto: '{texto}'")
    
    # Hacer predicción
    emocion_pred, probabilidades = predecir_emocion(texto)
    
    print(f"Emoción predicha: {emocion_pred}")
    if etiqueta_esperada != "Ambiguo":
        print(f"Emoción esperada: {etiqueta_esperada}")
        if emocion_pred == etiqueta_esperada:
            print("✓ PREDICCIÓN CORRECTA")
            aciertos += 1
        else:
            print("✗ PREDICCIÓN INCORRECTA")
        total_con_etiqueta += 1
    
    # Mostrar probabilidades (top 3)
    prob_ordenadas = sorted(probabilidades.items(), key=lambda x: x[1], reverse=True)
    print("Probabilidades (top 3):")
    for j, (emocion, prob) in enumerate(prob_ordenadas[:3], 1):
        print(f"  {j}. {emocion}: {prob:.4f} ({prob*100:.2f}%)")
    
    # Guardar resultado
    resultados_ejemplos.append({
        'Texto': texto,
        'Emocion_Predicha': emocion_pred,
        'Emocion_Esperada': etiqueta_esperada,
        'Probabilidad_Max': max(probabilidades.values()),
        'Correcto': emocion_pred == etiqueta_esperada if etiqueta_esperada != "Ambiguo" else None
    })
    
    print("-" * 80)

# Resumen de resultados
print(f"\nRESUMEN DE PREDICCIONES:")
print(f"Aciertos: {aciertos}/{total_con_etiqueta} ({aciertos/total_con_etiqueta*100:.2f}%)")

# Guardar resultados en CSV
df_ejemplos = pd.DataFrame(resultados_ejemplos)
df_ejemplos.to_csv(os.path.join(project_path, "ejemplos_predicciones_svm.csv"), 
                   index=False, encoding='utf-8')
print(f"\nResultados guardados en 'ejemplos_predicciones_svm.csv'")

# Función interactiva para probar textos personalizados
def modo_interactivo():
    """Permite al usuario ingresar textos personalizados para predicción"""
    print("\n" + "="*80)
    print("MODO INTERACTIVO - Ingresa tus propios textos")
    print("(Escribe 'salir' para terminar)")
    print("="*80)
    
    while True:
        texto_usuario = input("\nIngresa un texto para predecir su emoción: ")
        
        if texto_usuario.lower() in ['salir', 'exit', 'quit']:
            print("¡Hasta luego!")
            break
        
        if texto_usuario.strip():
            emocion_pred, probabilidades = predecir_emocion(texto_usuario)
            
            print(f"\nTexto: '{texto_usuario}'")
            print(f"Emoción predicha: {emocion_pred}")
            
            # Mostrar todas las probabilidades
            prob_ordenadas = sorted(probabilidades.items(), key=lambda x: x[1], reverse=True)
            print("Probabilidades por emoción:")
            for emocion, prob in prob_ordenadas:
                print(f"  {emocion}: {prob:.4f} ({prob*100:.2f}%)")
        else:
            print("Por favor, ingresa un texto válido.")

# Descomenta la siguiente línea si quieres activar el modo interactivo
# modo_interactivo()

print("\n¡Análisis de predicciones completado!")