import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,  
    roc_auc_score    
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Ruta del proyecto (ajústala si corres desde otro directorio)
project_path = os.path.dirname(os.path.abspath(__file__))

# Cargar datos
X_train = np.load(os.path.join(project_path, "X_train.npy"))
X_test = np.load(os.path.join(project_path, "X_test.npy"))
y_train = pd.read_csv(os.path.join(project_path, "y_train.csv")).squeeze()
y_test = pd.read_csv(os.path.join(project_path, "y_test.csv")).squeeze()

# Codificar etiquetas
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

# Binarizar etiquetas para ROC-AUC (una columna por clase)
y_test_bin = label_binarize(y_test_enc, classes=np.arange(len(label_encoder.classes_)))

# Para almacenar resultados
resultados = []

# Función para calcular métricas y añadirlas al resultado
def evaluar_modelo(nombre, modelo, X_test, y_test_enc, y_test_bin, resultados):
    # Predicciones
    y_pred = modelo.predict(X_test)

    # Matriz de confusion
    cm = confusion_matrix(y_test_enc, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title(f"Matriz de Confusión - {nombre}")
    plt.xlabel("Etiqueta Predicha")
    plt.ylabel("Etiqueta Verdadera")
    plt.tight_layout()
    plt.show()

    # Calcular scores probabilísticos para ROC-AUC si es posible
    if hasattr(modelo, 'predict_proba'):
        y_scores = modelo.predict_proba(X_test)
    elif hasattr(modelo, 'decision_function'):
        y_scores = modelo.decision_function(X_test)
    else:
        y_scores = None
    
    # Calcular AUC-ROC (si es posible)
    if y_scores is not None:
        try:
            auc_roc = roc_auc_score(y_test_bin, y_scores, average='macro', multi_class='ovr')
        except:
            auc_roc = None
    else:
        auc_roc = None
    
    # Añadir resultados
    resultados.append({
        "Modelo": nombre,
        "Precision": precision_score(y_test_enc, y_pred, average="weighted"),
        "Recall": recall_score(y_test_enc, y_pred, average="weighted"),
        "F1-Score": f1_score(y_test_enc, y_pred, average="weighted"),
        "Exactitud": accuracy_score(y_test_enc, y_pred),
        "AUC-ROC": auc_roc if auc_roc is not None else float('nan')
    })

# KNN con distintos valores de k
for k in [3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train_enc)
    evaluar_modelo(f"KNN (k={k})", knn, X_test, y_test_enc, y_test_bin, resultados)

# Naive Bayes (requiere valores positivos)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_scaled, y_train_enc)
evaluar_modelo("Naive Bayes", nb, X_test_scaled, y_test_enc, y_test_bin, resultados)

# Árboles de decisión
# ID3 (entropía)
tree_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42)
tree_entropy.fit(X_train, y_train_enc)
evaluar_modelo("Árbol de Decisión (ID3)", tree_entropy, X_test, y_test_enc, y_test_bin, resultados)

# C4.5 (entropía + poda)
tree_c45 = DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=42)
tree_c45.fit(X_train, y_train_enc)
evaluar_modelo("Árbol de Decisión (C4.5)", tree_c45, X_test, y_test_enc, y_test_bin, resultados)

# SVM con distintos kernels
for kernel in ["linear", "rbf"]:
    svm = SVC(kernel=kernel, probability=True, random_state=42)
    svm.fit(X_train, y_train_enc)
    evaluar_modelo(f"SVM (kernel={kernel})", svm, X_test, y_test_enc, y_test_bin, resultados)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train_enc)
evaluar_modelo("Random Forest", rf, X_test, y_test_enc, y_test_bin, resultados)

# Convertir a DataFrame y guardar resultados
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(
    os.path.join(project_path, "resultados_modelos_completos.csv"), index=False
)
print(df_resultados)

# Visualización de resultados (gráfico de barras)
plt.figure(figsize=(12, 8))
df_resultados.set_index("Modelo")[["Precision", "Recall", "F1-Score", "Exactitud", "AUC-ROC"]].plot(
    kind="bar", figsize=(14, 7)
)
plt.title("Comparación de Modelos de Clasificación")
plt.ylabel("Puntaje")
plt.ylim(0, 1)
plt.grid(True, axis="y")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(project_path, "comparacion_modelos_completa.png"))
plt.show()

# Matriz de confusión del mejor modelo según F1-Score
mejor = df_resultados.sort_values(by="F1-Score", ascending=False).iloc[0]["Modelo"]
print(f"Mejor modelo: {mejor}")

# Reconstruir el mejor modelo para mostrar su matriz de confusión
if "KNN" in mejor:
    k = int(mejor.split("=")[1].replace(")", ""))
    model = KNeighborsClassifier(n_neighbors=k)
elif "Naive Bayes" in mejor:
    model = MultinomialNB()
    X_train, X_test = X_train_scaled, X_test_scaled
elif "ID3" in mejor:
    model = DecisionTreeClassifier(criterion="entropy", random_state=42)
elif "C4.5" in mejor:
    model = DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=42)
elif "SVM" in mejor:
    kernel = mejor.split("=")[1].replace(")", "")
    model = SVC(kernel=kernel, probability=True, random_state=42)
elif "Random Forest" in mejor:
    model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train_enc)
y_pred_mejor = model.predict(X_test)
conf_mat = confusion_matrix(y_test_enc, y_pred_mejor)

plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_mat,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
plt.title(f"Matriz de Confusión - {mejor}")
plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Verdadera")
plt.tight_layout()
plt.savefig(os.path.join(project_path, "matriz_confusion.png"))
plt.show()
