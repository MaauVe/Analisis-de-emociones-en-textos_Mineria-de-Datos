import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from itertools import cycle

# Ruta del proyecto
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
n_classes = len(label_encoder.classes_)

# Binarizar etiquetas para ROC-AUC
y_test_bin = label_binarize(y_test_enc, classes=np.arange(n_classes))

# Escalar para Naive Bayes
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lista de modelos a evaluar
modelos = [
    ("SVM (kernel=rbf)", SVC(kernel="rbf", probability=True, random_state=42), X_train, X_test),
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42), X_train, X_test),
    ("KNN (k=5)", KNeighborsClassifier(n_neighbors=5), X_train, X_test),
    ("Naive Bayes", MultinomialNB(), X_train_scaled, X_test_scaled),
]

# Colores para las curvas
colors = cycle(['blue', 'red', 'green', 'purple', 'orange', 'cyan'])

# Figura para todas las curvas ROC
plt.figure(figsize=(12, 8))

# Para cada modelo
for (name, model, X_tr, X_te), color in zip(modelos, colors):
    # Entrenar modelo
    model.fit(X_tr, y_train_enc)
    
    # Preparar para generar curvas ROC
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_te)
    else:
        continue  # Saltamos modelos que no tienen predict_proba
    
    # Calcular curva ROC para cada clase y micro/macro promedios
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Calcular curva ROC para cada clase
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calcular micro-promedio
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Calcular macro-promedio
    # Primero agregamos todos los fpr en un solo array ordenado
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Interpolamos todas las curvas ROC en estos puntos
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Promediamos y calculamos AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Graficar curva ROC macro
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f'{name} (AUC = {roc_auc["macro"]:.4f})',
        color=color,
        linewidth=2,
    )

# Línea de referencia (aleatorio)
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
plt.title('Curvas ROC para diferentes modelos (Macro-promedio)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(project_path, "curvas_roc_modelos.png"), dpi=300)
plt.show()

# Ahora generamos curvas ROC por clase para el mejor modelo (SVM RBF)
plt.figure(figsize=(12, 8))

# Usamos el SVM como mejor modelo según análisis previo
best_model = SVC(kernel="rbf", probability=True, random_state=42)
best_model.fit(X_train, y_train_enc)
y_score = best_model.predict_proba(X_test)

# Calcular curva ROC para cada clase
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Graficar cada clase
for i, color, cls in zip(range(n_classes), colors, label_encoder.classes_):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        linewidth=2,
        label=f'ROC {cls} (AUC = {roc_auc[i]:.4f})'
    )

# Referencia (aleatorio)
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
plt.title('Curvas ROC por emoción para SVM (kernel=rbf)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(project_path, "curvas_roc_por_emocion.png"), dpi=300)
plt.show()

# Crear una tabla con los valores AUC-ROC por clase para cada modelo
auc_por_clase = {
    "Emoción": label_encoder.classes_
}

for name, model, X_tr, X_te in modelos:
    model.fit(X_tr, y_train_enc)
    
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_te)
        aucs = []
        
        for i in range(n_classes):
            fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            aucs.append(auc(fpr_i, tpr_i))
        
        auc_por_clase[name] = aucs

# Convertir a DataFrame y guardar
df_auc = pd.DataFrame(auc_por_clase)
print("AUC-ROC por clase y modelo:")
print(df_auc)
df_auc.to_csv(os.path.join(project_path, "auc_roc_por_clase.csv"), index=False, float_format="%.4f")