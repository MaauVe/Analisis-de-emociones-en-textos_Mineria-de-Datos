import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Ruta del proyecto
project_path = os.path.dirname(os.path.abspath(__file__))

# Cargar datos
X_train = np.load(os.path.join(project_path, "X_train.npy"))
y_train = pd.read_csv(os.path.join(project_path, "y_train.csv")).squeeze()

# Codificar etiquetas
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_train_bin = label_binarize(
    y_train_enc, classes=np.arange(len(label_encoder.classes_))
)

# Escalado para SVM
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Definir el modelo (mejor modelo: SVM con kernel RBF)
model = SVC(kernel="rbf", probability=True, random_state=42)

# Definir métricas para validación cruzada
scoring = {
    "precision_weighted": make_scorer(precision_score, average="weighted"),
    "recall_weighted": make_scorer(recall_score, average="weighted"),
    "f1_weighted": make_scorer(f1_score, average="weighted"),
}

# Validación cruzada con diferentes valores de k
kfolds = [5, 10]
resultados_cv = {}

for k in kfolds:
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        model,
        X_train_scaled,
        y_train_enc,
        cv=kfold,
        scoring=scoring,
        return_train_score=False,
    )
    resultados_cv[f"{k}-fold"] = {
        "Precision": np.mean(cv_scores["test_precision_weighted"]),
        "Recall": np.mean(cv_scores["test_recall_weighted"]),
        "F1-Score": np.mean(cv_scores["test_f1_weighted"]),
    }

# Convertir resultados a DataFrame y guardar
df_cv = pd.DataFrame(resultados_cv).T
df_cv.to_csv(os.path.join(project_path, "validacion_cruzada_resultados_paso_7.csv"))
print("Resultados de validación cruzada:")
print(df_cv)

# Comparación con rendimiento original (conjunto de prueba)
print("\nComparación con rendimiento en conjunto de prueba:")
df_test = pd.read_csv(os.path.join(project_path, "resultados_modelos_completos.csv"))
mejor_modelo = df_test.sort_values(by="F1-Score", ascending=False).iloc[0]
print(mejor_modelo)

# AUC-ROC multiclase manual
# AUC-ROC no se incluye porque cross_validate no lo soporta directamente para multiclase
print("\nCálculo de AUC-ROC multiclase en validación cruzada:")
auc_scores = []

for train_idx, test_idx in StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
).split(X_train_scaled, y_train_enc):
    X_tr, X_te = X_train_scaled[train_idx], X_train_scaled[test_idx]
    y_tr, y_te = y_train_enc[train_idx], y_train_enc[test_idx]
    y_te_bin = label_binarize(y_te, classes=np.arange(len(label_encoder.classes_)))

    model.fit(X_tr, y_tr)

    # Obtener probabilidades
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_te)
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_te)
    else:
        continue  # saltar si no se puede calcular

    auc = roc_auc_score(y_te_bin, y_scores, average="macro", multi_class="ovr")
    auc_scores.append(auc)

print(f"AUC-ROC promedio en 5-fold CV: {np.mean(auc_scores):.4f}")
