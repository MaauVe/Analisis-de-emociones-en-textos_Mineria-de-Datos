import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Cargamos los embeddings y etiquetas
    embeddings = np.load('embeddings.npy')                   # (n_samples, hidden_size)
    labels_df  = pd.read_csv('labels.csv', encoding='utf-8') # columna "emocion"
    y = labels_df['emocion']

    # Dividimos el dataset de forma estratificada 80% train / 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Verificamos tamaños y proporciones
    print(f"Total muestras:   {embeddings.shape[0]}")
    print(f"  → Train: {X_train.shape[0]}   ({100*X_train.shape[0]/embeddings.shape[0]:.1f}%)")
    print(f"  → Test:  {X_test.shape[0]}   ({100*X_test.shape[0]/embeddings.shape[0]:.1f}%)\n")

    dist_full  = y.value_counts(normalize=True).sort_index()
    dist_train = y_train.value_counts(normalize=True).sort_index()
    dist_test  = y_test.value_counts(normalize=True).sort_index()
    dist_df = pd.DataFrame({
        'Completo': dist_full,
        'Train':    dist_train,
        'Test':     dist_test
    })
    print("Distribución de clases (proporción):\n", dist_df, "\n")

    # Guardamos las particiones
    np.save('X_train.npy', X_train)
    np.save('X_test.npy',  X_test)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv',   index=False)

    print("Particiones guardadas en X_train.npy, X_test.npy, y_train.csv, y_test.csv")

if __name__ == '__main__':
    main()
