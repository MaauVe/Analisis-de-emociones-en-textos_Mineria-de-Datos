# pip install transformers torch numpy pandas tqdm (si no tienen alguno)
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

# 1. Carga del dataset preprocesado
df = pd.read_csv('dataset_emociones_preprocesado.csv', encoding='utf-8')
df['texto_final'] = df['texto_final'].fillna('')

# 2. Seleccionar RoBERTuito
MODEL_NAME = 'pysentimiento/robertuito-base-uncased'  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME)

# 3. Mean pooling sobre la última capa
def mean_pooling(token_embeddings, attention_mask):
    mask_exp = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed   = torch.sum(token_embeddings * mask_exp, 1)
    counts   = torch.clamp(mask_exp.sum(1), min=1e-9)
    return summed / counts

# 4. Generar embeddings por lotes
texts      = df['texto_final'].tolist()
batch_size = 32
all_emb    = []

model.eval()
with torch.no_grad():
    for i in tqdm(range(0, len(texts), batch_size), desc="Embeddings RoBERTuito"):
        batch = texts[i:i+batch_size]
        enc   = tokenizer(batch, padding=True, truncation=True,
                          max_length=128, return_tensors='pt')
        out   = model(**enc)
        emb   = mean_pooling(out.last_hidden_state, enc['attention_mask'])
        all_emb.append(emb.cpu().numpy())

embeddings = np.vstack(all_emb)  # (n_samples, hidden_size)

# 5. Guardar los resultados
np.save('embeddings.npy', embeddings)
df[['emocion']].to_csv('labels.csv', index=False)

print(f"Embeddings guardados en embeddings.npy con forma {embeddings.shape}")
print("Etiquetas guardadas en labels.csv")
