from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import joblib
import os

# 1. Carrega o dataset "emotion" do Hugging Face
dataset = load_dataset("emotion")
train_texts = dataset['train']['text']
train_labels = dataset['train']['label']
test_texts = dataset['test']['text']
test_labels = dataset['test']['label']

# 2. Usa encoder MPNet (melhor embedding contextual)
encoder = SentenceTransformer('all-mpnet-base-v2')  # mais preciso que MiniLM

# 3. Converte os textos em embeddings
X_train = encoder.encode(train_texts, show_progress_bar=True)
X_test = encoder.encode(test_texts, show_progress_bar=True)

# 4. Treina um classificador com balanceamento automático
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train, train_labels)

# 5. Avalia o desempenho
preds = clf.predict(X_test)
print(classification_report(test_labels, preds))

# 6. Salva encoder e modelo juntos em um único arquivo
os.makedirs("models", exist_ok=True)
joblib.dump({'model': clf, 'encoder': encoder}, "models/classifier_embed.joblib")


