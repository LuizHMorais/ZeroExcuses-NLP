import joblib
import os
import numpy as np

# Define as classes na mesma ordem usada durante o treino
CLASSES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Caminho absoluto para o modelo salvo
modelo_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'classifier_embed.joblib')
modelo_path = os.path.abspath(modelo_path)

# Carrega o dicionário contendo o modelo e o encoder
pacote = joblib.load(modelo_path)
encoder = pacote['encoder']
modelo = pacote['model']

def predict(texto: str):
    # Gera o embedding
    emb = encoder.encode([texto])[0].reshape(1, -1)

    # Predição da classe
    pred = modelo.predict(emb)[0]

    # Probabilidades para todas as classes
    probs = modelo.predict_proba(emb)[0]
    probs_dict = {CLASSES[i]: float(np.round(probs[i], 4)) for i in range(len(CLASSES))}

    return {
        "classe": CLASSES[pred],
        "probabilidades": probs_dict
    }

