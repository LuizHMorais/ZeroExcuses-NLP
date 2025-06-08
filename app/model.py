import joblib
import os
import numpy as np

# Caminho absoluto para garantir que o modelo seja carregado de qualquer lugar
modelo_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'classifier.joblib')
modelo_path = os.path.abspath(modelo_path)

# Carrega o modelo apenas uma vez
modelo = joblib.load(modelo_path)

# Lista das classes na mesma ordem que o modelo treinou
CLASSES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

def predict(texto: str):
    # Predição da classe
    predicao = modelo.predict([texto])[0]
    
    # Probabilidades para cada classe
    probs = modelo.predict_proba([texto])[0]
    
    # Mapeia classe com nome legível
    classe_nome = CLASSES[predicao]
    
    # Monta dict com probabilidades
    probs_dict = {CLASSES[i]: float(np.round(probs[i], 4)) for i in range(len(CLASSES))}
    
    return {
        "classe": classe_nome,
        "probabilidades": probs_dict
    }
