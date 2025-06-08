from fastapi import FastAPI
from pydantic import BaseModel
from app import model

# Instancia a aplicação FastAPI
app = FastAPI(
    title="ZeroExcuses NLP API",
    description="Classificação de emoções com embeddings e ML",
    version="1.0.0"
)

# Modelo de entrada para validação via Pydantic
class TextoEntrada(BaseModel):
    text: str

# Rota de predição
@app.post("/predict")
def prever_emocao(dados: TextoEntrada):
    resultado = model.predict(dados.text)
    return resultado

# Rota base para healthcheck e evitar erro 502 no Render
@app.get("/")
def root():
    return {"status": "API rodando", "documentacao": "/docs"}
