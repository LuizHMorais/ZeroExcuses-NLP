from fastapi import FastAPI
from pydantic import BaseModel
from app import model

app = FastAPI(title="ZeroExcuses NLP API")

class TextoEntrada(BaseModel):
    text: str

@app.post("/predict")
def prever_emocao(dados: TextoEntrada):
    resultado = model.predict(dados.text)
    return resultado
