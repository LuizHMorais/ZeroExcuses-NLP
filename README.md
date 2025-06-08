# ZeroExcuses-NLP

#zeroexcuses-nlp/
#│
#├── app/                  # Código da API
#│   ├── main.py           # FastAPI app
#│   └── model.py          # Carrega o modelo e faz predição
#│
#├── data/                 # Dados brutos e processados
#│   ├── raw/              # CSVs ou datasets baixados
#│   └── processed/        # Pronto pra treinar
#│
#├── models/               # Modelos salvos (.pkl ou .joblib)
#│   └── classifier.joblib
#│
#├── notebooks/            # Testes e protótipos (Jupyter ou colab)
#│   └── exploracao.ipynb
#│
#├── src/                  # Código de treino e pré-processamento
#│   ├── preprocess.py     # Limpeza de texto
#│   └── train.py          # Treinamento do modelo
#│
#├── requirements.txt      # Dependências do projeto
#├── README.md             # Descrição do projeto
#└── .gitignore            # Ignorar modelos grandes, pastas etc.
