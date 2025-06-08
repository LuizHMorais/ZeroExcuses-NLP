# ZeroExcuses-NLP

ZeroExcuses-NLP is a small project for emotion classification using text embeddings and a simple machine learning model. It provides scripts to train a classifier on the Hugging Face *emotion* dataset and exposes a FastAPI API for inference.

## Project structure

```
#zeroexcuses-nlp/
│
├── app/                  # API code
│   ├── main.py           # FastAPI app
│   └── model.py          # Loads the model and performs prediction
│
├── data/                 # Raw and processed data
│   ├── raw/              # CSVs or downloaded datasets
│   └── processed/        # Ready for training
│
├── models/               # Saved models (.pkl or .joblib)
│   └── classifier.joblib
│
├── notebooks/            # Experiments and prototypes
│   └── exploracao.ipynb
│
├── src/                  # Training and preprocessing code
│   ├── preprocess.py     # Text cleaning
│   └── train.py          # Model training
│
├── requirements.txt      # Project dependencies
├── README.md             # Project description
└── .gitignore            # Ignore large models, folders, etc.
```

## Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training the model

Run the training script to download the dataset, generate embeddings and fit the classifier:

```bash
python src/train.py
```

The trained model will be saved under `models/`.

## Running the API

Start the FastAPI server with Uvicorn:

```bash
uvicorn app.main:app
```

The interactive documentation will be available at `http://localhost:8000/docs`.

## `/predict` endpoint example

Send a POST request with JSON containing the text to classify:

```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{"text": "I am feeling great today!"}'
```

The API returns the predicted emotion and the probabilities for each class.
