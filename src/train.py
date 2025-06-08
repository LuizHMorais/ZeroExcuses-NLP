from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os

dataset = load_dataset("emotion")
train_texts = dataset['train']['text']
train_labels = dataset['train']['label']

test_texts = dataset['test']['text']
test_labels = dataset['test']['label']

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(train_texts, train_labels)

preds = pipeline.predict(test_texts)
print(classification_report(test_labels, preds))

os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, 'models/classifier.joblib')
