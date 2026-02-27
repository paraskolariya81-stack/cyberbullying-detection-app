import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("dataset.csv")

# Simple preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df['clean_text'] = df['text'].apply(preprocess)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Train model
model = LogisticRegression()
model.fit(X, y)

def predict(text):
    text = preprocess(text)
    vector = vectorizer.transform([text])
    return model.predict(vector)[0]