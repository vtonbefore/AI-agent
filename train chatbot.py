import json
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

nltk.download('punkt')

with open('intents.txt') as file:
    data = json.load(file)

training_sentences = []
labels = []
responses = {}
classes = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        labels.append(intent['tag'])
    responses[intent['tag']] = intent['responses']
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)
y = labels

model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer
pickle.dump(model, open("chat_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(responses, open("responses.pkl", "wb"))

print("Model trained and saved!")
