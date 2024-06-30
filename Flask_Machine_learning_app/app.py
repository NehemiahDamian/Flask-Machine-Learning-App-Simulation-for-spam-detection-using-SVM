import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

with open('model_pickle5', 'rb') as model_file:
    pipeline = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index1.html', prediction=None, positive_prob=None, negative_prob=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    probabilities = pipeline.predict_proba([text])[0]
    positive_prob = probabilities[1]
    negative_prob = probabilities[0]

    return render_template('index1.html', prediction=positive_prob > 0.5, positive_prob=positive_prob, negative_prob=negative_prob)

if __name__ == '__main__':
    app.run(debug=True)

