from flask import Flask, render_template, request
import pickle
import numpy as np


model = pickle.load(open('naive_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_text = request.form['news_text']
        transformed_text = vectorizer.transform([user_text])
        prediction = model.predict(transformed_text)[0]


        if prediction == 1:
            result = "REAL NEWS"
        else:
            result = "FAKE NEWS"


        return render_template('index.html', prediction=result, user_input=user_text)


if __name__ == '__main__':
    app.run(debug=True)