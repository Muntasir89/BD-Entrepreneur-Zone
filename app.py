from flask import Flask, request, jsonify
import pickle
import numpy

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():

    result = model.predict([[12, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])[0]
    if result == 1:
        return jsonify('Yes')
    else:
        return jsonify("No")

if __name__ == '__main__':
    app.run(debug=True)