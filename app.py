from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
    Age = request.form.get('Age')
    Gender = request.form.get('Gender')
    Married = request.form.get('Married')
    Housing = request.form.get('Housing')
    Dependents = request.form.get('Dependents')
    Education = request.form.get('Education')
    Income = request.form.get('Income')
    LoanAmount = request.form.get('LoanAmount')
    LoanDuration = request.form.get('LoanDuration')
    PropertyArea = request.form.get('PropertyArea')
    BusinessIdea = request.form.get('BusinessIdea')
    UnpaidLoan = request.form.get('UnpaidLoan')
    BusinessExp = request.form.get('BusinessExp')

    input_query = np.array([[Age, Gender, Married, Housing, Dependents, Education, Income, LoanAmount, LoanDuration, PropertyArea, BusinessIdea, UnpaidLoan, BusinessExp]])

    result = model.predict(input_query)[0]

    return jsonify({'result': str(result)})

if __name__ == '__main__':
    app.run(debug=True)