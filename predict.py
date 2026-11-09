import pickle

from flask import Flask, request, jsonify

model_file = 'model.bin'

threshold = 0.12

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('readmission')

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()

    X = dv.transform([patient])
    y_pred = model.predict_proba(X)[0, 1]
    readmit = y_pred >= threshold

    result = {
        'readmission_probability': float(y_pred),
        'readmit': bool(readmit)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)