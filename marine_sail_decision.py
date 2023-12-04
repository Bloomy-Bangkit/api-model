import numpy as np
from flask import jsonify, request

def marine_sail_decision(model):
    if request.method == 'POST':
        outlook = float(request.form['outlook'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        wind = float(request.form['wind'])
        data_predict = [[outlook, temperature, humidity, wind]]
        data_predict = np.array(data_predict, dtype=float)
        predicted = model.predict(data_predict)
        result = int(predicted[0][0])
        desicion = ['Boleh melaut' if result >= 0.6 else 'Tidak boleh melaut']
        return jsonify({
            'status': {
                'code': 200,
                'message': 'Success predicting',
                'data': { 'precentage': result, 'decision': desicion }
            }
        }), 200
    else:
        return jsonify({
            'status': {
                'code': 405,
                'message': 'Method not allowed'
            }
        }), 405