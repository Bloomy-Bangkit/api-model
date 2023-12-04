import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from flask import jsonify, request

def predict_price(data, model):
    scaler2 = MinMaxScaler(feature_range=(0,1))
    prediction = model.predict(data)
    prediction = scaler2.inverse_transform(prediction)
    prediction_float = prediction.item()
    return bulatkan_ke_kelipatan(round(prediction_float, 0),1000)

def bulatkan_ke_kelipatan(angka, kelipatan):
    ke_atas = kelipatan * ((angka + kelipatan - 1) // kelipatan)
    ke_bawah = kelipatan * (angka // kelipatan)
    if abs(angka - ke_atas) < abs(angka - ke_bawah):
        return ke_atas
    else:
        return ke_bawah

def marine_price_prediction(model):
    if request.method == 'POST':
        grade = float(request.form['grade'])
        catchingMethod = float(request.form['catchingMethod'])
        sustainability = float(request.form['sustainability'])
        actualPrice = float(request.form['actualPrice'])
        data_new = pd.DataFrame({
            'Grade': grade,
            'Catching Method': catchingMethod,
            'Sustainability': sustainability,
            'Actual Price': actualPrice
        }, index=[0])
        if data_new['Grade'].iloc[0] == 0 or data_new['Actual Price'].iloc[0] == 0:
            print('Masuk IF')
            result_array = np.array([0])
            price = float(result_array[0])
            return jsonify({
                'status': {
                    'code': 200,
                    'message': 'Success predicting',
                    'data': { 'price' : price }
                }
            }), 200
        else:
            print('Masuk ELSE')
            scaler = MinMaxScaler(feature_range=(0,1))
            data_new['Actual Price'] = scaler.transform(data_new[['Actual Price']])
            price = predict_price(data_new, model)
            price = bulatkan_ke_kelipatan(price, 1000)
            return jsonify({
                'status': {
                    'code': 200,
                    'message': 'Success predicting',
                    'data': { 'price' : price }
                }
            }), 200
