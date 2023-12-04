import os
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as tf_image
from sklearn.preprocessing import MinMaxScaler
from auth import auth

load_dotenv()
app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

app.config['MODEL_MARINE_CLASSIFICATION'] = 'models/marine_classification.h5'
app.config['MODEL_MARINE_GRADING'] = 'models/marine_grading.h5'
app.config['MODEL_MARINE_SAIL_DECISION'] = 'models/marine_sail_decision.h5'
app.config['MODEL_MARINE_PRICE'] = 'models/marine_price.h5'
app.config['MODEL_MARINE_PRICE_SCALER'] = 'models/price_scaler.pkl'
app.config['MODEL_MARINE_ACTUAL_PRICE_SCALER'] = 'models/actual_price_scaler.pkl'

model_marine_classification = load_model(app.config['MODEL_MARINE_CLASSIFICATION'], compile=False)
model_marine_grading = load_model(app.config['MODEL_MARINE_GRADING'], compile=False)
model_marine_sail_decision = load_model(app.config['MODEL_MARINE_SAIL_DECISION'], compile=False)
model_marine_price = load_model(app.config['MODEL_MARINE_PRICE'], compile=False)
model_marine_price_scaler = joblib.load(app.config['MODEL_MARINE_PRICE_SCALER'])
model_marine_actual_price_scaler = joblib.load(app.config['MODEL_MARINE_ACTUAL_PRICE_SCALER'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
           
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': {
            'code': 200,
            'message': 'Hello API Fish Grading!'
        }
    }), 200

@app.route('/marine/predict', methods=['POST'])
@auth.login_required()
def predict_marine_classification():
    if request.method == 'POST':
        reqImage = request.files['image']
        if reqImage and allowed_file(reqImage.filename):
            filename = secure_filename(reqImage.filename)
            reqImage.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = Image.open(image_path).convert("RGB")
            img = img.resize((150, 150))
            x = tf_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255
            classificationResult = model_marine_classification.predict(x, batch_size=1)
            class_list = ['Ikan', 'Udang']
            classification_class = class_list[np.argmax(classificationResult[0])]
            classes = model_marine_grading.predict(x, batch_size=1)
            class_list = ['A', 'B', 'C']
            predicted_class = class_list[np.argmax(classes[0])]
            
            return jsonify({
                'status': {
                    'code': 200,
                    'message': 'Success predicting',
                    'data': { 'class': classification_class, 'grade': predicted_class }
                }
            }), 200
        else:
            return jsonify({
                'status': {
                    'code': 400,
                    'message': 'Invalid file format. Please upload a JPG, JPEG, or PNG image.'
                }
            }), 400
    else:
        return jsonify({
            'status': {
                'code': 405,
                'message': 'Method not allowed'
            }
        }), 405

@app.route('/sail_decision/predict', methods=['POST'])
@auth.login_required()
def predict_marine_sail_decision():
    if request.method == 'POST':
        outlook = float(request.form['outlook'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        wind = float(request.form['wind'])
        data_predict = [[outlook, temperature, humidity, wind]]
        data_predict = np.array(data_predict, dtype=float)
        predicted = model_marine_sail_decision.predict(data_predict)
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

def predict_price(data, model):
    prediction = model.predict(data)
    prediction = model_marine_price_scaler.inverse_transform(prediction)
    prediction_float = prediction.item()
    return bulatkan_ke_kelipatan(round(prediction_float, 0),1000)

def bulatkan_ke_kelipatan(angka, kelipatan):
    ke_atas = kelipatan * ((angka + kelipatan - 1) // kelipatan)
    ke_bawah = kelipatan * (angka // kelipatan)
    if abs(angka - ke_atas) < abs(angka - ke_bawah):
        return ke_atas
    else:
        return ke_bawah

@app.route('/price/predict', methods=['POST'])
@auth.login_required()
def predict_marine_price_prediction():
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
            result_array = np.array([0])
            price = float(result_array[0])
            return jsonify({
                'status': {
                    'code': 200,
                    'message': 'Success predicting',
                    'data': { 'price' : int(price) }
                }
            }), 200
        else:
            data_new['Actual Price'] = model_marine_actual_price_scaler.transform(data_new[['Actual Price']])
            price = predict_price(data_new, model_marine_price)
            price = bulatkan_ke_kelipatan(price, 1000)
            return jsonify({
                'status': {
                    'code': 200,
                    'message': 'Success predicting',
                    'data': { 'price' : int(price) }
                }
            }), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))