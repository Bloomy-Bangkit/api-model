import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gdown
import joblib
import jwt
import numpy as np
import pandas as pd
from functools import wraps
from http import HTTPStatus
from PIL import Image
from flask import Flask, jsonify, request
from google.cloud import storage
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as tf_image
from zipfile import ZipFile 

gdown.download('https://drive.google.com/uc?id=1Yczy94kJKuywrJzL9PqHtcXHwmnDD6K4')
with ZipFile('./models.zip', 'r') as modelFolder: 
    modelFolder.extractall()

load_dotenv()

app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_MARINE_CLASSIFICATION'] = './models/marine_classification.h5'
app.config['MODEL_MARINE_GRADING_FISH'] = './models/marine_grading_fish.h5'
app.config['MODEL_MARINE_GRADING_SHRIMP'] = './models/marine_grading_shrimp.h5'
app.config['MODEL_MARINE_PRICE'] = './models/marine_price.h5'
app.config['MODEL_MARINE_SAIL_DECISION'] = './models/marine_sail_decision.h5'
app.config['MODEL_MARINE_SCALER_PRICE'] = './models/scaler_price.pkl'
app.config['MODEL_MARINE_SCALER_ACTUAL_PRICE'] = './models/scaler_actual_price.pkl'
app.config['GOOGLE_APPLICATION_CREDENTIALS'] = './credentials/sa-gcs.json'

model_marine_classification = load_model(app.config['MODEL_MARINE_CLASSIFICATION'], compile=False)
model_marine_grading_fish = load_model(app.config['MODEL_MARINE_GRADING_FISH'], compile=False)
model_marine_grading_shrimp = load_model(app.config['MODEL_MARINE_GRADING_SHRIMP'], compile=False)
model_marine_price = load_model(app.config['MODEL_MARINE_PRICE'], compile=False)
model_marine_sail_decision = load_model(app.config['MODEL_MARINE_SAIL_DECISION'], compile=False)
model_marine_scaler_price = joblib.load(app.config['MODEL_MARINE_SCALER_PRICE'])
model_marine_scaler_actual_price = joblib.load(app.config['MODEL_MARINE_SCALER_ACTUAL_PRICE'])

bucket_name = os.environ.get('BUCKET_NAME')
client = storage.Client.from_service_account_json(json_credentials_path=app.config['GOOGLE_APPLICATION_CREDENTIALS'])
bucket = storage.Bucket(client, bucket_name)

SECRET_KEY = os.environ.get('SECRET_KEY')
if SECRET_KEY is None:
    print("SECRET_KEY not found in environment variables.")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
           
def token_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        token = request.headers.get('Authorization', None)
        if not token:
            return jsonify({'message': 'Invalid token'}), 401
        try:
            token_prefix, token_value = token.split()
            if token_prefix.lower() != 'bearer':
                raise ValueError('Invalid token prefix')
            data = jwt.decode(token_value, SECRET_KEY, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token'}), 401
        except ValueError:
            return jsonify({'message': 'Invalid token format'}), 401
        return f(data, *args, **kwargs)
    return decorator

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'Message': 'بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ',
        'Data': {
            'Project': 'Capstone Bangkit 2023 Batch 2',
            'Tema': 'Ocean and Maritime Economy',
            'Judul': 'Bloomy',
            'Team': 'CH2-PS086',
            'Anggota': [
                { 'BangkitID': 'M128BSY0948', 'Nama': 'Heical Chandra Syahputra', 'Universitas': 'Politeknik Negeri Jakarta' },
                { 'BangkitID': 'M128BSY1852', 'Nama': 'Andra Rizki Pratama', 'Universitas': 'Politeknik Negeri Jakarta' },
                { 'BangkitID': 'M015BSY0866', 'Nama': 'Novebri Tito Ramadhani', 'Universitas': 'Universitas Negeri Yogyakarta' },
                { 'BangkitID': 'C256BSY3481', 'Nama': 'Aditya Bayu Aji', 'Universitas': 'Universitas Muhammadiyah Cirebon' },
                { 'BangkitID': 'C313BSX3054', 'Nama': 'Asrini Salsabila Putri', 'Universitas': 'Universitas Siliwangi' },
                { 'BangkitID': 'A258BSY2276', 'Nama': 'Ahmad Tiova Ian Avola', 'Universitas': 'Universitas Muhammadiyah Malang' },
                { 'BangkitID': 'A128BSY2319', 'Nama': 'Sandhi Karunia Sugihartana', 'Universitas': 'Politeknik Negeri Jakarta' },
            ],
            'Moto': 'Cihhh! Jangan meremehkan wibuuu, dasar Ninggen tidak bergunaa! >.< iKuzooo minnaa..',
            'CreatedBy': 'Aditya Bayu',
            'Copyright': '©2023 All Rights Reserved!'
        }
    }), HTTPStatus.OK

@app.route('/marine/predict', methods=['POST'])
@token_required
def predict_marine_classification(data):
    if data is None:
        return jsonify({
            'status': {
                'code': HTTPStatus.FORBIDDEN,
                'message': 'Akses dilarang',
            }
        }), HTTPStatus.FORBIDDEN
    if request.method == 'POST':
        reqImage = request.files['image']
        if reqImage and allowed_file(reqImage.filename):
            filename = secure_filename(reqImage.filename)
            reqImage.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = Image.open(image_path).convert('RGB')
            img = img.resize((160, 160))
            x = tf_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255
            classificationResult = model_marine_classification.predict(x, batch_size=1) # MODEL A
            class_list = ['Ikan', 'Udang']
            classification_class = class_list[np.argmax(classificationResult[0])]
            predicted_class = None
            if(classification_class == 'Ikan'):
                classes = model_marine_grading_fish.predict(x, batch_size=1) # MODEL B
                class_list = ['A', 'B', 'C']
                predicted_class = class_list[np.argmax(classes[0])]
            elif(classification_class == 'Udang'):
                classes = model_marine_grading_shrimp.predict(x, batch_size=1) # MODEL C
                class_list = ['A', 'B', 'C']
                predicted_class = class_list[np.argmax(classes[0])]
            else:
                predicted_class = 'Grade tidak tersedia'
            image_name = image_path.split('/')[-1]
            blob = bucket.blob('marine-images/' + image_name)
            blob.upload_from_filename(image_path) 
            os.remove(image_path)
            return jsonify({
                'status': {
                    'code': HTTPStatus.OK,
                    'message': 'Success predicting',
                    'data': { 'class': classification_class, 'grade': predicted_class }
                }
            }), HTTPStatus.OK
        else:
            return jsonify({
                'status': {
                    'code': HTTPStatus.BAD_REQUEST,
                    'message': 'Invalid file format. Please upload a JPG, JPEG, or PNG image.'
                }
            }), HTTPStatus.BAD_REQUEST
    else:
        return jsonify({
            'status': {
                'code': HTTPStatus.METHOD_NOT_ALLOWED,
                'message': 'Method not allowed'
            }
        }), HTTPStatus.METHOD_NOT_ALLOWED

@app.route('/sail_decision/predict', methods=['POST'])
@token_required
def predict_marine_sail_decision(data):
    if data is None:
        return jsonify({
            'status': {
                'code': HTTPStatus.FORBIDDEN,
                'message': 'Akses dilarang',
            }
        }), HTTPStatus.FORBIDDEN
    if request.method == 'POST':
        outlook = float(request.form['outlook'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        wind = float(request.form['wind'])
        data_predict = [[outlook, temperature, humidity, wind]]
        data_predict = np.array(data_predict, dtype=float)
        predicted = model_marine_sail_decision.predict(data_predict)
        result = float(predicted[0][0])
        result_class = ['Aman Untuk Berlayar' if result >= 0.6 else 'Tidak Aman Untuk Berlayar', 1 if result >= 0.6 else 0]
        return jsonify({
            'status': {
                'code': HTTPStatus.OK,
                'message': 'Success predicting',
                'data': { 'precentage': result, 'decision': result_class[0], 'class_predict': result_class[1]  }
            }
        }), HTTPStatus.OK
    else:
        return jsonify({
            'status': {
                'code': HTTPStatus.METHOD_NOT_ALLOWED,
                'message': 'Method not allowed'
            }
        }), HTTPStatus.METHOD_NOT_ALLOWED

def predict_price(data, model):
    prediction = model.predict(data)
    prediction = model_marine_scaler_price.inverse_transform(prediction)
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
@token_required
def predict_marine_price_prediction(data):
    if data is None:
        return jsonify({
            'status': {
                'code': HTTPStatus.FORBIDDEN,
                'message': 'Akses dilarang',
            }
        }), HTTPStatus.FORBIDDEN
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
                    'code': HTTPStatus.OK,
                    'message': 'Success predicting',
                    'data': { 'price' : int(price) }
                }
            }), HTTPStatus.OK
        else:
            data_new['Actual Price'] = model_marine_scaler_actual_price.transform(data_new[['Actual Price']])
            price = predict_price(data_new, model_marine_price)
            price = bulatkan_ke_kelipatan(price, 1000)
            return jsonify({
                'status': {
                    'code': HTTPStatus.OK,
                    'message': 'Success predicting',
                    'data': { 'price' : int(price) }
                }
            }), HTTPStatus.OK

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))