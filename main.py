import os
import numpy as np
from PIL import Image
from flask import Flask, jsonify
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

load_dotenv()
app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_MARINE_CLASSIFICATION'] = 'models/marine_classification.h5'
app.config['MODEL_MARINE_GRADING'] = 'models/marine_grading.h5'
app.config['MODEL_MARINE_PRICE_PREDICTION'] = 'models/marine_price_prediction.h5'
app.config['MODEL_MARINE_SAIL_DECISION'] = 'models/marine_sail_decision.h5'

model_marine_classification = load_model(app.config['MODEL_MARINE_CLASSIFICATION'], compile=False)
model_marine_grading = load_model(app.config['MODEL_MARINE_GRADING'], compile=False)
model_marine_price_prediction = load_model(app.config['MODEL_MARINE_PRICE_PREDICTION'], compile=False)
model_marine_sail_decision = load_model(app.config['MODEL_MARINE_SAIL_DECISION'], compile=False)

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

# @app.route('/marine/predict', methods=['POST'])
# @auth.login_required()
# def predict_marine_classification():
#     return marine(app, allowed_file, model_marine_classification, model_marine_grading)

# @app.route('/price/predict', methods=['POST'])
# @auth.login_required()
# def predict_marine_price_prediction():
#     return marine_price_prediction(model_marine_price_prediction)

# @app.route('/sail_decision/predict', methods=['POST'])
# @auth.login_required()
# def predict_marine_sail_decision():
#     return marine_sail_decision(model_marine_sail_decision)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))