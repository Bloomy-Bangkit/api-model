import os
import numpy as np
from PIL import Image
from flask import jsonify, request
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

def marine(app, allowed_file, model_classification, model_grading):
    if request.method == 'POST':
        reqImage = request.files['image']
        if reqImage and allowed_file(reqImage.filename):
            filename = secure_filename(reqImage.filename)
            reqImage.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = Image.open(image_path).convert("RGB")
            img = img.resize((150, 150))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255
            
            classificationResult = model_classification.predict(x, batch_size=1)
            class_list = ['Ikan', 'Udang']
            classification_class = class_list[np.argmax(classificationResult[0])]
            
            classes = model_grading.predict(x, batch_size=1)
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