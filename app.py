from flask import Flask, jsonify, request

import cv2
import numpy as np

from predict import face_detector, dog_detector, get_prediction


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        image_bytes = file.read()
        
        faces = face_detector(image_bytes, box=True)
        dog = dog_detector(image_bytes)
        entity = "unknown"

        prediction = ""
        if len(faces) > 0:
            entity = "human"
            
            image = np.asarray(bytearray(image_bytes), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            for (x, y, w, h) in faces:
                zoom_out_factor=.3
                h_factor = int(h*.3)
                w_factor = int(w*.3)
                # Select the region of interest that is the face in the image 
                roi = image[y-h_factor:y+h+h_factor, x-w_factor:x+w+w_factor]

                prediction = get_prediction(roi)

        elif dog:
            entity = "dog"
            prediction = get_prediction(image_bytes)
        
        return jsonify(**{'dog_class_name': prediction, 'detected_entity': entity})


if __name__ == '__main__':
    app.debug = True
    app.run()