
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from biobot.model import predict, get_model
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)
api = Api(app)

model = get_model()

class Diagnosis(Resource):
    def post(self):
        image = request.files['img']
        image_ = Image.open(image)
        new_width, new_height = 256, 256
        width, height = image_.size   # Get dimensions

        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2

        # Crop the center of the image
        image_cropped = image_.crop((left, top, right, bottom))
        im_file = BytesIO()
        # -*- coding: utf-8 -*-
        
        image_cropped.save(im_file, format='JPEG') 
        binary_data = im_file.getvalue() 
        io_image = base64.b64encode(binary_data)
        #io_image = base64.b64encode(image_cropped.read()).decode('utf-8')
        res1, res2 = predict(io_image, model)

        return { 'plant': res1, 'disease': res2}, 200

    
api.add_resource(Diagnosis, '/diagnosis')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)