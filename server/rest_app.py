import os
from flask import Flask, request, jsonify, render_template, make_response
from flask_restful import Resource, Api
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image

from biobot.model import predict, get_model
from biobot.qa import OpenAIPlayGround, get_suggested_question

app = Flask(__name__)
CORS(app)
api = Api(app)

model = get_model()
gpt3api = OpenAIPlayGround('.openaikey.txt')

class basic(Resource):
    def post(self):
        name = request.form['name']
        return make_response(render_template('index.html', t=name))
    def get(self):
        return make_response(render_template('index.html'))

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
        plant, disease = predict(model, io_image)
        suggested_initial_question = get_suggested_question(plant, disease)

        return { 'plant': plant, 'disease': disease, 'suggested_question': suggested_initial_question}, 200


class ChatBot(Resource):
    def post(self):
        data = request.get_json(force=True)
        new_text = data['question']
        chat_acumm = data['chat_acumm']
        response = gpt3api(new_text, chat_acumm)
        return response, 200


#api.add_resource(basic, '/')
api.add_resource(Diagnosis, '/diagnosis')
api.add_resource(ChatBot, '/chatbot')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)