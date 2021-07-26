import os
from flask import Flask, request, jsonify, render_template, make_response
from flask_restful import Resource, Api
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image

from biobot.model import predict, get_model
from biobot.qa import OpenAIPlayGround, get_suggested_question

from cloudant import Cloudant
import ibm_boto3
from ibm_botocore.client import Config

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
        data = request.get_json(force=True)
        image = data['img']
        #io_image = base64.b64encode(image).decode('utf-8')
        plant, disease = predict(model, image)
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