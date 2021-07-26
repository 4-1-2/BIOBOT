import os
from flask import Flask, request, jsonify, render_template, make_response
from flask_restful import Resource, Api
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image

from biobot.model import predict, get_model
from biobot.qa import OpenAIPlayGround

from cloudant import Cloudant
import ibm_boto3
from ibm_botocore.client import Config
import json

app = Flask(__name__)
CORS(app)
api = Api(app)

# Write image STORAGE IBM 
def cgsWriteImage(client, bucket, file, image):
    n = image.ndim
    if (n==3):
            img = Image.fromarray(image,'RGB')
    else:
        if (image.max()==1):
            img = Image.fromarray(image,'1').convert('RGB')  
        else:
            img = Image.fromarray(image,'L').convert('RGB')            
        
    bufImage = BytesIO()
    img.save(bufImage,"JPEG") 
    bufImage.seek(0)

    client.put_object(Bucket=bucket, 
                            Body = bufImage,
                            Key = file, 
                            ContentType = 'image/jpeg')
    print("""cgsWriteImage: 
        \n\tBucket=%s 
        \n\tFile=%s 
        \n\tArraySize=%d %s 
        RawSize=%d\n""" % (
            bucket, file, image.size, image.shape, bufImage.getbuffer().nbytes))

if os.path.exists('.credentials.json'):
    bucket = 'bucketbot'

    with open('.credentials.json') as f:
        data = json.load(f)

    # DB IBM 
    client = Cloudant.iam(
        data['API_KEY_DB'],
        data['KEY_DB'],
        connect=True
    )
    database_bot = client['biobot']

    # STORAGE IBM
    cgsClient = ibm_boto3.client(service_name='s3',
        ibm_api_key_id = data['ibm_api_key_id'],
        ibm_auth_endpoint= data['ibm_auth_endpoint'],
        config=Config(signature_version='oauth'),
        endpoint_url=data['endpoint_url'])

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
        #import pdb; pdb.set_trace()
        image = request.files['img']
        #io_image = base64.b64encode(image_cropped.read()).decode('utf-8')
        res1, res2 = predict(model, image)
        
        return { 'plant': res1, 'disease': res2}, 200

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