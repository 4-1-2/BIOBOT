# TODO: DEPRECATED
# use rest_app.py
from cloudant import Cloudant
from flask import Flask, render_template, request, jsonify
from ibm_botocore.client import Config
import ibm_boto3
import numpy as np
import atexit
import os
import json
import io as libio
from PIL import Image
app = Flask(__name__)
from biobot.model import predict, get_model
import base64
from io import BytesIO

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
        
    bufImage = libio.BytesIO()
    img.save(bufImage,"JPEG") 
    bufImage.seek(0)

    isr = client.put_object(Bucket=bucket, 
                            Body = bufImage,
                            Key = file, 
                            ContentType = 'image/jpeg')
    print("cgsWriteImage: \n\tBucket=%s \n\tFile=%s \n\tArraySize=%d %s RawSize=%d\n" % (bucket, file, image.size, image.shape, bufImage.getbuffer().nbytes))

# DB IBM 
client = Cloudant.iam(
    "0543c3c0-716a-4fe4-8deb-bb2fd61dcd8e-bluemix",
    "vQGV09OznRD9YzobsBtUgBjFzyeGBYOwfSrYZxWLGYwu",
    connect=True
)
database_bot = client['biobot']

# STORAGE IBM
cgsClient = ibm_boto3.client(service_name='s3',
    ibm_api_key_id = 'pEiJMOMCgmFyjPjOv0lBap1b8hPe9yOJENJYR1H3ZI7k',
    ibm_auth_endpoint='https://iam.cloud.ibm.com/identity/token',
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.ap.cloud-object-storage.appdomain.cloud')

#!im = numpy.array(pic)

# On IBM Cloud Cloud Foundry, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000
model = get_model()

port = int(os.getenv('PORT', 8000))

@app.route('/', methods=['GET', 'POST'])
def basic():
    if request.method == 'POST':
        name = request.form['name']
        partition_key = 'Humans'
        document_key = 'julia30'
        database_bot.create_document({
            '_id': ':'.join((partition_key, document_key)),
            'name': name
        })
        return render_template('index.html', t=name)
    return render_template('index.html')

# Diagnosis
@app.route('/diagnosis', methods=['GET', 'POST'])
def run_diagnosis():
    if request.method == 'POST':
        #import pdb; pdb.set_trace()
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
        
        return render_template('upload_image.html', image_up=res1+' '+ res2)
    return render_template('upload_image.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
