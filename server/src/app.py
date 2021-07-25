# DB 
from cloudant.client import Cloudant
from ibm_botocore.client import Config
import ibm_boto3
import os 
import sys
import types
import numpy
# now that the images is in a numpy.ndarray it needs to somehow be written to an object that represents a jpeg image
# the memory structure to hold that representation of the jpeg is a io.BytesIO object, suiteable for the Body arg of client.put_object
import io as libio
from PIL import Image
import cv2


def __iter__(self): return 0

# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
cgsClient = ibm_boto3.client(service_name='s3',
    ibm_api_key_id = 'pEiJMOMCgmFyjPjOv0lBap1b8hPe9yOJENJYR1H3ZI7k',
    ibm_auth_endpoint='https://iam.cloud.ibm.com/identity/token',
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.ap.cloud-object-storage.appdomain.cloud')


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

imgFile = 'ejm.jpg'
pic = Image.open(imgFile)
im = numpy.array(pic)

cgsWriteImage(cgsClient, 'bucketbot', imgFile, im)

'''

{
  "apikey": "53bPmE5u-1nbKfDF8kkj283BZwDeXljs20Q-D5CJut9t",
  "endpoints": "https://control.cloud-object-storage.cloud.ibm.com/v2/endpoints",
  "iam_apikey_description": "Auto-generated for key a1b631ad-02c0-4711-83b4-1856445feddc",
  "iam_apikey_name": "Service credentials-1",
  "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Manager",
  "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/21bcecb404444b53a002d19d087f0fd4::serviceid:ServiceId-5ea67e91-f851-40cf-9ee3-c790f3e6cfab",
  "resource_instance_id": "crn:v1:bluemix:public:cloud-object-storage:global:a/21bcecb404444b53a002d19d087f0fd4:9d6b22d4-160a-4174-8f37-3e7c5269a6e6::"
}


# client Storage IBMCLOUD 

def upload_file_cos(local_file_name,key):  
    cos = ibm_boto3.resource(service_name = 's3',
        ibm_api_key_id='53bPmE5u-1nbKfDF8kkj283BZwDeXljs20Q-D5CJut9t',
        ibm_service_instance_id='ServiceId-c1f0b923-ef69-458c-a78a-f78aba73e1b2',
        #ibm_auth_endpoint='s3.direct.ap.cloud-object-storage.appdomain.cloud',
        config=Config(signature_version='oauth'),
        endpoint_url='https://config.cloud-object-storage.cloud.ibm.com'
    )
    try:
        data = open(local_file_name, 'rb')
        cos.Bucket('my-bucket').put_object(Key=key, Body=data)
        #cos.upload_file(Filename=local_file_name, Bucket='bucketbiobot',Key=key)
    except Exception as e:
        print(Exception, e)
    else:
        print(' File Uploaded')

upload_file_cos('ejm.jpg', 'ejm.jpg')

# client DB IBMCLOUD
client = Cloudant.iam(
    "0543c3c0-716a-4fe4-8deb-bb2fd61dcd8e-bluemix",
    "vQGV09OznRD9YzobsBtUgBjFzyeGBYOwfSrYZxWLGYwu",
    connect=True
)
print(client.all_dbs())
'''
'''
import requests as req

config = {
    "apiKey": "AIzaSyAmFauSJcVZ6hDCHDe6R7u-udeFnDRsxgM",
    "authDomain": "agrobot-2477a.firebaseapp.com",
    "databaseURL": "https://agrobot-2477a-default-rtdb.firebaseio.com",
    "projectId": "agrobot-2477a",
    "storageBucket": "agrobot-2477a.appspot.com",
    "messagingSenderId": "498819222730",
    "appId": "1:498819222730:web:e5f8985474b9dbae463833",
    "measurementId": "G-4BSFR4WSL8"
}

firebase = pyrebase.initialize_app(config)

api_key = '827b9408e2582bbc7e4ec550027ccb62'

storage_ = firebase.storage()
db = firebase.database()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def basic():
    if request.method == 'POST':
        name = request.form['name']
        db.child("todo").push(name)
        todo = db.child("todo").get()
        to = todo.val()
        return render_template('index.html', t=to.values())
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image = request.files['img']
        lat = request.form['lat']
        lon = request.form['lon']
        #print(image.read())
        #!storage_.child('plants').put(image)
        url_weather = 'https://api.openweathermap.org/data/2.5/weather?lat={}&lon={}&appid={}'.format(lat, lon, api_key)
        res_weather = req.get(url_weather)
        return render_template('upload_image.html', image_up=res_weather.json())#res_weather)
    return render_template('upload_image.html')

#@app.route('/diagnosis', methods=['GET','POST'])
#def diagnosis():
#    if request.method == 'POST':
        
    #if request.method == 'POST':
    #    name = request.form
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
'''