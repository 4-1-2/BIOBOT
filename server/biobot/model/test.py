from .network import PlantDiseaseClassifier
import os 

# STORAGE IBM
#!cgsClient = ibm_boto3.client(service_name='s3',
#!    ibm_api_key_id = 'pEiJMOMCgmFyjPjOv0lBap1b8hPe9yOJENJYR1H3ZI7k', # api-key ibm
#!    ibm_auth_endpoint='https://iam.cloud.ibm.com/identity/token', #static
#!    config=Config(signature_version='oauth'), # static
#!    endpoint_url='https://s3.ap.cloud-object-storage.appdomain.cloud') # endpoints


#!def download_params():


#!if not os.path.exists('params'):
#!    os.mkdir('params')
    
#!if not os.path.exists('params/model.pth'):
#!    download_params()

DEVICE = 'cpu:0'
TEST_PATH = 'dataset/test'

#modules.test_classes_ids()
#modules.accuracy_performance(TEST_PATH, DEVICE)

#label, pred = modules.predict_random_image(TEST_PATH, DEVICE)

#print('Label: {:s}, Predicted: {:s}'.format(label, pred))


model = PlantDiseaseClassifier(DEVICE)


##filename1 = 'dataset/test/Corn_(maize)___Common_rust_/RS_Rust 1567.JPG'
##filename2 = 'dataset/test/Corn_(maize)___healthy/0a1a49a8-3a95-415a-b115-4d6d136b980b___R.S_HL 8216 copy.jpg'

def run(img_io):
    #with open(img_io, "rb") as img_file:
    #s1 = base64.b64encode(img_io.read()).decode('utf-8')
    return model.predict_disease(img_io)

#with open(filename1, "rb") as img_file:
#    s1 = base64.b64encode(img_file.read()).decode('utf-8')

#with open(filename2, "rb") as img_file:
#    s2 = base64.b64encode(img_file.read()).decode('utf-8')


