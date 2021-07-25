import modules
import network
import base64

DEVICE = 'cuda:0'
TEST_PATH = 'dataset/test'

modules.test_classes_ids()
modules.accuracy_performance(TEST_PATH, DEVICE)

label, pred = modules.predict_random_image(TEST_PATH, DEVICE)

print('Label: {:s}, Predicted: {:s}'.format(label, pred))


model = network.PlantDiseaseClassifier(DEVICE)

filename1 = 'dataset/test/Corn_(maize)___Common_rust_/RS_Rust 1567.JPG'
filename2 = 'dataset/test/Corn_(maize)___healthy/0a1a49a8-3a95-415a-b115-4d6d136b980b___R.S_HL 8216 copy.jpg'

with open(filename1, "rb") as img_file:
    s1 = base64.b64encode(img_file.read()).decode('utf-8')

with open(filename2, "rb") as img_file:
    s2 = base64.b64encode(img_file.read()).decode('utf-8')

print(model.predict_disease(s1))
print(model.predict_disease(s2))

