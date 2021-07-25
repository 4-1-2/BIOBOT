from .network import PlantDiseaseClassifier

def get_model():
    DEVICE = 'cpu:0'
    model = PlantDiseaseClassifier(DEVICE)
    return model

def predict(model, img_base64):
    return model.predict_disease(img_base64)