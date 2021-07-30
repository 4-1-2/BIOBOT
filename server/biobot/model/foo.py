from .network import PlantDiseaseClassifier

def get_model():
    """
    Returns  the 'torch.nn.Module' model (a classifier).
    """
    DEVICE = 'cpu:0'
    model = PlantDiseaseClassifier(DEVICE)
    return model

def predict(model, img_base64):
    """
    Returns the prediction for a given image.

    Params:
        model: the neural network (classifier).
    """
    return model.predict_disease(img_base64)