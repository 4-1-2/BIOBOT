import io
import os
import torch
import base64
import torch.nn as nn
from PIL import Image

import torchvision.transforms as transforms

from biobot.model import modules

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = modules.ConvBlock(in_channels, 64)
        self.conv2 = modules.ConvBlock(64, 128, pool=True) # out_dim : 128 x 64 x 64 
        self.res1 = nn.Sequential(
            modules.ConvBlock(128, 128), 
            modules.ConvBlock(128, 128))
        
        self.conv3 = modules.ConvBlock(128, 256, pool=True) # out_dim : 256 x 16 x 16
        self.conv4 = modules.ConvBlock(256, 512, pool=True) # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(
            modules.ConvBlock(512, 512), 
            modules.ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
        
class PlantDiseaseClassifier(nn.Module):
    def __init__(self, device):
        super().__init__()
        
        self.device = device
        self.model = ResNet9(3, 38).to(self.device)
        
        
        self.load_model(os.path.join('params', 'model.pth'))
        self.test_path = 'dataset/test'
        list_disease = ['Apple___Apple_scab',
                        'Apple___Black_rot',
                        'Apple___Cedar_apple_rust',
                        'Apple___healthy',
                        'Blueberry___healthy',
                        'Cherry_(including_sour)___Powdery_mildew',
                        'Cherry_(including_sour)___healthy',
                        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                        'Corn_(maize)___Common_rust_',
                        'Corn_(maize)___Northern_Leaf_Blight',
                        'Corn_(maize)___healthy',
                        'Grape___Black_rot',
                        'Grape___Esca_(Black_Measles)',
                        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                        'Grape___healthy',
                        'Orange___Haunglongbing_(Citrus_greening)',
                        'Peach___Bacterial_spot',
                        'Peach___healthy',
                        'Pepper,_bell___Bacterial_spot',
                        'Pepper,_bell___healthy',
                        'Potato___Early_blight',
                        'Potato___Late_blight',
                        'Potato___healthy',
                        'Raspberry___healthy',
                        'Soybean___healthy',
                        'Squash___Powdery_mildew',
                        'Strawberry___Leaf_scorch',
                        'Strawberry___healthy',
                        'Tomato___Bacterial_spot',
                        'Tomato___Early_blight',
                        'Tomato___Late_blight',
                        'Tomato___Leaf_Mold',
                        'Tomato___Septoria_leaf_spot',
                        'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Tomato___Target_Spot',
                        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                        'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy']
        self.classes = sorted(list_disease)
        self.t = transforms.ToTensor()

    def load_model(self, load_path):
        state_dict = torch.load(load_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def decoce_image(self, s):
        decoded_bytes = base64.b64decode(s)
        img = Image.open(io.BytesIO(decoded_bytes))
        return img

    def split_label(self, label):
        plant, disease = label.split('___')
        
        if disease[::-1].find('_') == 0:
             disease = disease[::-1].replace('_', '', 1)[::-1]
        plant = plant.replace('_', ' ')
        disease = disease.replace('_', ' ')

        return plant, disease

    def predict_disease(self, s):
        img = self.decoce_image(s)
        img = self.t(img)

        x = torch.Tensor(img).unsqueeze(0).to(self.device)
        y = self(x)
        _, preds  = torch.max(y, dim=1)
        return self.split_label(self.classes[preds[0].item()])

    def forward(self, x):
        x = self.model(x)
        return x