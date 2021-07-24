import torch
import torch.nn as nn
import numpy as np

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

import torchvision.transforms as transforms


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) # out_dim : 128 x 64 x 64 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True) # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True) # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
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

def create_network(device):
    model = ResNet9(3, 38).to(device)
    return model

def load_model(device, load_path):
    from torchvision.datasets import ImageFolder
    model = create_network(device)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint)
    return model

def get_db(path):
    test_db = ImageFolder(path, transform=transforms.ToTensor()) 
    return test_db, test_db.classes

def load_test_db(path):
    test_db, _ = get_db(path)
    test_dl = DataLoader(test_db, 1, num_workers=2, pin_memory=True)

    return test_dl, len(test_db)

def eval_test_db(model, path, device):
    data_generator, data_size = load_test_db(path)

    model.eval()

    acc = []
    for it, batch in enumerate(data_generator):
        image_batch, label_batch = batch
        y = model(image_batch.to(device))
        acc_it = (torch.max(y.detach().cpu(), dim=1)[1] == label_batch)
        acc.append(acc_it)
        print('\r[{:4d}/{:4d}] acc = {:3.2f}'.format(it, data_size, acc_it.item()))
    return np.mean(acc)

def predict_image(filename, model, classes_id, device):
    img = Image.open(filename)
    x = torch.Tensor(img).unsqueeze(0).to(device)
    y = model(x)
    _, preds  = torch.max(y, dim=1)
    return classes_id[preds[0].item()]

def predict_db(model, path, device):
    model.eval()
    test_db, classes_id = get_db(path)
    x, pred_y = test_db[np.random.randint(len(test_db))]

    y_hat = model(x.to(device).unsqueeze(0))
    _, pred_y_hat  = torch.max(y_hat, dim=1)

    return classes_id[pred_y], classes_id[pred_y_hat[0].item()]


