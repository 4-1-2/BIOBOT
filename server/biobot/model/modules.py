import os
import torch
import torch.nn as nn
import numpy as np
import base64

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from biobot.model import network

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

def create_network(device):
    model = network.ResNet9(3, 38).to(device)
    return model

def load_model(device, load_path):
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

def eval_accuracy(model, path, device):
    model.eval()
    data_generator, data_size = load_test_db(path)
    acc = []
    for it, batch in enumerate(data_generator):
        image_batch, label_batch = batch
        y = model(image_batch.to(device))
        acc_it = (torch.max(y.detach().cpu(), dim=1)[1] == label_batch)
        acc.append(float(acc_it.item()))
        print('\r[{:4d}/{:4d}] acc = {:3.5f}%'.format(
            it, data_size, np.mean(acc) * 100.), end='')
    accuracy = np.mean(acc)
    print('\r[{:4d}/{:4d}] acc = {:3.5f}%'.format(
        it, data_size, accuracy * 100.))
    return accuracy

def test_classes_ids():
    test_path = 'dataset/test'

    test1 = sorted(os.listdir(test_path))
    test2 = get_db(test_path)[1]
    for t1, t2 in zip(test1, test2):
        assert t1 == t2

def accuracy_performance(path, device):
    model = load_model(device, os.path.join('params', 'model.pth'))
    assert eval_accuracy(model, path, device) >= 0.95

def predict_random_image(path, device):
    model = load_model(device, os.path.join('params', 'model.pth'))
    model.eval()
    test_db, classes_id = get_db(path)
    x, pred_y = test_db[np.random.randint(len(test_db))]

    y_hat = model(x.to(device).unsqueeze(0))
    _, pred_y_hat  = torch.max(y_hat, dim=1)

    return classes_id[pred_y], classes_id[pred_y_hat[0].item()]



