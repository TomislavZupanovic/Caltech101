import torch
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def save(model, name):
    model_save = {'state_dict': model.state_dict(),
                  'class_to_index': model.class_to_idx,
                  'index_to_class': model.idx_to_class,
                  'fc': model.fc}
    torch.save(model_save, 'saved_models/' + name + '.pth')


def load(name):
    model_save = torch.load('saved_models/' + name)
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = model_save['fc']
    model.load_state_dict(model_save['state_dict'])
    model.class_to_idx = model_save['class_to_index']
    model.idx_to_class = model_save['index_to_class']
    model.cuda()
    return model


def show_image(image, image_path, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    title = image_path.split('/')[2]
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')
    return ax, image


def random_image():
    test_data_path = 'source/dataset/test/'
    category = random.choice(os.listdir(test_data_path))
    rand_image = random.choice(os.listdir(test_data_path + str(category) + '/'))
    img_path = test_data_path + str(category) + '/' + rand_image
    return img_path

