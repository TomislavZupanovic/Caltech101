import torch
from torchvision import transforms, datasets
import numpy as np


def process_data(directory):
    train_dir = directory + '/train'
    valid_dir = directory + '/val'
    test_dir = directory + '/test'
    train_transformation = transforms.Compose([transforms.RandomRotation(20),
                                               transforms.RandomResizedCrop(224),
                                               transforms.ColorJitter(),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])
    valid_transformation = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

    test_transformation = valid_transformation
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transformation)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transformation)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transformation)

    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_data = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    return train_data, valid_data, test_data, train_dataset


def process_image(img):
    width = img.size[0]
    height = img.size[1]
    if width > height:
        img = img.resize((int(width*256 / height), height))
    else:
        img = img.resize((width, int(height*256 / width)))
    left_margin = (img.size[0] - 224) / 2
    right_margin = left_margin + 224
    upper_margin = (img.size[1] - 224) / 2
    lower_margin = upper_margin + 224
    img = img.crop((left_margin, upper_margin, right_margin, lower_margin))
    img = np.array(img).transpose((2, 0, 1)) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img = (img - mean) / std
    tensor_img = torch.Tensor(img)
    return tensor_img
