import torch
from torchvision import models
from torch import nn, optim
from .utils import save, show_image
import pandas as pd
from .preprocess import process_image
import matplotlib.pyplot as plt
import numpy as np


class Model(object):
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

    def build(self, criterion, lr, train_dataset):
        self.lr = lr
        fc_layers = nn.Sequential(nn.Linear(2048, 512),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(512, 101),
                                  nn.LogSoftmax(dim=1))
        self.model.fc = fc_layers
        if criterion == 'NLLLoss':
            self.criterion = nn.NLLLoss()
        elif criterion == 'CEL':
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=self.lr)
        self.model.class_to_idx = train_dataset.class_to_idx
        self.model.idx_to_class = {idx: class_ for class_, idx in self.model.class_to_idx.items()}

    def fit(self, epochs, train_data, valid_data):
        min_valid_loss = np.Inf
        losses = []
        for e in range(epochs):
            self.model.train()
            self.model.cuda()
            train_loss = 0.0
            train_acc = 0
            for images, labels in train_data:
                images, labels = images.cuda(), labels.cuda()
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * images.size(0)
                _, pred = torch.max(output, dim=1)
                correct = pred.eq(labels.data.view_as(pred))
                accuracy = torch.mean(correct.type(torch.FloatTensor))
                train_acc += accuracy.item() * images.size(0)
            else:
                with torch.no_grad():
                    self.model.eval()
                    valid_loss = 0.0
                    valid_acc = 0
                    for images, labels in valid_data:
                        images, labels = images.cuda(), labels.cuda()
                        output = self.model(images)
                        loss = self.criterion(output, labels)
                        valid_loss += loss.item() * images.size(0)
                        _, pred = torch.max(output, dim=1)
                        correct = pred.eq(labels.data.view_as(pred))
                        accuracy = torch.mean(correct.type(torch.FloatTensor))
                        valid_acc += accuracy.item() * images.size(0)
                    train_loss = train_loss / len(train_data.dataset)
                    train_acc = train_acc / len(train_data.dataset)
                    valid_loss = valid_loss / len(valid_data.dataset)
                    valid_acc = valid_acc / len(valid_data.dataset)
                    losses.append([train_loss, valid_loss, train_acc, valid_acc])
                    print("Completed epoch {}/{}".format(e+1, epochs))
                    print("Train Loss: {:.3f}, Train accuracy: {:.1f}%".format(train_loss, train_acc * 100))
                    print("Valid Loss: {:.3f}, Valid accuracy: {:.1f}%".format(valid_loss, valid_acc * 100))
                    if valid_loss <= min_valid_loss:
                        print("Valid Loss decreased: {:.3f} --> {:.3f} Saving model...".format(
                            min_valid_loss, valid_loss))
                        save(self.model, 'model_Caltech101')
                        min_valid_loss = valid_loss
                    print('\n')
        losses = pd.DataFrame(losses, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
        return self.model, losses

    def load_weights(self, name):
        model_save = torch.load('saved_models/' + name + '.pth')
        self.model.fc = model_save['fc']
        self.model.load_state_dict(model_save['state_dict'])
        self.model.class_to_idx = model_save['class_to_index']
        self.model.idx_to_class = model_save['index_to_class']
        self.model.cuda()
        return self.model

    def evaluate(self, iterations, test_data):
        self.model.cuda()
        for i in range(iterations):
            with torch.no_grad():
                self.model.eval()
                test_acc = 0
                for images, labels in test_data:
                    images, labels = images.cuda(), labels.cuda()
                    output = self.model(images)
                    _, pred = torch.max(output, dim=1)
                    correct = pred.eq(labels.data.view_as(pred))
                    accuracy = torch.mean(correct.type(torch.FloatTensor))
                    test_acc += accuracy.item() * images.size(0)
                test_acc = test_acc / len(test_data.dataset)
                print('Iteration {}/{}'.format(i + 1, iterations))
                print('Test accuracy: {:.1f}%'.format(test_acc * 100))
        self.model.train()

    def predict(self, image_path, topk=5):
        self.model.eval()
        image = process_image(image_path)
        image = image.view(1, 3, 224, 224).cuda()
        with torch.no_grad():
            probability = torch.exp(self.model(image))
            top_prob, top_class = probability.topk(topk)
            classes = [self.model.idx_to_class[class_] for class_ in top_class.cpu().numpy()[0]]
            top_prob = top_prob.cpu().numpy()[0]
        image = image.cpu().squeeze()
        plt.figure(figsize=(10, 4))
        ax = plt.subplot(1, 2, 1)
        ax, img = show_image(image, image_path, ax=ax)
        ax.axis('off')
        ax1 = plt.subplot(1, 2, 2)
        y = np.arange(topk)
        ax1.set_yticks(y)
        ax1.set_yticklabels(classes)
        ax1.set_xlabel('Probability')
        ax1.invert_yaxis()
        ax1.barh(y, top_prob)
        plt.tight_layout()
        plt.show()


