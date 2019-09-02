from comet_ml import Experiment
from PIL import Image
import os
import os.path
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

svhn = datasets.SVHN(root='./dataset', download=False, transform=transform, split='extra')
svhn_vali = datasets.SVHN(root='./dataset', download=False, transform=transform, split='train')
svhn_test = datasets.SVHN(root='./dataset', download=False, transform=transform, split='test')


train_loader = torch.utils.data.DataLoader(dataset=svhn,
                                           batch_size=4096,
                                           shuffle=True,
                                           )

validation_loader = torch.utils.data.DataLoader(dataset=svhn_vali,
                                           batch_size=1024,
                                           shuffle=True,
                                           )
test_loader = torch.utils.data.DataLoader(dataset=svhn_test,
                                          batch_size=512,
                                          shuffle=True,
                                          )


## model definition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 32, 32)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),  # output shape (16,32,32)
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # (16,16,16)
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(  # input shape (16,16,16)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 16, 16)
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 8, 8)
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(  # input shape(32,8,8)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),  # (64,8,8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (64,4,4)
            nn.Dropout(0.2)
        )
        self.fc1 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        output = self.out(x)
        return output


# Hyper Parameters
EPOCH = 50  # train the training data n times, to save time, we just train 1 epoch
LR = 0.001  # learning rate
TRAIN_ON_GPU = True

cnn = CNN()

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="xxxxx",
                        project_name="xxxxx", workspace="xxxxx")
experiment.add_tags(["svhn epoch{}".format(EPOCH)])
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()
print(cnn)

if TRAIN_ON_GPU:
    cnn.cuda()

train_loss = []
vali_loss = []
valid_loss_min = 99999.9
for epoch in range(EPOCH):
    with experiment.train():
        cnn.train()
        total_loss = 0.0
        for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            if TRAIN_ON_GPU:
                b_x, b_y = b_x.cuda(), b_y.cuda()
                b_y = b_y.type(torch.cuda.LongTensor)
            else:
                b_y = b_y.type(torch.LongTensor)

            output = cnn(b_x)  # cnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            total_loss += loss.item()
        experiment.log_metric("train_loss", total_loss)

    with experiment.validate():
        with torch.no_grad():
            cnn.eval()
            total_validation_loss = 0.0
            for i, (batch_x, batch_y) in enumerate(validation_loader):
                if TRAIN_ON_GPU:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                    batch_y = batch_y.type(torch.cuda.LongTensor)
                else:
                    batch_y = batch_y.type(torch.LongTensor)

                output = cnn(batch_x)  # cnn output
                loss = loss_func(output, batch_y)  # cross entropy loss
                total_validation_loss += loss.item()
        experiment.log_metric("validation_loss",
                              total_validation_loss)  # this will be logged as validation accuracy based on the context.
        print('Epoch:{}, Training Loss: {:.6f}, Validation Loss: {:.6f}'.format(epoch, total_loss,total_validation_loss))

        # save model if validation loss has decreased
        if total_validation_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                total_validation_loss))
            model_name = os.path.join('/home/cili/svhn/models/model.pt'.format(epoch))
            torch.save(cnn.state_dict(), model_name)
            valid_loss_min = total_validation_loss

with experiment.test():
    with torch.no_grad():
        model = CNN()
        model.load_state_dict(torch.load('/home/cili/svhn/models/model.pt'))
        #model.load_state_dict('/home/cili/svhn/models/model.pt')
        model.eval()

        correct, total_item = 0, 0
        for i, (batch_x, batch_y) in enumerate(test_loader):
            if TRAIN_ON_GPU:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                batch_y = batch_y.type(torch.cuda.LongTensor)
            else:
                batch_y = batch_y.type(torch.LongTensor)

            test_output = model(batch_x)
            test_output = test_output.float()
            if TRAIN_ON_GPU:
                pred_y = torch.max(test_output, 1)[1].cuda().data  # data.squeeze()
            else:
                pred_y = torch.max(test_output, 1)[1].data.numpy()
            total_item += batch_y.shape[0]
            if TRAIN_ON_GPU:
                correct += (pred_y == batch_y).sum().item()
            else:
                correct += (pred_y == batch_y).astype(int).sum().item()

    accuracy = float(correct) / float(total_item)
    experiment.log_metric("test accuracy", 100 * accuracy)
    print('Epoch:{}, Accuracy: {}'.format(EPOCH, accuracy))
