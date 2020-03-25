import argparse

parser = argparse.ArgumentParser(description='Training APP')

parser.add_argument('data_dir', action="store", help="Set directory for training data")

parser.add_argument('--save_dir', action="store", dest= "save_dir", default='checkpoints', help="Set directory to save checkpoints")
parser.add_argument('--arch', action="store", dest="arch", default="vgg19", help="Choose architecture")
parser.add_argument('--learning_rate', action="store", dest="learning_rate", default=0.001, help="Set learning rate")
parser.add_argument('--hidden_units', action="store", dest="hidden_units", default=2048, help="Set hidden units")
parser.add_argument('--epochs', action="store", dest="epochs", default=3, help="Set epochs")
parser.add_argument('--gpu', action='store_true', default=False, dest='gpu', help="Set GPU mode")

print(parser.parse_args())
results = parser.parse_args()

import time
import matplotlib.pyplot as plt
import numpy as np
import os 
import random
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

data_dir = format(results.data_dir)
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = {
        'training' : transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ]),
        'validation' : transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ]),
        'testing' : transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])
}


# TODO: Load the datasets with ImageFolder
image_datasets = {
    'training' : datasets.ImageFolder(train_dir, transform=data_transforms['training']),
    'validation' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation']),
    'testing' : datasets.ImageFolder(test_dir, transform=data_transforms['testing'])  
}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    'training' : torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
    'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64, shuffle=False),
    'testing' : torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64, shuffle=False)
}

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
def build_model(input_size, hidden_size, output_size, arch):
    
    #if arch == 'vgg19':
        #model = models.vgg19(pretrained=True)
    #else:
       #print("Model " + arch + " not known")
       #exit()
    
    action = 'models.' + arch + '(pretrained=True)'
    model = eval(action)
    print('Using pretrained model ' + arch) 
    
    print("Building classifier with " + format(hidden_size) + " Hidden Units")
        
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(input_size, hidden_size)),
                                ('drop1', nn.Dropout(p=0.1)),
                                ('relu1', nn.ReLU()),
                                ('logits', nn.Linear(hidden_size, output_size)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))

    model.classifier = classifier
    return model

model = build_model(25088, results.hidden_units, 102, format(results.arch))

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=results.learning_rate)

epochs = int(results.epochs)
print_every = 40

def train(model, mode):
    steps = 0
    model.to(mode)

    for e in range(epochs):   
        running_loss = 0
        train_accuracy = 0

        model.train()

        for ii, (inputs, labels) in enumerate(dataloaders['training']):
            steps += 1

            inputs, labels = inputs.to(mode), labels.to(mode)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            ps = torch.exp(outputs).data
            equality = (labels.data == ps.max(dim=1)[1])
            train_accuracy += equality.type_as(torch.FloatTensor()).mean()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                        valid_loss = 0
                        valid_accuracy = 0
                        with torch.no_grad():
                            for images, labels in iter(dataloaders['validation']):

                                images, labels = images.to(mode), labels.to(mode)

                                output = model.forward(images)
                                valid_loss += criterion(output, labels).item()

                                ps = torch.exp(output)
                                equality = (labels.data == ps.max(dim=1)[1])
                                valid_accuracy += equality.type(torch.FloatTensor).mean()


                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Accuracy: {:.3f}.. ".format(train_accuracy/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(dataloaders['validation'])),
                      "Validation Accuracy: {:.3f}.. ".format(valid_accuracy/len(dataloaders['validation'])))

                running_loss = 0
                train_accuracy = 0
                model.train()
           
print("Training Model with " + format(results.epochs) + " Epochs and a learning rate of " + str(results.learning_rate))

if results.gpu == True:
    print('Using GPU mode')
    mode = 'cuda'
else:
    print('Using CPU Mode')
    mode = 'cpu'
    
train(model, mode)

def model_test(model, mode):
    
    model.eval()
    model.to(mode)
            
    with torch.no_grad():
        test_loss = 0
        test_accuracy = 0
        with torch.no_grad():
            for images, labels in iter(dataloaders['testing']):
            
                images, labels = images.to(mode), labels.to(mode)
            
                output = model.forward(images)
                test_loss += criterion(output, labels).item()

                ps = torch.exp(output)
                equality = (labels.data == ps.max(dim=1)[1])
                test_accuracy += equality.type(torch.FloatTensor).mean()

    print('Accuracy of the network on the test images: %d %%' % (100 * test_accuracy/len(dataloaders['testing'])))
   
print("Testing model")
model_test(model, mode)

print("Saving model")
model.class_to_idx = image_datasets['training'].class_to_idx
model.cpu()
checkpoint = {'arch': format(results.arch),
              'input_size': 25088,
              'output_size': 102,
              'hidden_layer': results.hidden_units,
              'state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict,
              'class_to_idx': model.class_to_idx,
              'epochs': epochs
             }

if not os.path.exists(results.save_dir):
        print('Creating dir {}'.format(results.save_dir))
        os.mkdir(results.save_dir)
 
savefile = results.save_dir + '/checkpoint.pth'
torch.save(checkpoint, savefile)

print("Model saved " + savefile)