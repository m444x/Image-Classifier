import argparse
import random
import os 
import random
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image


parser = argparse.ArgumentParser(description='Prediction APP')

parser.add_argument('image_path', action="store", help="Location of image to predict")
parser.add_argument('checkpoint', action="store", help="Location of last checkpoint for prediction")

parser.add_argument('--top_k', action="store", dest= "top_k", default=5, help="Number of most likely classes")
parser.add_argument('--category_names', action="store", dest="category_names", default='cat_to_name.json', help="Mapping of categories to real names")
parser.add_argument('--gpu', action='store_true', default=False, dest='gpu', help="Set GPU mode")

print(parser.parse_args())
results = parser.parse_args()

def build_model(input_size, hidden_size, output_size, arch):
    
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


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    model = build_model(checkpoint['input_size'], checkpoint['hidden_layer'], checkpoint['output_size'], checkpoint['arch'])

    model.class_to_idx = checkpoint['class_to_idx']    

    model.load_state_dict(checkpoint['state_dict'])
    return model




def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    im.load()
    im = im.resize((256,256))
    value = 0.5*(256-224)
    im = im.crop((value,value,256-value,256-value))
    
    im = np.array(im)/255
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225]) 
    im = (im - mean)/std
    
    im = im.transpose((2, 0, 1))
    return im

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    if gpu  == True:
        mode = 'cuda'
        print('Running in GPU Mode...')
    else:
        mode = 'cpu'
        print('Running in CPU Mode...')
     
    model.to(mode)

    model.eval()
    image = process_image(image_path)
    image = torch.from_numpy(np.array([image])).float()
    image = Variable(image).to(mode)
        
    output = model.forward(image)
    probabilities = torch.exp(output).data
   
    prob = torch.topk(probabilities, topk)[0].tolist()[0] 
    index = torch.topk(probabilities, topk)[1].tolist()[0]
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    # transfer index to label
    label = []
    for i in range(topk):
        label.append(ind[index[i]])

    return prob, label

import json

with open(format(results.category_names), 'r') as f:
    cat_to_name = json.load(f)

model_load = load_checkpoint(format(results.checkpoint))
model_load

image = format(results.image_path)

probs, classes = predict(image, model_load, int(results.top_k), results.gpu)

labels = []
for cl in classes:
    labels.append(cat_to_name[cl])

print(probs)
print(labels)

#How to run
# python predict.py ./flowers/test/1/image_06752.jpg checkpoint/checkpoint.pth --top_k 3 