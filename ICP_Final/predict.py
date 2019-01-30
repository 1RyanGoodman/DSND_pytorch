#%% Packages

from torchvision import models
from torch import nn
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from torch.autograd import Variable
import os
import argparse

#%% Argparse Model Setup

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument("-d","--data_dir",default = '/data/')
parser.add_argument("-i","--image_path",type = str, default = 'test/80/image_01983.jpg')
parser.add_argument("-ch","--checkpoint",type = str, default = 'checkpoint1.pth')
parser.add_argument("--topk",type = int, default = 5)
parser.add_argument("-g","--use_gpu",type = int, default = 1)


args = parser.parse_args()
data_dir = args.data_dir

def process_image(image):
    img_transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    pil_image = Image.open(image)
    pil_image = img_transform(pil_image).float()
    np_image = np.array(pil_image)    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
    return np_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
#    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    print(checkpoint["arch"])
    model.classifier = checkpoint['classifier']        
    model.state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu == 1 else "cpu")
    model = model.to(device)
    return model

def predict(image_path, model, n=5):
    model.eval()
    np_array = process_image(image_path)
    tensor = torch.from_numpy(np_array)
    if args.use_gpu == 1 and torch.cuda.is_available():
        # cuda
        model = model.cuda()
        inputs = Variable(tensor.float().cuda(), volatile=True)
        inputs = inputs.unsqueeze(0)
        output = model.forward(inputs)  
        predictions = torch.exp(output).data.topk(n)
        probabilities = predictions[0].cpu()
        classes = predictions[1].cpu()
    else:
        # cpu
        model = model.cpu()
        model.double()
        inputs = Variable(tensor, volatile=True)
        inputs = inputs.unsqueeze(0)
        output = model.forward(inputs)  
        predictions = torch.exp(output).data.topk(n)
        probabilities = predictions[0]
        classes = predictions[1]                                         
    class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    for label in classes.numpy()[0]:
        mapped_classes.append(class_to_idx_inverted[label])
    return probabilities.numpy()[0], mapped_classes

def show_predictions(test_image,model):
    probabilities, classes = predict(test_image, model)
    fig = plt.figure(figsize=(8,8))
    ax1 = plt.subplot2grid((20,10), (0,0), colspan=10, rowspan=10)
    ax2 = plt.subplot2grid((20,10), (10,2), colspan=5, rowspan=8)
    image = Image.open(test_image)
#    ax1.set_title(cat_to_name['80'])
    ax1.imshow(image)
    flower_labels = []
    for i in classes:
        flower_labels.append(cat_to_name[i])
    y_pos = np.arange(5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(flower_labels)
    ax2.invert_yaxis()
    ax2.set_xlabel('Probability')
    ax2.barh(y_pos, probabilities)
    plt.show()  
    
#%% Utilities
    
train_dir = data_dir + 'flowers/train/'
valid_dir = data_dir + 'flowers/valid/'
test_dir = data_dir + 'flowers/test/'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(valid_dir, transform=data_transforms['val'])
image_datasets['test'] = datasets.ImageFolder(test_dir, transform=data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4 if torch.cuda.is_available() else 0)
              for x in ['train', 'val','test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}

class_names = image_datasets['train'].classes

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu == 1 else "cpu")
print(device)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()

print(args.checkpoint)
print(os.getcwd())

model = load_checkpoint( args.checkpoint)

#Doesn't show images through command line here
# https://knowledge.udacity.com/questions/10729
#show_predictions(data_dir + '/flowers/' + args.image_path,model)

probabilities, labels = predict(data_dir + '/flowers/' + args.image_path, model,args.topk)

flower_labels = []
for i in labels:
    flower_labels.append(cat_to_name[i])
        
print(labels)
print(flower_labels)
print(probabilities)
