#%% Packages

from collections import OrderedDict
import json
import copy
from torchvision import models
from torch import nn
from torch import optim
import time
import torch
import os
import argparse
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#%% Argparse Model Setup
parser = argparse.ArgumentParser()

parser.add_argument("-d","--data_dir",default = '/data/')
parser.add_argument("-n","--num_epochs",type = int, default = 5)
parser.add_argument("-hid","--hidden_units",type = int, default = 4096)
parser.add_argument("-a","--arch",type = str, default = 'vgg11',choices=["vgg11", "densenet121"])
parser.add_argument("--lr",type = float, default = 0.001)
parser.add_argument("-m","--momentum",type = float, default = 0.9)
parser.add_argument("-ch","--checkpoint",type = str, default = 'chk3.pth')
parser.add_argument("-g","--use_gpu",type = int, default = 1)

args = parser.parse_args()
data_dir = args.data_dir

print(args.use_gpu)

#%% Functions
def train_model(model, criterion, optimizer,num_epochs = 5,denom = 10):

    best_model_weights = copy.deepcopy(model.state_dict()) # starting model weights
    best_acc = 0.0 # Starting accuracy - saves new weight if improved upon

    start_training = time.time()    
    
    for epoch in range(num_epochs):
        start_epoch = time.time()
        print('\n Epoch {} of {} \n'.format(epoch + 1, num_epochs))
        
        #### Training Phase
        phase = 'train'
        print('Train')
        model.train()  # Set model to training mode

        # Print Train Progress
        i = 0
        image_count = len(dataloaders[phase])
        progress_printer = int(image_count/denom)

        # Iterate over data
        for inputs, labels in dataloaders[phase]:
            i += 1
            if i % progress_printer == 0:
                print('{:.0f}% complete'.format(i/image_count*100))
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # zero the parameter gradients
            torch.set_grad_enabled(True) # Set gradients in train phase
            outputs = model(inputs) # forward pass
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward() # backprop
            optimizer.step() #optimize

        #### Validation Phase
        phase = 'val'
        print('Validation')
        model.eval()  # Set model to evaluation mode

        # zero running criterion loss and correct counts before iterating through images
        running_loss = 0.0 
        running_corrects = 0

        # Iterate over data
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            torch.set_grad_enabled(False) # Do not set gradients in val phase
            outputs = model(inputs) # forward pass
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]



        # save improved weights
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_weights = copy.deepcopy(model.state_dict())

        epoch_time = time.time() - start_epoch
        print('Epoch Time: {:.0f}m {:.0f}s'.format(epoch_time // 60, epoch_time % 60))
        print('Loss: {:.4f}'.format(epoch_loss))
        print('Acc: {:.4f}'.format(epoch_acc))
        
    training_time = time.time() - start_training
    print('\nTrain Time: {:.0f}m {:.0f}s'.format(training_time // 60, training_time % 60))    
    print('\nBest Validation Accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model

def validate(model,criterion,phase):
    optimizer.zero_grad()
    torch.set_grad_enabled(False)
    running_loss = 0
    running_corrects = 0
    
#    for inputs, labels in utils.dataloaders[phase]:
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs) # forward pass
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

#    print('{} loss: {:.4f}'.format(phase,running_loss / utils.dataset_sizes[phase]))
#    print('{} accuracy: {:.4f}'.format(phase,running_corrects.double() / utils.dataset_sizes[phase]))
    print('{} loss: {:.4f}'.format(phase,running_loss / dataset_sizes[phase]))
    print('{} accuracy: {:.4f}'.format(phase,running_corrects.double() / dataset_sizes[phase]))
    
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

def save_model(model,checkpoint_path):
    checkpoint = {
        'arch': args.arch,
        'class_to_idx': image_datasets['train'].class_to_idx,
        'state_dict': model.state_dict(),
        'hidden_units': args.hidden_units,
        'classifier':model.classifier
    }
    torch.save(checkpoint, checkpoint_path)

#%% Utilities
    
device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu == 1 else "cpu")
print(device)
    
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

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,shuffle=True, num_workers=4 if torch.cuda.is_available() else 0)
              for x in ['train', 'val','test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}

class_names = image_datasets['train'].classes

print(data_dir)
print(os.listdir(os.getcwd()))
print(os.getcwd())

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#Model Setup

model = getattr(models, args.arch)(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

if args.arch == 'densenet121':
    classifier_input_size = model.classifier.in_features
else:
    classifier_input_size = 25088

# vgg11 - input size 25088    
    
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(classifier_input_size,  args.hidden_units)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear( args.hidden_units, 102)),
    ('output', nn.LogSoftmax(dim=1))
    ]))
    
model.classifier = classifier

criterion = nn.CrossEntropyLoss()
model.class_to_idx = image_datasets['train'].class_to_idx

optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr, momentum=args.momentum)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Execution

model = train_model(model, criterion, optimizer,args.num_epochs,4) # Setup to print progress frequently
save_model(model,args.checkpoint)
