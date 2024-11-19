import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import time
from torch import nn, optim, tensor
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict
import json
from PIL import Image
from tqdm import tqdm

def input_args():
    parser = argparse.ArgumentParser(description="Train network on a dataset")
    
    parser.add_argument("--data_dir", type=str, default="flowers", help="Path to the dataset directory")
    parser.add_argument("--save_dir", type=str, default='flower_classifier_checkpoint.pth', help="Directory to save the model checkpoint")
    parser.add_argument("--arch", default="vgg16", help="Model architecture ('vgg16' or 'densenet')")
    parser.add_argument("--hidden_units", type=int, default=4096, help="Number of hidden units in the classifier")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--gpu", default='GPU', help="Use GPU or CPU for training")

    return parser.parse_args()


def train_model(data_dir, save_dir, arch, hidden_units, learning_rate, epochs, gpu):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'densenet':
        model = models.densenet121(pretrained=True)
        input_size = 1024
    else:
        raise ValueError("Unsupported architecture. Please choose 'vgg16' or 'densenet'.")
    
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.features[-3:].parameters():  
        param.requires_grad = True

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, len(image_datasets['train'].classes))),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier

    device = torch.device("cuda" if gpu.lower() == 'gpu' and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, labels in tqdm(dataloaders['train']):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        valid_loss, accuracy = validate_model(model, dataloaders['valid'], criterion, device)

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Training Loss: {train_loss/len(dataloaders['train']):.3f}, "
              f"Validation Loss: {valid_loss/len(dataloaders['valid']):.3f}, "
              f"Validation Accuracy: {accuracy:.3f}")

    test_accuracy = test_model(model, dataloaders['test'], device)
    print(f"Test Accuracy: {test_accuracy:.3f}")

    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'arch': arch,
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, save_dir)
    print("Model checkpoint saved!")

def validate_model(model, validloader, criterion, device):
    model.eval()
    valid_loss = 0
    accuracy = 0
    
    with torch.no_grad():
        for images, labels in validloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            valid_loss += criterion(outputs, labels).item()
            
            # Calculate accuracy
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return valid_loss, accuracy / len(validloader)

def test_model(model, testloader, device):
    model.eval()
    accuracy = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return accuracy / len(testloader)

if __name__ == "__main__":
    args = input_args()
    train_model(args.data_dir, args.save_dir, args.arch, args.hidden_units, args.learning_rate, args.epochs, args.gpu)
