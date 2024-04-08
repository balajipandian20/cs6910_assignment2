import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
import lightning.pytorch as pl
from torchmetrics.functional import accuracy

from lightning.pytorch.callbacks import ModelCheckpoint

wandb.login(key='fd85ae65c5e04fb6fea4d31c1180348532db32d6')
wandb_project= 'EE22M008_Assignment_02'

def Data_Preprocessing(augment_data, train_dir, test_dir, batch_size):
  if augment_data:
    # Define transformations for data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust mean and std as needed
    ])
  else:
      train_transform = transforms.Compose([
              transforms.Resize((256, 256)),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
          ])


  test_transform = transforms.Compose([
              transforms.Resize((256, 256)),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
          ])

  # Define dataset
  train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
  test_dataset= datasets.ImageFolder(root=test_dir, transform=test_transform)

  # Determine sizes for train and validation sets
  total_size = len(train_dataset)
  train_size = int(0.8 * total_size)
  val_size = total_size - train_size

  # Randomly split the dataset into train and validation sets
  train_set, val_set = random_split(train_dataset, [train_size, val_size])

  # Create data loaders for train and validation sets
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  return train_loader, val_loader, test_loader

class Convolutional_Neural_Networks(pl.LightningModule):
  def __init__(self, num_channels, num_conv_blocks, num_filters, filter_size, num_classes, activation_function, pooling_choice, num_fc_layers, dense_neurons, drop_rate, bn, optimizer, lr, wd):
    super(Convolutional_Neural_Networks, self).__init__()
    self.num_channels= num_channels
    self.num_conv_blocks = num_conv_blocks
    self.num_filters = num_filters
    self.filter_size = filter_size
    self.num_classes= num_classes
    self.pooling_layer = pooling_choice
    self.num_fc_layers = num_fc_layers
    self.dense_neurons = dense_neurons
    self.batch_normalization= bn
    self.optimisers= optimizer
    self.learning_rate= lr
    self.weight_decay= wd

    if activation_function=='ReLU':
      self.activation_function= nn.ReLU()
    elif activation_function=='GeLU':
      self.activation_function= nn.GELU()
    elif activation_function== 'SeLU':
      self.activation_function= nn.SELU()
    elif activation_function=='SiLU':
      self.activation_function= nn.SiLU()
    elif activation_function=='Mish':
      self.activation_function= nn.Mish()

    # Define convolutional layers with specified activation and pooling
    self.conv_layers = nn.ModuleList()
    for i in range(self.num_conv_blocks):
      # Determine input channels for the current conv layer
      in_channels = self.num_filters[i-1] if i > 0 else self.num_channels
      conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=self.num_filters[i], kernel_size=self.filter_size)
      self.conv_layers.append(conv_layer)
    self.drop_out_cn= nn.Dropout2d(drop_rate)
    self.drop_out_fc= nn.Dropout(drop_rate)
    self.softmax = nn.LogSoftmax(dim=1)

    self.save_hyperparameters()

  def forward(self, x):
    # Pass input through convolutional layers
    for layer in self.conv_layers:
      x = layer(x)
      if self.batch_normalization:
        bn= nn.BatchNorm2d(x.size(1))
        x= bn(x)
      x= self.drop_out_cn(x)
      x = self.activation_function(x)
      if self.pooling_layer=='max':
        pl= nn.MaxPool2d(kernel_size=2, stride=2)
      elif self.pooling_layer=='avg':
        pl= nn.AvgPool2d(kernel_size=2, stride=2)
      x= pl(x)
      x= self.drop_out_cn(x)
    # Flatten the output from the last convolution layers
    x = x.view(x.size(0), -1)

    # Pass through fully connected layers
    for i in range(self.num_fc_layers):
      fc = nn.Linear(x.size(1), self.dense_neurons[i])
      x = fc(x)
      # if self.batch_normalization=='yes':
      #   fc_bn= nn.BatchNorm1d(self.dense_neurons[i])
      #   x= fc_bn(x)
      # x= self.drop_out_fc(x)
      x = self.activation_function(x)
    # Apply softmax activation for classification
    last_layer= nn.Linear(x.size(1), self.num_classes)
    x= last_layer(x)
    x = self.softmax(x)
    return x

  def training_step(self, batch, batch_idx):
    img, label = batch
    logits = self(img)
    loss_function = nn.CrossEntropyLoss()
    loss = loss_function(logits, label)

    preds = torch.argmax(logits, dim=1)
    acc = accuracy(preds, label, 'multiclass', num_classes=self.num_classes)

    self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
    self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)

    return loss

  def validation_step(self, batch, batch_idx):
    img, label = batch
    logits = self(img)
    loss_function = nn.CrossEntropyLoss()
    loss = loss_function(logits, label)

    preds = torch.argmax(logits, dim=1)
    acc = accuracy(preds, label, 'multiclass', num_classes=self.num_classes)

    self.log('val_loss', loss, prog_bar=True)
    self.log('val_acc', acc, prog_bar=True)

    return loss

  def test_step(self, batch, batch_idx):
    img, label = batch
    logits = self(img)
    loss_function = nn.CrossEntropyLoss()
    loss = loss_function(logits, label)

    preds = torch.argmax(logits, dim=1)
    acc = accuracy(preds, label, 'multiclass', num_classes=self.num_classes)

    self.log('test_loss', loss, on_epoch=True, prog_bar=True)
    self.log('test_acc', acc, on_epoch=True, prog_bar=True)

    return loss

  def configure_optimizers(self):
    if self.optimisers=='adam':
      return torch.optim.Adam(self.parameters(), lr= self.learning_rate, weight_decay= self.weight_decay)
    elif self.optimisers=='nadam':
      return torch.optim.NAdam(self.parameters(), lr= self.learning_rate, weight_decay= self.weight_decay)
    elif self.optimisers=='RMSprop':
      return torch.optim.RMSprop(self.parameters(), lr= self.learning_rate, weight_decay= self.weight_decay)
    elif self.optimisers=='adadelta':
      return torch.optim.Adadelta(self.parameters(), lr= self.learning_rate, weight_decay= self.weight_decay)
    
sweep_config = {
  'name': 'Naturalist Classification',
    "metric": {
      "name":"val_acc",
      "goal": "maximize"
  },
  'method': 'grid',
  'parameters': {
        'optim': {
            'values': ['adam', 'nadam', 'RMSprop', 'adadelta']
        },
        'af_type':{
            'values': ['ReLU', 'SeLU', 'GeLU', 'SiLU', 'Mish']
        },
        'filters': {
            'values': ['same_32', 'same_64', 'doubling', 'halving']
        },
        'filter_size': {
           'values': [3, 5, 7]
        },
        'data_augmentation': {
            'values': [True, False]
        },
        'pooling_layer':{
            'values': ['max', 'avg']
        },
        'fc_size':{
            'values': [[128], [120, 84], [256]]           # The number of layer's in the FC can be decided by the length of the dense neurons in the list
        },
        'batch_normalization':{
            'values': [True, False]
        },
        'droprate':{
            'values':[0, 0.2, 0.3]
        },
        'epochs': {
            'values': [10]
        },
        'batch_size': {
            'values': [128]
        },
        'num_layers': {
            'values': [5]
        },
        'learning_rate':{
            'values': [0.01, 0.001]
        },
        'weight_decay':{
            'values': [0, 0.0005]
        }
    }
}

def models():
  config=wandb.config

  train_dir='C:\\Users\\balaj\\Documents\\Assignment_02\\inaturalist_12K\\train'
  test_dir='C:\\Users\\balaj\\Documents\\Assignment_02\\inaturalist_12K\\val'
  categories=['Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptilia']

  if config.filters=='same_32':
    num_filters= [32, 32, 32, 32, 32]
  elif config.filters=='same_64':
    num_filters= [64, 64, 64, 64, 64]
  elif config.filters=='doubling':
    num_filters= [16, 32, 64, 128, 256]
  elif config.filters=='halving':
    num_filters= [256, 128, 64, 32, 16]

  num_fc_layers= len(config.fc_size)

  num_channels= 3
  num_classes= len(categories)
  augment_data= config.data_augmentation
  wandb.run.name = (
        "bn" + str(config.batch_normalization) + "nf" + str(config.filters)
        + "fs" + str(config.filter_size) + "do" + str(config.droprate)
        + "opt" + str(config.optim) + "da" + str(augment_data)
        + "af" + str(config.af_type) + "fc" + str(num_fc_layers)
        + "pl" + str(config.pooling_layer)
        + "wd" + str(config.weight_decay) + "epochs" + str(config.epochs)
    )
  train_loader, val_loader, test_loader= Data_Preprocessing(augment_data, train_dir, test_dir, config.batch_size)

  model= Convolutional_Neural_Networks(num_channels, config.num_layers, num_filters, config.filter_size, num_classes, config.af_type, config.pooling_layer, num_fc_layers, config.fc_size, config.droprate, config.batch_normalization, config.optim, config.learning_rate, config.weight_decay)
  wandb_logger= WandbLogger(project=wandb_project, log_model='all')
  checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max')
  trainer = Trainer(max_epochs=config.epochs, logger=wandb_logger, callbacks=checkpoint_callback)
  trainer.fit(model, train_loader, val_loader)

  wandb.finish()

def sweeper(sweep_config,proj_name):
  sweep_id=wandb.sweep(sweep_config,project=wandb_project)
  wandb.agent(sweep_id,models,project=proj_name, count=1)