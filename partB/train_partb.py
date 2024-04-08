import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil
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
import torchvision.models as models
from lightning.pytorch.callbacks import ModelCheckpoint

from google.colab import drive
drive.mount('/content/drive')

wandb.login(key='fd85ae65c5e04fb6fea4d31c1180348532db32d6')
wandb_project= 'EE22M008_Assignment_02_Part_B'

class Convolutional_Neural_Networks(pl.LightningModule):
  def __init__(self, N_classes, af_type, N_dense, drop_rate, optimizers, lr, wd, pretrained_model_name):
    super(Convolutional_Neural_Networks, self).__init__()
    self.num_classes= N_classes
    self.dense_neurons = N_dense
    self.optimiser= optimizers
    self.learning_rate= lr
    self.weight_decay= wd

    if af_type=='ReLU':
      self.activation_function= nn.ReLU()
    elif af_type=='GeLU':
      self.activation_function= nn.GELU()
    elif af_type== 'SeLU':
      self.activation_function= nn.SELU()
    elif af_type=='SiLU':
      self.activation_function= nn.SiLU()
    elif af_type=='Mish':
      self.activation_function= nn.Mish()

    pretrained_model= load_base_model(pretrained_model_name)
    resnet = pretrained_model
    number_filters = resnet.fc.in_features
    resnet.fc = torch.nn.Sequential(torch.nn.Linear(number_filters, N_dense),
                                    self.activation_function,
                                    torch.nn.Dropout(drop_rate),
                                    torch.nn.Linear(N_dense, N_classes),
                                    torch.nn.LogSoftmax(dim=1))
    self.model = resnet

    self.save_hyperparameters()

  def forward(self, x):
    return self.model(x)

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

    self.log('val_loss', loss, on_epoch=True, prog_bar=True)
    self.log('val_acc', acc, on_epoch=True, prog_bar=True)

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
    if self.optimiser=='adam':
      return torch.optim.Adam(self.parameters(), lr= self.learning_rate, weight_decay= self.weight_decay)
    elif self.optimiser=='nadam':
      return torch.optim.NAdam(self.parameters(), lr= self.learning_rate, weight_decay= self.weight_decay)
    elif self.optimiser=='RMSprop':
      return torch.optim.RMSprop(self.parameters(), lr= self.learning_rate, weight_decay= self.weight_decay)
    elif self.optimiser=='adadelta':
      return torch.optim.Adadelta(self.parameters(), lr= self.learning_rate, weight_decay= self.weight_decay)
    
sweep_config = {
  'name': 'Naturalist Classification',
    "metric": {
      "name":"Accuracy_val",
      "goal": "maximize"
  },
  'method': 'grid',
  'parameters': {
        'pretrained': {
          #'values': ['googlenet', 'inceptionv3', 'resnet50', 'vgg16']
          'values': ['resnet50']
      },
        'filters': {
            'values': ['doubling']
        },
        'filter_size': {
           'values': [3]
        },
        'activation_function':{
            'values': ['ReLU']
        },
        'data_augmentation': {
            'values': [False]
        },
        'pooling':{
            'values': ['max']
        },
        'fc_size':{
            'values': [128]            # The number of layer's in the FC can be decided by the length of the dense neurons in the list
        },
        'batch_normalization':{
            'values': [True]
        },
        'droprate':{
            'values':[0.2]
        },
        'epochs': {
            'values': [10]
        },
        'batch_size': {
            'values': [64]
        },
        'num_layers': {
            'values': [5]
        },
        'optimizer': {
            'values': ['adam']
        },
        'learning_rate':{
            'values': [0.001]
        },
        'weight_decay':{
            'values': [0.0005]
        }
    }
}

def load_base_model(name, pretrained=True):
    import torchvision.models as models
    """
    Loads a base model from PyTorch's torchvision.models.

    Args:
    - name (str): Name of the base model (e.g., "GoogLeNet", "InceptionV3", "ResNet50", "VGG", "EfficientNetV2", "VisionTransformer").
    - pretrained (bool): Whether to load pre-trained weights or not.

    Returns:
    - model: Loaded base model.
    """
    name = name.lower()
    if name == "googlenet":
        return models.googlenet(pretrained=pretrained)
    elif name == "inceptionv3":
        return models.inception_v3(pretrained=pretrained)
    elif name == "resnet50":
        return models.resnet50(pretrained=pretrained)
    elif name == "vgg16":
        return models.vgg16(pretrained=pretrained)

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

def models():
  wandb.init(project= wandb_project)
  config=wandb.config

  train_dir='/content/drive/MyDrive/inaturalist_12K/train'
  test_dir='/content/drive/MyDrive/inaturalist_12K/val'
  categories=['Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptilia']

  num_conv_blocks= config.num_layers
  if config.filters=='same_32':
    num_filters= [32, 32, 32, 32, 32]
  elif config.filters=='same_64':
    num_filters= [64, 64, 64, 64, 64]
  elif config.filters=='doubling':
    num_filters= [16, 32, 64, 128, 256]
  elif config.filters=='halving':
    num_filters= [256, 128, 64, 32, 16]

  filter_size= config.filter_size
  activation_function= config.activation_function
  pooling_choice= config.pooling

  dense_neurons= config.fc_size
  num_epochs= config.epochs
  drop_rate= config.droprate
  bn= config.batch_normalization
  lr= config.learning_rate
  wd= config.weight_decay
  optimizer= config.optimizer
  pretrained_model_name= config.pretrained

  num_channels= 3
  num_classes= len(categories)
  augment_data= config.data_augmentation
  wandb.run.name = (
          "pretrained_model" + str(pretrained_model_name) + "do" + str(drop_rate)
        + "opt" + str(optimizer) + "da" + str(augment_data)
        + "af" + str(activation_function) + "fc" + str(dense_neurons)
        + "pl" + str(pooling_choice)
        + "wd" + str(wd) + "epochs" + str(num_epochs)
    )
  train_loader, val_loader, test_loader= Data_Preprocessing(augment_data, train_dir, test_dir, config.batch_size)
  model= Convolutional_Neural_Networks(num_classes, activation_function, dense_neurons, drop_rate, optimizer, lr, wd, pretrained_model_name)
  wandb_logger= WandbLogger(project=wandb_project, log_model='all')
  checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max')
  trainer = Trainer(max_epochs=num_epochs, logger=wandb_logger, callbacks=checkpoint_callback)
  trainer.fit(model, train_loader, val_loader)
  #Validating the model
  trainer.validate(model=model, dataloaders=val_loader)
  # Testing the model
  trainer.test(dataloaders=test_loader)

  wandb.finish()
  
  def sweeper(sweep_config,proj_name):
    sweep_id=wandb.sweep(sweep_config,project=wandb_project)
  wandb.agent(sweep_id,models,project=proj_name, count=1)

sweeper(sweep_config,wandb_project)