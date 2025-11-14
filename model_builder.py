"""model_builder.py - Construcción de modelos PyTorch"""
import torch
import torch.nn as nn
from torchvision import models

def build_densenet121(fast=False):
    print("[INFO] Descargando y construyendo modelo DenseNet121...")
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = not fast # Si fast=True, requires_grad=False (congelado)
    num_features = model.classifier.in_features
    #model.classifier = nn.Sequential(nn.Linear(num_features, 1), nn.Sigmoid())
    model.classifier = nn.Linear(num_features, 1)
    return model
