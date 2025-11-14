"""data_loader.py - Carga y preprocesamiento con PyTorch"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(train_dir, val_dir, test_dir, img_size=(224,224), batch_size=32):
    print("[INFO] Configurando DataLoaders...")
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

    class_counts = [0, 0] # [Normal, Pneumonia]
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    # Cálculo de pesos inversos
    total_samples = sum(class_counts)
    # Fórmula: N_samples / (N_clases * N_clase_i)
    weights = [total_samples / (2.0 * count) for count in class_counts]
    class_weights = torch.tensor(weights, dtype=torch.float32)
    
    print(f"[INFO] Clases: {train_dataset.classes}")
    print(f"[INFO] Conteos: {class_counts}")
    print(f"[INFO] Pesos (Normal, Pneumonia): {class_weights.tolist()}")

    print(f"[INFO] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader,  class_weights
