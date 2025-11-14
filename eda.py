"""
eda.py - Funciones para el Análisis Exploratorio de Datos (EDA) Extendido
"""
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import torch
from torchvision.utils import make_grid
import random

# --- Configuraciones de Normalización (Tomadas de data_loader.py) ---
# Necesarias para desnormalizar imágenes y calcular histogramas correctamente
NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
NORM_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# --- Funciones Auxiliares ---

def _denormalize(images):
    """Desnormaliza un tensor de imágenes de PyTorch a un rango [0, 1]."""
    unnorm_images = images * NORM_STD + NORM_MEAN
    return torch.clamp(unnorm_images, 0, 1)

def show_random_images(loader, num_per_class=5, class_names=['Normal', 'Pneumonia']):
    """
    Muestra una cuadrícula de imágenes aleatorias de ambas clases (muestreo estratificado).
    """
    collected_images = []
    collected_labels = []
    
    total_needed = num_per_class * 2
    
    print(f"Buscando {num_per_class} de cada clase (Total: {total_needed} imágenes) para visualización...")
    
    # 1. Iterar sobre lotes hasta obtener el número deseado de cada clase
    for images, labels in loader:
        images, labels = images.cpu(), labels.cpu()
        
        # Indices de las imágenes en el lote actual
        batch_normal_indices = (labels == 0).nonzero(as_tuple=True)[0]
        batch_pneumonia_indices = (labels == 1).nonzero(as_tuple=True)[0]
        
        # --- Recolectar Normal ---
        current_normal_count = sum(1 for l in collected_labels if l == 0)
        needed_normal = num_per_class - current_normal_count
        if needed_normal > 0 and len(batch_normal_indices) > 0:
            to_sample = min(needed_normal, len(batch_normal_indices))
            sampled_indices = random.sample(batch_normal_indices.tolist(), to_sample)
            collected_images.extend(images[sampled_indices])
            collected_labels.extend(labels[sampled_indices])

        # --- Recolectar Neumonía ---
        current_pneumonia_count = sum(1 for l in collected_labels if l == 1)
        needed_pneumonia = num_per_class - current_pneumonia_count
        if needed_pneumonia > 0 and len(batch_pneumonia_indices) > 0:
            to_sample = min(needed_pneumonia, len(batch_pneumonia_indices))
            sampled_indices = random.sample(batch_pneumonia_indices.tolist(), to_sample)
            collected_images.extend(images[sampled_indices])
            collected_labels.extend(labels[sampled_indices])
        
        # Detener si ya tenemos suficientes de ambas clases
        if len(collected_images) >= total_needed:
            break
            
    if not collected_images:
        print("Error: No se pudieron recolectar imágenes. Verifique si los DataLoaders están vacíos.")
        return
        
    final_images = torch.stack(collected_images)
    final_labels = torch.tensor(collected_labels)
    
    # 2. Desnormalizar para visualización
    unnorm_images = _denormalize(final_images)

    # 3. Crear la cuadrícula
    grid = make_grid(unnorm_images, nrow=num_per_class, padding=2)
    
    plt.figure(figsize=(15, 6))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')

    # 4. Añadir títulos (Etiquetas)
    titles = [f"{class_names[l.item()]}" for l in final_labels]
    
    # Asumiendo todas las imágenes tienen el mismo tamaño:
    img_height, img_width = final_images.shape[2:]
    
    for i, title in enumerate(titles):
        col = i % num_per_class
        row = i // num_per_class
        
        # Ajuste de posición (un poco de heurística basada en padding=2)
        x_pos = col * (img_width + 4) + img_width / 2 + 2 
        y_pos = row * (img_height + 4) + img_height + 15
        
        plt.text(x_pos, y_pos, title, ha='center', color='black', fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    plt.suptitle(f'Ejemplos Aleatorios de Imágenes ({len(final_images)} muestras estratificadas)', fontsize=14)
    plt.show()


def plot_pixel_intensity_histograms(loader, num_batches=15):
    """
    Compara la distribución de intensidad de píxeles entre las clases (Normal vs. Pneumonia).
    """
    normal_pixels = []
    pneumonia_pixels = []
    data_iter = iter(loader)
    
    print(f"Calculando histogramas de intensidad en {num_batches} lotes...")

    for i in range(num_batches):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            break
            
        images, labels = images.cpu(), labels.cpu()
        
        # Desnormalizar a un rango [0, 1] antes de aplanar
        unnorm_images = _denormalize(images)

        normal_mask = (labels == 0)
        pneumonia_mask = (labels == 1)
        
        # Aplanar todos los píxeles de todos los canales (R, G, B) en un solo vector
        if normal_mask.any():
            # .flatten() aplanará la imagen (C, H, W) en un vector de píxeles
            normal_pixels.extend(unnorm_images[normal_mask].numpy().flatten())
            
        if pneumonia_mask.any():
            pneumonia_pixels.extend(unnorm_images[pneumonia_mask].numpy().flatten())

    if not normal_pixels and not pneumonia_pixels:
        print("Advertencia: No se pudieron recolectar datos para los histogramas.")
        return

    plt.figure(figsize=(12, 6))
    
    # Trazar histograma Normal
    plt.hist(normal_pixels, bins=50, alpha=0.6, density=True, color='skyblue', label='Normal (0)', histtype='stepfilled')
    
    # Trazar histograma Pneumonia
    plt.hist(pneumonia_pixels, bins=50, alpha=0.6, density=True, color='salmon', label='Pneumonia (1)', histtype='stepfilled')
    
    plt.title('Distribución de Intensidad de Píxeles por Clase (Muestra Desnormalizada)')
    plt.xlabel('Intensidad de Píxel (0 = Negro, 1 = Blanco)')
    plt.ylabel('Densidad (Normalizada)')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.show()

# --- FUNCIÓN PRINCIPAL EXTENDIDA ---

def extended_eda(train_loader, val_loader, test_loader):
    """
    Realiza un EDA extendido: conteos, gráficos de desbalance, análisis de intensidad de píxeles 
    y visualización de imágenes.
    """
    print("--- 🔬 Análisis Exploratorio de Datos (EDA) Extendido ---")
    
    # 1. Obtener conteos de etiquetas
    train_counts = Counter(train_loader.dataset.targets)
    val_counts = Counter(val_loader.dataset.targets)
    test_counts = Counter(test_loader.dataset.targets)
    
    # NUEVO: Verificación de dimensiones y canales
    try:
        sample_img_shape = train_loader.dataset[0][0].shape
        print(f"\n[INFO] Dimensiones de la Imagen (C, H, W): {sample_img_shape}")
        print(f"[INFO] Se esperan 3 canales (RGB) y tamaño {sample_img_shape[-2]}x{sample_img_shape[-1]}.")
    except Exception as e:
        print(f"\n[ERROR] No se pudo obtener la forma de la imagen de muestra: {e}")

    print("\n[INFO] Distribución de Clases (Etiqueta 0: Normal, Etiqueta 1: Pneumonia):")
    print(f"  - Set de Entrenamiento: {train_counts}")
    print(f"  - Set de Validación: {val_counts}")
    print(f"  - Set de Prueba: {test_counts}")
    
    # 2. Visualización del desbalance en el set de entrenamiento
    labels = ['Normal (0)', 'Pneumonia (1)']
    counts = [train_counts.get(0, 0), train_counts.get(1, 0)]
    
    plt.figure(figsize=(7, 5))
    plt.bar(labels, counts, color=['skyblue', 'salmon'])
    plt.title('Distribución de Clases en el Set de Entrenamiento (Desbalance)')
    plt.ylabel('Número de Muestras')
    plt.show()

    # 3. Análisis de Intensidad de Píxeles (CLAVE)
    print("\n[INFO] Análisis de Intensidad de Píxeles por Clase (Indica Diferencias en Densidad/Opacidad):")
    plot_pixel_intensity_histograms(train_loader, num_batches=15) 

    # 4. Visualización de imágenes aleatorias (estratificado)
    print("\n[INFO] Visualización de Muestras Aleatorias (Test Set - Calidad y Contraste):")
    show_random_images(test_loader, num_per_class=5)
    
    print("\n--- 🏁 EDA Completo ---")