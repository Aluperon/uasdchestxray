"""evaluation.py - Evaluación PyTorch"""
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

def plot_roc_curve(y_true, y_probs, title="Curva ROC", save_path=None):
    """Calcula y grafica la curva ROC y el AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (False Positive Rate)')
    plt.ylabel('Tasa de Verdaderos Positivos (True Positive Rate)')
    plt.title(title)
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    return roc_auc

def evaluate_model(model, test_loader, device, model_tag="Modelo"):
    print(f"[INFO] Evaluando {model_tag}...")
    model.eval()
    model.to(device)
    y_true, y_probs = [], [] # y_probs para la curva ROC
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) # Estos son ahora logits (salida sin procesar)
            
            # === MODIFICACIÓN CLAVE: Aplicar Sigmoid ===
            # Convertir logits en probabilidades (requerido porque quitamos Sigmoid del modelo)
            probabilities = torch.sigmoid(outputs) 
            
            # Guardamos las probabilidades reales para la curva ROC y la predicción
            y_probs.extend(probabilities.cpu().numpy().flatten())
            y_true.extend(labels.cpu().numpy())
            
    y_probs = np.array(y_probs)
    y_true = np.array(y_true)
    
    # Predicciones binarias usando el umbral de 0.5 sobre las probabilidades
    y_pred = (y_probs > 0.5).astype(int) 

    # 1. Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=['Normal','Pneumonia'], yticklabels=['Normal','Pneumonia']
    )
    plt.title(f'Matriz de Confusión - {model_tag}')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Predicción del Modelo')
    plt.show()

    # 2. Clasificación Report (Accuracy, F1, etc.)
    report = classification_report(y_true, y_pred, target_names=['Normal','Pneumonia'], digits=4)
    print(report)
    report_dict = classification_report(y_true, y_pred, target_names=['Normal','Pneumonia'], output_dict=True)

    # 3. Curva ROC y AUC
    roc_auc = plot_roc_curve(y_true, y_probs, title=f"Curva ROC - {model_tag}")
    
    report_dict['auc'] = roc_auc # Añadir AUC al reporte para la tabla comparativa
    
    return cm, report_dict
