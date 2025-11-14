"""utils.py - Funciones auxiliares"""
import matplotlib.pyplot as plt
import pandas as pd

def plot_training_curves(csv_fast, csv_opt):
    print("[INFO] Generando gráficas comparativas...")
    df_fast = pd.read_csv(csv_fast)
    df_opt = pd.read_csv(csv_opt)
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    axes[0].plot(df_fast['epoch'], df_fast['train_loss'], label='Fast Train Loss')
    axes[0].plot(df_fast['epoch'], df_fast['val_loss'], label='Fast Val Loss')
    axes[0].plot(df_opt['epoch'], df_opt['train_loss'], label='Opt Train Loss')
    axes[0].plot(df_opt['epoch'], df_opt['val_loss'], label='Opt Val Loss')
    axes[0].legend()
    axes[0].set_title('Loss Comparison')
    axes[1].plot(df_fast['epoch'], df_fast['train_acc'], label='Fast Train Acc')
    axes[1].plot(df_fast['epoch'], df_fast['val_acc'], label='Fast Val Acc')
    axes[1].plot(df_opt['epoch'], df_opt['train_acc'], label='Opt Train Acc')
    axes[1].plot(df_opt['epoch'], df_opt['val_acc'], label='Opt Val Acc')
    axes[1].legend()
    axes[1].set_title('Accuracy Comparison')
    plt.show()
