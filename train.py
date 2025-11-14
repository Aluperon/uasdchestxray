import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from tqdm.auto import tqdm

# === Funciones Auxiliares ===

def _sync_if_cuda(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize()

def _fmt(secs: float) -> str:
    secs = int(secs)
    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def _ensure_unique_path(path: str) -> str:
    """Si path existe, agrega sufijos _001, _002, ... para no sobrescribir."""
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    i = 1
    while True:
        candidate = f"{root}_{i:03d}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1

# === Función Principal ===

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    fast_mode: bool,
    epochs: int,
    lr: float,
    save_path: str,
    class_weights: torch.Tensor = None, 
    model_name: str = "modelo_", 
    save_batch_csv: bool = False
):
    print(f"[INFO] Iniciando entrenamiento {'FAST' if fast_mode else 'OPTIMIZADO'}...")

    # ===== RUTAS =====
    results_dir = os.path.join(save_path, 'results')
    metrics_dir = os.path.join(results_dir, 'metricascsv')
    os.makedirs(metrics_dir, exist_ok=True)

    mode_dir = os.path.join(results_dir, 'fast' if fast_mode else 'optimized')
    os.makedirs(mode_dir, exist_ok=True)
    model_path = os.path.join(mode_dir, 'model.pth')

    modo_tag = 'fast' if fast_mode else 'Optimizado'
    base_name = f"{model_name}{modo_tag}"

    metrics_epoch_csv = _ensure_unique_path(os.path.join(metrics_dir, f"{base_name}_epoch.csv"))
    metrics_batch_csv = _ensure_unique_path(os.path.join(metrics_dir, f"{base_name}_batch.csv")) if save_batch_csv else None
    if save_batch_csv:
        pd.DataFrame(columns=[
            "epoch", "phase", "batch_idx", "batches_total",
            "loss", "acc", "seen_samples", "time_sec"
        ]).to_csv(metrics_batch_csv, index=False, encoding='utf-8-sig')


    # ===== ENTRENAMIENTO =====
    model = model.to(device)

    # === Configuración de Loss: BCEWithLogitsLoss con Pos_Weight ===
    if class_weights is not None and len(class_weights) == 2:
        n_normal = class_weights[0].to(device)
        n_pneumonia = class_weights[1].to(device)
        
        if n_pneumonia.item() > 0:
            pos_weight = n_normal / n_pneumonia
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
            print(f"[INFO] Usando BCEWithLogitsLoss con Pos_Weight: {pos_weight.item():.4f}")
        else:
            criterion = nn.BCEWithLogitsLoss()
            print("[INFO] Advertencia: Pos_Weight no aplicado.")
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.2)

    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
        'epoch_train_sec': [], 'epoch_val_sec': [], 'epoch_total_sec': []
    }

    global_start = time.perf_counter()

    for epoch in range(epochs):
        epoch_start = time.perf_counter()

        # ----------------- TRAIN -----------------
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        train_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", dynamic_ncols=True, leave=False
        )
        train_phase_start = time.perf_counter()
        total_batches_train = len(train_loader)

        for bidx, (inputs, labels) in enumerate(train_bar, start=1):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float() 
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            _sync_if_cuda(device)
            elapsed_train = time.perf_counter() - train_phase_start
            eta_epoch_train = (elapsed_train / max(bidx, 1)) * (total_batches_train - bidx)

            train_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(correct/max(total,1)):.4f}",
                "eta": _fmt(eta_epoch_train)
            })
            # ... (Guardado de CSV por batch) ...
        
        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        _sync_if_cuda(device)
        
        ### AJUSTE POR NAMERROR (1 de 2) ###
        train_time = time.perf_counter() - train_phase_start


        # ----------------- VAL -----------------
        model.eval()
        val_start = time.perf_counter()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_bar = tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ", dynamic_ncols=True, leave=False
        )
        total_batches_val = len(val_loader)

        with torch.no_grad():
            for bidx, (inputs, labels) in enumerate(val_bar, start=1):
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_bar.set_postfix({
                    "loss": f"{(val_loss/max(val_total,1)):.4f}",
                    "acc": f"{(val_correct/max(val_total,1)):.4f}"
                })
                # ... (Guardado de CSV por batch) ...

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        _sync_if_cuda(device)
        
        ### AJUSTE POR NAMERROR (2 de 2) ###
        val_time = time.perf_counter() - val_start

        scheduler.step(val_loss)

        # ----------------- TIEMPOS/ETA -----------------
        epoch_total = time.perf_counter() - epoch_start
        elapsed_global = time.perf_counter() - global_start
        epochs_done = epoch + 1
        eta_global = (elapsed_global / epochs_done) * (epochs - epochs_done)

        # ... (Guardado en history) ...
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epoch_train_sec'].append(train_time)
        history['epoch_val_sec'].append(val_time)
        history['epoch_total_sec'].append(epoch_total)

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
            f"Epoch Time: {_fmt(epoch_total)} | ETA Total: {_fmt(eta_global)}"
        )

    # ... (Guardado final de CSV y modelo) ...
    df_epoch = pd.DataFrame(history)
    df_epoch.to_csv(metrics_epoch_csv, index=False, encoding='utf-8-sig')
    print(f"[INFO] Métricas por epoch guardadas en: {metrics_epoch_csv}")

    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Modelo guardado en: {model_path}")

    total_time = time.perf_counter() - global_start
    print(f"[INFO] Tiempo total de entrenamiento: {_fmt(total_time)}")

    return history