import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
from sklearn.metrics import f1_score
import numpy as np


def get_optimizer(params, optimizer_name: str, lr: float, weight_decay: float = 0.01):
    if optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(optimizer, scheduler_name: str, num_warmup_steps: int, num_training_steps: int):
    if scheduler_name.lower() == "cosine_warmup":
        from transformers import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    elif scheduler_name.lower() == "none" or scheduler_name is None:
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

def train_epoch(model, data_loader, optimizer, criterion, device, scheduler=None, grad_clip_norm=None, aux_loss_coeff=0.01):
    model.train()
    total_loss, total_main_loss, total_aux_loss = 0, 0, 0
    all_preds, all_labels = [], []
    
    progress_bar = tqdm(data_loader, desc=f"Training {model.get_model_name()}", leave=False)
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        main_loss = criterion(outputs, labels)
        
        aux_loss = model.get_auxiliary_loss() * aux_loss_coeff if hasattr(model, 'get_auxiliary_loss') else 0.0
        loss = main_loss + aux_loss
        
        loss.backward()
        if grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item() * input_ids.size(0)
        total_main_loss += main_loss.item() * input_ids.size(0)
        if isinstance(aux_loss, torch.Tensor): total_aux_loss += aux_loss.item() * input_ids.size(0)
        
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix(loss=loss.item(), main_loss=main_loss.item(), acc=np.mean(np.array(all_preds) == np.array(all_labels)))

    num_samples = len(all_labels)
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    avg_main_loss = total_main_loss / num_samples if num_samples > 0 else 0
    avg_aux_loss = total_aux_loss / num_samples if num_samples > 0 else 0
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) if num_samples > 0 else 0
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) if num_samples > 0 else 0
    
    return avg_loss, avg_main_loss, avg_aux_loss, accuracy, f1

def evaluate_model(model, data_loader, criterion, device, return_latency=False):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    latencies = []
    
    progress_bar = tqdm(data_loader, desc=f"Evaluating {model.get_model_name()}", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            start_time = time.perf_counter()
            outputs = model(input_ids, attention_mask=attention_mask)
            end_time = time.perf_counter()
            latencies.append(end_time - start_time)
            
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix(loss=loss.item(), acc=np.mean(np.array(all_preds) == np.array(all_labels)))

    num_samples = len(all_labels)
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) if num_samples > 0 else 0
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) if num_samples > 0 else 0
    avg_latency = np.mean(latencies) if latencies and return_latency else 0.0
    
    model_specific_metrics = model.get_model_specific_metrics() if hasattr(model, 'get_model_specific_metrics') else {}

    if return_latency:
        return avg_loss, accuracy, f1, avg_latency, model_specific_metrics
    return avg_loss, accuracy, f1, model_specific_metrics


def gumbel_softmax(logits: torch.Tensor, tau: float = 1.0, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> torch.Tensor:
    gumbel_noise = -torch.log(eps - torch.log(torch.rand_like(logits) + eps)) # Numerically stable Gumbel noise
    y_soft = F.softmax((logits + gumbel_noise) / tau, dim=dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return (y_hard - y_soft).detach() + y_soft # Straight-through estimator
    return y_soft

def calculate_entropy(probs, eps=1e-10): # probs: (batch, num_classes)
    return -torch.sum(probs * torch.log(probs + eps), dim=-1).mean()

def calculate_cv(data): # Coefficient of variation
    mean = torch.mean(data)
    std = torch.std(data)
    return std / (mean + 1e-10) if mean != 0 else torch.tensor(0.0)
