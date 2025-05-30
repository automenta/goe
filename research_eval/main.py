import torch
import torch.nn as nn
import argparse
import json
import time
import pandas as pd
import os
import logging
import sys

# Ensure the project root is in sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))) 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from research_eval.datasets.providers import get_synthetic_data_loaders, get_real_world_data_loaders, SYNTHETIC_DATASETS, REAL_WORLD_DATASETS
from research_eval.models.dense_transformer import DenseTransformerClassifier
from research_eval.models.moe_transformer import MoETransformerClassifier
from research_eval.models.goe_transformer import GoEClassifier
from research_eval.models.goe_original_classifier import GoEOriginalClassifier
from research_eval.utils import train_epoch, evaluate_model, get_optimizer, get_scheduler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_MAP = {
    "dense": DenseTransformerClassifier,
    "moe": MoETransformerClassifier,
    "goe": GoEClassifier,
    "goe_original": GoEOriginalClassifier
}

MAX_PARAMS_SYNTHETIC = 30_000_000  # Example value
MAX_PARAMS_REAL_WORLD = 30_000_000 # Example value


def get_trainable_parameter_count(model: nn.Module) -> int:
    """Helper to count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_single_experiment_from_json(config: dict):
    """
    Runs a single experiment based on a JSON configuration object.
    """
    exp_name = config.get("experiment_name", f"json_exp_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
    logging.info(f"--- Starting Experiment: {exp_name} ---")
    logging.info(f"Full Configuration: {json.dumps(config, indent=2)}")

    results_data = {} 
    results_data['experiment_name'] = exp_name
    results_data['model_type'] = config['model_type']
    results_data['dataset_type'] = config['dataset_type']
    results_data['dataset_name'] = config['dataset_name']
    results_data['epochs'] = config['epochs']
    results_data['batch_size'] = config['batch_size']

    # --- 1. Dataset Setup ---
    padding_idx = 0 
    dataset_params = config.get('dataset_params', {})
    
    if config['dataset_type'] == "synthetic":
        req_syn_keys = ['num_samples', 'seq_len', 'vocab_size']
        if not all(k in dataset_params for k in req_syn_keys):
            missing_keys = [k for k in req_syn_keys if k not in dataset_params]
            raise ValueError(f"Synthetic dataset requires params: {', '.join(missing_keys)}. Provided: {dataset_params}")
        
        for k in req_syn_keys:
            if not isinstance(dataset_params[k], int):
                raise TypeError(f"Synthetic dataset param {k} must be int, got {type(dataset_params[k])}")

        train_loader, val_loader, vocab_size, num_classes = get_synthetic_data_loaders(
            dataset_name=config['dataset_name'], 
            batch_size=config['batch_size'], 
            num_samples=dataset_params['num_samples'],
            seq_len=dataset_params['seq_len'],
            vocab_size=dataset_params['vocab_size']
        )
    elif config['dataset_type'] == "real_world":
        req_rw_keys = ['max_length', 'train_samples', 'val_samples']
        if not all(k in dataset_params for k in req_rw_keys):
            missing_keys = [k for k in req_rw_keys if k not in dataset_params]
            raise ValueError(f"Real-world dataset requires params: {', '.join(missing_keys)}. Provided: {dataset_params}")
        
        for k in req_rw_keys:
             if not isinstance(dataset_params[k], int):
                raise TypeError(f"Real-world dataset param {k} must be int, got {type(dataset_params[k])}")

        train_loader, val_loader, vocab_size, num_classes, padding_idx_from_data = get_real_world_data_loaders(
            dataset_name=config['dataset_name'],
            batch_size=config['batch_size'],
            max_length=dataset_params['max_length'],
            train_samples=dataset_params['train_samples'],
            val_samples=dataset_params['val_samples']
        )
        padding_idx = padding_idx_from_data
    else:
        raise ValueError(f"Unknown dataset_type: {config['dataset_type']}")
    
    results_data['vocab_size'] = vocab_size
    results_data['num_classes'] = num_classes
    results_data['padding_idx'] = padding_idx
    logging.info(f"Dataset: {config['dataset_name']} ({config['dataset_type']}), Vocab: {vocab_size}, Classes: {num_classes}, Padding Idx: {padding_idx}")

    # --- 2. Model Instantiation ---
    model_type = config['model_type']
    if model_type not in MODELS_MAP:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model_constructor_params = config.get('model_params', {}).copy() 
    model_constructor_params['vocab_size'] = vocab_size
    model_constructor_params['num_classes'] = num_classes
    model_constructor_params['padding_idx'] = padding_idx
    
    if model_type == "moe" and "num_experts_per_layer" in model_constructor_params:
        model_constructor_params["num_experts"] = model_constructor_params.pop("num_experts_per_layer")
    if model_type == "goe":
        if "num_total_experts_pool" in model_constructor_params: # From JSON
             model_constructor_params["num_total_experts"] = model_constructor_params.pop("num_total_experts_pool")
        # Ensure num_goe_layers is present if it's a goe model, could be missing if old JSON config
        if "num_goe_layers" not in model_constructor_params and "num_layers" in model_constructor_params: # Compatibility
            model_constructor_params["num_goe_layers"] = model_constructor_params.pop("num_layers")
        elif "num_goe_layers" not in model_constructor_params: # Default if not specified
             model_constructor_params["num_goe_layers"] = 1 # Default GoE layers
    if model_type == "goe_original" and "max_visits" in model_constructor_params: # JSON config uses 'max_visits'
        model_constructor_params["max_visits_per_expert"] = model_constructor_params.pop("max_visits")

    model = MODELS_MAP[model_type](**model_constructor_params).to(DEVICE)
    
    trainable_params = get_trainable_parameter_count(model)
    results_data['param_count'] = trainable_params
    logging.info(f"Model: {model_type}, Trainable Parameters: {trainable_params}")

    # --- 3. Optimizer and Scheduler Setup ---
    optimizer_params = config.get('optimizer_params', {})
    scheduler_params = config.get('scheduler_params', {})

    if 'name' not in optimizer_params or 'lr' not in optimizer_params:
        raise ValueError("Optimizer 'name' and 'lr' must be specified in optimizer_params.")

    optimizer = get_optimizer(model.parameters(), optimizer_params['name'], optimizer_params['lr'], optimizer_params.get('weight_decay', 0.01))
    
    num_training_steps = len(train_loader) * config['epochs']
    warmup_factor = scheduler_params.get('warmup_factor', 0.1) 
    num_warmup_steps = int(num_training_steps * warmup_factor) if scheduler_params.get('name', 'none') != "none" else 0
    
    scheduler = get_scheduler(optimizer, scheduler_params.get('name', 'none'), num_warmup_steps, num_training_steps)
    
    results_data.update({f"optimizer_{k}": v for k,v in optimizer_params.items()})
    results_data.update({f"scheduler_{k}": v for k,v in scheduler_params.items()})

    # --- 4. Loss Criterion ---
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)

    # --- 5. Training Loop ---
    grad_clip_norm = optimizer_params.get('grad_clip_norm')
    aux_loss_coeff = optimizer_params.get('aux_loss_coeff', 0.0)
    
    best_val_f1_overall = 0.0
    experiment_start_time = time.time()

    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        train_loss, train_main_loss, train_aux_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE, scheduler, grad_clip_norm, aux_loss_coeff
        )
        val_loss, val_acc, val_f1, val_latency_batch, epoch_model_metrics = evaluate_model(
            model, val_loader, criterion, DEVICE, return_latency=True
        )
        epoch_duration = time.time() - epoch_start_time

        if val_f1 > best_val_f1_overall:
            best_val_f1_overall = val_f1

        logging.info(
            f"Experiment {exp_name} Epoch {epoch+1}/{config['epochs']} - "
            f"Train Loss: {train_loss:.3f} (Main: {train_main_loss:.3f}, Aux: {train_aux_loss:.3f}), Train Acc: {train_acc:.3f}, Train F1: {train_f1:.3f} | "
            f"Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f} | "
            f"Val Latency/batch: {val_latency_batch*1000:.2f}ms | Epoch Time: {epoch_duration:.2f}s"
        )
        if epoch_model_metrics:
            metrics_log_str = " | ".join([f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k,v in epoch_model_metrics.items()])
            logging.info(f"Experiment {exp_name} Epoch {epoch+1} Model Metrics: {metrics_log_str}")

    experiment_duration = time.time() - experiment_start_time
    results_data['total_training_time_seconds'] = experiment_duration
    
    # --- 6. Final Evaluation and Results Collection ---
    logging.info(f"Performing final evaluation for experiment: {exp_name}")
    final_loss, final_acc, final_f1, final_latency_batch, final_model_specific_metrics = evaluate_model(
        model, val_loader, criterion, DEVICE, return_latency=True
    )
    results_data['metric_final_val_loss'] = final_loss
    results_data['metric_final_val_f1'] = final_f1
    results_data['metric_final_val_acc'] = final_acc
    results_data['metric_inference_latency_ms_batch'] = final_latency_batch * 1000
    results_data['metric_best_val_f1_across_epochs'] = best_val_f1_overall

    if final_model_specific_metrics:
        for k, v_metric in final_model_specific_metrics.items():
            results_data[f'metric_model_specific_{k}'] = v_metric.item() if isinstance(v_metric, torch.Tensor) and v_metric.numel() == 1 else v_metric
    
    logging.info(f"\n--- Experiment Summary: {exp_name} ---")
    for key, value in results_data.items():
        if key.startswith("metric_"):
             logging.info(f"{key.replace('metric_', '').replace('_', ' ').capitalize()}: {value if isinstance(value, list) else (f'{value:.4f}' if isinstance(value, float) else value)}")

    # --- 7. Save Results to CSV ---
    csv_filename = "json_experiment_results.csv" 
    df_results = pd.DataFrame([results_data])
    
    file_exists = os.path.isfile(csv_filename)
    df_results.to_csv(csv_filename, mode='a', header=not file_exists, index=False)
    logging.info(f"Experiment results appended to {csv_filename}")
    logging.info(f"--- Experiment Finished: {exp_name} ---")

# --- Internal Helper Functions for Core Logic (to be used by JSON and Optuna) ---
def _setup_dataset_internal(dataset_type: str, dataset_name: str, batch_size: int, dataset_params_dict: dict, is_hpo_run: bool = False, trial: optuna.trial.Trial = None):
    logging.debug(f"Internal dataset setup: type={dataset_type}, name={dataset_name}, bs={batch_size}, params={dataset_params_dict}, hpo={is_hpo_run}")
    padding_idx = 0
    if dataset_type == "synthetic":
        if is_hpo_run and trial: # HPO suggests these params
            num_samples = trial.suggest_int("syn_num_samples", 500, 2000, step=500) # Reduced range for HPO
            seq_len = trial.suggest_int("syn_seq_len", 8, 32, step=4)
            vocab_size_val = trial.suggest_int("syn_vocab_size", 8, 16, step=4)
        else: # From JSON or fixed args
            num_samples = dataset_params_dict['num_samples']
            seq_len = dataset_params_dict['seq_len']
            vocab_size_val = dataset_params_dict['vocab_size']
        
        train_loader, val_loader, vocab_size, num_classes = get_synthetic_data_loaders(
            dataset_name=dataset_name, batch_size=batch_size, 
            num_samples=num_samples, seq_len=seq_len, vocab_size=vocab_size_val
        )
    elif dataset_type == "real_world":
        if is_hpo_run and trial:
            max_length = trial.suggest_categorical("rw_max_length", [32, 64, 128])
            # train_samples/val_samples are usually fixed, not HPO'd, but could be.
            # For now, assume they come from dataset_params_dict if needed even in HPO.
            train_samples = dataset_params_dict.get('train_samples', 1000) # Default if not in HPO search
            val_samples = dataset_params_dict.get('val_samples', 200)   # Default if not in HPO search
        else:
            max_length = dataset_params_dict['max_length']
            train_samples = dataset_params_dict['train_samples']
            val_samples = dataset_params_dict['val_samples']

        train_loader, val_loader, vocab_size, num_classes, padding_idx_from_data = get_real_world_data_loaders(
            dataset_name=dataset_name, batch_size=batch_size,
            max_length=max_length, train_samples=train_samples, val_samples=val_samples
        )
        padding_idx = padding_idx_from_data
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    return train_loader, val_loader, vocab_size, num_classes, padding_idx

def _create_model_internal(model_type: str, vocab_size: int, num_classes: int, padding_idx: int, model_params_dict: dict, device: torch.device):
    logging.debug(f"Internal model creation: type={model_type}, vocab={vocab_size}, classes={num_classes}, model_params={model_params_dict}")
    if model_type not in MODELS_MAP:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    constructor_params = model_params_dict.copy()
    constructor_params['vocab_size'] = vocab_size
    constructor_params['num_classes'] = num_classes
    constructor_params['padding_idx'] = padding_idx
    
    if model_type == "moe" and "num_experts_per_layer" in constructor_params:
        constructor_params["num_experts"] = constructor_params.pop("num_experts_per_layer")
    if model_type == "goe":
        if "num_total_experts_pool" in constructor_params: # From JSON
             constructor_params["num_total_experts"] = constructor_params.pop("num_total_experts_pool")
        # Ensure num_goe_layers is present if it's a goe model, could be missing if old JSON config
        if "num_goe_layers" not in constructor_params and "num_layers" in constructor_params: # Compatibility
            constructor_params["num_goe_layers"] = constructor_params.pop("num_layers")
        elif "num_goe_layers" not in constructor_params: # Default if not specified
             constructor_params["num_goe_layers"] = 1 # Default GoE layers
    if model_type == "goe_original" and "max_visits" in constructor_params:
        constructor_params["max_visits_per_expert"] = constructor_params.pop("max_visits")

    model = MODELS_MAP[model_type](**constructor_params).to(device)
    return model

def _setup_optimizer_scheduler_internal(model: nn.Module, optimizer_config: dict, scheduler_config: dict, train_loader_len: int, epochs: int):
    logging.debug(f"Internal optimizer/scheduler setup: opt_cfg={optimizer_config}, sched_cfg={scheduler_config}")
    optimizer = get_optimizer(model.parameters(), optimizer_config['name'], optimizer_config['lr'], optimizer_config.get('weight_decay', 0.01))
    
    num_training_steps = train_loader_len * epochs
    warmup_factor = scheduler_config.get('warmup_factor', 0.1)
    num_warmup_steps = int(num_training_steps * warmup_factor) if scheduler_config.get('name', 'none') != "none" else 0
    scheduler = get_scheduler(optimizer, scheduler_config.get('name', 'none'), num_warmup_steps, num_training_steps)
    return optimizer, scheduler

def _run_training_loop_internal(model, train_loader, val_loader, optimizer, scheduler, criterion, 
                               epochs, device, grad_clip_norm, aux_loss_coeff, 
                               log_prefix="Run", trial: optuna.trial.Trial = None): # Added trial for pruning
    logging.debug(f"Internal training loop: epochs={epochs}, prefix={log_prefix}")
    best_val_f1_overall = 0.0
    experiment_start_time = time.time()
    final_model_specific_metrics_for_trial = {}

    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_loss, train_main_loss, train_aux_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler, grad_clip_norm, aux_loss_coeff
        )
        val_loss, val_acc, val_f1, val_latency_batch, epoch_model_metrics = evaluate_model(
            model, val_loader, criterion, device, return_latency=True
        )
        epoch_duration = time.time() - epoch_start_time

        if val_f1 > best_val_f1_overall:
            best_val_f1_overall = val_f1
        
        if epoch == epochs -1: # Store metrics from last epoch for Optuna trial summary
            final_model_specific_metrics_for_trial = epoch_model_metrics

        logging.info(
            f"{log_prefix} Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.3f} (Main: {train_main_loss:.3f}, Aux: {train_aux_loss:.3f}), Train Acc: {train_acc:.3f}, Train F1: {train_f1:.3f} | "
            f"Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f} | "
            f"Val Latency/batch: {val_latency_batch*1000:.2f}ms | Epoch Time: {epoch_duration:.2f}s"
        )
        if epoch_model_metrics:
            metrics_log_str = " | ".join([f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k,v in epoch_model_metrics.items()])
            logging.info(f"{log_prefix} Epoch {epoch+1} Model Metrics: {metrics_log_str}")
        
        # Optuna trial pruning
        if trial: 
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()


    experiment_duration = time.time() - experiment_start_time
    
    # Final evaluation after all epochs
    final_loss, final_acc, final_f1, final_latency_batch, final_model_specific_metrics_final_eval = evaluate_model(
        model, val_loader, criterion, DEVICE, return_latency=True
    )
    
    return {
        "best_val_f1_overall": best_val_f1_overall,
        "final_val_loss": final_loss,
        "final_val_acc": final_acc,
        "final_val_f1": final_f1,
        "inference_latency_ms_batch": final_latency_batch * 1000,
        "training_duration_seconds": experiment_duration,
        "last_epoch_model_specific_metrics": final_model_specific_metrics_for_trial, # from loop
        "final_eval_model_specific_metrics": final_model_specific_metrics_final_eval # from separate final eval
    }


def objective(trial: optuna.trial.Trial, fixed_args: argparse.Namespace):
    """Optuna objective function."""
    trial_summary_dict = {}
    try:
        # --- Hyperparameter Suggestions ---
        # Dataset params (some can be fixed, others HPO'd)
        # For this example, we'll make seq_len HPO'd for synthetic, max_length for real_world
        # Other dataset params like num_samples, vocab_size (synthetic) or train/val_samples (real)
        # will be taken from fixed_args if not explicitly suggested.
        
        dataset_params_hpo = {}
        if fixed_args.dataset_type == "synthetic":
            dataset_params_hpo['seq_len'] = trial.suggest_int("syn_seq_len", 8, 64, step=4)
            dataset_params_hpo['num_samples'] = fixed_args.syn_num_samples if fixed_args.syn_num_samples else 2000 # Default if not in CLI
            dataset_params_hpo['vocab_size'] = fixed_args.syn_vocab_size if fixed_args.syn_vocab_size else 32 # Default if not in CLI
        elif fixed_args.dataset_type == "real_world":
            dataset_params_hpo['max_length'] = trial.suggest_categorical("rw_max_length", [32, 64, 128, 256])
            dataset_params_hpo['train_samples'] = fixed_args.rw_train_samples if fixed_args.rw_train_samples else 2000
            dataset_params_hpo['val_samples'] = fixed_args.rw_val_samples if fixed_args.rw_val_samples else 400


        # Model choice
        model_type_val = trial.suggest_categorical("model_type", list(MODELS_MAP.keys()))
        trial_summary_dict['hp_model_type'] = model_type_val

        # Shared model params
        model_params_dict = {}
        model_params_dict['embed_dim'] = trial.suggest_categorical("embed_dim", [16, 32, 64])
        model_params_dict['num_heads'] = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        model_params_dict['dim_feedforward_factor'] = trial.suggest_categorical("dim_feedforward_factor", [2, 4])
        model_params_dict['dropout'] = trial.suggest_float("dropout", 0.05, 0.3, step=0.05)
        
        # Ensure embed_dim is divisible by num_heads
        if model_params_dict['embed_dim'] % model_params_dict['num_heads'] != 0:
            # Adjust num_heads to be a divisor, or prune. Here, we'll pick the largest valid num_heads.
            valid_heads = [h for h in [1,2,4,8] if model_params_dict['embed_dim'] % h == 0 and h <= model_params_dict['embed_dim']]
            if not valid_heads: raise optuna.exceptions.TrialPruned("embed_dim not divisible by any valid num_heads.")
            model_params_dict['num_heads'] = valid_heads[-1] 
            trial.set_user_attr("adjusted_num_heads", model_params_dict['num_heads']) # Log adjustment


        # Model-specific params
        if model_type_val == "dense":
            model_params_dict['num_layers'] = trial.suggest_int("dt_num_layers", 1, 3)
        elif model_type_val == "moe":
            model_params_dict['num_layers'] = trial.suggest_int("moe_num_layers", 1, 3)
            num_experts = trial.suggest_int("moe_num_experts_per_layer", 2, 8, step=2)
            model_params_dict['num_experts_per_layer'] = num_experts # keep for logging, will be mapped by _create_model_internal
            model_params_dict['top_k_experts'] = trial.suggest_int("moe_top_k", 1, min(num_experts, 4))
        elif model_type_val == "goe":
            model_params_dict['num_goe_layers'] = trial.suggest_int("goe_num_layers", 1, 3) # Name in constructor
            num_total_experts = trial.suggest_int("goe_num_total_experts_pool", 2, 8, step=2)
            model_params_dict['num_total_experts_pool'] = num_total_experts # keep for logging, will be mapped
            model_params_dict['max_path_len'] = trial.suggest_int("goe_max_path_len", 1, min(num_total_experts, 4))
            model_params_dict['router_hidden_dim'] = trial.suggest_categorical("goe_router_hidden_dim", [d // 2 for d in [16,32,64] if d >= model_params_dict['embed_dim']//2] or [model_params_dict['embed_dim']//2])
            model_params_dict['gumbel_tau'] = trial.suggest_float("goe_gumbel_tau", 0.5, 2.0)
        elif model_type_val == "goe_original":
            num_total_experts_orig = trial.suggest_int("goe_original_num_total_experts", 2, 8, step=2)
            model_params_dict['num_total_experts'] = num_total_experts_orig # Direct name for constructor
            model_params_dict['max_path_len'] = trial.suggest_int("goe_original_max_path_len", 1, min(num_total_experts_orig, 4))
            model_params_dict['router_hidden_dim'] = trial.suggest_categorical("goe_original_router_hidden_dim", [d // 2 for d in [16,32,64] if d >= model_params_dict['embed_dim']//2] or [model_params_dict['embed_dim']//2])
            model_params_dict['expert_layers'] = trial.suggest_int("goe_original_expert_layers", 1, 4)
            model_params_dict['gumbel_tau'] = trial.suggest_float("goe_original_gumbel_tau", 0.5, 2.0)
            model_params_dict['path_penalty_coef'] = trial.suggest_float("goe_original_path_penalty_coef", 0.0, 0.1)
            model_params_dict['diversity_loss_coef'] = trial.suggest_float("goe_original_diversity_loss_coef", 0.0, 0.1)
            model_params_dict['contrastive_loss_coef'] = trial.suggest_float("goe_original_contrastive_loss_coef", 0.0, 0.1)
            model_params_dict['max_visits'] = trial.suggest_int("goe_original_max_visits", 1, 3) # will be mapped to max_visits_per_expert

        # Optimizer and Scheduler params
        optimizer_config = {
            "name": trial.suggest_categorical("optimizer_name", ["adamw", "sgd"]),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "grad_clip_norm": trial.suggest_float("grad_clip_norm", 0.5, 2.0, step=0.5) if trial.suggest_categorical("use_grad_clip", [True, False]) else None,
            "aux_loss_coeff": trial.suggest_float("aux_loss_coeff", 0.0, 0.1) if model_type_val in ["moe", "goe", "goe_original"] else 0.0
        }
        scheduler_config = {
            "name": trial.suggest_categorical("scheduler_name", ["cosine_warmup", "none"]),
            "warmup_factor": trial.suggest_float("warmup_factor", 0.01, 0.2) if optimizer_config["name"] != "none" else 0.0
        }
        
        # --- Call Internal Helper Functions ---
        train_loader, val_loader, vocab_size, num_classes, padding_idx = _setup_dataset_internal(
            fixed_args.dataset_type, fixed_args.dataset_name, fixed_args.batch_size, dataset_params_hpo, is_hpo_run=True, trial=trial
        )
        
        # Add fixed dataset params to trial_summary for complete logging
        trial_summary_dict.update({f"hp_dataset_{k}": v for k,v in dataset_params_hpo.items()})


        current_model = _create_model_internal(model_type_val, vocab_size, num_classes, padding_idx, model_params_dict, DEVICE)
        param_count = get_trainable_parameter_count(current_model)
        trial_summary_dict['param_count'] = param_count

        # Pruning based on parameter count
        max_params_for_type = MAX_PARAMS_REAL_WORLD if fixed_args.dataset_type == "real_world" else MAX_PARAMS_SYNTHETIC
        if param_count > max_params_for_type:
            logging.warning(f"Trial {trial.number} pruned: param_count {param_count} > {max_params_for_type}.")
            trial.set_user_attr("full_results", trial_summary_dict) # Store what we have
            return 0.0, float(param_count), 1e9 # Return worst F1, actual params, worst latency

        optimizer, scheduler = _setup_optimizer_scheduler_internal(
            current_model, optimizer_config, scheduler_config, len(train_loader), fixed_args.epochs
        )
        criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)

        # Store HPO'd params for this trial
        trial_summary_dict.update({f"hp_{k}": v for k,v in trial.params.items()}) # Optuna stores suggested params here
        trial_summary_dict.update({f"hp_optimizer_{k}": v for k,v in optimizer_config.items()})
        trial_summary_dict.update({f"hp_scheduler_{k}": v for k,v in scheduler_config.items()})
        
        # Run training loop
        training_results = _run_training_loop_internal(
            current_model, train_loader, val_loader, optimizer, scheduler, criterion,
            fixed_args.epochs, DEVICE, optimizer_config['grad_clip_norm'], optimizer_config['aux_loss_coeff'], 
            log_prefix=f"Trial {trial.number}", trial=trial # Pass trial object for pruning report
        )
        
        trial_summary_dict.update(training_results)
        trial.set_user_attr("full_results", trial_summary_dict)

        return training_results['final_val_f1'], float(param_count), training_results['inference_latency_ms_batch']

    except optuna.exceptions.TrialPruned:
        # Ensure partial results are stored if pruned mid-way
        trial.set_user_attr("full_results", trial_summary_dict)
        raise # Re-raise for Optuna to handle
    except Exception as e:
        logging.error(f"Trial {trial.number} failed with exception: {e}", exc_info=True)
        trial.set_user_attr("full_results", trial_summary_dict) # Store partial results
        # Return worst possible values for Optuna to mark as failed/worst.
        return 0.0, float(trial_summary_dict.get('param_count', MAX_PARAMS_SYNTHETIC * 2)), 1e9 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON-driven or Optuna HPO Experiment Execution for Research Eval.")
    parser.add_argument('--json_config_path', type=str, default=None, 
                        help='Path to a JSON file defining the experiment configuration. If provided, Optuna HPO is skipped.')
    
    # Optuna specific arguments (used if json_config_path is not provided)
    parser.add_argument("--study_name", type=str, default="research_eval_study", help="Name for the Optuna study.")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g., sqlite:///study.db).")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials to run.")

    # Common arguments that can be fixed for an Optuna study, or used by JSON config if not specified there.
    # These will be part of 'fixed_args' in the objective.
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.") # Default for HPO trials
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.") # Default for HPO trials
    
    # Dataset args - these are primarily for Optuna to know which dataset to run HPO on.
    # For JSON, these are specified inside the JSON.
    parser.add_argument("--dataset_type", type=str, default="synthetic", choices=["synthetic", "real_world"], help="Base dataset type for Optuna study.")
    parser.add_argument("--dataset_name", type=str, default="parity", help="Base dataset name for Optuna study.")

    # Minimal dataset params for Optuna's fixed_args if not HPO'd (can be overridden by HPO suggestions)
    parser.add_argument("--syn_num_samples", type=int, help="Fixed num_samples for synthetic data in HPO (if not HPO'd).")
    parser.add_argument("--syn_seq_len", type=int, help="Fixed seq_len for synthetic data in HPO (if not HPO'd).")
    parser.add_argument("--syn_vocab_size", type=int, help="Fixed vocab_size for synthetic data in HPO (if not HPO'd).")
    parser.add_argument("--rw_max_length", type=int, help="Fixed max_length for real-world data in HPO (if not HPO'd).")
    parser.add_argument("--rw_train_samples", type=int, help="Fixed train_samples for real-world data in HPO (if not HPO'd).")
    parser.add_argument("--rw_val_samples", type=int, help="Fixed val_samples for real-world data in HPO (if not HPO'd).")

    args = parser.parse_args()

    if args.json_config_path:
        if not os.path.exists(args.json_config_path):
            logging.error(f"Configuration file not found: {args.json_config_path}")
            sys.exit(1)
        try:
            with open(args.json_config_path, 'r') as f:
                config = json.load(f)
            logging.info(f"Successfully loaded JSON config from {args.json_config_path}")
            # Override JSON 'epochs' or 'batch_size' if explicitly provided on CLI with JSON
            if 'epochs' in config and args.epochs != parser.get_default("epochs"):
                logging.warning(f"Overriding JSON epochs ({config['epochs']}) with CLI value ({args.epochs}).")
                config['epochs'] = args.epochs
            if 'batch_size' in config and args.batch_size != parser.get_default("batch_size"):
                logging.warning(f"Overriding JSON batch_size ({config['batch_size']}) with CLI value ({args.batch_size}).")
                config['batch_size'] = args.batch_size

            run_single_experiment_from_json(config)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {args.json_config_path}: {e}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error loading or running experiment from configuration file {args.json_config_path}: {e}", exc_info=True)
            sys.exit(1)
    else:
        # Run Optuna HPO
        logging.info(f"Starting Optuna HPO: Study '{args.study_name}', Dataset '{args.dataset_name} ({args.dataset_type})'")
        logging.info(f"Running {args.n_trials} trials, each for {args.epochs} epochs with batch_size {args.batch_size}.")

        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
            directions=["maximize", "minimize", "minimize"] # F1, ParamCount, InferenceLatency
        )
        
        study.optimize(lambda trial: objective(trial, args), 
                       n_trials=args.n_trials, 
                       gc_after_trial=True,
                       show_progress_bar=True)

        logging.info(f"Optuna HPO finished. Study name: {study.study_name}")
        logging.info(f"Number of finished trials: {len(study.trials)}")
        logging.info("\nBest trials (Pareto front):")
        for i, trial in enumerate(study.best_trials):
            logging.info(f"  Pareto Trial {i+1} (Number {trial.number}):")
            logging.info(f"    Values (F1, ParamCount, Latency): {trial.values}")
            logging.info(f"    Params: {trial.params}")
            if "full_results" in trial.user_attrs:
                 logging.info(f"    Full Results (sample): {{'metric_final_val_f1': {trial.user_attrs['full_results'].get('metric_final_val_f1', 'N/A')}, ...}}")

        # Save full study results
        results_df_list = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                row = {"trial_number": trial.number, "state": trial.state.name}
                row.update(trial.params) # HPO'd params
                if trial.values:
                    row["value_f1"] = trial.values[0]
                    row["value_param_count"] = trial.values[1]
                    row["value_latency_ms"] = trial.values[2]
                
                if "full_results" in trial.user_attrs:
                    for k, v_attr in trial.user_attrs["full_results"].items():
                         row[k] = v_attr # Add all collected metrics and HPs
                results_df_list.append(row)
            elif trial.state != optuna.trial.TrialState.RUNNING: # Log pruned/failed trials too
                 row = {"trial_number": trial.number, "state": trial.state.name}
                 row.update(trial.params)
                 if "full_results" in trial.user_attrs: # Store partial results if any
                    for k, v_attr in trial.user_attrs["full_results"].items():
                         row[k] = v_attr
                 results_df_list.append(row)


        if results_df_list:
            df_study = pd.DataFrame(results_df_list)
            study_results_filename = f"optuna_study_{args.study_name}_{args.dataset_type}_{args.dataset_name}_full_results.csv"
            df_study.to_csv(study_results_filename, index=False)
            logging.info(f"Full Optuna study results saved to {study_results_filename}")
        else:
            logging.info("No completed Optuna trials to save.")
