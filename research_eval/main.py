import torch
import torch.nn as nn
import optuna
import argparse
import time
import pandas as pd
import os
import logging

from research_eval.datasets.providers import get_synthetic_data_loaders, get_real_world_data_loaders, SYNTHETIC_DATASETS
from research_eval.datasets.providers import get_synthetic_data_loaders, get_real_world_data_loaders, SYNTHETIC_DATASETS
from research_eval.models.dense_transformer import DenseTransformerClassifier
from research_eval.models.goe_transformer import GoEClassifier
from research_eval.models.moe_transformer import MoETransformerClassifier
from research_eval.models.goe_original_classifier import GoEOriginalClassifier # Import the new model
from research_eval.utils import train_epoch, evaluate_model, get_optimizer, get_scheduler

# Add the new model name to the list of available models
MODELS = ["dense", "moe", "goe", "goe_original"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_WORLD_DATASETS = ["ag_news", "imdb", "sst2", "trec"]
MAX_PARAMS_SYNTHETIC = 30_000_000
MAX_PARAMS_REAL_WORLD = MAX_PARAMS_SYNTHETIC

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.trial.Trial, dataset_type: str, dataset_name: str, epochs: int, batch_size: int):
    trial_results = {}
    padding_idx = 0 # Default, will be updated for real-world datasets

    if dataset_type == "synthetic":
        train_loader, val_loader, vocab_size, num_classes = get_synthetic_data_loaders(
            dataset_name=dataset_name, batch_size=batch_size, 
            num_samples=trial.suggest_int("syn_num_samples", 1000, 2000, step=500), # Smaller for faster trials
            seq_len=trial.suggest_int("syn_seq_len", 16, 32, step=4),
            vocab_size=trial.suggest_int("syn_vocab_size", 8, 16, step=4)
        )
    elif dataset_type == "real_world":
        train_loader, val_loader, vocab_size, num_classes, PADDING_IDX_FROM_DATA = get_real_world_data_loaders(
            dataset_name=dataset_name,
            max_length=trial.suggest_categorical("rw_max_length", [32, 64]),
            batch_size=batch_size,
            train_samples=trial.suggest_int("rw_train_samples", 800, 1500, step=100), # Small samples
            val_samples=trial.suggest_int("rw_val_samples", 200, 400, step=50)
        )
        padding_idx = PADDING_IDX_FROM_DATA
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    model_type = trial.suggest_categorical("model_type", MODELS)

    embed_dim = trial.suggest_categorical("embed_dim", [32, 64, 128])

    # Suggest num_heads from a fixed list of common values.
    # Then, validate against the chosen embed_dim.
    _fixed_num_heads_options = [1, 2, 4, 8]
    num_heads = trial.suggest_categorical("num_heads", _fixed_num_heads_options)

    if embed_dim % num_heads != 0:
        raise optuna.exceptions.TrialPruned(
            f"embed_dim ({embed_dim}) is not divisible by num_heads ({num_heads}). Pruning trial."
        )
    if num_heads > embed_dim and embed_dim > 0: # embed_dim > 0 to avoid issues if it could be 0
         raise optuna.exceptions.TrialPruned(
            f"num_heads ({num_heads}) cannot be greater than embed_dim ({embed_dim}). Pruning trial."
        )

    # Suggest a factor, then compute dim_feedforward. This keeps choices static.
    dim_feedforward_factor = trial.suggest_categorical("dim_feedforward_factor", [2, 4]) # Common factors
    dim_feedforward = embed_dim * dim_feedforward_factor

    dropout = trial.suggest_float("dropout", 0, 0.3, step=0.05)

    model = None
    if model_type == "dense":
        num_layers = trial.suggest_int("dt_num_layers", 1, 3)
        model = DenseTransformerClassifier(vocab_size, embed_dim, num_heads, num_layers, 
                                           dim_feedforward, num_classes, dropout, padding_idx).to(DEVICE)
    elif model_type == "moe":
        num_layers = trial.suggest_int("moe_num_layers", 1, 2) # Fewer layers for MoE to manage params
        num_experts = trial.suggest_int("moe_num_experts_per_layer", 2, 4)
        top_k_experts = trial.suggest_int("moe_top_k", 1, min(2, num_experts))
        model = MoETransformerClassifier(vocab_size, embed_dim, num_heads, num_layers, num_experts,
                                         dim_feedforward, num_classes, dropout, padding_idx, top_k_experts).to(DEVICE)
    elif model_type == "goe":
        num_total_experts = trial.suggest_int("goe_num_total_experts_pool", 2, 8)
        max_path_len = trial.suggest_int("goe_max_path_len", 2, num_total_experts)
        router_hidden_dim = trial.suggest_int("goe_router_hidden_dim", embed_dim // 2, embed_dim)
        gumbel_tau = trial.suggest_float("goe_gumbel_tau", 0.5, 1.5, step=0.25)
        model = GoEClassifier(vocab_size, embed_dim, num_heads, dim_feedforward, num_classes,
                              num_total_experts, max_path_len, router_hidden_dim,
                              dropout, padding_idx, gumbel_tau).to(DEVICE)
    elif model_type == "goe_original":
        # Hyperparameters specific to GoEOriginalClassifier
        num_total_experts = trial.suggest_int("goe_original_num_total_experts", 2, 8)
        max_path_len = trial.suggest_int("goe_original_max_path_len", 2, num_total_experts)
        router_hidden_dim = trial.suggest_int("goe_original_router_hidden_dim", embed_dim // 2, embed_dim)
        expert_layers = trial.suggest_int("goe_original_expert_layers", 1, 4) # Number of layers per expert
        gumbel_tau = trial.suggest_float("goe_original_gumbel_tau", 0.5, 2.0, step=0.25) # Wider range for tau
        path_penalty_coef = trial.suggest_float("goe_original_path_penalty_coef", 0.001, 0.05, log=True)
        diversity_loss_coef = trial.suggest_float("goe_original_diversity_loss_coef", 0.001, 0.1, log=True)
        contrastive_loss_coef = trial.suggest_float("goe_original_contrastive_loss_coef", 0.001, 0.1, log=True)
        max_visits_per_expert = trial.suggest_int("goe_original_max_visits", 1, 3) # Added max_visits_per_expert HP

        model = GoEOriginalClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads, # Reusing num_heads from shared HPs
            dim_feedforward=dim_feedforward, # Reusing dim_feedforward from shared HPs
            num_classes=num_classes,
            num_total_experts=num_total_experts,
            max_path_len=max_path_len,
            router_hidden_dim=router_hidden_dim,
            expert_layers=expert_layers,
            dropout=dropout, # Reusing dropout from shared HPs
            padding_idx=padding_idx,
            gumbel_tau=gumbel_tau,
            path_penalty_coef=path_penalty_coef,
            diversity_loss_coef=diversity_loss_coef,
            contrastive_loss_coef=contrastive_loss_coef,
            max_visits_per_expert=max_visits_per_expert # Pass new HP
        ).to(DEVICE)

    if model is None: raise ValueError("Model not instantiated")

    param_count = model.get_parameter_count()
    trial_results['param_count'] = param_count
    max_params = MAX_PARAMS_REAL_WORLD if dataset_type == "real_world" else MAX_PARAMS_SYNTHETIC
    if param_count > max_params :
        logging.warning(f"Trial {trial.number} pruned: param_count {param_count} > {max_params}.")
        # For multi-objective, we need to return values for all objectives.
        # Return worst possible accuracy, the high param_count, and high latency.
        return 0.0, float(param_count), 1e6 # Accuracy (maximize), ParamCount (minimize), Latency (minimize)

    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "sgd"])
    optimizer = get_optimizer(model.parameters(), optimizer_name, lr)
    
    scheduler_name = trial.suggest_categorical("scheduler", ["cosine_warmup", "none"])
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(num_training_steps * 0.1) if scheduler_name != "none" else 0
    scheduler = get_scheduler(optimizer, scheduler_name, num_warmup_steps, num_training_steps)
    
    grad_clip_norm = trial.suggest_float("grad_clip_norm", 0.5, 2.0, step=0.5) if trial.suggest_categorical("use_grad_clip", [True, False]) else None
    aux_loss_coeff = trial.suggest_float("aux_loss_coeff", 0.001, 0.1, log=True) if model_type in ["moe", "goe"] else 0.0

    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    trial_train_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_loss, train_main_loss, train_aux_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE, scheduler, grad_clip_norm, aux_loss_coeff
        )
        val_loss, val_acc, val_f1, val_latency_batch, model_metrics = evaluate_model(
            model, val_loader, criterion, DEVICE, return_latency=True
        )
        epoch_duration = time.time() - epoch_start_time
        
        logging.info(
            f"Trial {trial.number} Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.3f} (Main: {train_main_loss:.3f}, Aux: {train_aux_loss:.3f}), Train Acc: {train_acc:.3f}, Train F1: {train_f1:.3f} | "
            f"Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f} | "
            f"Val Latency/batch: {val_latency_batch*1000:.2f}ms | Epoch Time: {epoch_duration:.2f}s"
        )
        for k, v_metric in model_metrics.items(): # Renamed v to v_metric to avoid conflict
            # Ensure v_metric is a scalar before formatting
            log_value = v_metric.item() if isinstance(v_metric, torch.Tensor) else v_metric
            logging.info(f"Trial {trial.number} Epoch {epoch+1} - {k}: {log_value:.3f}")


        if val_f1 > best_val_f1: best_val_f1 = val_f1
        #trial.report(val_f1, epoch)
        #if trial.should_prune():
        #    raise optuna.exceptions.TrialPruned()

    trial_duration = time.time() - trial_train_start_time
    trial_results['training_time_seconds'] = trial_duration
    trial_results['best_val_f1'] = best_val_f1

    _, _, final_f1, final_latency, final_model_metrics = evaluate_model(model, val_loader, criterion, DEVICE, return_latency=True)
    trial_results['final_val_f1'] = final_f1
    trial_results['inference_latency_ms_batch'] = final_latency * 1000

    # Ensure model metrics are scalar before adding to results
    for k, v_metric in final_model_metrics.items():
        trial_results[k] = v_metric.item() if isinstance(v_metric, torch.Tensor) else v_metric


    for key, value in trial.params.items(): trial_results[f"param_{key}"] = value
    trial.set_user_attr("full_results", trial_results)

    return trial_results['final_val_f1'], float(param_count), trial_results['inference_latency_ms_batch']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Research Plan Evaluation with Optuna")
    parser.add_argument("--study_name", type=str, default="multiobj_study", help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g., sqlite:///study.db)")
    parser.add_argument("--dataset_type", type=str, default="synthetic", choices=["synthetic", "real_world"], help="Dataset type")
    parser.add_argument("--dataset_name", type=str, default="parity", help=f"Specific dataset name. Synthetic: {list(SYNTHETIC_DATASETS.keys())}. Real-world: {REAL_WORLD_DATASETS}")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs per trial")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    if args.dataset_type == "synthetic" and args.dataset_name not in SYNTHETIC_DATASETS:
        raise ValueError(f"Synthetic dataset '{args.dataset_name}' not found. Available: {list(SYNTHETIC_DATASETS.keys())}")
    if args.dataset_type == "real_world" and args.dataset_name not in REAL_WORLD_DATASETS:
        raise ValueError(f"Real-world dataset '{args.dataset_name}' not found. Available: {REAL_WORLD_DATASETS}")

    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Starting Optuna HPO: Study '{args.study_name}', Dataset Type '{args.dataset_type}', Dataset Name '{args.dataset_name}'")
    logging.info(f"Running {args.n_trials} trials, each {args.epochs} epochs with batch_size {args.batch_size}.")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True, # Load existing study if same name and storage
        directions=["maximize", "minimize", "minimize"],  # F1, ParamCount, InferenceLatency
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1, n_min_trials=3) # Prune after 1 epoch, if at least 3 trials done
    )
    
    study_start_time = time.time()
    try:
        study.optimize(lambda trial: objective(trial, args.dataset_type, args.dataset_name, args.epochs, args.batch_size), 
                       n_trials=args.n_trials, 
                       gc_after_trial=True, # Helps with VRAM
                       show_progress_bar=True) # Optuna progress bar
    except KeyboardInterrupt:
        logging.info("Optimization interrupted by user.")
    
    study_duration = time.time() - study_start_time
    logging.info(f"Optuna HPO finished in {study_duration:.2f} seconds.")
    
    logging.info(f"Study name: {study.study_name}")
    logging.info(f"Number of finished trials: {len(study.trials)}")

    logging.info("\nBest trials (Pareto front):")
    for i, trial in enumerate(study.best_trials):
        logging.info(f"  Pareto Trial {i+1} (Number {trial.number}):")
        logging.info(f"    Values (F1, ParamCount, Latency): {trial.values}")
        logging.info(f"    Params: {trial.params}")
        if "full_results" in trial.user_attrs:
             logging.info(f"    Full Results (sample): {{'final_val_f1': {trial.user_attrs['full_results'].get('final_val_f1', 'N/A')}, ...}}")


    results_df = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            row = {"trial_number": trial.number}
            row.update(trial.params) # Hyperparameters suggested by Optuna
            row["value_f1"] = trial.values[0] if trial.values else None
            row["value_param_count"] = trial.values[1] if trial.values and len(trial.values) > 1 else None
            row["value_latency_ms"] = trial.values[2] if trial.values and len(trial.values) > 2 else None

            # Add all other logged metrics from full_results, prefixing them to avoid clashes
            if "full_results" in trial.user_attrs:
                for k, v_attr in trial.user_attrs["full_results"].items():
                    # Avoid duplicating keys already present (like param_count from trial.params)
                    # or values already captured (like value_f1)
                    if not k.startswith("param_") and k not in ["final_val_f1", "param_count", "inference_latency_ms_batch"]:
                         row[f"metric_{k}"] = v_attr
            results_df.append(row)

    if results_df:
        df = pd.DataFrame(results_df)
        results_filename = f"{args.study_name}_{args.dataset_type}_{args.dataset_name}_results.csv"
        df.to_csv(results_filename, index=False)
        logging.info(f"Full study results saved to {results_filename}")
    else:
        logging.info("No completed trials to save.")
