import sqlite3
import pandas as pd
import json
import argparse
import copy
import random
import os
import logging
import sys
import hashlib

# Ensure the project root is in sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from research_eval.db_utils import DB_FILE
except ImportError:
    DB_FILE = "experiment_results.db"
    logging.warning("Could not import DB_FILE from research_eval.db_utils, using default 'experiment_results.db'")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def query_results(db_file, sql_query, params=()) -> pd.DataFrame:
    """
    Connects to the SQLite database, executes the given SQL query,
    and returns the results as a Pandas DataFrame.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        df = pd.read_sql_query(sql_query, conn, params=params)
        logging.info(f"Query executed successfully. Fetched {len(df)} rows.")
        return df
    except sqlite3.Error as e:
        logging.error(f"Error querying database {db_file}: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def get_top_configurations(db_file, dataset_name, model_type, metric_name='final_val_f1', top_n=5, higher_is_better=True) -> list[dict]:
    """
    Query the database for the top_n best performing experiment configurations.
    """
    logging.info(f"Retrieving top {top_n} configurations for dataset '{dataset_name}', model '{model_type}', metric '{metric_name}'.")
    
    order_direction = "DESC" if higher_is_better else "ASC"
    
    sql_query = f"""
    SELECT
        e.experiment_id,
        e.full_config_json,
        m.metric_value
    FROM experiments e
    JOIN metrics m ON e.experiment_id = m.experiment_id
    WHERE e.dataset_name = ?
      AND e.model_type = ?
      AND m.metric_name = ?
      AND (e.run_type = 'json' OR e.run_type = 'optuna_trial') 
    ORDER BY m.metric_value {order_direction}
    LIMIT ?
    """
    params = (dataset_name, model_type, metric_name, top_n)
    
    top_configs_df = query_results(db_file, sql_query, params=params)
    
    configurations = []
    if not top_configs_df.empty:
        for _, row in top_configs_df.iterrows():
            try:
                config_json_str = row['full_config_json']
                if config_json_str:
                    config_dict = json.loads(config_json_str)
                    config_dict['_original_experiment_id'] = row['experiment_id']
                    config_dict['_original_metric_value'] = row['metric_value']
                    configurations.append(config_dict)
                else:
                    logging.warning(f"Experiment ID {row['experiment_id']} has empty full_config_json.")
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON for experiment_id {row['experiment_id']}: {e}")
            except Exception as e:
                 logging.error(f"Unexpected error processing row for experiment_id {row['experiment_id']}: {e}")

    logging.info(f"Retrieved {len(configurations)} top configurations.")
    return configurations

def perturb_value(value, p_magnitude, is_int=False, is_dim=False, min_val=None, max_val=None, ensure_even=False):
    """Applies perturbation to a single value."""
    if isinstance(value, bool): # Don't perturb booleans
        return value

    original_value = value
    if is_int:
        if is_dim: # Larger range for dimensions
            perturbation = int(round(value * random.uniform(-p_magnitude, p_magnitude)))
            value += perturbation 
            value = max(1, value) # Ensure positive
            if ensure_even and value % 2 != 0:
                value = max(2, value + random.choice([-1,1])) # make it even, minimum 2
        else: # Smaller range for counts like num_layers
            value += random.choice([-2, -1, 0, 1, 2]) # Allow no change
            value = max(1, value)
    else: # Float
        value *= (1 + random.uniform(-p_magnitude, p_magnitude))

    if min_val is not None: value = max(min_val, value)
    if max_val is not None: value = min(max_val, value)
    
    # logging.debug(f"Perturbed {original_value} to {value} (is_int: {is_int}, is_dim: {is_dim})")
    return value


def generate_variations(base_config: dict, num_variations=3, perturbation_magnitude=0.1) -> list[dict]:
    """
    Generates variations of a base configuration by perturbing numerical hyperparameters.
    """
    variations = []
    
    # Define parameters to perturb and their characteristics
    # path_to_param: (is_int, is_dim_style_perturb, min_val, max_val, ensure_even_if_dim)
    perturbable_params_schema = {
        "model_params": {
            "embed_dim": (True, True, 8, 256, True),
            "num_heads": (True, False, 1, 8, False), # Will be adjusted relative to embed_dim later
            "dim_feedforward_factor": (True, False, 1, 8, False),
            "dropout": (False, False, 0.0, 0.7, False),
            "num_layers": (True, False, 1, 8, False), # For dense
            "dt_num_layers": (True, False, 1, 8, False), # For dense (if named differently)
            "moe_num_layers": (True, False, 1, 8, False),
            "num_experts_per_layer": (True, False, 1, 16, False),
            "top_k_experts": (True, False, 1, 4, False), # Will be adjusted by num_experts
            "num_goe_layers": (True, False, 1, 8, False),
            "num_total_experts_pool": (True, False, 2, 16, False), # Mapped to num_total_experts for GoE
            "num_total_experts": (True, False, 2, 16, False), # For GoEOriginal
            "max_path_len": (True, False, 1, 5, False),
            "router_hidden_dim": (True, True, 8, 128, True),
            "expert_layers": (True, False, 1, 4, False), # For GoEOriginal
            "gumbel_tau": (False, False, 0.1, 5.0, False),
            "path_penalty_coef": (False, False, 0.0, 0.1, False),
            "diversity_loss_coef": (False, False, 0.0, 0.1, False),
            "contrastive_loss_coef": (False, False, 0.0, 0.1, False),
            "max_visits": (True, False, 1, 5, False), # Mapped to max_visits_per_expert
        },
        "optimizer_params": {
            "lr": (False, False, 1e-6, 1e-2, False),
            "weight_decay": (False, False, 1e-7, 1e-1, False),
            "grad_clip_norm": (False, False, 0.1, 5.0, False),
            "aux_loss_coeff": (False, False, 0.0, 0.2, False)
        },
        "scheduler_params": {
            "warmup_factor": (False, False, 0.001, 0.3, False)
        },
        "dataset_params": { # Only perturb if they exist (e.g. synthetic vs real_world)
            "seq_len": (True, True, 8, 128, True), # For synthetic
            "max_length": (True, True, 32, 512, True) # For real_world
        }
    }

    original_exp_name = base_config.get("experiment_name", "unnamed_base")
    if '_var' in original_exp_name: # If already a variation, strip that part for new base name
        original_exp_name = original_exp_name.split('_var')[0]

    for i in range(num_variations):
        var_config = copy.deepcopy(base_config)
        
        # Remove original performance/ID markers
        var_config.pop('_original_experiment_id', None)
        var_config.pop('_original_metric_value', None)

        # Perturb parameters
        for section_key, params_to_perturb in perturbable_params_schema.items():
            if section_key in var_config:
                for param_key, P_ARGS in params_to_perturb.items():
                    if param_key in var_config[section_key]:
                        current_value = var_config[section_key][param_key]
                        is_int, is_dim, min_v, max_v, ensure_even = P_ARGS
                        var_config[section_key][param_key] = perturb_value(current_value, perturbation_magnitude, is_int, is_dim, min_v, max_v, ensure_even)
        
        # Post-perturbation adjustments / constraints
        if "model_params" in var_config:
            mp = var_config["model_params"]
            if "embed_dim" in mp and "num_heads" in mp:
                if mp["embed_dim"] % mp["num_heads"] != 0:
                    # Adjust num_heads to be a divisor of embed_dim
                    for h in range(mp["num_heads"], 0, -1): # Try smaller heads
                        if mp["embed_dim"] % h == 0:
                            mp["num_heads"] = h
                            break
                    if mp["embed_dim"] % mp["num_heads"] != 0: # If still not divisible (e.g. embed_dim is prime > 1)
                         mp["embed_dim"] = mp["num_heads"] * (mp["embed_dim"] // mp["num_heads"] or 1) # Adjust embed_dim instead

            if var_config.get("model_type") == "moe" and "num_experts_per_layer" in mp and "top_k_experts" in mp:
                mp["top_k_experts"] = max(1, min(mp["top_k_experts"], mp["num_experts_per_layer"]))
            
            if var_config.get("model_type") == "goe" and "num_total_experts_pool" in mp and "max_path_len" in mp:
                 mp["max_path_len"] = max(1, min(mp["max_path_len"], mp["num_total_experts_pool"]))
            
            if var_config.get("model_type") == "goe_original" and "num_total_experts" in mp and "max_path_len" in mp:
                 mp["max_path_len"] = max(1, min(mp["max_path_len"], mp["num_total_experts"]))


        # Update experiment name
        config_str_for_hash = json.dumps(var_config, sort_keys=True, ensure_ascii=False)
        config_hash = hashlib.md5(config_str_for_hash.encode('utf-8')).hexdigest()[:8]
        var_config["experiment_name"] = f"{original_exp_name}_var{i+1}_{config_hash}"
        
        variations.append(var_config)
    
    logging.info(f"Generated {len(variations)} variations for base config '{original_exp_name}'.")
    return variations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Suggest new experiment configurations based on top performers.")
    parser.add_argument('--dataset_name', type=str, required=True, help="Dataset name to analyze (e.g., 'parity').")
    parser.add_argument('--model_type', type=str, required=True, help="Model type to analyze (e.g., 'dense').")
    parser.add_argument('--metric_name', type=str, default='final_val_f1', help="Metric to optimize for selecting top configurations.")
    parser.add_argument('--top_n', type=int, default=3, help="Number of top configurations to select as base for variations.")
    parser.add_argument('--num_variations_per_config', type=int, default=2, help="Number of variations to generate per top configuration.")
    parser.add_argument('--perturbation_magnitude', type=float, default=0.1, help="Magnitude of perturbation for numerical parameters (e.g., 0.1 for +/- 10%).")
    parser.add_argument('--output_dir', type=str, default="suggested_configs/", help="Directory to save suggested JSON configuration files.")

    args = parser.parse_args()

    if not os.path.exists(DB_FILE):
        logging.error(f"Database file {DB_FILE} not found. Please run init_db.py first.")
        sys.exit(1)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"Created output directory: {args.output_dir}")

    higher_is_better_metrics = ['final_val_f1', 'final_val_acc', 'best_val_f1_across_epochs'] # Add more if needed
    is_metric_higher_better = args.metric_name in higher_is_better_metrics

    top_configs = get_top_configurations(
        DB_FILE, 
        args.dataset_name, 
        args.model_type, 
        metric_name=args.metric_name, 
        top_n=args.top_n,
        higher_is_better=is_metric_higher_better
    )

    if not top_configs:
        logging.info("No top configurations found to generate variations from.")
        sys.exit(0)

    total_variations_generated = 0
    for i, base_config in enumerate(top_configs):
        base_exp_name_for_log = base_config.get('experiment_name', base_config.get('_original_experiment_id', 'UnknownBase'))
        logging.info(f"\nGenerating variations for Top Configuration #{i+1} (Source ID: {base_config.get('_original_experiment_id')}, Metric: {base_config.get('_original_metric_value')}, Name: {base_exp_name_for_log}):")
        
        variations = generate_variations(
            base_config, 
            args.num_variations_per_config, 
            args.perturbation_magnitude
        )
        
        for var_idx, var_config in enumerate(variations):
            new_exp_name = var_config["experiment_name"] # This is now generated with a hash
            output_filename = os.path.join(args.output_dir, f"{new_exp_name}.json")
            try:
                with open(output_filename, 'w') as f:
                    json.dump(var_config, f, indent=2)
                logging.info(f"Saved variation: {output_filename}")
                total_variations_generated += 1
            except IOError as e:
                logging.error(f"Error saving variation {output_filename}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error saving variation {output_filename}: {e}")


    logging.info(f"\n--- Process Summary ---")
    logging.info(f"Retrieved {len(top_configs)} top configurations for {args.dataset_name} - {args.model_type} based on {args.metric_name}.")
    logging.info(f"Generated a total of {total_variations_generated} new configurations.")
    logging.info(f"Suggested configurations saved in: {args.output_dir}")
