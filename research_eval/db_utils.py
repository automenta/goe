import sqlite3
from sqlite3 import Error
import json
import uuid
import logging
import os
import hashlib

# Define the database file path (consistent with init_db.py)
# Allow overriding via environment variable for flexibility, otherwise default.
DB_FILE = os.environ.get("RESEARCH_EVAL_DB_PATH", "experiment_results.db")

def connect_db(db_path=DB_FILE):
    """ Create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
    except Error as e:
        logging.error(f"Error connecting to database {db_path}: {e}")
    return conn

def generate_experiment_id(config: dict, run_type: str, trial_info=None) -> str:
    """
    Generates a unique experiment ID.
    For Optuna, uses study name and trial number.
    For JSON, uses experiment_name from config + a short hash of the full config for uniqueness,
    or a UUID if experiment_name is not provided.
    """
    if run_type == "optuna_trial" and trial_info:
        # trial_info is expected to be a simple dict or Namespace with study_name and number
        return f"optuna_{trial_info.study_name}_{trial_info.number}"
    elif run_type == "json":
        exp_name = config.get("experiment_name")
        # Create a hash of the full config string to ensure uniqueness if names collide
        config_str = json.dumps(config, sort_keys=True, ensure_ascii=False)
        config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8] # Use part of the hash
        
        if exp_name:
            return f"json_{exp_name}_{config_hash}"
        else:
            return f"json_uuid_{str(uuid.uuid4())}_{config_hash}"
    else:
        # Fallback or raise error for unknown run_type
        logging.warning(f"Unknown run_type '{run_type}' for experiment ID generation. Using UUID.")
        return str(uuid.uuid4())

def log_experiment_to_db(experiment_data: dict, metrics_data: list[dict], db_path=DB_FILE):
    """
    Logs experiment configuration and its metrics to the SQLite database.
    Uses a transaction for atomicity.

    :param experiment_data: Dictionary containing data for the 'experiments' table.
                            Must include 'experiment_id'.
    :param metrics_data: List of dictionaries, each like 
                         {'metric_name': name, 'metric_value': value}.
                         Each dict will also have 'experiment_id' added.
    :param db_path: Path to the SQLite database file.
    :return: True if logging was successful, False otherwise.
    """
    if 'experiment_id' not in experiment_data:
        logging.error("Cannot log experiment: 'experiment_id' is missing from experiment_data.")
        return False

    conn = connect_db(db_path)
    if conn is None:
        return False

    try:
        cursor = conn.cursor()
        cursor.execute("BEGIN")

        # Insert into experiments table
        exp_cols = ', '.join(experiment_data.keys())
        exp_placeholders = ', '.join(['?'] * len(experiment_data))
        exp_sql = f"INSERT OR REPLACE INTO experiments ({exp_cols}) VALUES ({exp_placeholders})"
        # Using OR REPLACE to handle cases where an experiment ID might be re-logged (e.g., during debugging)
        # For production, OR IGNORE or more careful ID generation might be preferred.
        
        cursor.execute(exp_sql, list(experiment_data.values()))
        logging.info(f"Logged experiment core data for ID: {experiment_data['experiment_id']}")

        # Insert into metrics table
        for metric_dict in metrics_data:
            metric_dict['experiment_id'] = experiment_data['experiment_id'] # Ensure experiment_id is part of metric
            
            # Handle cases where metric_value might be a list (e.g. GoEOriginal expert usage)
            # SQLite doesn't directly support list types, so we serialize to JSON string
            if isinstance(metric_dict['metric_value'], list):
                metric_dict['metric_value'] = json.dumps(metric_dict['metric_value'])

            met_cols = ', '.join(metric_dict.keys())
            met_placeholders = ', '.join(['?'] * len(metric_dict))
            met_sql = f"INSERT INTO metrics ({met_cols}) VALUES ({met_placeholders})"
            cursor.execute(met_sql, list(metric_dict.values()))
        
        conn.commit()
        logging.info(f"Successfully logged {len(metrics_data)} metrics for experiment ID: {experiment_data['experiment_id']}")
        return True
    except Error as e:
        logging.error(f"Error logging experiment to database (experiment_id: {experiment_data.get('experiment_id')}): {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # Example Usage (for testing db_utils.py itself)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 1. Initialize database (ensure init_db.py has been run or call its function)
    # from research_eval.init_db import initialize_database
    # initialize_database(DB_FILE) # Make sure DB_FILE here matches the one in init_db
    
    # 2. Sample data
    sample_json_config = {
        "experiment_name": "sample_json_run_001",
        "model_type": "dense",
        "dataset_type": "synthetic",
        "dataset_name": "parity",
        "epochs": 1,
        "batch_size": 2,
        "model_params": {"embed_dim": 10, "num_heads": 1, "num_layers": 1, "dim_feedforward_factor": 2, "dropout": 0.1},
        "dataset_params": {"num_samples": 10, "seq_len": 4, "vocab_size": 8},
        "optimizer_params": {"name": "adamw", "lr": 1e-3}
    }
    
    exp_id_json = generate_experiment_id(sample_json_config, "json")
    
    experiment_data_json = {
        "experiment_id": exp_id_json,
        "run_type": "json",
        "experiment_name": sample_json_config["experiment_name"],
        "model_type": sample_json_config["model_type"],
        "dataset_name": sample_json_config["dataset_name"],
        "dataset_type": sample_json_config["dataset_type"],
        "epochs": sample_json_config["epochs"],
        "batch_size": sample_json_config["batch_size"],
        "full_config_json": json.dumps(sample_json_config),
    }
    metrics_data_json = [
        {"metric_name": "final_val_f1", "metric_value": 0.95},
        {"metric_name": "param_count", "metric_value": 12345},
        {"metric_name": "expert_usage", "metric_value": [10,5,8]} # Example of a list metric
    ]

    # Log the sample JSON experiment
    # Ensure DB_FILE exists and tables are created by running init_db.py first
    # if os.path.exists(DB_FILE): # Basic check
    #    log_experiment_to_db(experiment_data_json, metrics_data_json)
    # else:
    #    logging.warning(f"{DB_FILE} not found. Run init_db.py to create it before testing db_utils.")

    # Sample Optuna trial info (mimicking Optuna's trial object attributes)
    class MockStudy:
        def __init__(self, name):
            self.name = name
    class MockTrialInfo:
        def __init__(self, study_name, number, params):
            self.study = MockStudy(study_name)
            self.number = number
            self.params = params # Optuna trial.params

    sample_optuna_trial_info = MockTrialInfo("my_optuna_study", 1, {"lr": 0.001, "model_type": "moe"})
    exp_id_optuna = generate_experiment_id({}, "optuna_trial", trial_info=sample_optuna_trial_info)

    experiment_data_optuna = {
        "experiment_id": exp_id_optuna,
        "run_type": "optuna_trial",
        "experiment_name": sample_optuna_trial_info.study.name, # Using Optuna study name
        "model_type": sample_optuna_trial_info.params.get("model_type", "unknown"),
        "dataset_name": "from_fixed_args_or_trial", # This would come from fixed_args in objective
        "dataset_type": "from_fixed_args_or_trial",
        "epochs": 5, # Example fixed epoch
        "batch_size": 16, # Example fixed batch_size
        "full_config_json": json.dumps(sample_optuna_trial_info.params),
    }
    metrics_data_optuna = [
        {"metric_name": "final_val_f1", "metric_value": 0.88},
        {"metric_name": "param_count", "metric_value": 54321},
        {"metric_name": "inference_latency_ms_batch", "metric_value": 25.5},
    ]
    # if os.path.exists(DB_FILE):
    #    log_experiment_to_db(experiment_data_optuna, metrics_data_optuna)
    
    logging.info(f"Generated JSON experiment ID: {exp_id_json}")
    logging.info(f"Generated Optuna experiment ID: {exp_id_optuna}")
    logging.info(f"DB_FILE is set to: {DB_FILE}")
    logging.info("To test logging, uncomment the log_experiment_to_db calls and ensure init_db.py has been run.")
