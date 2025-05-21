# Graph-of-Experts (GoE) Language Model & Experimentation Framework

This repository implements a **Graph-of-Experts (GoE) Language Model** and a framework for running and evaluating experiments with this and other Transformer-based architectures. The primary script for execution is `research_eval/main.py`.

**Note:** The previous `research_eval/main.py` (which handled Optuna HPO and a different CLI structure) has been replaced by this version, which integrates both JSON-driven single experiments and Optuna HPO.

## Features

*   **JSON-Configurable Experiments:** Run single, reproducible experiments using detailed JSON configuration files (`research_eval/main.py`).
*   **Hyperparameter Optimization:** Integrated with Optuna for systematic hyperparameter optimization (`research_eval/main.py`).
*   **Multiple Model Architectures:** Supports:
    *   `dense`: Standard Dense Transformer.
    *   `moe`: Mixture-of-Experts (top-k gating).
    *   `goe`: Graph-of-Experts (simpler version, details may vary).
    *   `goe_original`: The specific Graph-of-Experts architecture with RL-routing and advanced features.
*   **Synthetic and Real-World Datasets:** Supports built-in synthetic datasets (e.g., `parity`, `addition`) and common real-world text classification datasets (e.g., `ag_news`, `imdb`).
*   **Centralized Results Management:** Experiment configurations and metrics are stored in an SQLite database for robust analysis and comparison.
*   **Automated Analysis:** Includes a script (`analyze_results.py`) to generate summary tables and plots from stored results.
*   **Configuration Suggestion:** Includes a script (`suggest_configs.py`) to generate new experiment configurations based on high-performing past runs.
*   **Detailed Logging & Results:** Outputs comprehensive logs during training and saves experiment results to CSV files (in addition to the database).
*   **Modular Codebase:** Designed for extensibility and research.

## Core Task: Language Modeling & Classification
     
* The primary task is language modeling (next-token prediction) for synthetic datasets and text classification for real-world datasets.
* The loss function is typically `nn.CrossEntropyLoss`.

## Model Architecture Highlights (GoEOriginalClassifier)

1.  **Input Processing:** Standard tokenization, embedding, positional encoding, and LayerNorm.
2.  **Expert Modules:** Each expert contains multiple Transformer-like blocks with GatedAttention and adaptive gating.
3.  **Routing Controller:** Uses RL-inspired mechanisms (Q-values, Gumbel-Softmax) to dynamically select a sequence of experts.
4.  **Dynamic Path Forward Pass:** Iteratively builds computational paths for each input sample.
5.  **Auxiliary Losses:** May include losses for router entropy, expert diversity, and contrastive learning to encourage specialization and efficient routing.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Initialize the results database (first time only):
    ```bash
    python research_eval/init_db.py
    ```
    (Note: `research_eval/main.py` will also attempt to initialize the database if it doesn't exist when an experiment is run.)

## Running Experiments with `main.py`

The `research_eval/main.py` script is the main entry point for running experiments. It supports two primary modes:

### 1. Running Single Experiments (JSON Configuration)

This mode allows you to run a single experiment with a specific configuration defined in a JSON file.

**Command:**

```bash
python research_eval/main.py --json_config_path path/to/your/experiment_config.json
```

**JSON Configuration File Structure:** 
(See `examples/configs/` for detailed examples.)
```json
{
  "experiment_name": "my_dense_parity_experiment", // Optional: A name for this specific run
  "model_type": "dense",                     // "dense", "moe", "goe", "goe_original"
  "dataset_type": "synthetic",               // "synthetic" or "real_world"
  "dataset_name": "parity",                  // e.g., "parity", "addition", "ag_news", "imdb"
  "epochs": 10,
  "batch_size": 32,
  
  "dataset_params": { /* ... */ },
  "model_params": { /* ... */ },
  "optimizer_params": { /* ... */ },
  "scheduler_params": { /* ... */ } // Optional
}
```
Refer to the "Developer Guide" section for details on `model_params` for specific model types.

**Output:**
*   Results are logged to the SQLite database (`experiment_results.db`).
*   A summary CSV (`json_experiment_results.csv`) is also appended with results from JSON-driven runs.

### 2. Hyperparameter Optimization (Optuna)

If `--json_config_path` is **not** provided, `main.py` runs in Optuna HPO mode.

**Command:**

```bash
python research_eval/main.py \
    --study_name "my_hpo_study" \
    --storage "sqlite:///my_hpo_study.db" \
    --n_trials 100 \
    --epochs 10 \
    --batch_size 32 \
    --dataset_type synthetic \
    --dataset_name parity \
    --syn_num_samples 10000 
    # Add other fixed parameters as needed
```

**Key CLI Arguments for Optuna Mode:**
*   `--study_name`, `--storage`, `--n_trials`: Optuna study configuration.
*   `--epochs`, `--batch_size`, `--dataset_type`, `--dataset_name`, and dataset-specific args (e.g., `--syn_num_samples`): Fixed parameters for the HPO study.

**Hyperparameter Search Space:** Defined within the `objective` function in `research_eval/main.py`.

**Output:**
*   Results for each trial are logged to the SQLite database.
*   Optuna study results are saved to `optuna_study_<study_name>_<dataset_type>_<dataset_name>_full_results.csv`.

## Experiment Results Management

Experiment configurations and detailed metrics are stored in an SQLite database (`experiment_results.db` by default, path defined in `research_eval.db_utils.DB_FILE`).

*   **Initialization (`research_eval/init_db.py`):**
    *   This script creates the database and its schema if it doesn't already exist.
    *   It is run automatically by `main.py` at the start of an experiment if the database file is not found.
    *   It can also be run manually: `python research_eval/init_db.py`.
*   **Database Schema:**
    *   **`experiments` Table:** Stores metadata for each experiment run.
        *   `experiment_id` (TEXT, PRIMARY KEY): Unique identifier for the run.
        *   `run_type` (TEXT): "json" or "optuna_trial".
        *   `timestamp` (DATETIME): Start time of the experiment.
        *   `experiment_name` (TEXT): User-defined name (for JSON runs) or Optuna study name.
        *   `model_type`, `dataset_name`, `dataset_type`, `epochs`, `batch_size` (TEXT/INTEGER): Key configuration parameters.
        *   `full_config_json` (TEXT): JSON string of the full configuration (either from the input JSON or Optuna's `trial.params`).
        *   `env_details_json` (TEXT): JSON string of basic environment details (e.g., Python version).
    *   **`metrics` Table:** Stores individual metrics for each experiment.
        *   `metric_id` (INTEGER, PRIMARY KEY AUTOINCREMENT).
        *   `experiment_id` (TEXT, FOREIGN KEY to `experiments` table).
        *   `metric_name` (TEXT): Name of the metric (e.g., "final_val_f1", "param_count").
        *   `metric_value` (REAL): Value of the metric. List-like metrics are stored as JSON strings.
*   **CSV Logs:** While the SQLite database is the primary store for analysis, `main.py` also appends to local CSV files (`json_experiment_results.csv` for JSON runs, and `optuna_study_...csv` for HPO studies) for quick, human-readable inspection of recent runs.

## Analyzing Results (`analyze_results.py`)

The `research_eval/analyze_results.py` script queries the SQLite database to generate summary tables and plots.

**Usage Example:**

```bash
python research_eval/analyze_results.py \
    --dataset_name parity \
    --model_types dense,moe \
    --output_dir analysis_reports/ \
    --run_all
```

**Key CLI Arguments:**
*   `--dataset_name` (required): Specifies the dataset to analyze (e.g., "parity", "ag_news").
*   `--model_types` (optional): Comma-separated list of model types to include (e.g., "dense,moe").
*   `--output_dir` (default: `analysis_reports/`): Directory where CSVs and plots will be saved.
*   `--metric_to_optimize` (default: `final_val_f1`): Metric used to select the "best" run for each model type in the summary table.
*   `--run_summary_table`: Flag to generate the summary table.
*   `--run_f1_vs_params_plot`: Flag to generate the F1 score vs. parameter count plot.
*   `--run_f1_vs_latency_plot`: Flag to generate the F1 score vs. inference latency plot.
*   `--run_all`: Flag to run all available analysis types.

**Outputs:**
*   **Summary Table:** A CSV file (e.g., `analysis_reports/parity_summary_table_best_finalvalf1.csv`) summarizing the best performing experiment for each model type based on the `--metric_to_optimize`.
*   **Plots:** PNG image files for F1 vs. Parameters and F1 vs. Latency, saved in the `--output_dir`.

## Suggesting New Configurations (`suggest_configs.py`)

The `research_eval/suggest_configs.py` script helps explore the hyperparameter space by querying the database for high-performing configurations and generating slight variations of them.

**Usage Example:**

```bash
python research_eval/suggest_configs.py \
    --dataset_name parity \
    --model_type dense \
    --metric_name final_val_f1 \
    --top_n 3 \
    --num_variations_per_config 2 \
    --output_dir suggested_configs/
```

**Key CLI Arguments:**
*   `--dataset_name`, `--model_type`: Specify the dataset and model type to find top configurations for.
*   `--metric_name` (default: `final_val_f1`): Metric used to rank existing configurations.
*   `--top_n` (default: 3): Number of top configurations to select as templates.
*   `--num_variations_per_config` (default: 2): Number of new variations to generate from each template.
*   `--perturbation_magnitude` (default: 0.1): Controls the degree of random change for numerical hyperparameters.
*   `--output_dir` (default: `suggested_configs/`): Directory where the new JSON configuration files will be saved.

**Outputs:**
New JSON configuration files (e.g., `suggested_configs/original_exp_name_var1_hash.json`), which can then be used to run new experiments with `main.py`:
```bash
python research_eval/main.py --json_config_path suggested_configs/some_suggested_config.json
```

## Developer Guide

This section provides guidance for extending the framework.

### Adding New Models

1.  **Create Model Class:**
    *   Define your new model in a Python file (e.g., `research_eval/models/my_new_model.py`).
    *   The class should ideally inherit from `research_eval.models.base_model.EvaluableModel` for API consistency, though not strictly enforced by an abstract base class in the current setup.
    *   Implement the following methods:
        *   `__init__(self, ...)`: Model constructor accepting necessary parameters.
        *   `forward(self, input_ids, attention_mask=None, ...)`: The forward pass of your model.
        *   `get_parameter_count(self) -> int`: Should return the total number of trainable parameters. The helper `get_trainable_parameter_count(self)` (defined in `main.py`) can be used.
        *   `get_model_name(self) -> str`: Should return a unique string identifier for your model type (e.g., "my_new_model"). This string is used as the key in `MODELS_MAP`.
        *   `get_auxiliary_loss(self)` (optional): If your model has auxiliary losses that should be added to the main task loss during training. Return `0.0` or `None` if not applicable.
        *   `get_model_specific_metrics(self)` (optional): Should return a dictionary of any model-specific metrics you want to log during evaluation (e.g., `{ "my_custom_metric": value }`). These will be stored in the database.
2.  **Import in `main.py`**:
    *   Add `from research_eval.models.my_new_model import MyNewModelClass` at the top of `research_eval/main.py`.
3.  **Add to `MODELS_MAP` in `main.py`**:
    *   Add an entry to the `MODELS_MAP` dictionary: `"your_model_type_string": MyNewModelClass`. The key should match the string returned by `get_model_name()`.
4.  **JSON Configuration (`model_params`)**:
    *   Define the structure for your model's parameters within the `model_params` section of the JSON configuration. These keys should directly match your model's `__init__` arguments.
    *   If your model's constructor arguments have different names than the keys you intend to use in the JSON `model_params` (e.g., for brevity or to match existing conventions like `num_experts_per_layer` vs. `num_experts`), you must handle this mapping within the `_create_model_internal` function in `research_eval/main.py`. See existing examples for MoE, GoE, and GoEOriginal.
5.  **Optuna HPO (`objective` function in `main.py`)**:
    *   To make your model and its hyperparameters optimizable by Optuna:
        *   Add an `elif model_type_val == "your_model_type_string":` block in the "Model-specific params" section of the `objective` function.
        *   Inside this block, use `trial.suggest_categorical`, `trial.suggest_int`, `trial.suggest_float` to define the search space for your model's unique hyperparameters. The keys used for `trial.suggest_*` (e.g., `trial.suggest_int("my_model_num_layers", ...)` should correspond to keys you expect in the `model_params_dict` that is passed to `_create_model_internal`.

### Adding New Datasets

1.  **Synthetic Datasets:**
    *   Create a new dataset class inheriting from `research_eval.datasets.base_dataset.SequenceBaseDataset` in a new file (e.g., `research_eval/datasets/my_synthetic_dataset.py`).
    *   Implement the `_generate_data(self)` method. This method should create `self.samples` (a list of dictionaries, each with `input_ids` and `label`) and `self.labels_list` (a list of all possible label values).
    *   The `__init__` method of your dataset class should call `super().__init__` and accept custom parameters (e.g., `num_samples`, `seq_len`, `vocab_size`, problem-specific parameters). These parameters will be passed from the `dataset_params` in the JSON config or suggested by Optuna.
    *   In `research_eval.datasets.providers.py`:
        *   Import your new dataset class.
        *   Add an entry to the `SYNTHETIC_DATASETS` dictionary: `"your_dataset_name": YourDatasetClass`.
    *   If your dataset has new HPO-tunable parameters (beyond those typically handled like `seq_len`), update the `objective` function in `research_eval/main.py` within the "Dataset params" section to include suggestions for them.
2.  **Real-World Datasets (from Hugging Face `datasets`):**
    *   In `research_eval.datasets.providers.py`:
        *   Add the dataset's Hugging Face identifier string to the `REAL_WORLD_DATASETS` list if it's not already present.
        *   In `get_real_world_data_loaders`, you might need to add dataset-specific logic for `text_field` and `label_field` if they differ from common defaults (e.g., "text", "label"). A more scalable approach would be to define these mappings in a configuration dictionary within `providers.py`.
        *   Handle any specific preprocessing or tokenization needs.
    *   If your dataset configuration requires new HPO-tunable parameters (e.g., a specific subset name or a custom preprocessing flag), update the `objective` function in `research_eval/main.py`.

**Code Comment Markers for Extension:**
*   In `research_eval/main.py`:
    *   Look for `MODELS_MAP = { ... }` to register new model classes.
    *   In the `objective` function, search for sections like `# --- Hyperparameter Suggestions ---` and `# Model-specific params` to add HPO logic for new models/datasets.
    *   In `_create_model_internal`, look for existing `if model_type == ...:` blocks to add parameter name mappings if needed for new models.
*   In `research_eval/datasets/providers.py`:
    *   Update `SYNTHETIC_DATASETS` dictionary for new synthetic datasets.
    *   Update `REAL_WORLD_DATASETS` list for new Hugging Face datasets and modify `get_real_world_data_loaders` if custom field mapping is needed.

## Contributing
We welcome contributions! Please open an issue to discuss potential changes or submit a pull request.
