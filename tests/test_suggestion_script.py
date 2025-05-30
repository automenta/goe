import unittest
import tempfile
import os
import subprocess
import json
import copy # Added import for deepcopy
# No direct sqlite3 needed for this test if we trust db_utils and init_db
# from unittest.mock import patch # Not patching DB_FILE directly in script, using ENV VAR

from research_eval.init_db import initialize_database
from research_eval.db_utils import log_experiment_to_db, generate_experiment_id

class TestSuggestConfigsScript(unittest.TestCase):

    def setUp(self):
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir_path = self.temp_dir_obj.name

        self.test_db_path = os.path.join(self.temp_dir_path, "test_suggestion.db")
        initialize_database(self.test_db_path)

        # Populate with sample data
        self.base_config = { # Made it an instance attribute
            "experiment_name": "dense_base_run",
            "model_type": "dense",
            "dataset_name": "sugg_dataset",
            "model_params": {
                "embed_dim": 32,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.1,
                "dim_feedforward_factor": 4
            },
            "optimizer_params": {"name": "adamw", "lr": 1e-4, "weight_decay": 0.01}, # Added optimizer name for completeness
            "dataset_params": {"seq_len": 64},
            # Added scheduler_params for more perturbation targets
            "scheduler_params": {"name": "cosine_warmup", "warmup_factor": 0.1, "num_training_steps_factor": 1.0}
        }

        exp_id = generate_experiment_id(self.base_config, "json") # Use self.base_config

        experiment_data = {
            "experiment_id": exp_id,
            "run_type": "json",
            "dataset_name": "sugg_dataset",
            "model_type": "dense",
            "experiment_name": "dense_base_run",
            "full_config_json": json.dumps(self.base_config), # Use self.base_config
            "timestamp": "2023-01-01 00:00:00",
            "epochs": 10, # Increased epochs for realism
            "batch_size": 32, # Increased batch_size for realism
            "dataset_type": "synthetic",
            "env_details_json": json.dumps({"test_env": "suggestion_test"})
        }

        metrics_data = [{"metric_name": "final_val_f1", "metric_value": 0.95}]
        log_experiment_to_db(experiment_data, metrics_data, db_path=self.test_db_path)

        # No direct patching of DB_FILE in suggest_configs.py, will use environment variable for subprocess

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_run_suggest_configs_generates_files(self):
        output_subdir = os.path.join(self.temp_dir_path, "suggested_out")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_dir, '..', 'research_eval', 'suggest_configs.py')

        cmd = [
            'python', script_path,
            '--dataset_name', 'sugg_dataset',
            '--model_type', 'dense',
            '--output_dir', output_subdir,
            '--top_n', '1', # Select the one base config we inserted
            '--num_variations_per_config', '2', # Generate 2 new configs
            '--metric_name', 'final_val_f1',
            '--perturbation_magnitude', '0.05'
        ]

        env = os.environ.copy()
        env['RESEARCH_EVAL_DB_PATH'] = self.test_db_path

        result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

        print(f"suggest_configs.py STDOUT:\n{result.stdout}")
        print(f"suggest_configs.py STDERR:\n{result.stderr}")

        self.assertEqual(result.returncode, 0, f"suggest_configs.py script failed with stderr: {result.stderr}")

        self.assertTrue(os.path.isdir(output_subdir), f"Output directory missing: {output_subdir}")

        generated_files = [f for f in os.listdir(output_subdir) if f.endswith('.json')]
        self.assertEqual(len(generated_files), 2, f"Expected 2 new config files, found {len(generated_files)}: {generated_files}")

        if len(generated_files) > 0:
            with open(os.path.join(output_subdir, generated_files[0]), 'r') as f:
                content = json.load(f)
            self.assertIn("experiment_name", content)
            self.assertTrue(content["experiment_name"].startswith(self.base_config["experiment_name"] + "_var"))
            self.assertIn("model_params", content)
            self.assertIn("optimizer_params", content)
            self.assertIn("dataset_params", content)
            self.assertIn("scheduler_params", content) # Check if new section is there

            # Check if at least one numeric param was perturbed (not equal to base)
            # This is a basic check, assumes at least one param is meaningfully perturbed.
            perturbed = False
            if content["model_params"]["embed_dim"] != self.base_config["model_params"]["embed_dim"]: perturbed = True
            if content["optimizer_params"]["lr"] != self.base_config["optimizer_params"]["lr"]: perturbed = True
            # Add more checks if necessary for other params

            # A simple way to check for perturbation is if the new config is different from base,
            # ignoring the experiment_name and original perf markers.
            content_copy = copy.deepcopy(content)
            content_copy.pop("experiment_name", None)
            content_copy.pop("_original_experiment_id", None) # In case the script adds this
            content_copy.pop("_original_metric_value", None)

            base_config_copy = copy.deepcopy(self.base_config) # Use self.base_config
            base_config_copy.pop("experiment_name", None)

            # This direct comparison might be too strict if script adds other default keys.
            # A more robust check would compare known perturbed fields.
            # For now, a basic check that something numerical changed.
            # Example: check if learning rate is different.
            self.assertNotEqual(content["optimizer_params"]["lr"], self.base_config["optimizer_params"]["lr"],
                                "Expected learning rate to be perturbed or at least one param.")
            # The above assertNotEqual might fail if perturbation by chance results in same value.
            # A better check is to ensure the experiment_name hash is different or that it's a new file.
            # The file count check (2 files) and naming convention already imply new variations.

if __name__ == '__main__':
    # import copy # No longer needed here as it's at the top
    unittest.main()
