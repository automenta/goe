import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import argparse
import json
import os
import pandas as pd
import optuna
import sys
import tempfile
import shutil
import logging

# Add the parent directory to sys.path to allow importing from research_eval
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Modules to be tested
from research_eval import main as main_module 
from research_eval.models.dense_transformer import DenseTransformerClassifier
from research_eval.models.moe_transformer import MoETransformerClassifier
from research_eval.models.goe_transformer import GoEClassifier
from research_eval.models.goe_original_classifier import GoEOriginalClassifier

# Suppress most logging output during tests for cleaner test results
logging.basicConfig(level=logging.ERROR) 


class TestMain2JSONExecution(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # Point the CSV output to the temp directory to avoid clutter and ensure it's cleaned up
        self.results_csv_path = os.path.join(self.test_dir, "json_experiment_results.csv")
        main_module.csv_filename = self.results_csv_path # Override global for testing

        self.dense_synthetic_config = {
            "experiment_name": "test_dense_synthetic_basic",
            "model_type": "dense",
            "dataset_type": "synthetic",
            "dataset_name": "parity",
            "epochs": 1,
            "batch_size": 2,
            "dataset_params": {
                "num_samples": 10, "seq_len": 4, "vocab_size": 8 
            },
            "model_params": {
                "embed_dim": 8, "num_heads": 1, "dim_feedforward_factor": 2, 
                "dropout": 0.1, "num_layers": 1 
            },
            "optimizer_params": {"name": "adamw", "lr": 1e-4, "weight_decay": 0.01},
            "scheduler_params": {"name": "none"}
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        # Reset global to its original if it was changed (though not strictly necessary if tests are isolated)
        main_module.csv_filename = "json_experiment_results.csv" 


    def test_json_run_dense_synthetic_basic(self):
        """Test a basic run using JSON config for a dense model and synthetic data."""
        config_path = os.path.join(self.test_dir, "dense_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.dense_synthetic_config, f)

        # Simulate running `python research_eval/main2.py --json_config_path <path>`
        test_args = argparse.Namespace(json_config_path=config_path)
        with patch('argparse.ArgumentParser.parse_args', return_value=test_args):
            try:
                main_module.run_single_experiment_from_json(self.dense_synthetic_config)
            except Exception as e:
                self.fail(f"run_single_experiment_from_json raised an exception: {e}\nConfig: {self.dense_synthetic_config}")

        self.assertTrue(os.path.exists(self.results_csv_path), f"Results CSV {self.results_csv_path} not created.")
        df = pd.read_csv(self.results_csv_path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df['experiment_name'].iloc[0], "test_dense_synthetic_basic")
        self.assertEqual(df['model_type'].iloc[0], "dense")
        self.assertIn('metric_final_val_f1', df.columns)
        self.assertIn('param_count', df.columns)

    @patch('research_eval.main2._run_training_loop_internal')
    def test_all_models_via_json_lightweight(self, mock_run_training_loop):
        """Test instantiation and basic processing for all models via JSON (lightweight)."""
        dummy_training_results = {
            "best_val_f1_overall": 0.1, "final_val_loss": 1.0, "final_val_acc": 0.5, 
            "final_val_f1": 0.1, "inference_latency_ms_batch": 10,
            "training_duration_seconds": 1, 
            "last_epoch_model_specific_metrics": {},
            "final_eval_model_specific_metrics": {}
        }
        mock_run_training_loop.return_value = dummy_training_results

        for model_type_key in main_module.MODELS_MAP.keys():
            with self.subTest(model_type=model_type_key):
                config = {
                    "experiment_name": f"test_{model_type_key}_light",
                    "model_type": model_type_key,
                    "dataset_type": "synthetic", "dataset_name": "parity", "epochs": 1, "batch_size": 1,
                    "dataset_params": {"num_samples": 4, "seq_len": 4, "vocab_size": 8},
                    "model_params": { "embed_dim": 8, "num_heads": 1, "dim_feedforward_factor": 2, "dropout": 0.1},
                    "optimizer_params": {"name": "adamw", "lr": 1e-4}
                }
                # Add minimal model-specific params
                if model_type_key == "dense": config["model_params"]["num_layers"] = 1
                elif model_type_key == "moe":
                    config["model_params"].update({"num_layers": 1, "num_experts_per_layer": 2, "top_k_experts": 1})
                elif model_type_key == "goe":
                    config["model_params"].update({"num_goe_layers": 1, "num_total_experts_pool": 2, "max_path_len": 1, "router_hidden_dim": 4})
                elif model_type_key == "goe_original":
                    config["model_params"].update({
                        "num_total_experts": 2, "max_path_len": 1, "router_hidden_dim": 4, 
                        "expert_layers": 1, "gumbel_tau": 1.0, "path_penalty_coef": 0.0,
                        "diversity_loss_coef": 0.0, "contrastive_loss_coef": 0.0, "max_visits": 1
                    })
                
                try:
                    main_module.run_single_experiment_from_json(config)
                except Exception as e:
                    self.fail(f"run_single_experiment_from_json for {model_type_key} raised: {e}\nConfig: {json.dumps(config, indent=2)}")
                
                self.assertTrue(mock_run_training_loop.called)
                mock_run_training_loop.reset_mock()

class TestMain2OptunaHPO(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.optuna_db_path = os.path.join(self.test_dir, "test_optuna_study.db")
        self.optuna_results_csv_path = os.path.join(self.test_dir, f"optuna_study_test_hpo_study_synthetic_parity_full_results.csv")
        # Override global CSV name for Optuna results for predictable cleanup
        main_module.optuna_study_csv_filename_template = os.path.join(self.test_dir, "optuna_study_{}_{}_{}_full_results.csv")


    def tearDown(self):
        shutil.rmtree(self.test_dir)
        # General cleanup in case specific files were missed
        if os.path.exists("optuna_study_test_hpo_study_synthetic_parity_full_results.csv"):
            os.remove("optuna_study_test_hpo_study_synthetic_parity_full_results.csv")

    @patch('research_eval.main2._run_training_loop_internal')
    def test_optuna_hpo_basic_run(self, mock_run_training_loop):
        dummy_training_results = {
            "best_val_f1_overall": 0.1, "final_val_loss": 1.0, "final_val_acc": 0.5, 
            "final_val_f1": 0.1, "inference_latency_ms_batch": 10,
            "training_duration_seconds": 1, "last_epoch_model_specific_metrics": {},
            "final_eval_model_specific_metrics": {}
        }
        mock_run_training_loop.return_value = dummy_training_results

        fixed_args = argparse.Namespace(
            json_config_path=None, study_name="test_hpo_study", storage=f"sqlite:///{self.optuna_db_path}",
            n_trials=1, epochs=1, batch_size=2, dataset_type="synthetic", dataset_name="parity",
            syn_num_samples=10, syn_seq_len=None, syn_vocab_size=None, # Let HPO suggest some
            rw_max_length=None, rw_train_samples=None, rw_val_samples=None
        )
        
        study = optuna.create_study(study_name=fixed_args.study_name, storage=fixed_args.storage, load_if_exists=True, directions=["maximize", "minimize", "minimize"])
        study.optimize(lambda trial: main_module.objective(trial, fixed_args), n_trials=fixed_args.n_trials, gc_after_trial=True)

        self.assertEqual(len(study.trials), 1)
        self.assertEqual(study.trials[0].state, optuna.trial.TrialState.COMPLETE)
        self.assertTrue(mock_run_training_loop.called)
        
        # Check CSV saving (this part relies on the main script's structure for saving)
        # For now, we check if the study object has the user_attr, as CSV part is in __main__
        self.assertIn("full_results", study.trials[0].user_attrs)


    @patch('research_eval.main2._create_model_internal') 
    @patch('research_eval.main2._setup_dataset_internal') # Mock dataset setup too
    def test_optuna_param_count_pruning(self, mock_setup_dataset, mock_create_model):
        mock_model_large = MagicMock(spec=DenseTransformerClassifier) # Use spec for hasattr checks
        mock_model_large.get_model_name.return_value = "mock_large_model"
        mock_model_large.get_auxiliary_loss.return_value = 0.0
        mock_model_large.get_model_specific_metrics.return_value = {}
        mock_create_model.return_value = mock_model_large
        
        # Mock dataset loader to return minimal valid values
        mock_train_loader = [1] # Dummy, just to have len()
        mock_val_loader = [1]
        mock_setup_dataset.return_value = (mock_train_loader, mock_val_loader, 10, 2, 0)


        with patch('research_eval.main2.get_trainable_parameter_count', return_value=main_module.MAX_PARAMS_SYNTHETIC + 1000):
            study = optuna.create_study(directions=["maximize", "minimize", "minimize"])
            fixed_args = argparse.Namespace(
                dataset_type="synthetic", dataset_name="parity", epochs=1, batch_size=2,
                syn_num_samples=10, syn_seq_len=None, syn_vocab_size=None, # Let HPO suggest
                rw_max_length=None, rw_train_samples=None, rw_val_samples=None
            )
            
            # We want to ensure that the trial.suggest_* calls within objective lead to
            # the creation of a model that *would* be too large.
            # The actual pruning happens based on the mocked get_trainable_parameter_count.
            
            # This test now checks the return values of objective for pruning indicators
            # as TrialPruned is caught internally in the objective function.
            trial = study.ask({
                "model_type": "dense", # Fixed for this test to control param suggestion
                "embed_dim": 2048, # Large embed_dim
                "num_heads": 8,    # Ensure divisibility
                "dim_feedforward_factor": 4,
                "dropout": 0.1,
                "dt_num_layers": 4 # Max layers for dense
            })

            f1, params, latency = main2.objective(trial, fixed_args)
            
            self.assertEqual(f1, 0.0) # Pruning returns 0.0 for the main metric
            self.assertEqual(params, main2.MAX_PARAMS_SYNTHETIC + 1000) # The mocked large param count
            self.assertEqual(latency, 1e9) # High latency indicates pruning due to params


if __name__ == '__main__':
    unittest.main()
