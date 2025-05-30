import unittest
import tempfile
import os
import subprocess
import json
import sqlite3 # Not strictly needed for test logic but good for reference/debugging
from unittest.mock import patch

from research_eval.init_db import initialize_database
from research_eval.db_utils import log_experiment_to_db, generate_experiment_id

# DB_FILE from analyze_results will be patched.
# matplotlib.pyplot will be patched.

class TestAnalyzeResultsScript(unittest.TestCase):

    def setUp(self):
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir_path = self.temp_dir_obj.name

        self.test_db_path = os.path.join(self.temp_dir_path, "test_analysis.db")
        initialize_database(self.test_db_path)

        # Populate with sample data
        exp_configs = [
            {"experiment_name": "dense_run1", "model_type": "dense", "dataset_name": "test_dataset1"},
            {"experiment_name": "moe_run1", "model_type": "moe", "dataset_name": "test_dataset1"}
        ]
        common_exp_data = {
            "run_type": "json",
            "dataset_type": "synthetic",
            "epochs": 1,
            "batch_size": 1,
            "timestamp": "2023-01-01 00:00:00", # Added timestamp
            "env_details_json": json.dumps({"test_env": "true"}) # Added env_details
        }
        metrics_map = {
            "dense_run1": [
                {"metric_name": "final_val_f1", "metric_value": 0.8},
                {"metric_name": "param_count", "metric_value": 1000},
                {"metric_name": "inference_latency_ms_batch", "metric_value": 10.0}
            ],
            "moe_run1": [
                {"metric_name": "final_val_f1", "metric_value": 0.9},
                {"metric_name": "param_count", "metric_value": 2000},
                {"metric_name": "inference_latency_ms_batch", "metric_value": 20.0}
            ]
        }

        for config in exp_configs:
            exp_id = generate_experiment_id(config, "json")
            current_exp_data = {
                **common_exp_data,
                **config,
                "experiment_id": exp_id,
                "full_config_json": json.dumps(config)
            }
            log_experiment_to_db(current_exp_data, metrics_map[config["experiment_name"]], db_path=self.test_db_path)

        # Patch DB_FILE in research_eval.analyze_results
        self.db_file_patcher = patch('research_eval.analyze_results.DB_FILE', self.test_db_path)
        self.mock_db_file = self.db_file_patcher.start()

        # Patch matplotlib.pyplot.show
        self.plt_show_patcher = patch('matplotlib.pyplot.show')
        self.mock_plt_show = self.plt_show_patcher.start()

    def tearDown(self):
        self.db_file_patcher.stop()
        self.plt_show_patcher.stop()
        self.temp_dir_obj.cleanup()

    def test_run_analyze_results_all_outputs(self):
        output_subdir = os.path.join(self.temp_dir_path, "analysis_out")

        # Correctly locate analyze_results.py relative to this test file
        # __file__ is tests/test_analysis_script.py
        # os.path.dirname(__file__) is tests/
        # So, '..' goes to project root, then 'research_eval/analyze_results.py'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_dir, '..', 'research_eval', 'analyze_results.py')

        cmd = [
            'python', script_path,
            '--dataset_name', 'test_dataset1',
            '--output_dir', output_subdir,
            '--run_all'
        ]

        # Set environment variable for the subprocess
        env = os.environ.copy()
        env['RESEARCH_EVAL_DB_PATH'] = self.test_db_path

        result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

        print(f"analyze_results.py STDOUT:\n{result.stdout}")
        print(f"analyze_results.py STDERR:\n{result.stderr}")

        self.assertEqual(result.returncode, 0, f"analyze_results.py script failed with stderr: {result.stderr}")

        # Assert expected files exist
        # Default metric_to_optimize is 'final_val_f1'
        expected_summary_table = os.path.join(output_subdir, "test_dataset1_summary_table_best_finalvalf1.csv")
        expected_f1_vs_params_plot = os.path.join(output_subdir, "test_dataset1_f1_vs_params.png")
        expected_f1_vs_latency_plot = os.path.join(output_subdir, "test_dataset1_f1_vs_latency.png")

        self.assertTrue(os.path.exists(expected_summary_table), f"Summary table missing: {expected_summary_table}")
        self.assertTrue(os.path.exists(expected_f1_vs_params_plot), f"F1 vs Params plot missing: {expected_f1_vs_params_plot}")
        self.assertTrue(os.path.exists(expected_f1_vs_latency_plot), f"F1 vs Latency plot missing: {expected_f1_vs_latency_plot}")

if __name__ == '__main__':
    unittest.main()
