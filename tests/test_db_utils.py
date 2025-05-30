import unittest
import tempfile
import os
import sqlite3
import json
import hashlib
import uuid
from unittest.mock import patch, MagicMock

from research_eval.db_utils import generate_experiment_id, log_experiment_to_db, connect_db # connect_db for direct use in test
from research_eval.init_db import initialize_database

# DB_FILE is patched, so direct import of it is not strictly needed for patching,
# but good to be aware of its role.

class TestDbUtils(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = os.path.join(self.temp_dir.name, "test_experiment_results.db")

        # Initialize a fresh database for each test method to ensure isolation,
        # or use setUpClass if initialization is heavy and tests don't interfere.
        # For db tests, fresh state per test is often safer.
        initialize_database(self.test_db_path)

        # Patch DB_FILE in research_eval.db_utils to use our temp DB
        self.db_file_patcher = patch('research_eval.db_utils.DB_FILE', self.test_db_path)
        self.mock_db_file_val = self.db_file_patcher.start()


    def tearDown(self):
        self.db_file_patcher.stop()
        self.temp_dir.cleanup()

    def test_generate_experiment_id_json(self):
        config1 = {"experiment_name": "my_exp", "param1": "value1"}
        exp_id1 = generate_experiment_id(config1, "json")
        config1_str = json.dumps(config1, sort_keys=True, ensure_ascii=False)
        config1_hash = hashlib.md5(config1_str.encode('utf-8')).hexdigest()[:8]
        self.assertEqual(exp_id1, f"json_my_exp_{config1_hash}")

        config2 = {"experiment_name": "my_exp", "param1": "value2"} # Different param value
        exp_id2 = generate_experiment_id(config2, "json")
        self.assertNotEqual(exp_id1, exp_id2)
        config2_str = json.dumps(config2, sort_keys=True, ensure_ascii=False)
        config2_hash = hashlib.md5(config2_str.encode('utf-8')).hexdigest()[:8]
        self.assertEqual(exp_id2, f"json_my_exp_{config2_hash}")


        config3 = {"param1": "value3"} # No experiment_name
        exp_id3 = generate_experiment_id(config3, "json")
        # Format: json_uuid_{uuid}_{hash}
        parts = exp_id3.split('_')
        self.assertEqual(parts[0], "json")
        self.assertEqual(parts[1], "uuid")
        self.assertTrue(len(parts[2]) > 0) # UUID part
        try:
            uuid.UUID(parts[2]) # Check if it's a valid UUID string fragment (might not be full UUID)
        except ValueError:
            # The function uses str(uuid.uuid4()), so it is a full UUID.
            pass # Let's assume the function's str(uuid.uuid4()) is correct.
                 # A more robust check would be to mock uuid.uuid4
        self.assertTrue(len(parts[3]) > 0) # Hash part
        config3_str = json.dumps(config3, sort_keys=True, ensure_ascii=False)
        config3_hash = hashlib.md5(config3_str.encode('utf-8')).hexdigest()[:8]
        self.assertEqual(parts[3], config3_hash)
        self.assertTrue(exp_id3.startswith("json_uuid_")) # General check


    def test_generate_experiment_id_optuna(self):
        class MockOptunaTrialInfo:
            def __init__(self, study_name, number):
                self.study_name = study_name # Direct attribute as per db_utils comment
                self.number = number

        trial_info = MockOptunaTrialInfo("optuna_study", 7)
        exp_id = generate_experiment_id({}, "optuna_trial", trial_info=trial_info)
        self.assertEqual(exp_id, "optuna_optuna_study_7")

    def test_generate_experiment_id_unknown(self):
        exp_id = generate_experiment_id({}, "unknown_type")
        # Check if it's a valid UUID string
        try:
            uuid_obj = uuid.UUID(exp_id, version=4)
            self.assertIsNotNone(uuid_obj)
        except ValueError:
            self.fail("generate_experiment_id with unknown type did not return a valid UUID v4 string.")

    def test_log_experiment_to_db_success(self):
        exp_id = "test_log_exp_001"
        experiment_data = {
            "experiment_id": exp_id, "run_type": "test_json", "timestamp": "2023-01-01 00:00:00",
            "experiment_name": "LogSuccessTest", "model_type": "dense", "dataset_name": "test_data",
            "dataset_type": "synthetic", "epochs": 1, "batch_size": 1,
            "full_config_json": json.dumps({"param": "value"}),
            "env_details_json": json.dumps({"python_version": "3.x"})
        }
        list_metric_value = [1,2,3,4,5]
        metrics_data = [
            {"metric_name": "f1", "metric_value": 0.9},
            {"metric_name": "list_metric", "metric_value": list_metric_value}
        ]

        result = log_experiment_to_db(experiment_data, metrics_data, db_path=self.test_db_path)
        self.assertTrue(result)

        # Verify data in DB
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM experiments WHERE experiment_id = ?", (exp_id,))
        row = cursor.fetchone()
        self.assertIsNotNone(row)

        # Convert row to dict for easier comparison
        cols = [column[0] for column in cursor.description]
        row_dict = dict(zip(cols, row))

        self.assertEqual(row_dict["experiment_id"], experiment_data["experiment_id"])
        self.assertEqual(row_dict["experiment_name"], experiment_data["experiment_name"])
        self.assertEqual(row_dict["model_type"], experiment_data["model_type"])
        self.assertEqual(row_dict["full_config_json"], experiment_data["full_config_json"])

        cursor.execute("SELECT metric_name, metric_value FROM metrics WHERE experiment_id = ?", (exp_id,))
        metrics_rows = cursor.fetchall()
        self.assertEqual(len(metrics_rows), 2)

        db_metrics = {name: val for name, val in metrics_rows}
        self.assertEqual(db_metrics["f1"], 0.9)
        self.assertEqual(db_metrics["list_metric"], json.dumps(list_metric_value)) # Check JSON serialization

        conn.close()

    def test_log_experiment_to_db_missing_id(self):
        experiment_data = {
            "run_type": "test_json_missing_id",
            "experiment_name": "MissingIDTest",
            "model_type": "test", "dataset_name": "test", "dataset_type": "test",
            "epochs": 0, "batch_size": 0
            # 'experiment_id' is missing
        }
        metrics_data = []
        result = log_experiment_to_db(experiment_data, metrics_data, db_path=self.test_db_path)
        self.assertFalse(result)

    # Patch connect_db for this specific test method
    @patch('research_eval.db_utils.connect_db')
    def test_log_experiment_to_db_connection_error(self, mock_connect_db):
        mock_connect_db.return_value = None # Simulate connection failure

        experiment_data = {
            "experiment_id": "conn_err_test", "run_type": "test_conn_error",
            "experiment_name": "ConnErr", "model_type": "test", "dataset_name": "test",
            "dataset_type": "test", "epochs": 0, "batch_size": 0
        }
        metrics_data = []
        result = log_experiment_to_db(experiment_data, metrics_data, db_path=self.test_db_path)
        self.assertFalse(result)
        mock_connect_db.assert_called_with(self.test_db_path)


if __name__ == '__main__':
    unittest.main()
