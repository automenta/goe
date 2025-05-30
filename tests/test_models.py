import unittest
import torch

# Assuming DenseTransformerClassifier and EvaluableModel are in research_eval.models
from research_eval.models.dense_transformer import DenseTransformerClassifier
from research_eval.models.base_model import EvaluableModel
from research_eval.models.moe_transformer import MoETransformerClassifier
from research_eval.models.goe_transformer import GoEClassifier
from research_eval.models.goe_original_classifier import GoEOriginalClassifier # Added import

torch.manual_seed(0)

class TestDenseTransformerClassifier(unittest.TestCase):
    def test_model_construction(self):
        config = {
            "embed_dim": 16,
            "vocab_size": 100,
            "num_layers": 1,
            "num_heads": 1,
            "dim_feedforward": 32,
            "dropout": 0.1,
            "num_classes": 2,
        }
        model = DenseTransformerClassifier(**config).to('cpu')

        self.assertIsInstance(model, DenseTransformerClassifier)
        self.assertIsInstance(model, EvaluableModel)

        self.assertEqual(model.get_model_name(), "DenseTransformer")
        self.assertGreater(model.get_parameter_count(), 0)
        self.assertGreater(model.get_trainable_parameter_count(), 0)
        self.assertIsInstance(model.get_parameter_count(), int)
        self.assertIsInstance(model.get_trainable_parameter_count(), int)


    def test_forward_pass_classification(self):
        config = {
            "vocab_size": 10,
            "embed_dim": 8,
            "num_heads": 1,
            "num_layers": 1,
            "dim_feedforward": 16,
            "num_classes": 3,
            "dropout": 0.1,
        }
        model = DenseTransformerClassifier(**config).to('cpu')

        batch_size = 2
        # seq_len is determined by the input data, not a model config
        input_seq_len = 5
        vocab_size = 10
        num_classes = 3

        input_ids = torch.randint(0, vocab_size, (batch_size, input_seq_len)).to('cpu')
        attention_mask = torch.ones(batch_size, input_seq_len, dtype=torch.int64).to('cpu')

        logits = model(input_ids, attention_mask=attention_mask)

        self.assertEqual(logits.shape, (batch_size, num_classes))
        self.assertEqual(model.get_auxiliary_loss(), 0.0)

        model_specific_metrics = model.get_model_specific_metrics()
        self.assertIsInstance(model_specific_metrics, dict)


class TestMoETransformerClassifier(unittest.TestCase):
    def test_model_construction(self):
        config = {
            "vocab_size": 10,
            "embed_dim": 8,
            "num_heads": 1,
            "num_layers": 1,
            "dim_feedforward": 12,
            "num_classes": 3,
            "num_experts": 2,
            "top_k_experts": 1,
        }
        model = MoETransformerClassifier(**config).to('cpu')

        self.assertIsInstance(model, MoETransformerClassifier)
        self.assertIsInstance(model, EvaluableModel)

        self.assertEqual(model.get_model_name(), "MoETransformer")
        self.assertGreater(model.get_parameter_count(), 0)
        self.assertGreater(model.get_trainable_parameter_count(), 0)
        self.assertIsInstance(model.get_parameter_count(), int)
        self.assertIsInstance(model.get_trainable_parameter_count(), int)

    def test_forward_pass_classification(self):
        config = {
            "vocab_size": 10,
            "embed_dim": 8,
            "num_heads": 1,
            "num_layers": 1,
            "dim_feedforward": 12,
            "num_classes": 3,
            "num_experts": 2,
            "top_k_experts": 1,
        }
        model = MoETransformerClassifier(**config).to('cpu')

        batch_size = 2
        input_seq_len = 5
        vocab_size = config["vocab_size"]
        num_classes = config["num_classes"]

        input_ids = torch.randint(0, vocab_size, (batch_size, input_seq_len)).to('cpu')
        attention_mask = torch.ones(batch_size, input_seq_len, dtype=torch.int64).to('cpu')

        logits = model(input_ids, attention_mask=attention_mask)

        self.assertEqual(logits.shape, (batch_size, num_classes))

        aux_loss = model.get_auxiliary_loss()
        self.assertIsInstance(aux_loss.item(), float)

        model_specific_metrics = model.get_model_specific_metrics()
        self.assertIsInstance(model_specific_metrics, dict)
        if config["num_layers"] > 0 and config["num_experts"] > 0 :
            self.assertTrue(any(k.startswith("moe_layer_") and k.endswith("_expert_load_cv") for k in model_specific_metrics.keys()))

class TestGoEClassifier(unittest.TestCase):
    def test_model_construction(self):
        config = {
            "vocab_size": 10,
            "embed_dim": 8,
            "num_heads": 1,
            "dim_feedforward": 16,
            "num_classes": 3,
            "num_total_experts": 2,
            "max_path_len": 1,
            "router_hidden_dim": 4,
            "dropout": 0.1,
        }
        model = GoEClassifier(**config).to('cpu')

        self.assertIsInstance(model, GoEClassifier)
        self.assertIsInstance(model, EvaluableModel)

        self.assertEqual(model.get_model_name(), "GoEClassifier")
        self.assertGreater(model.get_parameter_count(), 0)
        self.assertGreater(model.get_trainable_parameter_count(), 0)
        self.assertIsInstance(model.get_parameter_count(), int)
        self.assertIsInstance(model.get_trainable_parameter_count(), int)

    def test_forward_pass_classification(self):
        config = {
            "vocab_size": 10,
            "embed_dim": 8,
            "num_heads": 1,
            "dim_feedforward": 16,
            "num_classes": 3,
            "num_total_experts": 2,
            "max_path_len": 1,
            "router_hidden_dim": 4,
            "dropout": 0.1,
        }
        model = GoEClassifier(**config).to('cpu')

        batch_size = 2
        input_seq_len = 5
        vocab_size = config["vocab_size"]
        num_classes = config["num_classes"]

        input_ids = torch.randint(0, vocab_size, (batch_size, input_seq_len)).to('cpu')
        attention_mask = torch.ones(batch_size, input_seq_len, dtype=torch.int64).to('cpu')

        logits = model(input_ids, attention_mask=attention_mask)

        self.assertEqual(logits.shape, (batch_size, num_classes))

        aux_loss = model.get_auxiliary_loss()
        self.assertIsInstance(aux_loss.item(), float)

        model_specific_metrics = model.get_model_specific_metrics()
        self.assertIsInstance(model_specific_metrics, dict)
        self.assertIn('goe_avg_router_entropy', model_specific_metrics)
        if config["max_path_len"] > 0:
            self.assertTrue(any(k.startswith("goe_step_") and k.endswith("_expert_util_cv") for k in model_specific_metrics.keys()))

class TestGoEOriginalClassifier(unittest.TestCase):
    def test_model_construction(self):
        config = {
            "vocab_size": 10,
            "embed_dim": 8,
            "num_heads": 1,
            "dim_feedforward": 8 * 2, # embed_dim * dim_feedforward_factor
            "num_classes": 3,
            "num_total_experts": 2,
            "max_path_len": 1,
            "router_hidden_dim": 4,
            "expert_layers": 1,
            "gumbel_tau": 1.0,
            "path_penalty_coef": 0.01,
            "diversity_loss_coef": 0.01,
            "contrastive_loss_coef": 0.01,
            "max_visits_per_expert": 1, # Corrected from max_visits
            "dropout": 0.1, # Added dropout as it's a constructor arg
        }
        model = GoEOriginalClassifier(**config).to('cpu')

        self.assertIsInstance(model, GoEOriginalClassifier)
        self.assertIsInstance(model, EvaluableModel)

        self.assertEqual(model.get_model_name(), "GoEOriginalClassifier")
        self.assertGreater(model.get_parameter_count(), 0)
        self.assertGreater(model.get_trainable_parameter_count(), 0)
        self.assertIsInstance(model.get_parameter_count(), int)
        self.assertIsInstance(model.get_trainable_parameter_count(), int)

    def test_forward_pass_classification(self):
        config = {
            "vocab_size": 10,
            "embed_dim": 8,
            "num_heads": 1,
            "dim_feedforward": 8 * 2, # embed_dim * dim_feedforward_factor
            "num_classes": 3,
            "num_total_experts": 2,
            "max_path_len": 1,
            "router_hidden_dim": 4,
            "expert_layers": 1,
            "gumbel_tau": 1.0,
            "path_penalty_coef": 0.01,
            "diversity_loss_coef": 0.01,
            "contrastive_loss_coef": 0.01,
            "max_visits_per_expert": 1, # Corrected from max_visits
            "dropout": 0.1, # Added dropout
        }
        model = GoEOriginalClassifier(**config).to('cpu')

        batch_size = 2
        input_seq_len = 5
        vocab_size = config["vocab_size"]
        num_classes = config["num_classes"]

        input_ids = torch.randint(0, vocab_size, (batch_size, input_seq_len)).to('cpu')
        attention_mask = torch.ones(batch_size, input_seq_len, dtype=torch.int64).to('cpu')

        logits = model(input_ids, attention_mask=attention_mask)

        self.assertEqual(logits.shape, (batch_size, num_classes))

        aux_loss = model.get_auxiliary_loss()
        self.assertIsInstance(aux_loss.item(), float)

        model_specific_metrics = model.get_model_specific_metrics()
        self.assertIsInstance(model_specific_metrics, dict)
        # Check for expected metrics based on GoEOriginalClassifier.get_model_specific_metrics
        self.assertIn('goe_original_avg_router_entropy', model_specific_metrics)
        self.assertIn('goe_original_avg_path_length', model_specific_metrics)
        self.assertIn('goe_original_expert_usage_cv', model_specific_metrics)
        self.assertIn('goe_original_expert_usage_counts', model_specific_metrics)
        self.assertIsInstance(model_specific_metrics['goe_original_expert_usage_counts'], list)


if __name__ == "__main__":
    unittest.main()
