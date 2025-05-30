import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import math # For log in entropy test

# Functions to test from research_eval.utils
from research_eval.utils import (
    get_optimizer,
    get_scheduler,
    gumbel_softmax,
    calculate_entropy,
    calculate_cv
)
# get_cosine_schedule_with_warmup is imported dynamically by get_scheduler,
# but we might need to import transformers if its type hint is used by LambdaLR or for reference.
# For now, we'll rely on the dynamic import. If transformers is missing, get_scheduler will fail.

torch.manual_seed(0)

class TestUtils(unittest.TestCase):

    def test_get_optimizer(self):
        params = [nn.Parameter(torch.randn(2, 2))]

        optimizer_adamw = get_optimizer(params, optimizer_name="adamw", lr=1e-3, weight_decay=0.01)
        self.assertIsInstance(optimizer_adamw, optim.AdamW)
        self.assertEqual(optimizer_adamw.defaults['lr'], 1e-3)
        self.assertEqual(optimizer_adamw.defaults['weight_decay'], 0.01)

        optimizer_sgd = get_optimizer(params, optimizer_name="sgd", lr=1e-2, weight_decay=0.001)
        self.assertIsInstance(optimizer_sgd, optim.SGD)
        self.assertEqual(optimizer_sgd.defaults['lr'], 1e-2)
        self.assertEqual(optimizer_sgd.defaults['weight_decay'], 0.001)

        with self.assertRaises(ValueError):
            get_optimizer(params, optimizer_name="unknown_opt", lr=1e-3)

    def test_get_scheduler(self):
        dummy_model_param = nn.Parameter(torch.randn(2,2))
        optimizer = optim.SGD([dummy_model_param], lr=0.1)

        # Test cosine_warmup scheduler
        scheduler_cosine = get_scheduler(optimizer, scheduler_name="cosine_warmup", num_warmup_steps=10, num_training_steps=100)
        # The actual type is transformers.optimization.LambdaLR,
        # so just checking it's not None is a basic check.
        # A more specific check would require importing LambdaLR from transformers.
        self.assertIsNotNone(scheduler_cosine)

        # Test "none" scheduler
        scheduler_none = get_scheduler(optimizer, scheduler_name="none", num_warmup_steps=10, num_training_steps=100)
        self.assertIsNone(scheduler_none)

        scheduler_none_explicit = get_scheduler(optimizer, scheduler_name=None, num_warmup_steps=10, num_training_steps=100)
        self.assertIsNone(scheduler_none_explicit)


        with self.assertRaises(ValueError):
            get_scheduler(optimizer, scheduler_name="unknown_scheduler", num_warmup_steps=10, num_training_steps=100)

    def test_gumbel_softmax(self):
        torch.manual_seed(0) # Reset seed for this specific test
        logits = torch.randn(2, 3) # batch_size=2, num_classes=3

        # Test soft Gumbel-Softmax
        y_soft = gumbel_softmax(logits, tau=1.0, hard=False)
        self.assertEqual(y_soft.shape, (2, 3))
        self.assertTrue(torch.allclose(y_soft.sum(dim=-1), torch.tensor([1.0, 1.0])))

        # Test hard Gumbel-Softmax
        torch.manual_seed(0) # Reset seed again for comparable hard output if needed, or ensure different noise
        y_hard = gumbel_softmax(logits, tau=1.0, hard=True)
        self.assertEqual(y_hard.shape, (2, 3))
        self.assertTrue(torch.allclose(y_hard.sum(dim=-1), torch.tensor([1.0, 1.0])))
        # Check for one-hot property (exactly one 1.0 per row)
        is_one_hot = torch.all((y_hard == 0) | (y_hard == 1), dim=-1) & \
                     torch.all(torch.sum(y_hard, dim=-1) == 1)
        self.assertTrue(torch.all(is_one_hot))


    def test_calculate_entropy(self):
        torch.manual_seed(0)

        # Test with a uniform distribution
        probs_uniform = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        expected_entropy_uniform = -1 * (0.25 * math.log(0.25)) * 4 # log base e
        # calculate_entropy returns mean over batch, so for single item batch, it's just the item's entropy
        entropy_uniform = calculate_entropy(probs_uniform)
        self.assertTrue(torch.isclose(entropy_uniform, torch.tensor(expected_entropy_uniform)))

        # Test with a peaked distribution
        probs_peaked = torch.tensor([[0.7, 0.1, 0.1, 0.1]])
        # Sum of p*log(p) for peaked:
        expected_entropy_peaked = - (0.7 * math.log(0.7) + 3 * (0.1 * math.log(0.1)))
        entropy_peaked = calculate_entropy(probs_peaked)
        self.assertTrue(torch.isclose(entropy_peaked, torch.tensor(expected_entropy_peaked)))

        self.assertLess(entropy_peaked.item(), entropy_uniform.item())

    def test_calculate_cv(self):
        # Test with zero variation
        data_zero_cv = torch.tensor([5.0, 5.0, 5.0])
        expected_cv_zero = torch.tensor(0.0)
        cv_zero = calculate_cv(data_zero_cv)
        self.assertTrue(torch.isclose(cv_zero, expected_cv_zero))

        # Test with some variation
        data_some_cv = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        # Mean = 3.0
        # Sample Std = sqrt( ( (1-3)^2 + ... + (5-3)^2 ) / (5-1) ) = sqrt(10/4) = sqrt(2.5)
        # CV = sqrt(2.5) / 3.0
        expected_cv_some = math.sqrt(2.5) / 3.0
        cv_some = calculate_cv(data_some_cv)
        self.assertTrue(torch.isclose(cv_some, torch.tensor(expected_cv_some)))
        self.assertGreater(cv_some.item(), 0.0)

        # Test with zero mean (and zero std)
        data_zero_mean_zero_std = torch.tensor([0.0, 0.0, 0.0])
        expected_cv_zero_mean = torch.tensor(0.0) # As per implementation
        cv_zero_mean = calculate_cv(data_zero_mean_zero_std)
        self.assertTrue(torch.isclose(cv_zero_mean, expected_cv_zero_mean))

        # Test with zero mean but non-zero std (should not happen with real numbers, but for robustness)
        # The implementation has 1e-10 in denominator, so it won't be undefined.
        # If mean is exactly 0, it returns 0.0.
        data_zero_mean_some_std = torch.tensor([-1.0, 0.0, 1.0]) # Mean is 0
        expected_cv_zero_mean_some_std = torch.tensor(0.0)
        cv_zero_mean_some_std = calculate_cv(data_zero_mean_some_std)
        self.assertTrue(torch.isclose(cv_zero_mean_some_std, expected_cv_zero_mean_some_std))


if __name__ == "__main__":
    unittest.main()
