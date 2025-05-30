import unittest
import torch
from torch.utils.data import Dataset, DataLoader
from unittest.mock import patch, MagicMock

# Import specific dataset classes from providers.py
from research_eval.datasets.providers import SequenceElementParityDataset, SequenceCopyingDataset, get_real_world_data_loaders
import datasets as hf_datasets
from transformers import AutoTokenizer

torch.manual_seed(0)

class TestSyntheticDatasetProviders(unittest.TestCase):
    def test_synthetic_parity_dataset_direct_init(self):
        num_samples = 10
        seq_len = 4
        vocab_size_param = 3
        dataset = SequenceElementParityDataset(
            num_samples=num_samples,
            seq_len=seq_len,
            vocab_size=vocab_size_param
        )

        self.assertIsNotNone(dataset)
        self.assertIsInstance(dataset, torch.utils.data.Dataset)
        self.assertEqual(len(dataset), num_samples)

        sample = dataset[0]
        self.assertIsInstance(sample, dict)
        self.assertIn('input_ids', sample)
        self.assertIn('labels', sample)

        self.assertIsInstance(sample['input_ids'], torch.Tensor)
        self.assertEqual(sample['input_ids'].shape, (seq_len,))

        self.assertTrue(torch.all(sample['input_ids'] >= 1))
        self.assertTrue(torch.all(sample['input_ids'] <= vocab_size_param))

        self.assertIsInstance(sample['labels'].item(), int)
        self.assertIn(sample['labels'].item(), [0, 1])

        self.assertEqual(dataset.vocab_size, vocab_size_param)
        self.assertEqual(dataset.seq_len, seq_len)
        self.assertEqual(dataset.num_classes, 2)

    def test_synthetic_copying_dataset_direct_init(self):
        num_samples = 10
        seq_len = 6
        vocab_size_param = 5
        pattern_len_param = 3

        dataset = SequenceCopyingDataset(
            num_samples=num_samples,
            seq_len=seq_len,
            vocab_size=vocab_size_param,
            pattern_len=pattern_len_param
        )

        self.assertIsNotNone(dataset)
        self.assertIsInstance(dataset, torch.utils.data.Dataset)
        self.assertEqual(len(dataset), num_samples)

        sample = dataset[0]
        self.assertIsInstance(sample, dict)
        self.assertIn('input_ids', sample)
        self.assertIn('labels', sample)

        self.assertIsInstance(sample['input_ids'], torch.Tensor)
        self.assertEqual(sample['input_ids'].shape, (seq_len,))

        self.assertTrue(torch.all(sample['input_ids'] >= 1))
        self.assertTrue(torch.all(sample['input_ids'] <= vocab_size_param))

        self.assertIsInstance(sample['labels'].item(), int)
        self.assertIn(sample['labels'].item(), [0, 1])

        self.assertEqual(dataset.vocab_size, vocab_size_param)
        self.assertEqual(dataset.seq_len, seq_len)
        self.assertEqual(dataset.num_classes, 2)
        self.assertEqual(dataset.pattern_len, pattern_len_param)
        self.assertIsNotNone(dataset.pattern_to_copy)
        self.assertEqual(dataset.pattern_to_copy.shape, (pattern_len_param,))


class TestRealWorldDatasetProviderActualSignature(unittest.TestCase):
    mock_load_dataset_patcher = None
    mock_hf_load_dataset = None
    mock_tokenizer_patcher = None
    mock_hf_tokenizer = None

    @classmethod
    def setUpClass(cls):
        # Patch 'load_dataset' and 'AutoTokenizer.from_pretrained'
        # where they are looked up by the providers module.
        cls.mock_load_dataset_patcher = patch('research_eval.datasets.providers.load_dataset')
        cls.mock_hf_load_dataset = cls.mock_load_dataset_patcher.start()

        cls.mock_tokenizer_patcher = patch('research_eval.datasets.providers.AutoTokenizer.from_pretrained')
        cls.mock_hf_tokenizer = cls.mock_tokenizer_patcher.start()

    @classmethod
    def tearDownClass(cls):
        if cls.mock_load_dataset_patcher:
            cls.mock_load_dataset_patcher.stop()
        if cls.mock_tokenizer_patcher:
            cls.mock_tokenizer_patcher.stop()

    def test_get_real_world_data_loaders_agnews(self):
        # Define mock Hugging Face DatasetDict for 'ag_news'
        # Note: The features must match what the actual ag_news dataset would have for num_classes.
        mock_features = hf_datasets.Features({
            'text': hf_datasets.Value(dtype='string', id=None),
            'label': hf_datasets.ClassLabel(num_classes=4, names=['World', 'Sports', 'Business', 'Sci/Tech'])
        })
        # Using more samples initially in the mock Dataset to allow .select to work as expected.
        mock_train_ds = hf_datasets.Dataset.from_dict({'text': ['ag news text 1']*20, 'label': [0]*20}, features=mock_features)
        mock_val_ds = hf_datasets.Dataset.from_dict({'text': ['ag news text 2']*20, 'label': [1]*20}, features=mock_features)
        mock_hf_ds_dict = hf_datasets.DatasetDict({'train': mock_train_ds, 'validation': mock_val_ds, 'test': mock_val_ds})

        self.mock_hf_load_dataset.return_value = mock_hf_ds_dict

        # Create a mock tokenizer instance
        mock_tokenizer_instance = MagicMock(spec=AutoTokenizer) # Use spec for better mocking
        mock_tokenizer_instance.vocab_size = 30522
        mock_tokenizer_instance.pad_token_id = 0

        def tokenizer_side_effect(texts, padding, truncation, max_length):
            batch_size = len(texts)
            # This needs to return a BatchEncoding or dict-like object
            # Ensure tensors are on CPU as models/data will be moved there in other tests.
            return {
                'input_ids': torch.randint(0, mock_tokenizer_instance.vocab_size, (batch_size, max_length), dtype=torch.long).cpu(),
                'attention_mask': torch.ones((batch_size, max_length), dtype=torch.long).cpu()
            }
        # Use .side_effect for the mock instance itself if it's directly callable
        # If it's a method like .__call__ that should have the side_effect:
        # mock_tokenizer_instance.__call__.side_effect = tokenizer_side_effect
        # Given AutoTokenizer instances are callable, mock_tokenizer_instance.side_effect is correct.
        mock_tokenizer_instance.side_effect = tokenizer_side_effect

        self.mock_hf_tokenizer.return_value = mock_tokenizer_instance

        # Call the function under test
        train_loader, val_loader, vocab_size, num_classes, pad_token_id = get_real_world_data_loaders(
            dataset_name='ag_news',
            max_length=12,
            batch_size=2,
            train_samples=5,
            val_samples=5
        )

        self.mock_hf_load_dataset.assert_called_with('ag_news', name=None, trust_remote_code=True)
        self.mock_hf_tokenizer.assert_called_with("prajjwal1/bert-tiny")

        # Check that the tokenizer instance was called (e.g., by the .map function)
        self.assertGreater(mock_tokenizer_instance.call_count, 0)

        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertEqual(len(train_loader.dataset), 5) # train_samples
        self.assertEqual(len(val_loader.dataset), 5) # val_samples

        batch = next(iter(train_loader))
        self.assertIsInstance(batch, dict)
        self.assertIn('input_ids', batch)
        self.assertIn('attention_mask', batch)
        self.assertIn('labels', batch) # This is added by tokenize_and_format

        self.assertEqual(batch['input_ids'].shape, (2, 12)) # batch_size, max_length
        self.assertEqual(batch['labels'].shape, (2,))    # batch_size

        self.assertEqual(vocab_size, mock_tokenizer_instance.vocab_size)
        self.assertEqual(num_classes, 4) # From mock_features
        self.assertEqual(pad_token_id, mock_tokenizer_instance.pad_token_id)

if __name__ == "__main__":
    unittest.main()
