"""Tests for data loading and preprocessing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from configs.base import DataConfig


class TestDataConfig:
    def test_default_config(self):
        config = DataConfig()
        assert config.dataset_name == "sst2"
        assert config.hf_path == "stanfordnlp/sst2"
        assert config.split == "validation"
        assert config.toxicity_threshold == 0.5

    def test_custom_config(self):
        config = DataConfig(dataset_name="toxicity", hf_path="google/civil_comments")
        assert config.dataset_name == "toxicity"


class TestLoader:
    @patch("src.data.loader.hf_load_dataset")
    def test_load_sst2(self, mock_load):
        from src.data.loader import load_dataset_splits
        # Mock HuggingFace dataset
        mock_data = [
            {"sentence": "Great movie!", "label": 1},
            {"sentence": "Terrible movie.", "label": 0},
            {"sentence": "Amazing film!", "label": 1},
            {"sentence": "Awful film.", "label": 0},
            {"sentence": "Loved it!", "label": 1},
            {"sentence": "Hated it.", "label": 0},
            {"sentence": "Wonderful!", "label": 1},
            {"sentence": "Horrible.", "label": 0},
            {"sentence": "Best ever!", "label": 1},
            {"sentence": "Worst ever.", "label": 0},
        ]
        mock_load.return_value = mock_data

        config = DataConfig(probe_train_ratio=0.8)
        splits = load_dataset_splits(config)

        assert "train_texts" in splits
        assert "train_labels" in splits
        assert "test_texts" in splits
        assert "test_labels" in splits
        assert len(splits["train_texts"]) + len(splits["test_texts"]) == 10


class TestPreprocessing:
    def test_get_last_token_position(self):
        """Test that we correctly identify last non-padding token."""
        import torch
        # Simulate attention mask: [1,1,1,0,0] -> last token at index 2
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
        positions = attention_mask.sum(dim=1) - 1
        assert positions[0].item() == 2
        assert positions[1].item() == 3
