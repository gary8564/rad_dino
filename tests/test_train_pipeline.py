import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
import shutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from rad_dino.train.trainer import Trainer

class DummyDataset(Dataset):
    def __init__(self, n_samples=10, n_classes=2):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.image_ids = list(range(n_samples))
        self.df = pd.DataFrame(
            np.ones((n_samples, n_classes), dtype=np.int64),
            index=self.image_ids,
            columns=[f"label{i + 1}" for i in range(n_classes)]
        )
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        img = torch.randn(3, 224, 224)
        target = torch.randint(0, 2, (self.n_classes,)).float()
        return img, target, f"id{idx}"

class DummyModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        # Return tuple like the real model (logits, attentions)
        # Create dummy attentions for compatibility
        batch_size = x.size(0)
        dummy_attentions = torch.randn(12, batch_size, 12, 197, 197)
        return logits, dummy_attentions

class TestTrainPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        cls.args = MagicMock(
            task="multilabel",
            data="VinDr-CXR",
            model="rad_dino",
            unfreeze_backbone=False,
            optimize_compute=False,
            resume=False,
            output_dir=tempfile.mkdtemp()
        )
        
        # Mock wandb
        cls.wandb_mock = MagicMock()
        cls.wandb_patcher = patch('rad_dino.train.trainer.wandb', cls.wandb_mock)
        cls.wandb_patcher.start()
        
        # Initialize common components
        cls.model = DummyModel(num_classes=2)
        cls.criterion = nn.BCEWithLogitsLoss()
        cls.eval_metrics = {
            "classification": {
                "acc": MagicMock(return_value=torch.tensor(0.8)),
                "auroc": MagicMock(return_value=torch.tensor(0.7)),
                "f1_score": MagicMock(return_value=torch.tensor(0.75)),
                "ap": MagicMock(return_value=torch.tensor(0.65))
            }
        }
        cls.train_config = MagicMock(
            batch_size=4,
            epochs=2,
            optim=MagicMock(base_lr=0.001, weight_decay=0.01),
            lr_scheduler=MagicMock(warmup_ratio=0.1),
            early_stopping=MagicMock(patience=2, min_delta=0.001, mode="max")
        )
        cls.accelerator = Accelerator()
        cls.checkpoint_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        cls.wandb_patcher.stop()
        if hasattr(cls.args, 'output_dir'):
            shutil.rmtree(cls.args.output_dir)
        if hasattr(cls, 'checkpoint_dir'):
            shutil.rmtree(cls.checkpoint_dir)

    def setUp(self):
        """Set up test fixtures before each test."""
        self.trainer = Trainer(
            model=self.model,
            criterion=self.criterion,
            eval_metrics=self.eval_metrics,
            train_config=self.train_config,
            accelerator=self.accelerator,
            checkpoint_dir=self.checkpoint_dir,
            args=self.args
        )
        self.dummy_ds = DummyDataset(n_samples=10, n_classes=2)
        self.data_loader = DataLoader(self.dummy_ds, batch_size=4)
        self.data_loader = self.accelerator.prepare(self.data_loader)

    def test_trainer_initialization(self):
        """Test Trainer class initialization."""
        self.assertEqual(self.trainer.model, self.model)
        self.assertEqual(self.trainer.criterion, self.criterion)
        self.assertEqual(self.trainer.eval_metrics, self.eval_metrics)
        self.assertEqual(self.trainer.train_config, self.train_config)
        self.assertEqual(self.trainer.accelerator, self.accelerator)
        self.assertEqual(self.trainer.checkpoint_dir, self.checkpoint_dir)
        self.assertEqual(self.trainer.args, self.args)

    def test_train_per_epoch(self):
        """Test training for one epoch."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        
        self.model.train()
        
        loss, acc, auroc = self.trainer.train_per_epoch(
            curr_epoch=0,
            model=self.model,
            data_loader=self.data_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            log_prefix="test/"
        )

        # Convert metrics to tensors properly
        loss_tensor = torch.tensor(loss, device=self.accelerator.device)
        acc_tensor = torch.tensor(acc, device=self.accelerator.device)
        auroc_tensor = torch.tensor(auroc, device=self.accelerator.device)

        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertIsInstance(auroc, float)
        self.assertGreaterEqual(loss, 0)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)
        self.assertGreaterEqual(auroc, 0)
        self.assertLessEqual(auroc, 1)

    def test_eval_per_epoch(self):
        """Test evaluation for one epoch."""
        loss, acc, f1, ap, auroc = self.trainer.eval_per_epoch(
            model=self.model,
            data_loader=self.data_loader
        )

        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertIsInstance(f1, float)
        self.assertIsInstance(ap, float)
        self.assertIsInstance(auroc, float)
        self.assertGreaterEqual(loss, 0)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)
        self.assertGreaterEqual(f1, 0)
        self.assertLessEqual(f1, 1)
        self.assertGreaterEqual(ap, 0)
        self.assertLessEqual(ap, 1)
        self.assertGreaterEqual(auroc, 0)
        self.assertLessEqual(auroc, 1)

    @patch('torch.onnx.export')
    def test_export_onnx(self, mock_onnx_export):
        """Test ONNX model export."""
        # Save model state dict
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "best.pt"))
        
        # Test export
        self.trainer.export_onnx(self.model, "rad_dino")
        
        # Verify ONNX export was called
        mock_onnx_export.assert_called_once()
        
        # Check if model state dict was saved
        self.assertTrue(os.path.exists(os.path.join(self.checkpoint_dir, "best.pt")))

if __name__ == "__main__":
    unittest.main()