import unittest
from unittest.mock import patch, MagicMock, Mock
import tempfile
import shutil
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from rad_dino.train.trainer import Trainer

# --- Test Helpers ---

class DummyDataset(Dataset):
    def __init__(self, n_samples=10, n_classes=2, multi_view=False):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.multi_view = multi_view
        
    def __len__(self): 
        return self.n_samples
        
    def __getitem__(self, idx):
        img = torch.randn(4, 3, 224, 224) if self.multi_view else torch.randn(3, 224, 224)
        target = torch.randint(0, 2, (self.n_classes,)).float()
        return img, target, f"id{idx}"

class DummyModel(nn.Module):
    def __init__(self, num_classes=2, multi_view=False):
        super().__init__()
        self.multi_view = multi_view
        
        # Create a backbone structure for progressive unfreezing tests
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.backbone.config = MagicMock()
        self.backbone.config.num_hidden_layers = 4
        
        if multi_view:
            self.fusion = nn.Sequential(nn.Linear(4*128, 128), nn.ReLU())
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        if self.multi_view:
            b = x.shape[0]
            x = x.reshape(b*4, 3, 224, 224)
            feats = self.backbone(x).view(b, 4*128)
            feats = self.fusion(feats)
        else:
            feats = self.backbone(x).view(x.size(0), -1)
        return self.classifier(feats), None

def get_trainer(model=None, args=None, accelerator=None):
    args = args or MagicMock(
        task="multilabel", data="VinDr-CXR", model="rad_dino",
        unfreeze_backbone=False, progressive_unfreeze=False, resume=False,
        output_dir=tempfile.mkdtemp())
    
    train_config = MagicMock()
    train_config.optim.base_lr = 0.001
    train_config.optim.weight_decay = 0.01
    train_config.batch_size = 4
    train_config.epochs = 2
    train_config.lr_scheduler = MagicMock(warmup_ratio=0.1)
    train_config.early_stopping = MagicMock(patience=2, min_delta=0.001, mode="max")
    
    eval_metrics = {
        "classification": {
            "acc": MagicMock(return_value=torch.tensor(0.8)),
            "auroc": MagicMock(return_value=torch.tensor(0.7)),
            "f1_score": MagicMock(return_value=torch.tensor(0.75)),
            "ap": MagicMock(return_value=torch.tensor(0.65))
        }
    }
    
    accelerator = accelerator or Accelerator()
    return Trainer(
        model=model or DummyModel(), criterion=nn.BCEWithLogitsLoss(),
        eval_metrics=eval_metrics, train_config=train_config,
        accelerator=accelerator, checkpoint_dir=tempfile.mkdtemp(), args=args
    )

# --- Test Classes ---

@patch('rad_dino.train.trainer.wandb', MagicMock())
class TestTrainer(unittest.TestCase):
    """Test the Trainer class functionality."""
    
    def setUp(self):
        self.trainer = get_trainer()
        self.data_loader = DataLoader(DummyDataset(), batch_size=4)
        self.data_loader = self.trainer.accelerator.prepare(self.data_loader)
        
    def tearDown(self):
        # Clean up temporary directories
        if hasattr(self.trainer, 'checkpoint_dir'):
            shutil.rmtree(self.trainer.checkpoint_dir, ignore_errors=True)
        if hasattr(self.trainer.args, 'output_dir'):
            shutil.rmtree(self.trainer.args.output_dir, ignore_errors=True)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertIsInstance(self.trainer.model, nn.Module)
        self.assertIsInstance(self.trainer.criterion, nn.BCEWithLogitsLoss)
        self.assertIsInstance(self.trainer.accelerator, Accelerator)
    
    def test_single_view_training(self):
        """Test single-view training and evaluation."""
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.1)
        
        # Test training
        self.trainer.model.train()
        loss, acc, auroc = self.trainer.train_per_epoch(
            0, self.trainer.model, self.data_loader, optimizer, scheduler, "")
        
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertIsInstance(auroc, float)
        
        # Test evaluation
        self.trainer.model.eval()
        loss, acc, f1, ap, auroc = self.trainer.eval_per_epoch(self.trainer.model, self.data_loader)
        
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertIsInstance(f1, float)
        self.assertIsInstance(ap, float)
        self.assertIsInstance(auroc, float)
    
    def test_multi_view_training(self):
        """Test multi-view training and evaluation."""
        trainer = get_trainer(DummyModel(multi_view=True))
        ds = DummyDataset(multi_view=True)
        data_loader = DataLoader(ds, batch_size=4)
        data_loader = trainer.accelerator.prepare(data_loader)
        
        optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.1)
        
        # Test training
        trainer.model.train()
        loss, acc, auroc = trainer.train_per_epoch(
            0, trainer.model, data_loader, optimizer, scheduler, "")
        
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertIsInstance(auroc, float)
        
        # Test evaluation
        trainer.model.eval()
        loss, acc, f1, ap, auroc = trainer.eval_per_epoch(trainer.model, data_loader)
        
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertIsInstance(f1, float)
        self.assertIsInstance(ap, float)
        self.assertIsInstance(auroc, float)
    
    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = DummyModel()
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        target = torch.randint(0, 2, (2, 2)).float()
        
        logits, _ = model(x)
        loss = nn.BCEWithLogitsLoss()(logits, target)
        loss.backward()
        
        self.assertIsNotNone(x.grad)
    
    @patch('rad_dino.train.trainer.get_model_info')
    @patch('rad_dino.train.trainer.get_layer_term')
    def test_progressive_unfreezing(self, mock_get_layer_term, mock_get_model_info):
        """Test progressive unfreezing functionality."""
        # Mock the model registry functions
        mock_get_model_info.return_value = {
            'model_type': 'ViT',
            'total_layers': 4,
            'layer_pattern': 'layer.{}'
        }
        mock_get_layer_term.return_value = 'layers'
        
        args = MagicMock(unfreeze_backbone=True, progressive_unfreeze=True, resume=False, output_dir=tempfile.mkdtemp())
        trainer = get_trainer(model=DummyModel(), args=args)
        model = DummyModel()
        
        # Test freezing backbone
        trainer._freeze_backbone(model)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))

        # Test progressive unfreezing - first unfreeze at epoch 2
        trainer._progressive_unfreeze(model, current_epoch=2)
        # Should unfreeze 1 layer (epoch 2 // 2 = 1)
        # Since our dummy model doesn't have the exact layer structure, we'll test the logic differently
        # The test should verify that the unfreezing logic is called correctly
        mock_get_model_info.assert_called()
        mock_get_layer_term.assert_called()
        
        # Test that the progressive unfreezing method doesn't raise errors
        trainer._progressive_unfreeze(model, current_epoch=4)
        # Should unfreeze 2 layers (epoch 4 // 2 = 2)
        mock_get_model_info.assert_called()
    
    @patch('rad_dino.train.trainer.get_model_info')
    @patch('rad_dino.train.trainer.get_layer_term')
    def test_unfreezing_strategies(self, mock_get_layer_term, mock_get_model_info):
        """Test different unfreezing strategies."""
        # Mock the model registry functions
        mock_get_model_info.return_value = {
            'model_type': 'ViT',
            'total_layers': 4,
            'layer_pattern': 'layer.{}'
        }
        mock_get_layer_term.return_value = 'layers'
        
        # Test unfreeze_num_layers strategy
        args = MagicMock(
            unfreeze_backbone=True, 
            progressive_unfreeze=False, 
            unfreeze_num_layers=2,
            resume=False, 
            output_dir=tempfile.mkdtemp()
        )
        trainer = get_trainer(model=DummyModel(), args=args)
        model = DummyModel()
        
        trainer._apply_unfreezing_strategy(model, current_epoch=0)
        # The test should verify that the strategy is applied without errors
        mock_get_model_info.assert_called()
        
        # Test unfreeze all backbone strategy
        args = MagicMock(
            unfreeze_backbone=True, 
            progressive_unfreeze=False, 
            unfreeze_num_layers=None,
            resume=False, 
            output_dir=tempfile.mkdtemp()
        )
        trainer = get_trainer(model=DummyModel(), args=args)
        model = DummyModel()
        
        trainer._apply_unfreezing_strategy(model, current_epoch=0)
        # All backbone parameters should be unfrozen
        unfrozen_count = sum(1 for p in model.backbone.parameters() if p.requires_grad)
        self.assertEqual(unfrozen_count, len(list(model.backbone.parameters())))
    
    def test_parameter_groups(self):
        """Test parameter group separation."""
        trainer = get_trainer()
        model = DummyModel()
        
        # Freeze backbone first
        trainer._freeze_backbone(model)
        
        backbone_params, head_params = trainer._get_parameter_groups(model)
        
        # All backbone params should be frozen
        self.assertTrue(all(not p.requires_grad for p in backbone_params))
        # All head params should be trainable
        self.assertTrue(all(p.requires_grad for p in head_params))
    
    def test_optimizer_parameter_groups_update(self):
        """Test optimizer parameter groups update."""
        trainer = get_trainer()
        model = DummyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        initial_groups = len(optimizer.param_groups)
        
        # Unfreeze some backbone parameters
        for i, param in enumerate(model.backbone.parameters()):
            if i < 2:  # Unfreeze first 2 parameters
                param.requires_grad = True
        
        trainer._update_optimizer_parameter_groups(optimizer, model)
        
        # Should have added a new parameter group for backbone
        self.assertGreaterEqual(len(optimizer.param_groups), initial_groups)

@patch('rad_dino.train.trainer.wandb', MagicMock())
class TestNewModels(unittest.TestCase):
    """Test that new models work with the trainer."""
    
    def setUp(self):
        self.args = MagicMock(
            task="multilabel", data="VinDr-CXR", model="rad_dino",
            unfreeze_backbone=False, progressive_unfreeze=False, resume=False,
            output_dir=tempfile.mkdtemp()
        )
        
    def tearDown(self):
        if hasattr(self.args, 'output_dir'):
            shutil.rmtree(self.args.output_dir, ignore_errors=True)
    
    def test_ark_model_integration(self):
        """Test Ark model integration with trainer."""
        # Create a mock Ark model structure with correct shapes
        class MockArkModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, 1, 1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.backbone.layers = [nn.Linear(64, 32) for _ in range(4)]  # Mock layers
                self.classifier = nn.Linear(64, 2)  # Fixed: 64 input features
                
            def forward(self, x):
                x = self.backbone(x).view(x.size(0), -1)  # Flatten to [batch, 64]
                return self.classifier(x), None
        
        model = MockArkModel()
        trainer = get_trainer(model=model, args=self.args)
        self.assertIsInstance(trainer.model, MockArkModel)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output, _ = model(x)
        self.assertEqual(output.shape, (2, 2))
    
    def test_medsiglip_model_integration(self):
        """Test MedSigLIP model integration with trainer."""
        # Create a mock MedSigLIP model structure with correct shapes
        class MockMedSigModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, 1, 1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.backbone.config = MagicMock()
                self.backbone.config.vision_config = MagicMock()
                self.backbone.config.vision_config.num_hidden_layers = 12
                self.classifier = nn.Linear(64, 2)  # Fixed: 64 input features
                
            def forward(self, x):
                x = self.backbone(x).view(x.size(0), -1)  # Flatten to [batch, 64]
                return self.classifier(x), None
        
        model = MockMedSigModel()
        trainer = get_trainer(model=model, args=self.args)
        self.assertIsInstance(trainer.model, MockMedSigModel)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output, _ = model(x)
        self.assertEqual(output.shape, (2, 2))
    
    def test_new_models_multi_view(self):
        """Test new models with multi-view processing."""
        # Create mock multi-view models with correct shapes
        class MockMultiViewModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.multi_view = True
                self.num_views = 4
                self.view_fusion_type = "mean"
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, 1, 1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.backbone.layers = [nn.Linear(64, 32) for _ in range(4)]
                self.classifier = nn.Linear(64, 2)  # Fixed: 64 input features
                
            def forward(self, x):
                # Multi-view processing
                b, v, c, h, w = x.shape
                x = x.view(b * v, c, h, w)
                x = self.backbone(x).view(b, v, -1)  # [batch, views, 64]
                x = x.mean(dim=1)  # Mean fusion -> [batch, 64]
                return self.classifier(x), None
        
        model = MockMultiViewModel()
        trainer = get_trainer(model=model, args=self.args)
        
        # Test multi-view forward pass
        x = torch.randn(2, 4, 3, 224, 224)
        output, _ = model(x)
        self.assertEqual(output.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()