import unittest
from unittest.mock import patch, MagicMock, ANY
import types
from datetime import datetime
import argparse
import shutil
import tempfile
import os
import torch
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset
from accelerate import Accelerator
from rad_dino.train.train import (
    load_data,
    setup,
    initialize_fold,
    train_per_epoch,
    main,
    OptimizerConfig,
    LRSchedulerConfig,
    EarlyStoppingConfig,
    TrainConfig,
    DataConfig
)
from rad_dino.utils.utils import EarlyStopping
from rad_dino.data.VinDrCXR.data import VinDrCXR_Dataset

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
        img = torch.randn(3,16,16)
        target = torch.randint(0,2,(self.n_classes,)).float()
        return img, target, f"id{idx}"

class TestTrainPipeline(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(
            task="multilabel",
            data="VinDr-CXR",
            model="rad_dino",
            unfreeze_backbone=False,
            optimize_compute=False,
            resume=False,
            output_dir=tempfile.mkdtemp(),
        )
        self.class_labels = ["label1", "label2"]
        self.batch_size = 4
        self.num_workers = 0
        self.device = "cpu"
        self.data_root_folder = tempfile.mkdtemp()  # Create temporary directory for data
        
        # Create Pydantic config objects instead of dictionaries
        self.train_config = TrainConfig(
            batch_size=self.batch_size,
            epochs=2,
            optim=OptimizerConfig(
                base_lr=1e-3,
                weight_decay=0.01
            ),
            lr_scheduler=LRSchedulerConfig(
                warmup_ratio=0.1
            ),
            early_stopping=EarlyStoppingConfig(
                patience=1
            )
        )
        
        # Mock wandb
        self.wandb_mock = types.SimpleNamespace(
            init=lambda *a, **k: types.SimpleNamespace(
                config=types.SimpleNamespace(batch_size=self.batch_size),
                log=lambda *a, **k: None
            ),
            log=lambda *a, **k: None,
            config=types.SimpleNamespace(batch_size=self.batch_size),
            Error=Exception,
            _sentry=types.SimpleNamespace(reraise=lambda e: e),
            sdk=types.SimpleNamespace(
                wandb_init=types.SimpleNamespace(
                    setup_run_log_directory=lambda *a, **k: None
                )
            )
        )
        self.wandb_patcher = patch('rad_dino.train.train.wandb', self.wandb_mock)
        self.wandb_patcher.start()

        # Create necessary directory structure and files for VinDrCXR_Dataset
        ann_dir = os.path.join(self.data_root_folder, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_root_folder, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.data_root_folder, "test"), exist_ok=True)

        # Create dummy annotation files
        cols = ["image_id"] + self.class_labels
        df_train = pd.DataFrame([[f"img{i}", 1, 1] for i in range(10)], columns=cols).set_index("image_id")
        df_test = pd.DataFrame([[f"img{i}", 1, 1] for i in range(10)], columns=cols).set_index("image_id")
        
        # Save annotation files
        df_train.to_csv(os.path.join(ann_dir, "image_labels_train.csv"))
        df_test.to_csv(os.path.join(ann_dir, "image_labels_test.csv"))
        
        # Create annotations files
        annot_cols = ["image_id", "class_name"]
        df_train_annot = pd.DataFrame([[f"img{i}", label] for i in range(10) for label in self.class_labels], columns=annot_cols)
        df_test_annot = pd.DataFrame([[f"img{i}", label] for i in range(10) for label in self.class_labels], columns=annot_cols)
        df_train_annot.to_csv(os.path.join(ann_dir, "annotations_train.csv"), index=False)
        df_test_annot.to_csv(os.path.join(ann_dir, "annotations_test.csv"), index=False)

        # Create dummy DICOM files
        for i in range(10):
            open(os.path.join(self.data_root_folder, "train", f"img{i}.dicom"), "wb").close()
            open(os.path.join(self.data_root_folder, "test", f"img{i}.dicom"), "wb").close()

        # Mock filter_subset_annot_labels
        self.filter_patcher = patch('rad_dino.data.VinDrCXR.data.filter_subset_annot_labels', return_value=self.class_labels)
        self.filter_patcher.start()

        # Mock dicom2array
        self.dicom_patcher = patch('rad_dino.data.VinDrCXR.data.dicom2array', return_value=np.ones((16, 16), dtype=np.uint8))
        self.dicom_patcher.start()

    def tearDown(self):
        self.wandb_patcher.stop()
        self.filter_patcher.stop()
        self.dicom_patcher.stop()
        if hasattr(self, 'args') and hasattr(self.args, 'output_dir'):
            shutil.rmtree(self.args.output_dir)
        if hasattr(self, 'data_root_folder'):
            shutil.rmtree(self.data_root_folder)
    
    def test_early_stopping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "tmp_best.pt")
            es = EarlyStopping(patience=2, min_delta=0.0, mode="max", ckpt_path=ckpt_path)

            class DummyModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lin = torch.nn.Linear(1,1)

            model = DummyModel()

            # First call sets best_score, should not stop
            stop = es.step(0.5, model, torch.optim.SGD(model.parameters(), lr=0.1), None, epoch=1)
            self.assertFalse(stop)
            self.assertTrue(os.path.exists(ckpt_path))

            # One bad epoch, still within patience
            stop = es.step(0.4, model, torch.optim.SGD(model.parameters(), lr=0.1), None, epoch=2)
            self.assertFalse(stop)

            # Second bad epoch → patience exceeded → early_stop=True
            stop = es.step(0.3, model, torch.optim.SGD(model.parameters(), lr=0.1), None, epoch=3)
            self.assertTrue(es.early_stop)

    def test_initialize_fold_resume(self):
        # Test initialize_fold creates correct KFold and resumes when checkpoint exists
        with tempfile.TemporaryDirectory() as tmpdir:
            # create dummy checkpoint
            ckpt_dir = os.path.join(tmpdir, "fold_1")
            os.makedirs(ckpt_dir)
            best_pt = os.path.join(ckpt_dir, "best.pt")
            dummy_model = torch.nn.Linear(1,1)
            dummy_opt = torch.optim.SGD(dummy_model.parameters(), lr=0.1)
            
            # Create accelerator
            accelerator = Accelerator()
            
            # Save checkpoint in main process
            if accelerator.is_main_process:
                torch.save({
                    "epoch": 3,
                    "model_state": dummy_model.state_dict(),
                    "optimizer_state": dummy_opt.state_dict(),
                    "scheduler_state": None,
                    "best_metric": 0.7
                }, best_pt)
            
            # Wait for main process to finish saving
            accelerator.wait_for_everyone()
            
            # dummy loaders
            dummy_ds = TensorDataset(torch.randn(5,3,16,16), torch.randn(5,3))
            train_loader = DataLoader(dummy_ds, batch_size=2)
            val_loader   = DataLoader(dummy_ds, batch_size=2)
            
            # Create a minimal TrainConfig for testing
            test_train_config = TrainConfig(
                batch_size=2,
                epochs=10,
                optim=OptimizerConfig(
                    base_lr=0.1,
                    weight_decay=0.0
                )
            )
            
            # Mock reduce to return different values for start_epoch and best_metric
            def mock_reduce(tensor, reduction="mean"):
                # Convert tensor to float for comparison
                val = float(tensor.item())
                if abs(val - 3.0) < 1e-6:  # start_epoch tensor
                    return torch.tensor(3.0)
                elif abs(val - 0.7) < 1e-6:  # best_metric tensor
                    return torch.tensor(0.7)
                elif abs(val - (-float("inf"))) < 1e-6:  # default best_metric
                    return torch.tensor(-float("inf"))
                return torch.tensor(0.0)  # default case

            with patch.object(accelerator, 'reduce', side_effect=mock_reduce):
                # Test normal resume case
                kf = initialize_fold(
                    args=argparse.Namespace(resume=True, unfreeze_backbone=False),
                    base_model=dummy_model,
                    train_config=test_train_config,
                    checkpoint_dir=tmpdir,
                    fold_idx=1,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    accelerator=accelerator
                )
                
                # Verify that start_epoch and best_metric are properly synchronized
                self.assertEqual(kf.start_epoch, 3)
                self.assertAlmostEqual(kf.best_metric, 0.7)
            
            # Test error handling when checkpoint doesn't exist
            with self.assertRaises(RuntimeError):
                initialize_fold(
                    args=argparse.Namespace(resume=True, unfreeze_backbone=False),
                    base_model=dummy_model,
                    train_config=test_train_config,
                    checkpoint_dir=tmpdir,
                    fold_idx=2,  # Use a different fold that doesn't have a checkpoint
                    train_loader=train_loader,
                    val_loader=val_loader,
                    accelerator=accelerator
                )
            
            # Test behavior with empty tensors (simulating no main process)
            def mock_reduce_non_main(tensor, reduction="mean"):
                return torch.tensor(0.0)

            # Create a new accelerator instance for non-main process testing
            non_main_accelerator = Accelerator()
            with patch.object(non_main_accelerator, 'reduce', side_effect=mock_reduce_non_main):
                kf = initialize_fold(
                    args=argparse.Namespace(resume=True, unfreeze_backbone=False),
                    base_model=dummy_model,
                    train_config=test_train_config,
                    checkpoint_dir=tmpdir,
                    fold_idx=1,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    accelerator=non_main_accelerator
                )
                # Should use default values when no main process
                self.assertEqual(kf.start_epoch, 0)
                self.assertEqual(kf.best_metric, 0.0)

    def test_load_data_single(self):
        # Test load_data for single-split returns two DataLoaders
        train_loader, val_loader = load_data(
            data_root_folder=self.data_root_folder,
            class_labels=self.class_labels, 
            batch_size=4,
            train_transforms=None, 
            val_transforms=None,
            num_workers=0, 
            kfold=None
        )
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        # total samples
        total = len(train_loader.dataset) + len(val_loader.dataset)
        self.assertEqual(total, 10)

    def test_load_data_kfold(self):
        # Test load_data with kfold=2 returns list of 2 folds
        with patch('rad_dino.train.train.MultilabelStratifiedKFold') as mock_kfold:
            mock_kfold.return_value.split.return_value = [
                (np.array([0,1,2,3,4]), np.array([5,6,7,8,9])),
                (np.array([5,6,7,8,9]), np.array([0,1,2,3,4]))
            ]
            folds = load_data(
                data_root_folder=self.data_root_folder,
                class_labels=self.class_labels, 
                batch_size=4,
                train_transforms=None, 
                val_transforms=None,
                num_workers=0, 
                kfold=2, 
                seed=0
            )
            self.assertIsInstance(folds, list)
            self.assertEqual(len(folds), 2)
            for tr, va in folds:
                self.assertIsInstance(tr, DataLoader)
                self.assertIsInstance(va, DataLoader)
                self.assertEqual(len(tr.dataset) + len(va.dataset), 10)
                                
    def test_train_per_epoch(self):
        # Create a dummy model that matches the expected input/output shapes
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Sequential(
                    torch.nn.Linear(3 * 16 * 16, 768),  # Flatten and project to expected dimension
                    torch.nn.ReLU()
                )
                self.classifier = torch.nn.Linear(768, 2)  # Output 2 classes
            
            def forward(self, x):
                batch_size = x.size(0)
                x = x.view(batch_size, -1)  # Flatten
                x = self.backbone(x)
                return self.classifier(x)

        model = DummyModel()
        
        # Create dummy data with image IDs
        images = torch.randn(5, 3, 16, 16)  # 5 samples, 3 channels, 16x16
        labels = torch.randn(5, 2)  # 5 samples, 2 classes
        dummy_ids = torch.zeros(5)  # Create tensor of size 5 to match batch size
        dummy_ds = TensorDataset(images, labels, dummy_ids)
        data_loader = DataLoader(dummy_ds, batch_size=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        # Create a proper scheduler instance
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Create dummy metrics
        eval_metrics = {
            "classification": {
                "acc": MagicMock(return_value=torch.tensor(0.8))
            }
        }
        
        # Create train config
        train_config = TrainConfig(
            batch_size=2,
            epochs=2,
            optim=OptimizerConfig(
                base_lr=0.1,
                weight_decay=0.01
            )
        )
        
        # Create accelerator
        accelerator = Accelerator()
        
        # Prepare model, optimizer, and data loader with accelerator
        data_loader, model, optimizer = accelerator.prepare(data_loader, model, optimizer)
        
        # Test train_per_epoch
        loss, acc = train_per_epoch(
            curr_epoch=0,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            eval_metrics=eval_metrics,
            accelerator=accelerator,
            train_config=train_config
        )
        
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(loss, 0)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)

    @patch('torch.jit.script')
    @patch('torch.jit.trace')
    def test_main_torchscript_fallback(self, mock_trace, mock_script):
        # Setup test environment
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create args
            args = argparse.Namespace(
                model="rad_dino",
                output_dir=tmpdir,
                task="multilabel",
                data="VinDr-CXR",
                unfreeze_backbone=False,
                optimize_compute=False,
                resume=False
            )
            
            # Mock the train_model function to return a dummy model
            dummy_model = torch.nn.Linear(1, 1)
            with patch('rad_dino.train.train.train_model', return_value=dummy_model):
                # Mock scripting to fail
                mock_script.side_effect = RuntimeError("Scripting failed")
                
                # Mock trace to succeed
                mock_traced_model = MagicMock()
                mock_trace.return_value = mock_traced_model
                
                # Run main
                main(args)
                
                # Verify that script was attempted
                mock_script.assert_called_once_with(dummy_model)
                
                # Verify that trace was called as fallback
                mock_trace.assert_called_once()
                
                # Verify the correct input size was used for tracing
                call_args = mock_trace.call_args[0]
                self.assertEqual(len(call_args), 2)  # model and dummy input
                self.assertEqual(call_args[0], dummy_model)
                self.assertEqual(call_args[1][0].shape, (1, 3, 518, 518))  # RadDINO input size
                
                # Verify the model was saved with the correct path structure
                save_calls = mock_traced_model.save.call_args_list
                self.assertTrue(len(save_calls) == 1, "Model should be saved exactly once")
                actual_path = save_calls[0][0][0]
                self.assertTrue(actual_path.endswith("rad_dino_final_scripted.pt"), 
                              "Save path should end with rad_dino_final_scripted.pt")
                self.assertTrue("checkpoints_rad_dino_" in actual_path,
                              "Save path should contain checkpoints_rad_dino_")
                self.assertTrue(any(c.isdigit() for c in actual_path),
                              "Save path should contain timestamp")

if __name__ == "__main__":
    unittest.main()