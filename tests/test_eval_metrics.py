import numpy as np
import tempfile
import os
from unittest.mock import Mock

# Mock accelerator for testing
class MockAccelerator:
    def __init__(self):
        self.is_main_process = True

def test_binary_metrics():
    """Test binary classification metrics"""
    print("Testing binary classification metrics...")
    
    # Create mock data
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred_prob = np.array([0.1, 0.8, 0.2, 0.9, 0.7, 0.3, 0.6, 0.1])
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        accelerator = MockAccelerator()
        
        # Import the function
        from rad_dino.utils.metrics.compute_metrics import compute_binary_metrics
        
        # Test the function
        metrics = compute_binary_metrics(y_true, y_pred_prob, temp_dir, accelerator)
        
        # Check that metrics are computed
        assert "AUROC" in metrics
        assert "AUPRC" in metrics
        assert isinstance(metrics["AUROC"], float)
        assert isinstance(metrics["AUPRC"], float)
        
        print(f"✓ Binary metrics: AUROC={metrics['AUROC']:.3f}, AUPRC={metrics['AUPRC']:.3f}")

def test_multiclass_metrics():
    """Test multiclass classification metrics"""
    print("Testing multiclass classification metrics...")
    
    # Create mock data for 3 classes
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    y_pred_prob = np.array([
        [0.8, 0.1, 0.1],  # Class 0
        [0.1, 0.7, 0.2],  # Class 1
        [0.2, 0.1, 0.7],  # Class 2
        [0.9, 0.05, 0.05], # Class 0
        [0.1, 0.8, 0.1],   # Class 1
        [0.1, 0.2, 0.7],   # Class 2
        [0.7, 0.2, 0.1],   # Class 0
        [0.2, 0.6, 0.2]    # Class 1
    ])
    
    class_labels = ["Class_A", "Class_B", "Class_C"]
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        accelerator = MockAccelerator()
        
        # Import the function
        from rad_dino.utils.metrics.compute_metrics import compute_multiclass_metrics
        
        # Test the function
        metrics = compute_multiclass_metrics(y_true, y_pred_prob, class_labels, temp_dir, accelerator)
        
        # Check that metrics are computed
        assert "overall" in metrics
        assert "f1_macro" in metrics["overall"]
        assert "f1_micro" in metrics["overall"]
        assert "roc_auc_macro_ovr" in metrics["overall"]
        assert "roc_auc_micro_ovr" in metrics["overall"]
        
        # Check per-class metrics
        for cls in class_labels:
            assert cls in metrics
            assert "AUROC" in metrics[cls]
            assert "AUPRC" in metrics[cls]
        
        print(f"✓ Multiclass metrics: F1-macro={metrics['overall']['f1_macro']:.3f}, F1-micro={metrics['overall']['f1_micro']:.3f}")
        print(f"  Macro ROC-AUC: {metrics['overall']['roc_auc_macro_ovr']:.3f}")
        print(f"  Micro ROC-AUC: {metrics['overall']['roc_auc_micro_ovr']:.3f}")

def test_multilabel_metrics():
    """Test multilabel classification metrics"""
    print("Testing multilabel classification metrics...")
    
    # Create mock data for 2 labels
    y_true = np.array([
        [1, 0],  # Label 1 present, Label 2 absent
        [0, 1],  # Label 1 absent, Label 2 present
        [1, 1],  # Both labels present
        [0, 0],  # Both labels absent
        [1, 0],  # Label 1 present, Label 2 absent
    ])
    
    y_pred_prob = np.array([
        [0.8, 0.2],  # High confidence for Label 1
        [0.1, 0.9],  # High confidence for Label 2
        [0.7, 0.8],  # High confidence for both
        [0.1, 0.1],  # Low confidence for both
        [0.9, 0.1],  # High confidence for Label 1
    ])
    
    class_labels = ["Label_1", "Label_2"]
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        accelerator = MockAccelerator()
        
        # Import the function
        from rad_dino.utils.metrics.compute_metrics import compute_multilabel_metrics
        
        # Test the function
        metrics = compute_multilabel_metrics(y_true, y_pred_prob, class_labels, temp_dir, accelerator)
        
        # Check that metrics are computed for each label
        for cls in class_labels:
            assert cls in metrics
            assert "AUROC" in metrics[cls]
            assert "AUPRC" in metrics[cls]
        
        print(f"✓ Multilabel metrics:")
        for cls in class_labels:
            print(f"  {cls}: AUROC={metrics[cls]['AUROC']:.3f}, AUPRC={metrics[cls]['AUPRC']:.3f}")

def test_main_function():
    """Test the main compute_evaluation_metrics function"""
    print("Testing main compute_evaluation_metrics function...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        accelerator = MockAccelerator()
        
        # Import the function
        from rad_dino.utils.metrics.compute_metrics import compute_evaluation_metrics
        
        # Test binary
        y_true_binary = np.array([0, 1, 0, 1])
        y_pred_binary = np.array([0.1, 0.8, 0.2, 0.9])
        metrics_binary = compute_evaluation_metrics(
            y_true_binary, y_pred_binary, "binary", [], temp_dir, accelerator
        )
        assert "AUROC" in metrics_binary
        
        # Test multiclass
        y_true_multi = np.array([0, 1, 2])
        y_pred_multi = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        class_labels = ["A", "B", "C"]
        metrics_multi = compute_evaluation_metrics(
            y_true_multi, y_pred_multi, "multiclass", class_labels, temp_dir, accelerator
        )
        assert "overall" in metrics_multi
        
        # Test multilabel
        y_true_ml = np.array([[1, 0], [0, 1]])
        y_pred_ml = np.array([[0.8, 0.2], [0.1, 0.9]])
        ml_labels = ["Label_1", "Label_2"]
        metrics_ml = compute_evaluation_metrics(
            y_true_ml, y_pred_ml, "multilabel", ml_labels, temp_dir, accelerator
        )
        assert "Label_1" in metrics_ml
        
        print("✓ Main function works for all task types")

if __name__ == "__main__":
    print("Testing refactored metrics computation...")
    print("=" * 50)
    
    try:
        test_binary_metrics()
        test_multiclass_metrics()
        test_multilabel_metrics()
        test_main_function()
        print("\n" + "=" * 50)
        print("✓ All tests passed! The refactored code works correctly.")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 