import os
import json
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Any
from accelerate import Accelerator
from rad_dino.utils.metrics.compute_metrics import compute_evaluation_metrics
from rad_dino.configs.config import OutputPaths
import logging

logger = logging.getLogger(__name__)

class EvaluationProcessor:
    """Handles processing and saving of inference evaluation metrics"""
    
    def __init__(self, accelerator: Accelerator, output_paths: OutputPaths, 
                 task: str, class_labels: List):
        self.accelerator = accelerator
        self.output_paths = output_paths
        self.task = task
        self.class_labels = class_labels
        
        # Initialize result storage
        self.all_ids = []
        self.all_trues = []
        self.all_preds_prob = []
        
    def _prepare_labels_for_csv(self, Y_true: np.ndarray, Y_pred_prob: np.ndarray) -> tuple:
        """Prepare true and predicted labels for CSV output"""
        if self.task == "multiclass":
            true_col = Y_true.astype(int).tolist()
            pred_indices = np.argmax(Y_pred_prob, axis=1)
            pred_labels = [self.class_labels[i] for i in pred_indices]
        elif self.task == "binary":
            true_col = Y_true.squeeze().astype(int).tolist()
            prob_pos = Y_pred_prob.squeeze()
            pred_labels = (prob_pos >= 0.5).astype(int).tolist()
        else:  # multilabel
            true_col = [list(map(int, row)) for row in Y_true]
            threshold = 0.5
            pred_labels = [
                [self.class_labels[i] for i, p in enumerate(row) if p >= threshold]
                for row in Y_pred_prob
            ]
        
        return true_col, pred_labels 

    def _prepare_class_labels_for_metrics(self) -> List:
        """Return class labels for metrics computation"""
        return self.class_labels 
    
    def add_batch_results(self, image_ids: List[str], targets: torch.Tensor, 
                         logits: torch.Tensor) -> None:
        """Add results from a single batch"""
        # Process predictions
        if self.task == "multiclass":
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        else:
            probs = torch.sigmoid(logits).cpu().numpy()
        
        trues = targets.cpu().numpy()
        
        self.all_ids.extend(image_ids)
        self.all_trues.append(trues)
        self.all_preds_prob.append(probs)
    
    def process_and_save_results(self) -> Dict[str, Any]:
        """Process all results and save to files"""
        if not self.accelerator.is_main_process:
            return {}
        
        # Concatenate all results
        Y_true = np.concatenate(self.all_trues, axis=0)
        Y_pred_prob = np.concatenate(self.all_preds_prob, axis=0)
        
        # Validation checks
        assert len(self.all_ids) == Y_true.shape[0], f"len(all_ids) = {len(self.all_ids)}, Y_true.shape[0] = {Y_true.shape[0]}"
        assert Y_pred_prob.shape[0] == Y_true.shape[0], f"Y_pred_prob.shape[0] = {Y_pred_prob.shape[0]}, Y_true.shape[0] = {Y_true.shape[0]}"
        
        # Prepare labels for CSV
        true_col, pred_labels = self._prepare_labels_for_csv(Y_true, Y_pred_prob)
        
        # Create and save DataFrame
        df = pd.DataFrame({
            "image_id": self.all_ids,
            "true_labels": true_col,
            "pred_labels": pred_labels,
            "pred_probs": [list(row) for row in Y_pred_prob],
        })
        
        # Save predictions CSV
        output_csv = os.path.join(self.output_paths.table, 'predictions.csv')
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved predictions to {output_csv}!")
        
        # Convert class labels for metrics computation (matches original fix)
        metrics_class_labels = self._prepare_class_labels_for_metrics()
        
        # Compute and save metrics
        metrics = compute_evaluation_metrics(Y_true, Y_pred_prob, self.task, 
                                           metrics_class_labels, self.output_paths.figs, 
                                           self.accelerator)
        
        metrics_path = os.path.join(self.output_paths.table, "metrics.json")
        with open(metrics_path, "w") as jf:
            json.dump(metrics, jf, indent=4)
        logger.info(f"Saved ROC/PR curves to {self.output_paths.figs} and metrics in JSON to {metrics_path}!")
        
        return metrics
    
    