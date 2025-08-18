import os
import argparse
import torch
import logging
import json
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from rad_dino.loggings.setup import init_logging
from rad_dino.data.data_loader import create_test_loader
from rad_dino.utils.config_utils import setup_configs
from rad_dino.eval.evaluation_processor import EvaluationProcessor
from rad_dino.configs.config import OutputPaths
from rad_dino.models.ark import load_prtrained_ark_model
from rad_dino.utils.transforms import get_transforms
from rad_dino.data.label_mapping import class_labels_mapping
from rad_dino.configs.ark_zero_shot_config import ARK_PRETRAINED_TASKS
from rad_dino.utils.zero_shot_transfer.ark_zero_shot_postprocess import build_target_to_pretrained_ark_indices, aggregate_targeted_pred_probs
from rad_dino.utils.zero_shot_transfer.rsna_postprocess import rsna_multiclass_logits_to_binary_logits

init_logging()
logger = logging.getLogger(__name__)

class MedSigLIPZeroShotClassifier:    
    def __init__(self, device: str = "cuda"):
        """
        Initialize the MedSigLIP zero-shot classifier.
        
        Args:
            model_name: HuggingFace model name for MedSigLIP
            device: Device to run inference on
        """
        if not torch.cuda.is_available() and device == "cuda":
            logging.warning("CUDA is not available. Using CPU for inference.")
            device = "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained("google/medsiglip-448")
        self.model = AutoModel.from_pretrained("google/medsiglip-448")
        self.model.to(self.device)
        self.model.eval()
            
    def predict(self, images: torch.Tensor, text_prompts: List[str]) -> torch.Tensor:
        """
        Perform zero-shot prediction on a batch of pre-processed images.
        
        Args:
            images: Pre-processed image tensors from dataset [B, C, H, W]
            text_prompts: List of text prompts for classification
            
        Returns:
            logits_per_image: Image-text similarity scores
        """
        # Process text prompts only
        text_inputs = self.tokenizer(
            text_prompts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        # Move images to device
        images = images.to(self.device)
        
        # Forward pass with pre-processed images and tokenized text
        with torch.no_grad():
            outputs = self.model(
                pixel_values=images,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"]
            )
        
        logits_per_image = outputs.logits_per_image # image-text similarity scores
        return logits_per_image
    
    def run_zero_shot_inference(self, 
                                data_loader: DataLoader, 
                                evaluation_processor: EvaluationProcessor, 
                                text_prompts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run zero-shot inference on the dataset.
        
        Args:
            data_loader: DataLoader for test data
            evaluation_processor: EvaluationProcessor for saving results
            text_prompts: Text prompts for classification (for MedSigLIP)
            
        Returns:
            metrics
        """
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Zero-shot inference...")):
            images = batch["pixel_values"]
            labels = batch["labels"]
            image_ids = batch["sample_ids"]
            # Get predictions 
            logits = self.predict(images, text_prompts)
            # For binary tasks, extract the positive-class logit
            if evaluation_processor.task == "binary" and text_prompts is not None and len(text_prompts) >= 2:
                # Apply softmax to get probabilities 
                probs = torch.softmax(logits, dim=1)
                # Take the positive-class probability
                # Ensure that the prompts order is negative then positive 
                prob_pos = probs[:, -1]
                eps = 1e-6
                logits = torch.logit(torch.clamp(prob_pos, eps, 1 - eps)).unsqueeze(1)
            # Add batch results to evaluation processor
            evaluation_processor.add_batch_results(image_ids, labels, logits)
        # Process and save results using evaluation processor
        metrics = evaluation_processor.process_and_save_results()
        return metrics
    
class ArkZeroShotClassifier:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize the Ark zero-shot classifier.
        
        Args:
            checkpoint_path: Path to the Ark checkpoint file
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        
        # Mapping state between downstream labels and pretrained Ark labels
        self._target_to_pretrained_label_indices = None 
        
        # Load Ark model with all pretrained task heads
        self.model = load_prtrained_ark_model(
            checkpoint_path=checkpoint_path,
            num_classes_list=[14, 14, 14, 3, 6, 1],  # Ark+ pretrained tasks
            img_size=768,
            patch_size=4,
            window_size=12,
            embed_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48),
            projector_features=1376,
            use_mlp=False,
            return_attention=False,
            device=self.device
        )
        self.model.to(self.device)
        self.model.eval()
            
    def predict(self, images: torch.Tensor, head_index: int) -> torch.Tensor:
        """
        Perform zero-shot prediction using Ark's pretrained task heads.
        
        Args:
            images: Pre-processed image tensors from dataset [B, C, H, W]
            head_index: Index of the task head to use for prediction
            
        Returns:
            logits: Classification logits from the appropriate task head
        """
        images = images.to(self.device)
        with torch.no_grad():
            logits, _ = self.model(images, head_n=head_index)
        return logits
    
    def predict_all_heads(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get predictions from all Ark task heads.
        
        Args:
            images: Pre-processed image tensors from dataset [B, C, H, W]
            
        Returns:
            logits: Tensor of shape [B, num_heads, num_classes]
        """
        images = images.to(self.device)
        with torch.no_grad():
            # model returns list of outputs from all heads and attention maps
            pre_logits, _ = self.model(images) 
            logits = torch.cat(pre_logits, dim=1)
        return logits
    
    def run_zero_shot_inference(self, 
                                data_loader: DataLoader, 
                                evaluation_processor: EvaluationProcessor,
                                dataset_name: str,
                                task_type: str,
                                downstream_target_labels: Optional[List[str]]) -> Dict[str, Any]:
        """
        Run zero-shot inference on the dataset.
        
        Args:
            data_loader: DataLoader for test data
            evaluation_processor: EvaluationProcessor for saving results
            dataset_name: Name of the dataset
            task_type: Type of task
            downstream_target_labels: List of downstream target labels
            
        Returns:
            metrics
        """
        # Build mapping between downstream labels and matched pretrained Ark labels
        self._target_to_pretrained_label_indices = build_target_to_pretrained_ark_indices(
            dataset_name, task_type, downstream_target_labels
        )
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Zero-shot inference...")):
            images = batch["pixel_values"]
            labels = batch["labels"]
            image_ids = batch["sample_ids"]
            
            # Get all task head predictions and aggregate
            head_logits = self.predict_all_heads(images)
            head_probs = torch.sigmoid(head_logits)
            agg_probs = aggregate_targeted_pred_probs(
                head_probs,
                target_to_pretrained_label_indices=self._target_to_pretrained_label_indices,
                task_type=task_type,
                downstream_target_labels=downstream_target_labels,
            )
            eps = 1e-8
            if task_type == "multiclass":
                # L1-normalize per-sample so probabilities sum to 1
                prob_sum = agg_probs.sum(dim=1, keepdim=True)
                prob_norm = agg_probs / torch.clamp(prob_sum, min=eps)
                # Pass log-probs so that later in EvaluationProcessor, softmax yields the normalized probs as is.
                agg_logits = torch.log(torch.clamp(prob_norm, min=eps))
            else:
                # For binary/multilabel, convert probs back to logits so that later in EvaluationProcessor,
                # add_batch_results() yields the averaged probs as is.
                agg_logits = torch.logit(torch.clamp(agg_probs, eps, 1 - eps))
            evaluation_processor.add_batch_results(image_ids, labels, agg_logits)

        metrics = evaluation_processor.process_and_save_results()
        return metrics

    def run_rsna_head_zero_shot_inference(self,
                                          data_loader: DataLoader,
                                          evaluation_processor: EvaluationProcessor) -> Dict[str, Any]:
        """
        Run RSNA-Pneumonia zero-shot inference using the pretrained RSNA task head
        and convert outputs to binary logits for pneumonia.
        """
        rsna_head_index = ARK_PRETRAINED_TASKS["RSNA-Pneumonia"]["head_index"]
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Zero-shot inference (RSNA task head)...")):
            images = batch["pixel_values"]
            labels = batch["labels"]
            image_ids = batch["sample_ids"]

            raw_logits = self.predict(images, head_index=rsna_head_index)
            logits = rsna_multiclass_logits_to_binary_logits(raw_logits)
            evaluation_processor.add_batch_results(image_ids, labels, logits)

        metrics = evaluation_processor.process_and_save_results()
        return metrics
    
def get_args_parser() -> argparse.ArgumentParser:
    """Create argument parser for zero-shot inference script"""
    parser = argparse.ArgumentParser(description="Zero-shot inference")
    parser.add_argument('--task', type=str, required=True, 
                       choices=['multilabel', 'multiclass', 'binary'])
    parser.add_argument('--model', type=str, required=True, 
                       choices=['medsiglip', 'ark']) 
    parser.add_argument('--data', type=str, required=True, 
                       choices=['VinDr-CXR', 'RSNA-Pneumonia', 'TAIX-Ray'])
    parser.add_argument('--output-path', type=str, required=True,
                       help="Output directory for results")
    parser.add_argument('--batch-size', type=int, default=16,
                       help="Batch size for processing")
    parser.add_argument('--device', type=str, default="cuda",
                       choices=['cuda', 'cpu'],
                       help="Device to run zero-shot inference on")
    parser.add_argument('--custom-text-prompts', type=str, default=None,
                       help="Path to JSON file with custom text prompts. This flag must be specified with `--model medsiglip`.")
    parser.add_argument('--ark-checkpoint-path', type=str, default=None,
                       help="Path to Ark checkpoint file. This flag must be specified with `--model ark`.")
    parser.add_argument('--use-rsna-head', action='store_true', default=False,
                       help="Use pretrained RSNA task head and convert to binary task. This flag only works when `--model ark --data RSNA-Pneumonia --task binary`.")
    return parser

def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments"""
    if args.model == "medsiglip":
        if args.custom_text_prompts is None:
            raise ValueError("Custom text prompts must be specified with `--model medsiglip`.")
        if not os.path.exists(args.custom_text_prompts):
            raise ValueError(f"Custom text prompts file does not exist: {args.custom_text_prompts}")
    if args.model == "ark":
        if args.ark_checkpoint_path is None:
            raise ValueError("Ark checkpoint path must be specified with `--model ark`.")
        if not os.path.exists(args.ark_checkpoint_path):
            raise ValueError(f"Ark checkpoint file does not exist: {args.ark_checkpoint_path}")
        if args.use_rsna_head:
            if args.data != "RSNA-Pneumonia" or args.task != "binary":
                raise ValueError("--use-rsna-head requires --data RSNA-Pneumonia and --task binary.")
        
def get_text_prompts(prompt_file: str, dataset: str, task: str) -> List[str]:
    """
    Load custom text prompts from JSON file and extract prompts for the given dataset and task.
    
    Args:
        prompt_file: Path to JSON file containing prompts
        dataset: Dataset name
        task: Classification task type
        
    Returns:
        List of text prompts
    """
    with open(prompt_file, 'r') as f:
        prompts = json.load(f)
    return prompts.get(dataset, {}).get(task, [])

def create_output_directories(output_base_dir: str, accelerator: Accelerator) -> OutputPaths:
    """Create output directories and return paths for zero-shot inference"""
    if accelerator.is_main_process:
        os.makedirs(f"{output_base_dir}/figs", exist_ok=True)
        os.makedirs(f"{output_base_dir}/table", exist_ok=True)
    
    return OutputPaths(
        base=output_base_dir,
        figs=f"{output_base_dir}/figs",
        table=f"{output_base_dir}/table",
        gradcam=None,  
        attention=None,  
        lrp=None
    )

def main():
    """Main function for zero-shot inference"""
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Validate args
    validate_args(args)
    
    # Setup accelerator
    accelerator = Accelerator()
    
    # Create output directories
    output_base_dir = os.path.join(args.output_path, args.data, args.model)
    output_paths = create_output_directories(output_base_dir, accelerator)
    
    # Initialize classifier based on model type
    if args.model == "ark":
        # Initialize Ark classifier
        classifier = ArkZeroShotClassifier(
            checkpoint_path=args.ark_checkpoint_path,
            device=args.device
        )
        text_prompts = None  
        
    elif args.model == "medsiglip":
        # Initialize MedSigLIP classifier
        classifier = MedSigLIPZeroShotClassifier(
            device=args.device
        )
        # Load custom prompts if provided
        text_prompts = get_text_prompts(args.custom_text_prompts, args.data, args.task)
        if len(text_prompts) == 0:
            raise ValueError(f"No prompts available for dataset '{args.data}' and task '{args.task}'")
        
    else:
        raise ValueError(f"Model '{args.model}' is not supported for zero-shot inference")
    
    # Setup configs
    data_config, _ = setup_configs(args.data, args.task)
    
    # Get data root folder from config
    data_root_folder = data_config.get_data_root_folder(False)  # No multi-view for zero-shot
    
    # Setup dataset
    # For MedSigLIP, use AutoImageProcessor
    # For Ark, use torchvision.transforms.Compose
    if args.model == "ark":
        model_name = None
        _, test_transforms = get_transforms(model_name=args.model)
    else:
        model_name = args.model
        test_transforms = None
    
    test_loader = create_test_loader(
        data_root_folder=data_root_folder,
        task=args.task,
        test_transforms=test_transforms,
        batch_size=args.batch_size,
        multi_view=False,
        model_name=model_name
    )
    dataset = test_loader.dataset
    
    # Prepare the data loader with accelerator
    test_loader = accelerator.prepare(test_loader)
    
    # Determine class labels for evaluation processor
    if args.task == "binary":
        class_labels = None
    elif args.task == "multiclass":
        raw_class_labels = list(set(dataset.labels))
        class_labels = class_labels_mapping(args.data, raw_class_labels)
    else:
        class_labels = dataset.labels

    # Initialize evaluation processor
    evaluation_processor = EvaluationProcessor(
        accelerator, output_paths, args.task, class_labels
    )
        
    # Run zero-shot inference
    logger.info(f"Starting zero-shot inference on {len(dataset)} samples")
    if args.model == "medsiglip":
        results = classifier.run_zero_shot_inference(test_loader, evaluation_processor, text_prompts)
    elif args.model == "ark":
        if args.use_rsna_head:
            results = classifier.run_rsna_head_zero_shot_inference(
                test_loader,
                evaluation_processor,
            )
        else:
            results = classifier.run_zero_shot_inference(
                test_loader,
                evaluation_processor,
                dataset_name=args.data,
                task_type=args.task,
                downstream_target_labels=class_labels,
            )
    logger.info(f"Zero-shot inference with dataset {args.data} using pretrained {args.model} completed!")
    logger.info(f"Saved per-class metrics and curves under {output_paths.table} and {output_paths.figs}")
    logger.info(f"Evaluation results: {results}")
    
if __name__ == "__main__":
    main()

