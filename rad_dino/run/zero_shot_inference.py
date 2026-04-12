import os
import argparse
import torch
import torch.nn.functional as F
import logging
import json
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv, find_dotenv

from rad_dino.loggings.setup import init_logging
from rad_dino.data.data_loader import create_test_loader
from rad_dino.utils.config_utils import setup_configs
from rad_dino.eval.evaluation_processor import EvaluationProcessor
from rad_dino.configs.config import OutputPaths
from rad_dino.models.ark import load_pretrained_ark_model
from rad_dino.models.medimageinsight import load_medimageinsight_model
from rad_dino.models.biomedclip import load_biomedclip_model, get_biomedclip_tokenizer
from rad_dino.utils.transforms import get_transforms
from rad_dino.data.label_mapping import class_labels_mapping
from rad_dino.configs.ark_zero_shot_config import ARK_PRETRAINED_TASKS
from rad_dino.utils.zero_shot_transfer.ark_zero_shot_postprocess import build_target_to_pretrained_ark_indices, aggregate_targeted_pred_probs
from rad_dino.utils.zero_shot_transfer.rsna_postprocess import rsna_multiclass_logits_to_binary_logits

load_dotenv(find_dotenv())

init_logging()
logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_MEDIMAGEINSIGHT_PATH = os.path.normpath(os.path.join(CURR_DIR, "..", "models", "MedImageInsights"))

# VLM zero-shot classifiers
class BaseVLMZeroShotClassifier(ABC):
    """
    Abstract base for CLIP-style VLM zero-shot classifiers.
    """

    def __init__(self, device: str = "cuda"):
        if not torch.cuda.is_available() and device == "cuda":
            logging.warning("CUDA is not available. Using CPU for inference.")
            device = "cpu"
        self.device = torch.device(device)

    @abstractmethod
    def predict(self, images: torch.Tensor, text_prompts: List[str]) -> torch.Tensor:
        """
        Compute image-text similarity logits for a single batch.

        Args:
            images: Pre-processed image tensors ``[B, C, H, W]``.
            text_prompts: List of text prompts for classification.

        Returns:
            logits_per_image: ``[B, num_prompts]``
        """
        ...

    @abstractmethod
    def encode_image(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images into L2-normalised embeddings.

        Args:
            images: ``[B, C, H, W]``

        Returns:
            ``(image_features [B, D], logit_scale scalar)``
        """
        ...

    @abstractmethod
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """
        Encode text prompts into L2-normalised embeddings.

        Args:
            text_prompts: List of text strings.

        Returns:
            text_features: ``[P, D]``
        """
        ...

    def _compute_multiview_logits(
        self,
        images: torch.Tensor,
        text_features: torch.Tensor,
        num_views: int,
    ) -> torch.Tensor:
        """
        Embedding-level mean fusion for multi-view images.

        1. Flatten [B, V, C, H, W] -> [B*V, C, H, W]
        2. Encode each view -> [B*V, D]
        3. Reshape -> [B, V, D], average-pool -> [B, D] -> L2-normalise
        4. Compute scale * img @ text.T -> [B, P]
        """
        batch_size = images.shape[0]
        flat = images.view(batch_size * num_views, *images.shape[2:])
        img_feats, logit_scale = self.encode_image(flat)           # [B*V, D]
        img_feats = img_feats.view(batch_size, num_views, -1)      # [B, V, D]
        img_feats = img_feats.mean(dim=1)                          # [B, D]
        img_feats = F.normalize(img_feats, p=2, dim=1)
        return logit_scale * img_feats @ text_features.t()         # [B, P]

    def run_zero_shot_inference(
        self,
        data_loader: DataLoader,
        evaluation_processor: EvaluationProcessor,
        text_prompts: Optional[Union[List[str], Dict[str, Any]]] = None,
        multi_view: bool = False,
        num_views: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run zero-shot inference over the full dataset.

        When ``multi_view=True``, per-view image embeddings are average-pooled before comparing to text embeddings (embedding-level fusion).

        For multilabel tasks with paired templates, 
        ``text_prompts`` should be a dict with keys ``positive_template``, ``negative_template``, and ``class_names``. 
        For each class the model computes softmax between the positive and negative prompt to obtain a calibrated probability.

        For binary tasks, ``text_prompts`` should be a list of two prompts [negative, positive] (ordered negative-first, positive-last).  

        For multiclass tasks, ``text_prompts`` should be a list of prompts per class. 
        Then, ``EvaluationProcessor`` applies softmax across all classes to obtain the class probabilities.
        """
        task = evaluation_processor.task
        is_multilabel = task == "multilabel"
        is_binary = task == "binary"

        # Validate prompt format matches task type
        if is_multilabel:
            if not isinstance(text_prompts, dict):
                raise ValueError(
                    "Multilabel task requires text_prompts to be a dict with keys "
                    "'positive_template', 'negative_template', and 'class_names'. "
                    f"Got {type(text_prompts).__name__} instead."
                )
            for key in ("positive_template", "negative_template", "class_names"):
                if key not in text_prompts:
                    raise ValueError(
                        f"Multilabel paired-template dict is missing required key '{key}'. "
                        f"Available keys: {list(text_prompts.keys())}"
                    )
            class_names = text_prompts["class_names"]
            pos_prompts = [text_prompts["positive_template"].format(c) for c in class_names]
            neg_prompts = [text_prompts["negative_template"].format(c) for c in class_names]
            all_prompts = pos_prompts + neg_prompts   # length 2N
            n_classes = len(class_names)
            logger.info(
                "Using paired-template zero-shot inference for multilabel task with %d class labels: %s",
                n_classes, class_names,
            )
        if is_binary:
            if not isinstance(text_prompts, list) or len(text_prompts) < 2:
                raise ValueError(
                    "Binary task requires text_prompts to be a list of at least 2 prompts "
                    "[negative, positive]. "
                    f"Got {type(text_prompts).__name__} with length {len(text_prompts) if isinstance(text_prompts, list) else 'N/A'}."
                )

        # Pre-encode text once for multi-view (avoids repeated tokenization)
        multi_view_text_features = None
        if multi_view:
            if is_multilabel:
                multi_view_text_features = self.encode_text(all_prompts)
            else:
                multi_view_text_features = self.encode_text(text_prompts)
            logger.info(
                "Multi-view enabled: %d views, embedding-level mean fusion", num_views
            )

        for batch in tqdm(data_loader, desc="Zero-shot inference..."):
            images = batch["pixel_values"]
            labels = batch["labels"]
            image_ids = batch["sample_ids"]

            if is_multilabel:
                # Compute similarities to all 2N prompts at once: [B, 2N]
                if multi_view:
                    logits_all = self._compute_multiview_logits(
                        images, multi_view_text_features, num_views
                    )
                else:
                    logits_all = self.predict(images, all_prompts)
                pos_logits = logits_all[:, :n_classes]   # [B, N]
                neg_logits = logits_all[:, n_classes:]   # [B, N]

                # Paired softmax per class softmax([neg, pos]) and extract P(positive)
                paired = torch.stack([neg_logits, pos_logits], dim=-1)  # [B, N, 2]
                probs = torch.softmax(paired, dim=-1)[:, :, 1]         # [B, N]

                evaluation_processor.add_batch_results(
                    image_ids, labels, logits=logits_all, probs=probs
                )
            elif is_binary:
                if multi_view:
                    logits = self._compute_multiview_logits(
                        images, multi_view_text_features, num_views
                    )
                else:
                    logits = self.predict(images, text_prompts)

                probs = torch.softmax(logits, dim=1)
                prob_pos = probs[:, -1].unsqueeze(1)  # [B, 1]
                evaluation_processor.add_batch_results(
                    image_ids, labels, logits=logits, probs=prob_pos
                )
            else:
                if multi_view:
                    logits = self._compute_multiview_logits(
                        images, multi_view_text_features, num_views
                    )
                else:
                    logits = self.predict(images, text_prompts)
                evaluation_processor.add_batch_results(image_ids, labels, logits)

        metrics = evaluation_processor.process_and_save_results()
        return metrics

class MedSigLIPZeroShotClassifier(BaseVLMZeroShotClassifier):
    """Zero-shot classifier using MedSigLIP (HuggingFace SigLIP)."""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.tokenizer = AutoTokenizer.from_pretrained("google/medsiglip-448")
        self.model = AutoModel.from_pretrained("google/medsiglip-448")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, images: torch.Tensor, text_prompts: List[str]) -> torch.Tensor:
        text_inputs = self.tokenizer(
            text_prompts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        images = images.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                pixel_values=images,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
        return outputs.logits_per_image

    def encode_image(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        images = images.to(self.device)
        with torch.no_grad():
            feats = self.model.get_image_features(pixel_values=images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            scale = self.model.logit_scale.exp()
        return feats, scale

    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        text_inputs = self.tokenizer(
            text_prompts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            feats = self.model.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

class MedImageInsightZeroShotClassifier(BaseVLMZeroShotClassifier):
    """Zero-shot classifier using MedImageInsight (UniCL two-tower model)."""

    def __init__(self, model_dir: str, device: str = "cuda"):
        super().__init__(device)
        self.model = load_medimageinsight_model(model_dir, device=device)
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        self.max_length = self.model.conf_lang_encoder["CONTEXT_LENGTH"]

    def _tokenize(self, text_prompts: List[str]) -> dict:
        """Tokenize text prompts and move tensors to device."""
        tokens = self.tokenizer(
            text_prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in tokens.items()}

    def predict(self, images: torch.Tensor, text_prompts: List[str]) -> torch.Tensor:
        text_tokens = self._tokenize(text_prompts)
        images = images.to(self.device)

        with torch.no_grad():
            image_features, text_features, temperature = self.model(
                image=images, text=text_tokens
            )
            logits_per_image = image_features @ text_features.t() * temperature
        return logits_per_image

    def encode_image(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        images = images.to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(images, norm=True)
            scale = self.model.logit_scale.exp()
        return feats, scale

    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        text_tokens = self._tokenize(text_prompts)
        with torch.no_grad():
            feats = self.model.encode_text(text_tokens, norm=True)
        return feats

class BiomedCLIPZeroShotClassifier(BaseVLMZeroShotClassifier):
    """Zero-shot classifier using BiomedCLIP (open_clip ViT-B/16 + PubMedBERT)."""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.model, _ = load_biomedclip_model(device=device)
        self.model.eval()
        self.tokenizer = get_biomedclip_tokenizer()
        self.context_length = 256  # BiomedCLIP default context length

    def predict(self, images: torch.Tensor, text_prompts: List[str]) -> torch.Tensor:
        texts = self.tokenizer(
            text_prompts, context_length=self.context_length
        ).to(self.device)
        images = images.to(self.device)

        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(images, texts)
            logits_per_image = logit_scale * image_features @ text_features.t()
        return logits_per_image

    def encode_image(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        images = images.to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            scale = self.model.logit_scale.exp()
        return feats, scale

    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        texts = self.tokenizer(
            text_prompts, context_length=self.context_length
        ).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(texts)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

# Ark zero-shot classifier
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
        self.model = load_pretrained_ark_model(
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
            logits: Tensor of shape [B, total_pretrained_classes]
        """
        images = images.to(self.device)
        with torch.no_grad():
            pre_logits, _ = self.model(images) 
            logits = torch.cat(pre_logits, dim=1)
        return logits

    def predict_all_heads_multiview(self, images: torch.Tensor, num_views: int) -> torch.Tensor:
        """
        Multi-view embedding-level mean fusion through Ark's backbone and task heads.

        1. Flatten ``[B, V, C, H, W]`` -> ``[B*V, C, H, W]``
        2. ``forward_features`` + ``projector`` -> ``[B*V, D]``
        3. Reshape ``[B, V, D]``, mean-pool -> ``[B, D]``
        4. Pass fused embedding through each ``omni_head`` -> concatenated logits

        Returns:
            logits: ``[B, total_pretrained_classes]``
        """
        batch_size = images.shape[0]
        flat = images.view(batch_size * num_views, *images.shape[2:]).to(self.device)
        with torch.no_grad():
            x = self.model.forward_features(flat)        # [B*V, encoder_dim]
            x = self.model.projector(x)                  # [B*V, proj_dim]
            x = x.view(batch_size, num_views, -1)        # [B, V, proj_dim]
            x = x.mean(dim=1)                            # [B, proj_dim]
            head_outputs = [head(x) for head in self.model.omni_heads]
            logits = torch.cat(head_outputs, dim=1)      # [B, total_classes]
        return logits

    def run_zero_shot_inference(self, 
                                data_loader: DataLoader, 
                                evaluation_processor: EvaluationProcessor,
                                dataset_name: str,
                                task_type: str,
                                downstream_target_labels: Optional[List[str]],
                                multi_view: bool = False,
                                num_views: Optional[int] = None) -> Dict[str, Any]:
        """
        Run zero-shot inference on the dataset.

        When ``multi_view=True``, per-view embeddings are average-pooled before
        being passed through pretrained task heads (embedding-level fusion).
        """
        self._target_to_pretrained_label_indices = build_target_to_pretrained_ark_indices(
            dataset_name, task_type, downstream_target_labels
        )
        if multi_view:
            logger.info("Multi-view enabled: %d views, embedding-level mean fusion", num_views)

        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Zero-shot inference...")):
            images = batch["pixel_values"]
            labels = batch["labels"]
            image_ids = batch["sample_ids"]
            
            if multi_view:
                head_logits = self.predict_all_heads_multiview(images, num_views)
            else:
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
                evaluation_processor.add_batch_results(
                    image_ids, labels, logits=torch.zeros_like(prob_norm), probs=prob_norm
                )
            else:
                # Binary/multilabel: pass aggregated probs directly
                evaluation_processor.add_batch_results(
                    image_ids, labels, logits=torch.zeros_like(agg_probs), probs=agg_probs
                )

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
                       choices=['medsiglip', 'ark', 'medimageinsight', 'biomedclip']) 
    parser.add_argument('--data', type=str, required=True, 
                       choices=['TAIX-Ray', 'VinDr-Mammo', 'VinDr-SpineXR', 'TBX11K', 'NODE21'])
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
    parser.add_argument('--medimageinsight-path', type=str, default=DEFAULT_MEDIMAGEINSIGHT_PATH,
                       help="Path to the cloned lion-ai/MedImageInsights repository. "
                            "This flag must be specified with `--model medimageinsight`.")
    parser.add_argument('--use-rsna-head', action='store_true', default=False,
                       help="Use pretrained RSNA task head and convert to binary task. This flag only works when `--model ark --data RSNA-Pneumonia --task binary`.")
    parser.add_argument('--multi-view', action='store_true', default=False,
                       help="Enable multi-view processing for mammography data. "
                            "Image embeddings from all views are average pooled before comparison with text embeddings.")
    return parser

def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments"""
    if args.model in ("medsiglip", "medimageinsight", "biomedclip"):
        if args.custom_text_prompts is None:
            raise ValueError(f"Custom text prompts must be specified with `--model {args.model}`.")
        if not os.path.exists(args.custom_text_prompts):
            raise ValueError(f"Custom text prompts file does not exist: {args.custom_text_prompts}")
    if args.model == "medimageinsight":
        if not os.path.isdir(args.medimageinsight_path):
            raise ValueError(
                f"MedImageInsight repo not found at '{args.medimageinsight_path}'. "
                "Clone it with: git lfs install && git clone https://huggingface.co/lion-ai/MedImageInsights"
            )
    if args.model == "ark":
        if args.ark_checkpoint_path is None:
            raise ValueError("Ark checkpoint path must be specified with `--model ark`.")
        if not os.path.exists(args.ark_checkpoint_path):
            raise ValueError(f"Ark checkpoint file does not exist: {args.ark_checkpoint_path}")
        if args.use_rsna_head:
            if args.data != "RSNA-Pneumonia" or args.task != "binary":
                raise ValueError("--use-rsna-head requires --data RSNA-Pneumonia and --task binary.")
        if args.data in ("VinDr-Mammo", "VinDr-SpineXR"):
            raise ValueError(
                f"Ark's pretrained task heads are trained exclusively on chest X-ray "
                f"datasets and cannot be used for zero-shot transfer on '{args.data}' "
                f"(out-of-distribution). Use a VLM model instead."
            )
    if args.multi_view and args.data != "VinDr-Mammo":
        raise ValueError("--multi-view is only supported for VinDr-Mammo.")
        
def get_text_prompts(prompt_file: str, dataset: str, task: str) -> Union[List[str], Dict[str, Any]]:
    """
    Load custom text prompts from JSON file and extract prompts for the given dataset and task.
    
    For multilabel tasks with paired templates, returns a dict with keys:
    ``positive_template``, ``negative_template``, ``class_names``.
    For binary/multiclass tasks, returns a flat list of text prompts.
    
    Args:
        prompt_file: Path to JSON file containing prompts
        dataset: Dataset name
        task: Classification task type
        
    Returns:
        Dict (paired templates for multilabel) or List[str] (binary/multiclass)
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
        text_prompts = get_text_prompts(args.custom_text_prompts, args.data, args.task)
    
    elif args.model == "medimageinsight":
        # Initialize MedImageInsight zero-shot classifier
        classifier = MedImageInsightZeroShotClassifier(
            model_dir=args.medimageinsight_path,
            device=args.device
        )
        text_prompts = get_text_prompts(args.custom_text_prompts, args.data, args.task)

    elif args.model == "biomedclip":
        # Initialize BiomedCLIP zero-shot classifier
        classifier = BiomedCLIPZeroShotClassifier(
            device=args.device
        )
        text_prompts = get_text_prompts(args.custom_text_prompts, args.data, args.task)

    else:
        raise ValueError(f"Model '{args.model}' is not supported for zero-shot inference")
    
    # Setup configs
    data_config, _ = setup_configs(args.data, args.task)
    
    # Get data root folder from config
    data_root_folder = data_config.get_data_root_folder(args.multi_view)
    # If VinDr-Mammo binary, route to binary preprocessed folder if available
    if args.data == "VinDr-Mammo" and args.task == "binary":
        candidate_path = data_root_folder.replace("/birads/", "/binary/")
        if candidate_path != data_root_folder and os.path.exists(candidate_path):
            data_root_folder = candidate_path
            logger.info(f"Using VinDr-Mammo binary preprocessed data at: {data_root_folder}")
    
    # Setup dataset
    # For MedSigLIP, use AutoImageProcessor
    # For Ark, MedImageInsight, and BiomedCLIP, use torchvision.transforms.Compose
    if args.model in ("ark", "medimageinsight", "biomedclip"):
        model_name = None
        _, test_transforms = get_transforms(model_name=args.model)
    else:
        model_name = args.model
        test_transforms = None
    
    # Resolve multi-view config
    multi_view_config = data_config.get_multi_view_config(args.multi_view)
    num_views = multi_view_config.num_views if multi_view_config else None

    test_loader = create_test_loader(
        data_root_folder=data_root_folder,
        task=args.task,
        test_transforms=test_transforms,
        batch_size=args.batch_size,
        multi_view=args.multi_view,
        model_name=model_name
    )
    dataset = test_loader.dataset
    
    # Prepare the data loader with accelerator
    test_loader = accelerator.prepare(test_loader)
    
    # Determine class labels for evaluation processor
    if args.task == "binary":
        class_labels = None
    elif args.task == "multiclass":
        # Keep deterministic class-index order aligned with numeric dataset labels.
        raw_class_labels = sorted(set(dataset.labels))
        class_labels = class_labels_mapping(args.data, raw_class_labels)
    else:
        class_labels = dataset.labels

    # Validate text prompts for VLM models
    if args.model in ("medsiglip", "medimageinsight", "biomedclip"):
        if isinstance(text_prompts, dict):
            # Paired-template mode (multilabel): validate required keys and class count
            for key in ("positive_template", "negative_template", "class_names"):
                if key not in text_prompts:
                    raise ValueError(
                        f"Paired-template prompts for '{args.data}/{args.task}' missing required key '{key}'"
                    )
            n_prompts = len(text_prompts["class_names"])
            n_labels = len(class_labels) if class_labels else 0
            if n_prompts != n_labels:
                raise ValueError(
                    f"Number of class names in text prompts ({n_prompts}) does not match "
                    f"number of dataset class labels ({n_labels}). "
                    f"Prompt class names: {text_prompts['class_names']}, "
                    f"Dataset labels: {class_labels}"
                )
        elif isinstance(text_prompts, list):
            if len(text_prompts) == 0:
                raise ValueError(f"No prompts available for dataset '{args.data}' and task '{args.task}'")
            if args.task == "multiclass" and class_labels is not None:
                if len(text_prompts) != len(class_labels):
                    raise ValueError(
                        f"Number of text prompts ({len(text_prompts)}) does not match "
                        f"number of dataset class labels ({len(class_labels)}) for "
                        f"multiclass task on '{args.data}'. "
                        f"Prompts: {text_prompts}, Class labels: {class_labels}"
                    )
        else:
            raise ValueError(f"Unexpected text_prompts type: {type(text_prompts)}")

    # Initialize evaluation processor
    evaluation_processor = EvaluationProcessor(
        accelerator, output_paths, args.task, class_labels
    )
        
    # Run zero-shot inference
    logger.info(f"Starting zero-shot inference on {len(dataset)} samples")
    if args.model in ("medsiglip", "medimageinsight", "biomedclip"):
        results = classifier.run_zero_shot_inference(
            test_loader, evaluation_processor, text_prompts,
            multi_view=args.multi_view, num_views=num_views,
        )
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
                multi_view=args.multi_view,
                num_views=num_views,
            )
    logger.info(f"Zero-shot inference with dataset {args.data} using pretrained {args.model} completed!")
    logger.info(f"Saved per-class metrics and curves under {output_paths.table} and {output_paths.figs}")
    logger.info(f"Evaluation results: {results}")
    
if __name__ == "__main__":
    main()

