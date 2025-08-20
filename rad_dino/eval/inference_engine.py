import torch
import numpy as np
import onnxruntime
from typing import Optional, List, Tuple, Any
from accelerate import Accelerator
from rad_dino.models.dino import DinoClassifier
from rad_dino.models.siglip import MedSigClassifier
from rad_dino.models.ark import ArkClassifier
import logging

logger = logging.getLogger(__name__)

def _calculate_attention_shape(backbone_config: Any, 
                               images: torch.Tensor, 
                               multi_view: bool, 
                               num_layers: int, 
                               num_heads: int) -> Tuple[torch.Size, int]:
    """Calculate attention tensor shape and sequence length"""
    if multi_view and images.shape[1] == 4:
        # Multi-view: images shape is [B, 4, C, H, W]
        img_height, img_width = images.shape[3], images.shape[4]
    else:
        # Single-view: images shape is [B, C, H, W]
        img_height, img_width = images.shape[2], images.shape[3]
    
    patch_size = backbone_config.patch_size
    num_patches_h = img_height // patch_size
    num_patches_w = img_width // patch_size
    seq_len = num_patches_h * num_patches_w + 1
    
    if multi_view and images.shape[1] == 4:
        # Multi-view attention shape: [num_layers, B, 4, num_heads, seq_len, seq_len]
        attention_shape = (num_layers, images.shape[0], 4, num_heads, seq_len, seq_len)
    else:
        # Single-view attention shape: [num_layers, B, num_heads, seq_len, seq_len]
        attention_shape = (num_layers, images.shape[0], num_heads, seq_len, seq_len)
    
    return attention_shape, seq_len

def _validate_input_shape(images: torch.Tensor, multi_view: bool) -> None:
    """Validate input tensor shape for multi-view vs single-view"""
    if multi_view:
        if len(images.shape) != 5 or images.shape[1] != 4:
            raise ValueError(f"Multi-view inference expects images shape [B, 4, C, H, W], got {images.shape}")
        logger.debug(f"Multi-view batch shape: {images.shape}")
    else:
        if len(images.shape) != 4:
            raise ValueError(f"Single-view inference expects images shape [B, C, H, W], got {images.shape}")
        logger.debug(f"Single-view batch shape: {images.shape}")

def _run_onnx_inference(session: onnxruntime.InferenceSession, 
                        input_name: str, 
                        output_names: List[str], 
                        images: torch.Tensor, show_attention: bool, 
                        accelerator: Accelerator, 
                        num_classes: int,
                        backbone_config: Any, 
                        multi_view: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """ONNX inference workflow"""
    batch_size = images.shape[0]
    attentions = None

    # Validate ONNX model input shape compatibility (excluding batch dimension)
    onnx_input_shape = session.get_inputs()[0].shape
    expected_shape = tuple(onnx_input_shape[1:])
    actual_shape = tuple(images.shape[1:])
    if expected_shape != actual_shape:
        raise ValueError(
            f"Shape mismatch: ONNX model expects input shape {expected_shape}, "
            f"but got {actual_shape} from data. "
            "Check if you are using the correct model for single-view or multi-view inference."
        )

    use_gpu = 'CUDAExecutionProvider' in session.get_providers()
    many_attn_outputs = show_attention and len(output_names) > 2

    with torch.no_grad():
        if use_gpu and not many_attn_outputs:
            # GPU with IOBinding
            io_binding = session.io_binding()
            device_id = accelerator.device.index if accelerator.device.index is not None else 0

            # Bind input
            io_binding.bind_input(
                name=input_name,
                device_type='cuda',
                device_id=device_id,
                element_type=np.float32,
                shape=tuple(images.shape),
                buffer_ptr=images.data_ptr()
            )

            # Bind logits output
            logits = torch.empty((batch_size, num_classes), dtype=torch.float32, device=accelerator.device)
            io_binding.bind_output(
                name=output_names[0],
                device_type='cuda',
                device_id=device_id,
                element_type=np.float32,
                shape=(batch_size, num_classes),
                buffer_ptr=logits.data_ptr()
            )

            # Bind attention output if present
            if show_attention and len(output_names) > 1:
                num_layers = backbone_config.num_hidden_layers
                num_heads = backbone_config.num_attention_heads
                attention_shape, _ = _calculate_attention_shape(
                    backbone_config, images, multi_view, num_layers, num_heads
                )
                attentions = torch.empty(attention_shape, dtype=torch.float32, device=accelerator.device)
                io_binding.bind_output(
                    name=output_names[1],
                    device_type='cuda',
                    device_id=device_id,
                    element_type=np.float32,
                    shape=attention_shape,
                    buffer_ptr=attentions.data_ptr()
                )

            session.run_with_iobinding(io_binding)

        else:
            # CPU fallback
            input_data = images.cpu().numpy()
            onnx_outputs = session.run(output_names, {input_name: input_data})
            logits = torch.from_numpy(onnx_outputs[0]).to(accelerator.device)

            if show_attention and len(onnx_outputs) > 1:
                # Stacking multiple attention outputs to [num_layers, batch_size, num_heads, seq_len, seq_len]
                attn_arrays = [out for name, out in zip(output_names, onnx_outputs) if name != output_names[0]]
                torch_tensors: List[torch.Tensor] = []
                for arr in attn_arrays:
                    t = torch.from_numpy(arr).to(accelerator.device)
                    if t.dim() == 3:  # [B, S, S] -> [B, 1, S, S]
                        t = t.unsqueeze(1)
                    torch_tensors.append(t)
                try:
                    attentions = torch.stack(torch_tensors, dim=0)
                except Exception as e:
                    raise Exception(f"Failed to stack attention outputs: {e}.")

    return logits, attentions

def _run_pytorch_inference(model: DinoClassifier | MedSigClassifier | ArkClassifier, 
                           images: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    PyTorch inference workflow
    Returns (logits, attentions, attention_pooling) where attention_pooling is from SigLIP pooler if available.
    """
    with torch.no_grad():
        logits, attentions = model(images)
    # Try to fetch attention pooling weights if present (SigLIP only)
    attention_pooling = getattr(model, 'last_pooler_attn', None)
    return logits, attentions, attention_pooling

class InferenceEngine:
    """Unified inference engine for both ONNX and PyTorch models"""    
    def __init__(self, model_wrapper, accelerator: Accelerator, show_attention: bool):
        self.model_wrapper = model_wrapper
        self.accelerator = accelerator
        self.is_onnx = model_wrapper.model_type == 'onnx'
        self.multi_view = model_wrapper.multi_view
        self.show_attention = show_attention
        
        if self.is_onnx:
            self.session = model_wrapper.session
            self.input_name = model_wrapper.input_name
            self.output_names = model_wrapper.output_names
            self.backbone_config = model_wrapper.config
            
            # Validate ONNX attention outputs
            if show_attention and len(self.output_names) < 2:
                raise ValueError("ONNX model does not have attention outputs, which is required for attention visualization.")
                
            logger.info(f"Running inference with ONNX model (multi_view={self.multi_view})")
        else:
            self.model = model_wrapper.model
            self.backbone_config = model_wrapper.config
            self.model.eval()
            logger.info(f"Running inference with PyTorch model (multi_view={self.multi_view})")
    
    def run_inference(self, images: torch.Tensor, num_classes: int = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run inference with the appropriate backend - pure inference only"""
        # Validate input shapes
        _validate_input_shape(images, self.multi_view)
        
        # Clear CUDA cache before inference
        if self.accelerator.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Run inference based on model type
        if self.is_onnx:
            logits, attentions = _run_onnx_inference(
                self.session, self.input_name, self.output_names, images, self.show_attention,
                self.accelerator, num_classes, self.backbone_config, self.multi_view)
            pooler_attn = None
        else:
            logits, attentions, raw_pooler_attn = _run_pytorch_inference(self.model, images)
            # Reshape pooler attention to [B*V or B, H, 1, N] -> [B, H, N] or [B, V, H, N]
            pooler_attn = None
            if raw_pooler_attn is not None:
                raw_pooler_attn = raw_pooler_attn.squeeze(2) # squeeze query dim
                batch_size = images.shape[0]
                if self.multi_view and images.dim() == 5:
                    num_views = images.shape[1]
                    num_heads = raw_pooler_attn.shape[1]
                    num_tokens = raw_pooler_attn.shape[-1]
                    pooler_attn = raw_pooler_attn.view(batch_size, num_views, num_heads, num_tokens)
                else:
                    pooler_attn = raw_pooler_attn  # [B, H, N]
        
        # Clear CUDA cache after inference
        if self.accelerator.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return logits, attentions, pooler_attn 