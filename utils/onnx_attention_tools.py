import onnx
import logging
from rad_dino.loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)



def _dim_to_value(dim):
    """Convert an ONNX Dimension to an int or str."""
    if hasattr(dim, 'dim_value'):
        return dim.dim_value
    elif hasattr(dim, 'dim_param'):
        return dim.dim_param
    else:
        return None


def _get_value_info_shape(model, tensor_name):
    """Get the shape of a tensor from model info."""
    for value_info in model.graph.value_info:
        if value_info.name == tensor_name:
            return [_dim_to_value(dim) for dim in value_info.type.tensor_type.shape.dim]
    
    # Check inputs
    for input_info in model.graph.input:
        if input_info.name == tensor_name:
            return [_dim_to_value(dim) for dim in input_info.type.tensor_type.shape.dim]
    
    # Check outputs
    for output_info in model.graph.output:
        if output_info.name == tensor_name:
            return [_dim_to_value(dim) for dim in output_info.type.tensor_type.shape.dim]
    
    return None


def find_attention_softmax_nodes(model):
    """
    Heuristically find Softmax nodes that likely correspond to attention matrices.
    
    Args:
        model: ONNX model
        
    Returns:
        List of (tensor_name, node_name) tuples for attention softmax outputs
    """
    attention_softmax_nodes = []
    
    for node in model.graph.node:
        if node.op_type == 'Softmax':
            if len(node.output) == 0:
                continue
                
            output_name = node.output[0]
            shape = _get_value_info_shape(model, output_name)
            
            if shape is None:
                continue
                
            # Attention matrices typically have rank >= 3 and last two dimensions are equal
            if len(shape) >= 3:
                # Check if last two dimensions are equal (common in attention)
                if shape[-1] == shape[-2]:
                    attention_softmax_nodes.append((output_name, node.name or 'unnamed_softmax'))
                # Also check for cases where dimensions might be symbolic but equal
                elif shape[-1] is not None and shape[-2] is not None and str(shape[-1]) == str(shape[-2]):
                    attention_softmax_nodes.append((output_name, node.name or 'unnamed_softmax'))
    
    return attention_softmax_nodes


def _build_graph_indices(model):
    """Build producer and consumer maps for tensor names."""
    producers = {}  # tensor_name -> node_index
    consumers = {}  # tensor_name -> set of node_indices
    
    for i, node in enumerate(model.graph.node):
        # Record producers
        for output_name in node.output:
            producers[output_name] = i
        
        # Record consumers
        for input_name in node.input:
            if input_name not in consumers:
                consumers[input_name] = set()
            consumers[input_name].add(i)
    
    return producers, consumers


def _follow_simple_ops(consumers, start_tensor, max_depth=3):
    """
    Follow simple pass-through operations (Transpose, Reshape, Identity, Dropout, Cast)
    to collect reachable tensor names.
    """
    reachable = set()
    to_visit = [(start_tensor, 0)]
    
    while to_visit:
        current_tensor, depth = to_visit.pop(0)
        
        if depth >= max_depth or current_tensor in reachable:
            continue
            
        reachable.add(current_tensor)
        
        if current_tensor not in consumers:
            continue
            
        for node_idx in consumers[current_tensor]:
            # We need to get the actual node to check its type
            # For now, we'll assume it's a simple op and add its outputs
            # This is a simplified version - in practice you'd want to check the actual node type
            pass
    
    return reachable


def find_post_softmax_matmul_nodes(model, prefer_concat=False):
    """
    Find MatMul nodes that come after Softmax operations.
    
    Args:
        model: ONNX model
        prefer_concat: If True, prefer to return Concat nodes that come after the MatMul
    
    Returns:
        List of (tensor_name, node_name) tuples for post-softmax MatMul outputs
    """
    # Build producer and consumer maps
    producers, consumers = _build_graph_indices(model)
    
    # Find all Softmax nodes
    softmax_nodes = []
    for node in model.graph.node:
        if node.op_type == 'Softmax':
            softmax_nodes.append(node)
    
    post_softmax_matmul_nodes = []
    
    for softmax_node in softmax_nodes:
        if len(softmax_node.output) == 0:
            continue
            
        softmax_output = softmax_node.output[0]
        
        # Look for MatMul operations that directly consume softmax outputs
        if softmax_output in consumers:
            for node_idx in consumers[softmax_output]:
                consumer_node = model.graph.node[node_idx]
                if consumer_node.op_type == 'MatMul':
                    if len(consumer_node.output) > 0:
                        matmul_output = consumer_node.output[0]
                        
                        if prefer_concat:
                            # Look for downstream Concat nodes
                            concat_nodes = _follow_simple_ops(consumers, matmul_output, max_depth=3)
                            for concat_name in concat_nodes:
                                # Check if this is actually a Concat node
                                for node in model.graph.node:
                                    if node.op_type == 'Concat' and node.output[0] == concat_name:
                                        post_softmax_matmul_nodes.append((concat_name, node.name or 'unnamed_concat'))
                                        break
                            else:
                                # No Concat found, use MatMul output
                                post_softmax_matmul_nodes.append((matmul_output, consumer_node.name or 'unnamed_matmul'))
                        else:
                            post_softmax_matmul_nodes.append((matmul_output, consumer_node.name or 'unnamed_matmul'))
    
    return post_softmax_matmul_nodes


def augment_onnx_add_attention_outputs(input_onnx_path, output_onnx_path=None, mode='softmax'):
    """
    Augment an ONNX model to add attention-related tensors as outputs.
    
    Args:
        input_onnx_path: Path to input ONNX model
        output_onnx_path: Path to save augmented ONNX model. If None, the model is only returned without saving.
        mode: Extraction mode ('softmax', 'matmul', or 'concat')
    
    Returns:
        List of added output names
    """
    # Load the model
    model = onnx.load(input_onnx_path)
    
    # Infer shapes if needed
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        logger.warning(f"Warning: Could not infer shapes: {e}")
    
    # Find candidate attention tensors based on mode
    if mode == 'softmax':
        candidate_tensors = find_attention_softmax_nodes(model)
    elif mode == 'matmul':
        candidate_tensors = find_post_softmax_matmul_nodes(model, prefer_concat=False)
    elif mode == 'concat':
        candidate_tensors = find_post_softmax_matmul_nodes(model, prefer_concat=True)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'softmax', 'matmul', or 'concat'")
    
    if not candidate_tensors:
        logger.warning(f"No attention tensors found for mode '{mode}'")
        return []
    
    logger.info(f"Found {len(candidate_tensors)} attention tensors for mode '{mode}'")
    
    # Add tensors as outputs
    added_outputs = []
    for tensor_name, node_name in candidate_tensors:
        # Get tensor info
        tensor_info = None
        
        # Check value_info first
        for value_info in model.graph.value_info:
            if value_info.name == tensor_name:
                tensor_info = value_info
                break
        
        # If not found in value_info, check if it's an input or output
        if tensor_info is None:
            for input_info in model.graph.input:
                if input_info.name == tensor_name:
                    tensor_info = input_info
                    break
        
        if tensor_info is None:
            for output_info in model.graph.output:
                if output_info.name == tensor_name:
                    tensor_info = output_info
                    break
        
        if tensor_info is None:
            print(f"Warning: Could not find tensor info for {tensor_name}")
            continue
        
        # Create new output
        new_output = onnx.ValueInfoProto()
        new_output.name = tensor_name
        new_output.type.CopyFrom(tensor_info.type)
        
        # Check if this output already exists
        if not any(output.name == tensor_name for output in model.graph.output):
            model.graph.output.append(new_output)
            added_outputs.append(tensor_name)
            logger.info(f"Added output: {tensor_name} (from {node_name})")
        else:
            logger.info(f"Output already exists: {tensor_name}")
    
    if output_onnx_path is not None:
        # Save the augmented model
        onnx.save(model, output_onnx_path)
        logger.info(f"Augmented onnx model saved to: {output_onnx_path}")
    
    return added_outputs



