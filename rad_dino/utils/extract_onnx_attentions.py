import os
import onnx
import re
from onnx import helper, TensorProto, shape_inference
from typing import Dict, List, Optional, Tuple, Set

def _dim_to_value(dim) -> Optional[int | str]:
    """Convert an ONNX Dimension to an int or str for helper APIs."""
    if dim is None:
        return None
    if dim.dim_value is not None and dim.dim_value != 0:
        return int(dim.dim_value)
    if dim.dim_param:
        return str(dim.dim_param)
    return None


def _get_value_info_shape(model: onnx.ModelProto, tensor_name: str) -> Optional[List[Optional[int | str]]]:
    """Return the shape list for a given tensor name from model value_info, inputs or outputs."""
    def find_vi() -> Optional[onnx.ValueInfoProto]:
        for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
            if vi.name == tensor_name:
                return vi
        return None

    vi = find_vi()
    if vi is None:
        return None
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return None
    return [_dim_to_value(d) for d in tt.shape.dim]

def _downstream_has_matmul(model: onnx.ModelProto, tensor_name: str, max_hops: int = 6) -> bool:
    producer, consumers = _build_graph_indices(model)
    passthrough_ops = {"Div", "Mul", "Add", "Sub", "Transpose", "Reshape", "Identity", "Cast", "Dropout"}
    seen: Set[str] = set([tensor_name])
    frontier: List[Tuple[str, int]] = [(tensor_name, 0)]
    while frontier:
        t, d = frontier.pop(0)
        if d > max_hops:
            continue
        for cons in consumers.get(t, []):
            if cons.op_type == "MatMul":
                return True
            if cons.op_type in passthrough_ops:
                for out in cons.output:
                    if out not in seen:
                        seen.add(out)
                        frontier.append((out, d + 1))
    return False


def find_attention_softmax_nodes(model: onnx.ModelProto) -> List[Tuple[str, List[Optional[int | str]]]]:
    """Find Softmax(QK^T) tensors that represent multi-head attention weights.

    Rules for a valid attention Softmax:
    - Preferably, its input comes (within a few trivial ops) from a MatMul (Q·K^T). If not resolvable, we still allow it.
    - Its output must feed (within a few trivial ops) into a MatMul with V (i.e., context computation).
    - If shape info exists, the last two dims must be equal (seq_len x seq_len).
    """
    candidates: List[Tuple[str, List[Optional[int | str]]]] = []

    for node in model.graph.node:
        if node.op_type != "Softmax" or len(node.output) == 0:
            continue
        out_name = node.output[0]
        softmax_in = node.input[0] if len(node.input) > 0 else None
        if softmax_in is None:
            continue

        # Must lead to Attn·V; upstream MatMul preferred but optional for robustness
        if not _downstream_has_matmul(model, out_name, max_hops=6):
            continue

        shape = _get_value_info_shape(model, out_name) or []
        if shape and len(shape) >= 3:
            last, second_last = shape[-1], shape[-2]
            if last is not None and second_last is not None and last != second_last:
                continue
        candidates.append((out_name, shape))

    # Sort by layer index if present in names (e.g., '/layer.3/...')
    try:
        def layer_key(item: Tuple[str, List[Optional[int | str]]]) -> Tuple[int, str]:
            name = item[0]
            m = re.search(r"/layer\.(\d+)/", name)
            return (int(m.group(1)) if m else 9999, name)
        candidates.sort(key=layer_key)
    except Exception:
        pass

    return candidates


def _build_graph_indices(model: onnx.ModelProto) -> Tuple[Dict[str, onnx.NodeProto], Dict[str, List[onnx.NodeProto]]]:
    """Create producer and consumer maps for tensor names."""
    producer: Dict[str, onnx.NodeProto] = {}
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for out in node.output:
            producer[out] = node
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)
    return producer, consumers

def augment_onnx_add_attention_outputs(
    input_onnx_path: str,
    output_onnx_path: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Expose Softmax(QK^T) tensors as additional graph outputs.

    Returns (saved_model_path, new_output_names)
    """
    if not os.path.isfile(input_onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {input_onnx_path}")

    # Load and run shape inference to annotate shapes
    model = onnx.load(input_onnx_path)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        # Shape inference may fail for some ops; proceed without it
        pass

    # Build a set of existing outputs to avoid duplicates
    existing_outputs = {o.name for o in model.graph.output}

    # Find candidate attention tensors (Softmax only)
    candidates = find_attention_softmax_nodes(model)
    if not candidates:
        # Fallback: directly collect known Softmax node outputs by naming pattern
        fallback = [
            (n.output[0], _get_value_info_shape(model, n.output[0]) or [])
            for n in model.graph.node
            if n.op_type == "Softmax" and len(n.output) > 0 and "/attention/attention/Softmax" in (n.name or "")
        ]
        if not fallback:
            raise RuntimeError("No attention Softmax nodes (QK^T) found in the ONNX graph.")
        candidates = fallback

    new_output_names: List[str] = []
    for tensor_name, shape in candidates:
        if tensor_name in existing_outputs:
            continue
        # Default to FLOAT, unknown dims if shapes unavailable
        if not shape:
            vi = helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, None)
        else:
            vi = helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, shape)
        model.graph.output.extend([vi])
        new_output_names.append(tensor_name)

    # Save model
    if output_onnx_path is None:
        base, ext = os.path.splitext(input_onnx_path)
        output_onnx_path = f"{base}_with_attn{ext}"
    onnx.save(model, output_onnx_path)

    return output_onnx_path, new_output_names


if __name__ == "__main__":
    import argparse
    import onnxruntime
    import numpy as np

    parser = argparse.ArgumentParser(description="Expose attention Softmax (QK^T) tensors as ONNX outputs and run a quick test.")
    parser.add_argument("--onnx-path", required=True, type=str, help="Path to original ONNX model")
    parser.add_argument("--out", required=False, type=str, help="Path to save augmented ONNX model")
    parser.add_argument("--dry-run", action="store_true", help="Only modify graph without running a test inference")
    args = parser.parse_args()

    out_path, new_outputs = augment_onnx_add_attention_outputs(args.onnx_path, args.out)
    print(f"Saved augmented ONNX: {out_path}")
    print(f"Added {len(new_outputs)} attention outputs")

    if not args.dry_run:
        # smoke test: load session and run a 1-image dummy inference
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = onnxruntime.InferenceSession(out_path, providers=providers)
        inp = sess.get_inputs()[0]
        input_name = inp.name
        in_shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        if in_shape[0] != 1:
            in_shape[0] = 1
        x = np.random.rand(*in_shape).astype(np.float32)
        out_names = [o.name for o in sess.get_outputs()]
        outputs = sess.run(out_names, {input_name: x})
        print(f"Model produced {len(outputs)} outputs")
        for name, arr in zip(out_names, outputs):
            tag = "Attention tensor" if name in new_outputs else "Original output"
            print(f"{tag}: {name} shape={arr.shape}")


