"""Export a Brax PPO checkpoint to ONNX (actor only).

Loads `<checkpoint>.pkl` produced by train.py, extracts the running-statistics
normalizer and the policy MLP weights, and writes a self-contained ONNX graph:

    Sub(obs_mean) -> Div(obs_std) -> [MatMul, Add, activation] x N
                                  -> MatMul, Add -> Slice([:action_size]) -> Tanh

The slice strips the log_std half of Brax PPO's `mean ⊕ log_std` output so the
exported actor returns a deterministic, [-1, 1]-bounded action.

Usage:
    python export.py --checkpoint checkpoints/final.pkl --output actor.onnx
"""

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


_ACTIVATION_ONNX = {
    "relu": "Relu",
    "tanh": "Tanh",
    "sigmoid": "Sigmoid",
    "elu": "Elu",
    "leaky_relu": "LeakyRelu",
    "swish": "swish",   # custom: x * sigmoid(x)
    "silu": "swish",
    "gelu": "Gelu",     # ONNX 20+
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="actor.onnx")
    p.add_argument("--obs-size", type=int, default=10)
    p.add_argument("--action-size", type=int, default=2)
    return p.parse_args()


def _unwrap(d: Any) -> Any:
    """Strip Flax FrozenDict / single-key 'params' wrappers."""
    try:
        d = dict(d)
    except Exception:
        pass
    if isinstance(d, dict) and set(d.keys()) == {"params"}:
        return _unwrap(d["params"])
    return d


def _extract_normalizer(normalizer_params, obs_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) as float32 arrays of shape (obs_size,)."""
    if hasattr(normalizer_params, "mean") and hasattr(normalizer_params, "std"):
        mean = normalizer_params.mean
        std = normalizer_params.std
        if isinstance(mean, dict):
            key = "state" if "state" in mean else next(iter(mean.keys()))
            mean = mean[key]
            std = std[key]
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
    else:
        print("WARNING: normalizer params missing — defaulting to identity (zeros/ones)")
        mean = np.zeros(obs_size, dtype=np.float32)
        std = np.ones(obs_size, dtype=np.float32)

    std = np.maximum(std, 1e-6)
    return mean, std


def _extract_mlp(policy_params: Any) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """Walk the Flax param tree and return (hidden_kernels, hidden_biases, out_kernel, out_bias).

    Brax 0.14.x names every dense layer `hidden_<i>`, including the final
    output layer. The last hidden layer's output dim is `action_size` (or
    `2*action_size` for PPO mean⊕log_std), so we treat the final hidden_N
    as the output layer.
    """
    pp = _unwrap(policy_params)
    if not isinstance(pp, dict):
        raise ValueError(f"policy_params is not a dict after unwrap: {type(pp)}")

    hidden_kernels, hidden_biases = [], []
    i = 0
    while f"hidden_{i}" in pp:
        layer = pp[f"hidden_{i}"]
        hidden_kernels.append(np.asarray(layer["kernel"], dtype=np.float32))
        hidden_biases.append(np.asarray(layer["bias"], dtype=np.float32))
        i += 1

    out_kernel = out_bias = None
    for key in ("logits", "output", "out", "final"):
        if key in pp:
            out_kernel = np.asarray(pp[key]["kernel"], dtype=np.float32)
            out_bias = np.asarray(pp[key]["bias"], dtype=np.float32)
            break

    if out_kernel is None:
        # Brax 0.14.x convention: last `hidden_N` is the output layer.
        if hidden_kernels:
            out_kernel = hidden_kernels.pop()
            out_bias = hidden_biases.pop()
        else:
            # Last resort: any non-hidden_* layer with a kernel
            for k, v in pp.items():
                if not k.startswith("hidden_") and isinstance(v, dict) and "kernel" in v:
                    out_kernel = np.asarray(v["kernel"], dtype=np.float32)
                    out_bias = np.asarray(v["bias"], dtype=np.float32)
                    break

    if out_kernel is None:
        raise ValueError(f"No output layer found. Keys: {list(pp.keys())}")
    return hidden_kernels, hidden_biases, out_kernel, out_bias


def build_onnx(
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    hidden_kernels: List[np.ndarray],
    hidden_biases: List[np.ndarray],
    out_kernel: np.ndarray,
    out_bias: np.ndarray,
    activation: str,
    obs_size: int,
    action_size: int,
) -> onnx.ModelProto:
    """Build the ONNX policy graph: Sub→Div→[MatMul→Add→act]×N→MatMul→Add→Slice→Tanh."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    inits.append(numpy_helper.from_array(obs_mean, name="obs_mean"))
    inits.append(numpy_helper.from_array(obs_std, name="obs_std"))

    # Normalize
    nodes.append(helper.make_node("Sub", ["observation", "obs_mean"], ["norm_sub"]))
    nodes.append(helper.make_node("Div", ["norm_sub", "obs_std"], ["norm"]))
    cur = "norm"

    act_op = _ACTIVATION_ONNX.get(activation, "Relu")

    def emit_act(in_name: str, out_name: str, idx: int):
        if act_op == "swish":  # x * sigmoid(x)
            nodes.append(helper.make_node("Sigmoid", [in_name], [f"sig_{idx}"]))
            nodes.append(helper.make_node("Mul", [in_name, f"sig_{idx}"], [out_name]))
        else:
            nodes.append(helper.make_node(act_op, [in_name], [out_name]))

    for i, (k, b) in enumerate(zip(hidden_kernels, hidden_biases)):
        inits.append(numpy_helper.from_array(k.astype(np.float32), name=f"h{i}_W"))
        inits.append(numpy_helper.from_array(b.astype(np.float32), name=f"h{i}_b"))
        nodes.append(helper.make_node("MatMul", [cur, f"h{i}_W"], [f"h{i}_mm"]))
        nodes.append(helper.make_node("Add", [f"h{i}_mm", f"h{i}_b"], [f"h{i}_pre"]))
        emit_act(f"h{i}_pre", f"h{i}_act", i)
        cur = f"h{i}_act"

    # Output layer
    inits.append(numpy_helper.from_array(out_kernel.astype(np.float32), name="out_W"))
    inits.append(numpy_helper.from_array(out_bias.astype(np.float32), name="out_b"))
    nodes.append(helper.make_node("MatMul", [cur, "out_W"], ["out_mm"]))
    nodes.append(helper.make_node("Add", ["out_mm", "out_b"], ["raw"]))

    # If output produces mean ⊕ log_std (Brax PPO default), slice off log_std
    if out_kernel.shape[1] == 2 * action_size:
        inits.extend([
            numpy_helper.from_array(np.array([0], dtype=np.int64), name="slice_starts"),
            numpy_helper.from_array(np.array([action_size], dtype=np.int64), name="slice_ends"),
            numpy_helper.from_array(np.array([-1], dtype=np.int64), name="slice_axes"),
        ])
        nodes.append(helper.make_node(
            "Slice",
            ["raw", "slice_starts", "slice_ends", "slice_axes"],
            ["mean"],
        ))
        head = "mean"
    elif out_kernel.shape[1] == action_size:
        head = "raw"
    else:
        raise ValueError(
            f"Unexpected output size {out_kernel.shape[1]} (expected {action_size} or {2*action_size})"
        )

    nodes.append(helper.make_node("Tanh", [head], ["action"]))

    in_t = helper.make_tensor_value_info("observation", TensorProto.FLOAT, [obs_size])
    out_t = helper.make_tensor_value_info("action", TensorProto.FLOAT, [action_size])
    graph = helper.make_graph(nodes, "policy_network", [in_t], [out_t], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    print(f"Loading {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        ckpt: Dict[str, Any] = pickle.load(f)

    params = ckpt["params"]
    cfg = ckpt.get("config", {})
    activation = (
        cfg.get("training", {}).get("brax_ppo", {}).get("network", {}).get("activation", "swish")
    )
    print(f"  activation: {activation}")

    # Brax PPO returns (normalizer, network_params, ...) — accept tuple of 2 or 3
    if isinstance(params, (tuple, list)) and len(params) >= 2:
        normalizer_params = params[0]
        policy_params = params[1]
    else:
        raise ValueError(f"Unrecognised params type: {type(params)}")

    obs_mean, obs_std = _extract_normalizer(normalizer_params, args.obs_size)
    print(f"  obs_mean shape: {obs_mean.shape}")

    hidden_k, hidden_b, out_k, out_b = _extract_mlp(policy_params)
    print(f"  hidden layers: {[k.shape for k in hidden_k]}")
    print(f"  output layer:  {out_k.shape}")

    model = build_onnx(
        obs_mean, obs_std, hidden_k, hidden_b, out_k, out_b,
        activation=activation,
        obs_size=args.obs_size,
        action_size=args.action_size,
    )

    out_path = Path(args.output)
    onnx.save(model, str(out_path))
    print(f"Saved ONNX model: {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
