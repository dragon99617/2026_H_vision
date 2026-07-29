#!/usr/bin/env python3
"""Patch YOLO26 one-class end-to-end ONNX for TensorRT 8.5.

TensorRT 8.5 cannot import this ONNX Mod. For a one-class graph ``index % 1`` is
always zero. Replace only that node with ``index - index``. This preserves the
shape and dtype without changing a shared constant or triggering TensorRT's
incorrect zero-Mul/Concat optimization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import onnx
    from onnx import numpy_helper
except ImportError as exc:
    raise RuntimeError(
        "ONNX is required only while building the engine. "
        "Install requirements-build.txt first."
    ) from exc

MOD_OP = b"\x22\x03Mod"
SUB_OP = b"\x22\x03Sub"


def patch_model(data: bytes) -> bytes:
    try:
        model = onnx.load_model_from_string(data)
    except Exception as exc:
        raise RuntimeError(
            "ONNX is not the expected one-class YOLO26 end-to-end graph"
        ) from exc
    mod_nodes = [node for node in model.graph.node if node.op_type == "Mod"]
    metadata_checks = {
        "single ball metadata": data.count(b"{0: 'ball'}"),
        "end-to-end metadata": data.count(b"end2end"),
    }
    invalid_metadata = [
        "%s=%d" % (name, count)
        for name, count in metadata_checks.items()
        if count != 1
    ]
    if len(mod_nodes) != 1 or invalid_metadata:
        raise RuntimeError(
            "ONNX is not the expected one-class YOLO26 end-to-end graph: "
            + ", ".join(
                ["Mod operators=%d" % len(mod_nodes)] + invalid_metadata
            )
        )

    mod_node = mod_nodes[0]
    if len(mod_node.input) != 2:
        raise RuntimeError("expected Mod to have exactly two inputs")
    divisor_name = mod_node.input[1]
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    divisor_tensor = initializers.get(divisor_name)
    divisor_node = next(
        (
            node
            for node in model.graph.node
            if divisor_name in node.output
        ),
        None,
    )
    if divisor_tensor is not None:
        divisor = numpy_helper.to_array(divisor_tensor)
    elif divisor_node is not None and divisor_node.op_type == "Constant":
        value_attributes = [
            attr for attr in divisor_node.attribute if attr.name == "value"
        ]
        if len(value_attributes) != 1:
            raise RuntimeError("Mod Constant divisor has no unique tensor value")
        divisor = numpy_helper.to_array(value_attributes[0].t)
    else:
        raise RuntimeError(
            "Mod divisor is neither an initializer nor a Constant tensor"
        )
    if divisor.size != 1 or int(divisor.reshape(-1)[0]) != 1:
        raise RuntimeError("expected one-class Mod divisor to be scalar 1")
    consumers = [
        node
        for node in model.graph.node
        if divisor_name in node.input
    ]
    if consumers != [mod_node]:
        raise RuntimeError("Mod divisor is unexpectedly shared")

    index_input = mod_node.input[0]
    mod_node.op_type = "Sub"
    del mod_node.attribute[:]
    del mod_node.input[:]
    mod_node.input.extend([index_input, index_input])
    if divisor_tensor is not None:
        model.graph.initializer.remove(divisor_tensor)
    else:
        model.graph.node.remove(divisor_node)
    onnx.checker.check_model(model)
    patched = model.SerializeToString()
    if patched.count(MOD_OP) != 0 or patched.count(SUB_OP) < data.count(SUB_OP) + 1:
        raise RuntimeError("structured compatibility patch did not replace Mod")
    return patched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original = args.input.read_bytes()
    patched = patch_model(original)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(patched)
    print(
        "patched TensorRT 8.5 compatibility model: %s -> %s (%d bytes)"
        % (args.input, args.output, len(patched))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
