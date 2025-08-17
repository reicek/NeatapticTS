# ONNX Export Plan for NeatapticTS

## Overview

This document outlines the plan for exporting NeatapticTS neural networks to the ONNX (Open Neural Network Exchange) format.

## Supported Architectures

- **Single-layer MLP:** Input fully connected to output, standard activations (ReLU, Tanh, Sigmoid, Identity).
- **Single hidden layer MLP:** Input fully connected to hidden, hidden fully connected to output, standard activations (ReLU, Tanh, Sigmoid, Identity) for both hidden and output layers.
- **Multi-hidden-layer MLP:** Any number of contiguous, fully connected hidden layers, standard activations (ReLU, Tanh, Sigmoid, Identity) for all layers. All weights, biases, and activations are mapped to ONNX nodes and initializers.

## Mapping Plan

- **Inputs/Outputs:**
  - Map `Network.input` and `Network.output` to ONNX graph inputs/outputs.
- **Nodes/Layers:**
  - Each Neataptic node becomes an ONNX node (operator) with attributes:
    - `type` → ONNX operator type (e.g., `Add`, `MatMul`, `Relu`, `Tanh`, etc.)
    - `bias` → ONNX `Add` node or bias initializer
    - `squash` (activation) → ONNX activation operator (e.g., `Relu`, `Tanh`, `Sigmoid`)
    - Custom activations: fallback to `Identity` or raise warning. Custom activations must be registered using the exported `registerCustomActivation` function.
- **Connections:**
  - All connections are represented as ONNX `MatMul` (weights) and `Add` (biases)
  - Gated connections and evolutionary features: not directly supported; will be ignored or raise warning
- **Parameters:**
  - Weights and biases exported as ONNX initializers (tensors)
- **Graph Structure:**
  - Feedforward: straightforward mapping for single-layer and multi-layer MLPs
  - Recurrent: not supported yet

## TypeScript Implementation Notes

- ONNX export uses explicit TypeScript interfaces for ONNX tensors, nodes, and graphs.
- The export function returns an object with a `.graph` property, matching the ONNX model structure expected by tests and consumers.
- The `Activation` object only contains activation functions; custom registration is handled by the exported `registerCustomActivation(name, fn)` function.

## Limitations

- Only single-layer and multi-layer MLPs (fully connected, contiguous hidden layers) are supported.
- Gating, custom evolutionary connections, and user-defined activations are not supported in ONNX and will be ignored or replaced with `Identity`.
- Only standard activations (ReLU, Tanh, Sigmoid, Identity) are mapped directly.
- Recurrent, non-fully-connected, and non-contiguous hidden layer architectures are not supported yet.

## Next Steps

- Add support for recurrent and more advanced architectures.
- Add/expand tests for multi-layer MLP ONNX export structure and compatibility.
- Document any unsupported features in the export process.

---

For questions or contributions, see the `toONNX()` method in `src/architecture/network.ts` and the ONNX TypeScript types in the same file.
