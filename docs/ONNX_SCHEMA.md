# NeatapticTS ONNX JSON Schema (Phases 1–4 Groundwork + Experimental Fused Recurrent Nodes)

This document describes the lightweight ONNX JSON structure produced by `exportToONNX` through Phase 4 groundwork additions (manual Conv2D mapping, pooling insertion, optional flatten bridging, weight sharing validation, heuristic convolution inference metadata) plus the experimental fused LSTM / GRU heuristic emission layer. Earlier Phase 1–2–3 features (feed‑forward MLP, partial connectivity, mixed activation decomposition, recurrent single‑step) are retained; new Phase 4 deltas are clearly marked.

## High-Level Shape

```ts
interface OnnxModel {
  ir_version?: number; // present when includeMetadata=true
  opset_import?: { version: number; domain: string }[]; // metadata flag
  producer_name?: string; // metadata flag
  producer_version?: string; // metadata flag
  doc_string?: string; // metadata flag
  metadata_props?: { key: string; value: string }[]; // (unused Phase 1)
  graph: OnnxGraph;
}

interface OnnxGraph {
  inputs: TensorValueInfo[]; // 'input' plus optional recurrent previous-state inputs per recurrent hidden layer
  outputs: TensorValueInfo[]; // single item: 'output'
  initializer: OnnxTensor[]; // [W0,B0, (R0?), W1,B1,(R1?), ...]
  node: OnnxNode[]; // Per layer: (Recurrent: Gemm_in, Gemm_rec, Add, Activation) else Gemm->Activation or decomposed per-neuron + Concat
}

interface TensorValueInfo {
  name: string; // 'input' | 'output'
  type: {
    tensor_type: {
      elem_type: 1; // float32
      shape: { dim: Array<{ dim_value?: number; dim_param?: string }> };
    };
  };
}

interface OnnxTensor {
  name: string; // W{layerIndex} | B{layerIndex}
  data_type: 1; // float
  dims: number[]; // W: [rows=currentLayerNeurons, cols=prevLayerNeurons]; B: [rows]
  float_data: number[]; // row-major weights or bias values
}

type ActivationOp = 'Relu' | 'Sigmoid' | 'Tanh' | 'Identity';
interface OnnxNode {
  op_type:
    | 'Gemm'
    | ActivationOp
    | 'Concat'
    | 'Add'
    | 'LSTM'
    | 'GRU'
    | 'Conv'
    | 'MaxPool'
    | 'AveragePool'
    | 'Flatten';
  // Gemm (dense) unified: input [PrevOut, Wk, Bk]
  // Conv: input [PrevOut, ConvWk, ConvBk]
  // Pool: input [Layer_k] (post activation or post conv activation) -> Pool_k
  // Flatten: input [Pool_k]
  // Recurrent: Gemm_in / Gemm_rec / Add / Activation as before
  input: string[];
  output: string[];
  name: string; // Extended: conv_l{k}, act_conv_l{k}, pool_after_l{k}, flatten_after_l{k}
  attributes?: Array<{
    name: string;
    type: string;
    f?: number;
    i?: number;
    ints?: number[];
  }>; // Conv/Pool include kernel_shape,strides,pads; Flatten axis
}
```

## Layer & Spatial Mapping

Two representations:

1. Unified (homogeneous activations) — Phase 1 style:

- Weight tensor `W{layer}` dims `[n_L, n_{L-1}]`.
- Bias tensor `B{layer}` dims `[n_L]`.
- Gemm: `[PrevOut, W{layer}, B{layer}]` → `Gemm_{L}`.
- Activation: `Gemm_{L}` → `Layer_{L}` (op_type from first neuron's activation).

2. Decomposed (mixed activations with `allowMixedActivations=true`) — Phase 2:

- For neuron i in layer L:
  - Weight tensor `W{layer}_n{i}` dims `[1, n_{L-1}]`.
  - Bias tensor `B{layer}_n{i}` dims `[1]`.
  - Gemm: `[PrevOut, W{layer}_n{i}, B{layer}_n{i}]` → `Gemm_{L}_n{i}`.
  - Activation: `Gemm_{L}_n{i}` → `Layer_{L}_n{i}` (individual op_type mapping).
- Concat: `[Layer_{L}_n0, ..., Layer_{L}_n{n_L-1}]` → `Layer_{L}` with `axis = (batchDimension?1:0)`.

### Convolutional (Phase 4 Groundwork – Manual Mapping)

When an export-layer index `k` (1-based among hidden layers; output layer currently excluded) is declared via `conv2dMappings`, the dense layer is replaced in the emitted graph by:

Initializers:

```
ConvW{k-1}: [outChannels, inChannels, kernelHeight, kernelWidth]
ConvB{k-1}: [outChannels]
```

Nodes:

```
Conv (name: conv_l{k}) inputs [PrevOut, ConvW{k-1}, ConvB{k-1}] -> Conv_{k}
Activation (name: act_conv_l{k}) input [Conv_{k}] -> Layer_{k}
```

Attributes (Conv): `kernel_shape=[kH,kW]`, `strides=[sH,sW]`, `pads=[padTop,padLeft,padBottom,padRight]`.

Neuron ordering assumptions for dimensional consistency & reconstruction heuristics:

```
Input feature index = ic * (H * W) + ih * W + iw
Output neuron index = oc * (outH * outW) + oh * outW + ow
```

Import currently expands Conv initializers back into equivalent dense weights (sliding window, zero outside padded bounds) to preserve existing dense forward parity. True convolution semantics are not re-simulated separately; correctness relies on the original dense layer truly representing a shared-kernel pattern (sharing validation optional).

### Pooling (Phase 4 Groundwork – Structural Annotation)

After any layer (dense or conv) with export-layer index `k`, if a `pool2dMappings` entry exists:

```
Pool (name: pool_after_l{k}) input [Layer_{k}] -> Pool_{k}
```

Attributes mirror ONNX: `kernel_shape`, `strides`, `pads`.

Import currently treats pooling as metadata only; no weight or shape transformation is applied. Forward parity remains identical to the pre-pooling dense path (pooling is a no-op placeholder for now).

### Flatten Bridge (Phase 4 Extension)

If `flattenAfterPooling=true`, a `Flatten` node is inserted immediately after each emitted pooling node:

```
Flatten (name: flatten_after_l{k}) input [Pool_{k}] -> PoolFlat_{k}
```

Attribute: `axis=1` (standard ONNX semantics). This is structural only; import records layer index in `flatten_layers` metadata but does not alter internal network tensor dimensions yet.

`PrevOut` is `'input'` for the first non-input layer, else `Layer_{L-1}`. For recurrent layers a _second_ previous-state input is present: first recurrent layer uses `hidden_prev`, others `hidden_prev_l{k}`.

## Ordering

Default: Gemm precedes Activation (industry convention). Legacy mode flips ordering to preserve historical project snapshots.

## Batch Dimension

When `batchDimension=true` the input and output shapes add a leading symbolic dimension `{ dim_param: 'N' }`.

Example input shape: `[ { dim_param: 'N' }, { dim_value: 5 } ]` for a 5-feature network.

## Activations

Mapped internal functions → ONNX op_type:

- tanh → `Tanh`
- logistic/sigmoid → `Sigmoid`
- relu → `Relu`
- unknown/custom → `Identity` with a console warning.

## Import Assumptions

- Unified path: weight tensors `W0..Wk`, biases `B0..Bk` (k = hiddenCount + output layer - 1).
- Decomposed path: per-neuron `W{layer}_n{idx}`, `B{layer}_n{idx}` with a corresponding activation per neuron and a `concat_l{layer}` node assembling the layer output.
- Layer sizes optionally stored in `metadata_props` key `layer_sizes` (array of hidden layer sizes) to disambiguate decomposed exports.
- Without metadata, importer infers sizes by grouping weight tensor prefixes.
- Conv: If `conv2d_layers` + `conv2d_specs` present and matching ConvW/ConvB initializers exist, convolution weights are expanded into dense incoming connection weights (sliding window) overriding any previously assigned W/B for that layer.
- Pool: `_onnxPooling` object attached to the imported network containing layer & spec arrays (no numerical effect yet).
- Flatten: recorded only via metadata; no structural flattening applied during import (future enhancement will align dense layer widths with flattened spatial extents).

## Numerical Fidelity

Round-trip tests enforce MSE < 1e-12 for deterministic forward passes (see `onnx.roundtrip.test.ts`).

## Partial Connectivity (Phase 2)

Missing connections are encoded as implicit zero weights inside the dense matrices (unified) or per-neuron rows (decomposed). No explicit sparse encoding yet; placeholder option `allowPartialConnectivity` simply prevents validation error and fills zeros.

## Metadata Additions (Phase 2–4)

When `includeMetadata=true`, keys may include:

Phase 2:

- `layer_sizes`: `[h1,h2,...]` for hidden layers.

Phase 3 (recurrent):

- `recurrent_single_step`: array of recurrent hidden export-layer indices.
- `lstm_groups_stub`, `lstm_emitted_layers`, `gru_emitted_layers`, `rnn_pattern_fallback`: experimental fused recurrent diagnostics (see section below).

Phase 4 (spatial groundwork):

- `conv2d_layers`: array of export-layer indices explicitly mapped to Conv.
- `conv2d_specs`: array of `Conv2DMapping` specs `{ layerIndex, inHeight, inWidth, inChannels, kernelHeight, kernelWidth, strideHeight, strideWidth, padTop?, padBottom?, padLeft?, padRight?, outHeight, outWidth, outChannels, activation? }`.
- `pool2d_layers`: array of export-layer indices after which a pooling op was inserted.
- `pool2d_specs`: array of `Pool2DMapping` specs `{ afterLayerIndex, type, kernelHeight, kernelWidth, strideHeight, strideWidth, padTop?, padBottom?, padLeft?, padRight?, activation? }`.
- `conv2d_sharing_verified`: layers where weight sharing validation passed (optional, when `validateConvSharing` true).
- `conv2d_sharing_mismatch`: layers where validation detected divergence (advisory; export still succeeds).
- `conv2d_inferred_layers`: layers heuristically recognized as potential conv (single-channel square + 2x2/3x3 kernel) – metadata only.
- `conv2d_inferred_specs`: corresponding heuristic spec objects (not auto-promoted to Conv yet).
- `flatten_layers`: export-layer indices where a Flatten bridge was inserted after pooling (`flattenAfterPooling`).

## Recurrent Single-Step (Phase 3 Baseline)

When `allowRecurrent && recurrentSingleStep`:

- For each hidden layer k (1-based among hidden layers) containing at least one self-connection, we add:
  - Input: `hidden_prev` (k=1) or `hidden_prev_l{k}` shape `[N?, size_k]`.
  - Initializer: `R{k-1}` dims `[size_k, size_k]` (currently diagonal weights from self-connections; off-diagonal zero reserved for future dense recurrence).
  - Nodes (in order):
    1. `gemm_in_l{k}`: forward weights.
    2. `gemm_rec_l{k}`: recurrent weights.
    3. `add_recurrent_l{k}`: elementwise sum.
    4. `act_l{k}`: activation.
- Metadata `recurrent_single_step` stores JSON array of recurrent hidden layer indices (e.g. `[2]`).

Importer logic:

- Reads `layer_sizes` first (Phase 2 metadata).
- Parses `recurrent_single_step` indices; for each index k loads `R{k-1}` and applies diagonal entries as self-connection weights (creating connections if absent).

Limitations:

- Only self-connections (no inter-neuron recurrent edges) – matrices shaped for forward compatibility.
- Mixed activations disallowed in recurrent layers (must use homogeneous activation there; mixed + recurrence combination deferred).

## Experimental Fused Recurrent Heuristics (Phase 3 Extension)

Status: EXPERIMENTAL (shape & metadata may change; importer best-effort). These fused nodes are emitted **in addition to** the unfused Gemm + Activation path (no pruning yet) to preserve transparency while development continues.

### Detection Heuristics

Applied only when `allowRecurrent:true`:

- LSTM: Hidden layer size divisible by 5 (>=10). Interpreted as 5 equal partitions: input gate, forget gate, cell (memory), output gate, output block. Memory (cell) slice must have self-connections (one each) to qualify.
- GRU: Hidden layer size divisible by 4 (>=8). Interpreted as update gate, reset gate, candidate (with self-connections), output block.
- Near-miss sizes (e.g., between thresholds) are recorded with metadata key `rnn_pattern_fallback` for diagnostics.

### Initializers (Simplified)

For layer export index `k` (1-based among hidden layers):

| Tensor                 | Name Pattern  | Shape (current) | Notes                                                                                            |
| ---------------------- | ------------- | --------------- | ------------------------------------------------------------------------------------------------ |
| LSTM input weights     | `LSTM_W{k-1}` | `[4*H, I]`      | 4 gates \* hidden_size (H) vs input_size (I). Gate order (current): input, forget, cell, output. |
| LSTM recurrent weights | `LSTM_R{k-1}` | `[4*H, H]`      | Only diagonal for cell group populated (self connections); others zero.                          |
| LSTM biases            | `LSTM_B{k-1}` | `[4*H]`         | Only input (Wb) bias; recurrent (Rb) bias implicitly zero.                                       |
| GRU input weights      | `GRU_W{k-1}`  | `[3*H, I]`      | Gate order: update, reset, candidate.                                                            |
| GRU recurrent weights  | `GRU_R{k-1}`  | `[3*H, H]`      | Only candidate diagonal populated (self connections).                                            |
| GRU biases             | `GRU_B{k-1}`  | `[3*H]`         | Simplified (no Rb segment).                                                                      |

ONNX spec normally expects shapes with a leading num_directions dimension and bias of length `2*gates*H`; this simplified form is intentional for interim experimentation and will be aligned later.

### Nodes

An emitted fused node appears as:

```
// LSTM
op_type: 'LSTM'
name:   'lstm_l{k}'
input:  [PrevOut, LSTM_W{k-1}, LSTM_R{k-1}, LSTM_B{k-1}]
output: ['Layer_{k}_lstm_hidden']
attributes: [{ name:'hidden_size', i:H }, { name:'layout', i:0 }]

// GRU
op_type: 'GRU'
name:   'gru_l{k}'
input:  [PrevOut, GRU_W{k-1}, GRU_R{k-1}, GRU_B{k-1}]
output: ['Layer_{k}_gru_hidden']
attributes: [{ name:'hidden_size', i:H }, { name:'layout', i:0 }]
```

`PrevOut` is `'input'` for first hidden layer else `Layer_{k-1}`. Sequence axis (time dimension) is not represented; single-step only.

### Metadata Keys (New)

| Key                    | Value Shape                              | Purpose                                                                              |
| ---------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------ |
| `lstm_groups_stub`     | JSON array of `{ layerIndex, unitSize }` | Heuristic groupings detected pre-emission (for analysis even if fused node skipped). |
| `lstm_emitted_layers`  | JSON array of export layer indices       | Layers where an `LSTM` node + tensors were emitted.                                  |
| `gru_emitted_layers`   | JSON array of export layer indices       | Layers where a `GRU` node + tensors were emitted.                                    |
| `rnn_pattern_fallback` | JSON objects (multiple entries)          | Near-miss pattern diagnostics (e.g., size thresholds not met exactly).               |

### Import Reconstruction

Importer looks for `lstm_emitted_layers` / `gru_emitted_layers` and, if present with matching tensors, replaces the raw hidden node block with a `Layer.lstm(H)` or `Layer.gru(H)` instance, copying weights/biases and restoring diagonal recurrent weights for the cell/candidate group only. Any failure to parse shapes silently skips reconstruction (network still imports via unfused tensors).

### Limitations / TODO (Experimental)

- Bias splitting (Wb|Rb) and full gate recurrent weight matrices not yet implemented.
- Gate ordering may differ from ONNX canonical expectations; future alignment will introduce reordering attribute or canonical internal ordering.
- Original unfused Gemm/Activation nodes are retained (could be pruned/fused later to avoid duplication).
- Multi-directional (bidirectional) recurrent layers, sequence length >1, and advanced features (peepholes, projection) are out of scope for this phase.
- Mixed activations + fused recurrent emission unsupported simultaneously.
- Validation tests for fused nodes pending (planned Step 6 of sub‑plan).

## Phase 4 Mapping Interfaces

### Conv2DMapping (export option)

```
interface Conv2DMapping {
  layerIndex: number;        // export-layer index (1-based hidden)
  inHeight: number; inWidth: number; inChannels: number;
  kernelHeight: number; kernelWidth: number;
  strideHeight: number; strideWidth: number;
  padTop?: number; padBottom?: number; padLeft?: number; padRight?: number;
  outHeight: number; outWidth: number; outChannels: number;
  activation?: string;       // overrides per-layer detected activation (optional)
}
```

Constraints: `(inHeight*inWidth*inChannels) == previousLayerWidth` and `(outHeight*outWidth*outChannels) == layerWidth` else mapping is skipped with a console warning.

### Pool2DMapping (export option)

```
interface Pool2DMapping {
  afterLayerIndex: number;   // export-layer index whose output is pooled
  type: 'MaxPool' | 'AveragePool';
  kernelHeight: number; kernelWidth: number;
  strideHeight: number; strideWidth: number;
  padTop?: number; padBottom?: number; padLeft?: number; padRight?: number;
  activation?: string; // (reserved – not applied yet)
}
```

### Additional Export Options (Phase 4)

```
conv2dMappings?: Conv2DMapping[];
pool2dMappings?: Pool2DMapping[];
validateConvSharing?: boolean;   // weight sharing validation across spatial positions
flattenAfterPooling?: boolean;   // insert Flatten after each Pool and record metadata
```

## Planned Extensions (Future Phases)

- Sparse encoding for partially connected layers.
- Multiple inputs/outputs.
- Dense intra-layer recurrence & Scan-based sequence unrolling.
- Convolutional ops (multi-channel inference & automatic promotion, dilation, groups, depthwise) **(partially begun)**.
- Pooling shape simulation & true forward parity (currently metadata only).
- Flatten-aware dense remapping after spatial layers.
- Quantization operators (QLinearMatMul, etc.).
- Custom activation domain ops.
- ONNX-compliant LSTM/GRU (proper bias layout, full recurrence, optional sequence axes, pruning of duplicate unfused layers).

---

Update this file when new fields or ordering rules are introduced.
