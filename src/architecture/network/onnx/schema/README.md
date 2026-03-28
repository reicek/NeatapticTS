# architecture/network/onnx/schema

Leaf JSON wire-format schema for NeatapticTS ONNX-like import and export.

This chapter holds the stable, serializable payload shapes shared by both the
exporter and importer: model containers, graph nodes, tensors, metadata
records, and explicit Conv/Pool mapping declarations.

Keep this file leaf-only. It should describe the persisted document shape,
not runtime execution contexts such as node internals, layer traversal state,
or importer/exporter orchestration helpers.

Example:

```ts
const model: OnnxModel = {
  graph: {
    inputs: [],
    outputs: [],
    initializer: [],
    node: [],
  },
};
```

## architecture/network/onnx/schema/network.onnx.schema.types.ts

### Conv2DMapping

Mapping declaration for treating a fully-connected layer as a 2D convolution during export.

This does **not** magically turn an MLP into a convolutional network at runtime.
It annotates a particular export-layer index with a conv interpretation so that:
- The exported graph uses conv-shaped tensors/operators, and
- Import can re-attach pooling/flatten metadata appropriately.

Pitfall: mappings must match the actual layer sizes. If `inHeight * inWidth * inChannels`
does not correspond to the prior layer width (and similarly for outputs), export or import
may reject the model.

### OnnxAttribute

ONNX node attribute payload.

This simplified JSON-first shape is enough for the operators emitted by the
current exporter. It intentionally avoids protobuf-level complexity while
still preserving the attribute variants needed by the importer.

### OnnxDimension

One dimension inside an ONNX tensor shape.

Use `dim_value` for fixed numeric widths and `dim_param` for symbolic names
such as a batch dimension.

### OnnxGraph

Graph body of an ONNX-like model.

The exporter writes three main collections here:
- `inputs` and `outputs` describe graph boundaries,
- `initializer` stores constant tensors such as weights and biases,
- `node` stores the ordered operator payloads that consume those tensors.

### OnnxMetadataProperty

Canonical metadata key-value pair used in ONNX model metadata_props.

### OnnxModel

ONNX-like model container (JSON-serializable).

This is the main “wire format” object in this folder. Persist it as JSON text:

```ts
const jsonText = JSON.stringify(model);
const restoredModel = JSON.parse(jsonText) as OnnxModel;
```

Notes:
- `metadata_props` contains NeatapticTS-specific keys (layer sizes, recurrent flags,
  conv/pool mappings, etc.). This is where most round-trip hints live.
- Initializers currently store floating-point weights in `float_data`.

Security/trust boundary:
- Treat this as untrusted input if it comes from outside your process.

### OnnxNode

One ONNX operator invocation inside the graph.

Nodes connect named tensors rather than object references, which keeps the
exported payload easy to serialize, inspect, and diff as plain JSON.

### OnnxShape

ONNX tensor type shape.

### OnnxTensor

Serialized tensor payload stored inside graph initializers.

NeatapticTS currently writes floating-point parameter vectors and matrices to
`float_data`, along with the tensor name, element type, and logical shape.

### OnnxTensorType

ONNX tensor type.

### OnnxValueInfo

ONNX value info (input/output description).

### Pool2DMapping

Mapping describing a pooling operation inserted after a given export-layer index.

This is represented as metadata and optional graph nodes during export.
Import uses it to attach pooling-related runtime metadata back onto the reconstructed
network (when supported).
