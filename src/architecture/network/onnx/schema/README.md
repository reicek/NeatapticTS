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

Canonical metadata key-value pair used by `OnnxModel.metadata_props`.

Keys are exporter-defined semantic hints (for example layout or fallback
reasons) and values are serialized as plain strings.

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
- Initializers currently store floating-point weights in `float_data`, and the
  Phase 7 storage-fp16 lane can pack half-precision words into `int32_data`
  while keeping the logical tensor shape stable.

Security/trust boundary:

- Treat this as untrusted input if it comes from outside your process.

### OnnxNode

One ONNX operator invocation inside the graph.

Nodes connect named tensors rather than object references, which keeps the
exported payload easy to serialize, inspect, and diff as plain JSON.

### OnnxShape

Tensor-shape envelope used by ONNX value and initializer descriptors.

The `dim` array preserves rank and per-axis declarations in order.

### OnnxTensor

Serialized tensor payload stored inside graph initializers.

NeatapticTS currently writes floating-point parameter vectors and matrices to
`float_data`, while the storage-fp16 lane can pack float16 words into
`int32_data` for JSON-first persistence without changing the logical tensor
shape.

### OnnxTensorType

ONNX tensor type descriptor combining element type and shape metadata.

### OnnxValueInfo

Input or output boundary descriptor for an ONNX graph.

Export uses this to declare the tensor contracts expected at graph entry and
produced at graph exit.

### Pool2DMapping

Mapping describing a pooling operation inserted after a given export-layer index.

This is represented as metadata and optional graph nodes during export.
Import uses it to attach pooling-related runtime metadata back onto the reconstructed
network (when supported).

## architecture/network/onnx/schema/network.onnx.schema.binary.utils.ts

### serializeOnnxModelToBinary

```ts
serializeOnnxModelToBinary(
  onnxModel: OnnxModel,
): Uint8Array<ArrayBufferLike>
```

Serialize an ONNX-like model into protobuf ModelProto bytes.

This helper turns the shared exporter model view into the deterministic
binary `ModelProto` surface that Phase 8 and Phase 9 treat as the primary
runtime-validated artifact for the approved subset.
It intentionally uses ONNX's typed tensor storage fields such as
`float_data`, `int32_data`, and `int64_data` rather than widening into
`raw_data` or `external_data` yet.

Parameters:

- `onnxModel` - ONNX-like model payload.

Returns: Binary protobuf ModelProto bytes.

## architecture/network/onnx/schema/network.onnx.schema.tensor-data.utils.ts

### createFloat16StoragePayload

```ts
createFloat16StoragePayload(
  floatValues: number[],
): Pick<OnnxTensor, "data_type" | "float_data" | "int32_data">
```

Create a float16-backed tensor payload from float32 values.
This helper emits the exact storage fields expected by ONNX initializer writers so callers can downgrade precision while keeping exporter and importer tensor contracts structurally consistent.

Parameters:

- `floatValues` - Float32-domain values to pack.

Returns: ONNX tensor storage fields for a float16 initializer.

### decodeFloat16Bits

```ts
decodeFloat16Bits(
  packedValue: number,
): number
```

Decode one packed float16 word into a float32-domain value.

Parameters:

- `packedValue` - Packed float16 bits.

Returns: Decoded float value.

### decodeFloat16Int32Data

```ts
decodeFloat16Int32Data(
  packedValues: number[],
): number[]
```

Decode packed float16 words stored as int32 entries into float32-domain values.
Decoder output is normalized to JavaScript number values so upstream importer logic can reuse one scalar path regardless of original tensor storage precision.

Parameters:

- `packedValues` - Packed float16 words.

Returns: Decoded float values.

### encodeFloat16Bits

```ts
encodeFloat16Bits(
  floatValue: number,
): number
```

Encode one float32-domain value into one float16 word.

Parameters:

- `floatValue` - Float value to encode.

Returns: Packed float16 bits.

### encodeFloat16Int32Data

```ts
encodeFloat16Int32Data(
  floatValues: number[],
): number[]
```

Encode float32-domain values into packed float16 words stored as int32 entries.
Packing through this utility keeps round-trip behavior aligned with the paired decoder used by import and schema audit paths, including special-value handling.

Parameters:

- `floatValues` - Float values to encode.

Returns: Packed float16 words.

### ONNX_FLOAT_DATA_TYPE

ONNX TensorProto numeric data-type enum value used for float32 tensors.

### ONNX_FLOAT16_DATA_TYPE

ONNX TensorProto numeric data-type enum value used for float16 tensors.

### readOnnxTensorFloatData

```ts
readOnnxTensorFloatData(
  tensor: Pick<OnnxTensor, "data_type" | "float_data" | "int32_data">,
): number[]
```

Read a tensor's floating-point values regardless of whether it is stored as float32 or float16.
This abstraction gives importer and analysis utilities one read entrypoint that transparently handles native float shelves and packed float16 compatibility shelves.

Parameters:

- `tensor` - Source ONNX tensor.

Returns: Decoded floating-point values.
