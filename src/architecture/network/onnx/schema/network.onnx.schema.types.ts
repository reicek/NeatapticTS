/**
 * Leaf JSON wire-format schema for NeatapticTS ONNX-like import and export.
 *
 * This chapter holds the stable, serializable payload shapes shared by both the
 * exporter and importer: model containers, graph nodes, tensors, metadata
 * records, and explicit Conv/Pool mapping declarations.
 *
 * Keep this file leaf-only. It should describe the persisted document shape,
 * not runtime execution contexts such as node internals, layer traversal state,
 * or importer/exporter orchestration helpers.
 *
 * Example:
 *
 * ```ts
 * const model: OnnxModel = {
 *   graph: {
 *     inputs: [],
 *     outputs: [],
 *     initializer: [],
 *     node: [],
 *   },
 * };
 * ```
 */

/**
 * Mapping declaration for treating a fully-connected layer as a 2D convolution during export.
 *
 * This does **not** magically turn an MLP into a convolutional network at runtime.
 * It annotates a particular export-layer index with a conv interpretation so that:
 * - The exported graph uses conv-shaped tensors/operators, and
 * - Import can re-attach pooling/flatten metadata appropriately.
 *
 * Pitfall: mappings must match the actual layer sizes. If `inHeight * inWidth * inChannels`
 * does not correspond to the prior layer width (and similarly for outputs), export or import
 * may reject the model.
 */
export interface Conv2DMapping {
  layerIndex: number;
  inHeight: number;
  inWidth: number;
  inChannels: number;
  kernelHeight: number;
  kernelWidth: number;
  strideHeight: number;
  strideWidth: number;
  padTop?: number;
  padBottom?: number;
  padLeft?: number;
  padRight?: number;
  outHeight: number;
  outWidth: number;
  outChannels: number;
  activation?: string;
}

/**
 * Mapping describing a pooling operation inserted after a given export-layer index.
 *
 * This is represented as metadata and optional graph nodes during export.
 * Import uses it to attach pooling-related runtime metadata back onto the reconstructed
 * network (when supported).
 */
export interface Pool2DMapping {
  afterLayerIndex: number;
  type: 'MaxPool' | 'AveragePool';
  kernelHeight: number;
  kernelWidth: number;
  strideHeight: number;
  strideWidth: number;
  padTop?: number;
  padBottom?: number;
  padLeft?: number;
  padRight?: number;
  activation?: string;
}

/**
 * One dimension inside an ONNX tensor shape.
 *
 * Use `dim_value` for fixed numeric widths and `dim_param` for symbolic names
 * such as a batch dimension.
 */
export type OnnxDimension = {
  dim_value?: number;
  dim_param?: string;
};

/** ONNX tensor type shape. */
export type OnnxShape = {
  dim: OnnxDimension[];
};

/** ONNX tensor type. */
export type OnnxTensorType = {
  elem_type: number;
  shape: OnnxShape;
};

/** ONNX value info (input/output description). */
export type OnnxValueInfo = {
  name: string;
  type: {
    tensor_type: OnnxTensorType;
  };
};

/**
 * ONNX node attribute payload.
 *
 * This simplified JSON-first shape is enough for the operators emitted by the
 * current exporter. It intentionally avoids protobuf-level complexity while
 * still preserving the attribute variants needed by the importer.
 */
export type OnnxAttribute = {
  name: string;
  type?: string;
  f?: number;
  i?: number;
  s?: string;
  t?: OnnxTensor;
  g?: OnnxGraph;
  floats?: number[];
  ints?: number[];
  strings?: string[];
};

/**
 * ONNX-like model container (JSON-serializable).
 *
 * This is the main “wire format” object in this folder. Persist it as JSON text:
 *
 * ```ts
 * const jsonText = JSON.stringify(model);
 * const restoredModel = JSON.parse(jsonText) as OnnxModel;
 * ```
 *
 * Notes:
 * - `metadata_props` contains NeatapticTS-specific keys (layer sizes, recurrent flags,
 *   conv/pool mappings, etc.). This is where most round-trip hints live.
 * - Initializers currently store floating-point weights in `float_data`.
 *
 * Security/trust boundary:
 * - Treat this as untrusted input if it comes from outside your process.
 */
export type OnnxModel = {
  ir_version?: number;
  opset_import?: { version: number; domain: string }[];
  producer_name?: string;
  producer_version?: string;
  doc_string?: string;
  metadata_props?: { key: string; value: string }[];
  graph: OnnxGraph;
};

/**
 * Graph body of an ONNX-like model.
 *
 * The exporter writes three main collections here:
 * - `inputs` and `outputs` describe graph boundaries,
 * - `initializer` stores constant tensors such as weights and biases,
 * - `node` stores the ordered operator payloads that consume those tensors.
 */
export type OnnxGraph = {
  inputs: OnnxValueInfo[];
  outputs: OnnxValueInfo[];
  initializer: OnnxTensor[];
  node: OnnxNode[];
};

/**
 * Serialized tensor payload stored inside graph initializers.
 *
 * NeatapticTS currently writes floating-point parameter vectors and matrices to
 * `float_data`, along with the tensor name, element type, and logical shape.
 */
export type OnnxTensor = {
  name: string;
  data_type: number;
  dims: number[];
  float_data: number[];
};

/**
 * One ONNX operator invocation inside the graph.
 *
 * Nodes connect named tensors rather than object references, which keeps the
 * exported payload easy to serialize, inspect, and diff as plain JSON.
 */
export type OnnxNode = {
  op_type: string;
  input: string[];
  output: string[];
  name: string;
  attributes?: OnnxAttribute[];
};

/** Canonical metadata key-value pair used in ONNX model metadata_props. */
export type OnnxMetadataProperty = {
  key: string;
  value: string;
};
