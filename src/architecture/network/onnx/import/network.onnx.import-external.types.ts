import type { OnnxActivationOperation } from '../network.onnx.utils.types';

/**
 * Union type for raw byte fields emitted by `onnx-proto` object conversion; accepts string, Uint8Array, or number-array representations.
 */
export type OnnxDecodedBytes = string | Uint8Array | number[];

/**
 * Union type for ONNX 64-bit integer fields decoded by `onnx-proto`; includes numeric, string, and toString-capable object forms to handle platform-specific long encoding.
 */
export type OnnxDecodedLongLike = number | string | { toString(): string };

/**
 * Decoded ONNX dimension payload used by external import parsing to preserve symbolic and numeric shape information from protobuf conversion.
 * Import normalization relies on this shape to reconstruct rank and axis semantics before tensor compatibility checks run.
 */
export type DecodedExternalOnnxDimension = {
  dimValue?: OnnxDecodedLongLike | null;
  dimParam?: string | null;
};

/**
 * Decoded ONNX tensor-type payload describing element type and optional shape dimensions after external binary decode.
 * This type bridges raw protobuf decode output and importer-owned tensor validation routines.
 */
export type DecodedExternalOnnxTensorType = {
  elemType?: number | null;
  shape?: {
    dim?: DecodedExternalOnnxDimension[] | null;
  } | null;
};

/**
 * Decoded ONNX value-info payload carrying named tensor metadata for graph inputs, outputs, and intermediate value descriptors.
 * It allows importer passes to align tensor names, element types, and shapes across graph boundaries.
 */
export type DecodedExternalOnnxValueInfo = {
  name?: string | null;
  type?: {
    tensorType?: DecodedExternalOnnxTensorType | null;
  } | null;
};

/**
 * Decoded ONNX attribute payload preserving scalar, integer, and byte-string forms emitted by the protobuf decoder.
 * Attribute decoding uses this shape before operation-specific coercion into importer contracts.
 */
export type DecodedExternalOnnxAttribute = {
  name?: string | null;
  f?: number | null;
  i?: OnnxDecodedLongLike | null;
  s?: OnnxDecodedBytes | null;
};

/**
 * Decoded ONNX node payload describing operator identity, wiring, and decoded attribute list for importer normalization passes.
 * Node-level validation and operator support checks consume this schema directly.
 */
export type DecodedExternalOnnxNode = {
  name?: string | null;
  domain?: string | null;
  opType?: string | null;
  input?: string[] | null;
  output?: string[] | null;
  attribute?: DecodedExternalOnnxAttribute[] | null;
};

/**
 * Decoded ONNX tensor payload containing name, data type, shape dimensions, and raw or float initializer storage fields.
 * Initializer extraction and shape-matching code paths depend on this decoded tensor contract.
 */
export type DecodedExternalOnnxTensor = {
  name?: string | null;
  dataType?: number | null;
  dims?: OnnxDecodedLongLike[] | null;
  floatData?: number[] | null;
  rawData?: OnnxDecodedBytes | null;
};

/**
 * Decoded ONNX graph payload containing decoded graph interfaces, initializer tables, and ordered node records for external import orchestration.
 * Graph traversal, initializer indexing, and topology validation all begin from this representation.
 */
export type DecodedExternalOnnxGraph = {
  input?: DecodedExternalOnnxValueInfo[] | null;
  output?: DecodedExternalOnnxValueInfo[] | null;
  valueInfo?: DecodedExternalOnnxValueInfo[] | null;
  initializer?: DecodedExternalOnnxTensor[] | null;
  node?: DecodedExternalOnnxNode[] | null;
};

/**
 * Decoded ONNX operator-set import payload carrying the domain string and version number used by external import compatibility checks.
 */
export type DecodedExternalOnnxOpsetImport = {
  domain?: string | null;
  version?: OnnxDecodedLongLike | null;
};

/**
 * Decoded ONNX model payload containing optional graph content and operator-set imports used to validate supported external import lanes.
 * External import entrypoints decode into this shape before compatibility and topology checks proceed.
 */
export type DecodedExternalOnnxModel = {
  graph?: DecodedExternalOnnxGraph | null;
  opsetImport?: DecodedExternalOnnxOpsetImport[] | null;
};

/**
 * Canonical single-layer payload for the external dense import lane, carrying input and output widths, weight values, biases, and the resolved activation operator.
 */
export type OnnxExternalDenseLayer = {
  inputWidth: number;
  outputWidth: number;
  weightValues: number[];
  biasValues: number[];
  activation: OnnxActivationOperation;
};

/**
 * Canonical importer-owned dense chain derived from an accepted external binary graph, collecting opset version, IO widths, and an ordered layer list.
 */
export type OnnxExternalDenseChain = {
  opsetVersion: number;
  inputWidth: number;
  outputWidth: number;
  layers: OnnxExternalDenseLayer[];
};

/**
 * Named rejection category set for the external import lane; each string label identifies a distinct failure class so callers can route errors without string matching.
 */
export type OnnxExternalImportErrorCategory =
  | 'invalid-binary'
  | 'invalid-model'
  | 'unsupported-opset'
  | 'unsupported-domain'
  | 'unsupported-topology'
  | 'unsupported-node'
  | 'unsupported-attribute'
  | 'unsupported-activation'
  | 'unsupported-tensor-type'
  | 'missing-initializer'
  | 'duplicate-initializer'
  | 'shape-mismatch'
  | 'rank-mismatch';

/**
 * Error raised when an external ONNX binary falls outside the first supported import lane.
 */
export class OnnxExternalImportError extends Error {
  readonly category: OnnxExternalImportErrorCategory;

  /**
   * @param category Named rejection category.
   * @param message Human-readable rejection reason.
   */
  constructor(category: OnnxExternalImportErrorCategory, message: string) {
    super(message);
    this.category = category;
    this.name = 'OnnxExternalImportError';
  }
}
