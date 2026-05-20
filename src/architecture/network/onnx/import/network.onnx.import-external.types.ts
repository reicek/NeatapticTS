import type { OnnxActivationOperation } from '../network.onnx.utils.types';

/** Plain decoded bytes field from `onnx-proto` object conversion. */
export type OnnxDecodedBytes = string | Uint8Array | number[];

/** Decoded ONNX long-like field represented by `onnx-proto`. */
export type OnnxDecodedLongLike = number | string | { toString(): string };

/** Decoded ONNX dimension payload. */
export type DecodedExternalOnnxDimension = {
  dimValue?: OnnxDecodedLongLike | null;
  dimParam?: string | null;
};

/** Decoded ONNX tensor-type payload. */
export type DecodedExternalOnnxTensorType = {
  elemType?: number | null;
  shape?: {
    dim?: DecodedExternalOnnxDimension[] | null;
  } | null;
};

/** Decoded ONNX value-info payload. */
export type DecodedExternalOnnxValueInfo = {
  name?: string | null;
  type?: {
    tensorType?: DecodedExternalOnnxTensorType | null;
  } | null;
};

/** Decoded ONNX attribute payload. */
export type DecodedExternalOnnxAttribute = {
  name?: string | null;
  f?: number | null;
  i?: OnnxDecodedLongLike | null;
  s?: OnnxDecodedBytes | null;
};

/** Decoded ONNX node payload. */
export type DecodedExternalOnnxNode = {
  name?: string | null;
  domain?: string | null;
  opType?: string | null;
  input?: string[] | null;
  output?: string[] | null;
  attribute?: DecodedExternalOnnxAttribute[] | null;
};

/** Decoded ONNX tensor payload. */
export type DecodedExternalOnnxTensor = {
  name?: string | null;
  dataType?: number | null;
  dims?: OnnxDecodedLongLike[] | null;
  floatData?: number[] | null;
  rawData?: OnnxDecodedBytes | null;
};

/** Decoded ONNX graph payload. */
export type DecodedExternalOnnxGraph = {
  input?: DecodedExternalOnnxValueInfo[] | null;
  output?: DecodedExternalOnnxValueInfo[] | null;
  valueInfo?: DecodedExternalOnnxValueInfo[] | null;
  initializer?: DecodedExternalOnnxTensor[] | null;
  node?: DecodedExternalOnnxNode[] | null;
};

/** Decoded ONNX operator-set import payload. */
export type DecodedExternalOnnxOpsetImport = {
  domain?: string | null;
  version?: OnnxDecodedLongLike | null;
};

/** Decoded ONNX model payload. */
export type DecodedExternalOnnxModel = {
  graph?: DecodedExternalOnnxGraph | null;
  opsetImport?: DecodedExternalOnnxOpsetImport[] | null;
};

/** Canonical one-layer affine-plus-activation payload for the external dense lane. */
export type OnnxExternalDenseLayer = {
  inputWidth: number;
  outputWidth: number;
  weightValues: number[];
  biasValues: number[];
  activation: OnnxActivationOperation;
};

/** Canonical importer-owned dense chain derived from an accepted external binary graph. */
export type OnnxExternalDenseChain = {
  opsetVersion: number;
  inputWidth: number;
  outputWidth: number;
  layers: OnnxExternalDenseLayer[];
};

/** Named rejection categories for the first external import lane. */
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
