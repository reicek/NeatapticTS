import Connection from '../connection';

/**
 * Runtime interface for accessing node internal properties.
 * Nodes have runtime properties for connections, bias, and squash that aren't in the public interface.
 */
export interface NodeInternals {
  connections: {
    in: Connection[];
    out: Connection[];
    self: Connection[];
  };
  bias: number;
  squash: ((x: number, derivate?: boolean) => number) & { name?: string };
}

/** Options controlling ONNX export behavior (Phase 1). */
export interface OnnxExportOptions {
  opset?: number;
  includeMetadata?: boolean;
  batchDimension?: boolean;
  legacyNodeOrdering?: boolean;
  producerName?: string;
  producerVersion?: string;
  docString?: string;
  allowPartialConnectivity?: boolean;
  allowMixedActivations?: boolean;
  allowRecurrent?: boolean;
  recurrentSingleStep?: boolean;
  conv2dMappings?: Conv2DMapping[];
  pool2dMappings?: Pool2DMapping[];
  validateConvSharing?: boolean;
  flattenAfterPooling?: boolean;
}

/** Mapping declaration for treating a fully-connected layer as a 2D convolution during export. */
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

/** Mapping describing a pooling operation inserted after a given export-layer index. */
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

/** ONNX tensor type shape dimension. */
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

/** ONNX node attribute. */
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

export type OnnxModel = {
  ir_version?: number;
  opset_import?: { version: number; domain: string }[];
  producer_name?: string;
  producer_version?: string;
  doc_string?: string;
  metadata_props?: { key: string; value: string }[];
  graph: OnnxGraph;
};

export type OnnxGraph = {
  inputs: OnnxValueInfo[];
  outputs: OnnxValueInfo[];
  initializer: OnnxTensor[];
  node: OnnxNode[];
};

export type OnnxTensor = {
  name: string;
  data_type: number;
  dims: number[];
  float_data: number[];
};

export type OnnxNode = {
  op_type: string;
  input: string[];
  output: string[];
  name: string;
  attributes?: OnnxAttribute[];
};
