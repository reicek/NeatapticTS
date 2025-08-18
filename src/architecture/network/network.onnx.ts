/**
 * ONNX export/import utilities for a constrained, documented subset of networks.
 *
 * Phase Coverage (incremental roadmap implemented so far):
 *  - Phase 1: Deterministic layered MLP export (Gemm + Activation pairs) with basic metadata.
 *  - Phase 2: Optional partial connectivity (missing edges -> 0 weight) and mixed per-neuron activations
 *              (decomposed into per-neuron Gemm + Activation + Concat) via `allowPartialConnectivity` /
 *              `allowMixedActivations`.
 *  - Phase 3 (baseline): Multi-layer self‑recurrence single‑step representation (`allowRecurrent` +
 *              `recurrentSingleStep`) adding per-recurrent-layer previous state inputs and diagonal R matrices.
 *  - Phase 3 (experimental extension): Heuristic detection + emission of simplified LSTM / GRU fused nodes
 *              (no sequence axis, simplified bias & recurrence handling) while retaining original Gemm path.
 *
 * Scope & Assumptions (current):
 *  - Network must be strictly layered and acyclic (feed‑forward between layers; optional self recurrence within
 *    hidden layers when enabled).
 *  - Homogeneous activation per layer unless `allowMixedActivations` is true (then per-neuron decomposition used).
 *  - Only a minimal ONNX tensor / node subset is emitted (no external ONNX proto dependency; pure JSON shape).
 *  - Recurrent support limited to: (a) self-connections mapped to diagonal Rk matrices (single step),
 *    (b) experimental fused LSTM/GRU heuristics relying on equal partition patterns (not spec-complete).
 *  - LSTM / GRU biases currently single segment (Wb only) and recurrent bias (Rb) implicitly zero; ordering of
 *    gates documented in code comments (may differ from canonical ONNX gate ordering and will be normalized later).
 *
 * Metadata Keys (may appear in `model.metadata_props` when `includeMetadata` true):
 *  - `layer_sizes`: JSON array of hidden layer sizes.
 *  - `recurrent_single_step`: JSON array of 1-based hidden layer indices with exported self recurrence.
 *  - `lstm_groups_stub`: Heuristic grouping stubs for prospective LSTM layers (pre-emission discovery data).
 *  - `lstm_emitted_layers` / `gru_emitted_layers`: Arrays of export-layer indices where fused nodes were emitted.
 *  - `rnn_pattern_fallback`: Records near-miss pattern sizes for diagnostic purposes.
 *
 * Design Goals:
 *  - Zero heavy runtime dependencies; the structure is intentionally lightweight & serializable.
 *  - Early, explicit structural validation with actionable error messages.
 *  - Transparent, stepwise transform for testability and deterministic round-tripping.
 *
 * Limitations / TODO (tracked for later phases):
 *  - Proper ONNX-compliant LSTM/GRU biases (split Wb/Rb) & complete gate ordering alignment.
 *  - Pruning or replacing redundant Gemm graph segments when fused recurrent ops are emitted (currently both kept).
 *  - Multi-time-step sequence handling (currently single-step recurrent representation only).
 *  - Richer recurrence (off-diagonal intra-layer connectivity) and gating reconstruction fidelity.
 *
 * NOTE: Import is only guaranteed to work for models produced by {@link exportToONNX}; arbitrary ONNX graphs are
 * NOT supported. Experimental fused recurrent nodes are best-effort and may silently degrade if shapes mismatch.
 */

import * as methods from '../../methods/methods';
import type Network from '../network';
import Connection from '../connection';

// ---------------------------------------------------------------------------
// Phase 1 Enhancements (metadata + options + ordering normalization)
// ---------------------------------------------------------------------------

/** Options controlling ONNX export behavior (Phase 1). */
export interface OnnxExportOptions {
  /** ONNX opset version (default 18). */
  opset?: number;
  /** Emit ModelProto-level metadata (ir_version, opset_import, producer fields). */
  includeMetadata?: boolean;
  /** Add a symbolic batch dimension ("N") to input/output shapes. */
  batchDimension?: boolean;
  /** Preserve legacy Activation-before-Gemm node ordering (default false => Gemm then Activation). */
  legacyNodeOrdering?: boolean;
  /** Producer name override (defaults to 'neataptic-ts'). */
  producerName?: string;
  /** Producer version override (defaults to package.json version when available). */
  producerVersion?: string;
  /** Optional doc string override. */
  docString?: string;
  /** Allow partial (non fully-connected) layers by inserting 0 weights for missing connections (Phase 2). */
  allowPartialConnectivity?: boolean;
  /** Allow heterogeneous activations within a layer (currently downgraded to Identity with warning if true; placeholder for future per-neuron export). */
  allowMixedActivations?: boolean;
  /**
   * Enable recurrent export logic (Phase 3 baseline + experimental extensions).
   * When combined with `recurrentSingleStep`, per-hidden-layer previous state inputs and diagonal R matrices
   * (self connections) are emitted. Also unlocks heuristic LSTM/GRU detection & fused node emission.
   */
  allowRecurrent?: boolean;
  /** Emit single-step recurrent form (adds per-recurrent-layer previous state inputs + Rk diagonal recurrent matrices). */
  recurrentSingleStep?: boolean;
  /**
   * Phase 4 (groundwork): Explicit 2D convolution layer mappings.
   * Provide an array of mapping specs declaring that certain export-layer indices (the same indices used for Gemm layers: 1-based hidden, final output at hiddenCount+1)
   * should be serialized as ONNX Conv nodes instead of Gemm+Activation. This is a manual seed before heuristic detection exists.
   * IMPORTANT: Import currently does not reconstruct Conv; models relying on Conv export will not round-trip to convolution semantics yet.
   */
  conv2dMappings?: Conv2DMapping[];
  /**
   * Phase 4: Explicit 2D pooling mappings. Each mapping injects a pooling node (MaxPool or AveragePool)
   * immediately AFTER the specified export-layer activation output (Layer_{index} or act_conv_l{index}).
   * Import currently ignores pooling (dense expansion deferred); use for structural experimentation only.
   */
  pool2dMappings?: Pool2DMapping[];
  /** When true, validate declared Conv2D mappings for weight sharing across all spatial positions (best-effort). */
  validateConvSharing?: boolean;
  /** When true, insert a Flatten node immediately after each emitted pooling node (Phase 4 extension). */
  flattenAfterPooling?: boolean;
}

/**
 * Mapping declaration for treating a fully-connected layer as a 2D convolution during export.
 * This assumes the dense layer was originally synthesized from a convolution with weight sharing; we reconstitute spatial metadata.
 * Each mapping references an export-layer index (1-based across hidden layers, output layer would be hiddenCount+1) and supplies spatial/kernel hyperparameters.
 * Validation ensures that input spatial * channels product equals the previous layer width and that output channels * output spatial equals the current layer width.
 */
export interface Conv2DMapping {
  /** Export-layer index to reinterpret as Conv (1-based hidden index; cannot be the output layer for this groundwork stage). */
  layerIndex: number;
  /** Input spatial height. */
  inHeight: number;
  /** Input spatial width. */
  inWidth: number;
  /** Number of input channels (so previous layer width must equal inHeight*inWidth*inChannels). */
  inChannels: number;
  /** Kernel height. */
  kernelHeight: number;
  /** Kernel width. */
  kernelWidth: number;
  /** Stride along height. */
  strideHeight: number;
  /** Stride along width. */
  strideWidth: number;
  /** Padding (top,bottom,left,right) – symmetric simplified representation used for forward shape math, exported as pads attribute: [pt, pl, pb, pr]. */
  padTop?: number;
  padBottom?: number;
  padLeft?: number;
  padRight?: number;
  /** Output spatial height. */
  outHeight: number;
  /** Output spatial width. */
  outWidth: number;
  /** Number of output channels (so outChannels*outHeight*outWidth must equal this layer's neuron count). */
  outChannels: number;
  /** Activation op_type to apply post Conv (defaults to per-layer activation detection). */
  activation?: string;
}

/** Mapping describing a pooling operation inserted after a given export-layer index. */
export interface Pool2DMapping {
  afterLayerIndex: number; // layer index whose output is pooled
  type: 'MaxPool' | 'AveragePool';
  kernelHeight: number;
  kernelWidth: number;
  strideHeight: number;
  strideWidth: number;
  padTop?: number;
  padBottom?: number;
  padLeft?: number;
  padRight?: number;
  activation?: string; // optional activation after pool (not yet used)
}

// --- Lightweight ONNX type aliases (minimal subset used for export/import) ---
export type OnnxModel = {
  ir_version?: number;
  opset_import?: { version: number; domain: string }[];
  producer_name?: string;
  producer_version?: string;
  doc_string?: string;
  metadata_props?: { key: string; value: string }[];
  graph: OnnxGraph;
};
type OnnxGraph = {
  inputs: any[];
  outputs: any[];
  initializer: OnnxTensor[];
  node: OnnxNode[];
};
type OnnxTensor = {
  name: string;
  data_type: number;
  dims: number[];
  float_data: number[];
};
type OnnxNode = {
  op_type: string;
  input: string[];
  output: string[];
  name: string;
  attributes?: any[];
};

// ---------------------------------------------------------------------------
// Internal helpers (not exported)
// ---------------------------------------------------------------------------

/** Rebuild the network's flat connections array from each node's outgoing list (avoids circular import). */
function rebuildConnectionsLocal(networkLike: any): void {
  /** Set used to deduplicate connection objects. */
  const uniqueConnections = new Set<any>();
  networkLike.nodes.forEach((node: any) =>
    node.connections?.out.forEach((conn: any) => uniqueConnections.add(conn))
  );
  networkLike.connections = Array.from(uniqueConnections);
}

/** Map an internal activation function (squash) to an ONNX op_type, defaulting to Identity. */
function mapActivationToOnnx(squash: any): string {
  const upperName = (squash?.name || '').toUpperCase();
  if (upperName.includes('TANH')) return 'Tanh';
  if (upperName.includes('LOGISTIC') || upperName.includes('SIGMOID'))
    return 'Sigmoid';
  if (upperName.includes('RELU')) return 'Relu';
  if (squash)
    console.warn(
      `Unsupported activation function ${squash.name} for ONNX export, defaulting to Identity.`
    );
  return 'Identity';
}

/** Infer strictly layered ordering from a network, ensuring feed-forward fully-connected structure. */
function inferLayerOrdering(network: Network): any[][] {
  /** All input nodes (first layer). */
  const inputNodes = network.nodes.filter((n: any) => n.type === 'input');
  /** All output nodes (final layer). */
  const outputNodes = network.nodes.filter((n: any) => n.type === 'output');
  /** All hidden nodes requiring layer inference. */
  const hiddenNodes = network.nodes.filter((n: any) => n.type === 'hidden');
  if (hiddenNodes.length === 0) return [inputNodes, outputNodes];
  /** Remaining hidden nodes to allocate. */
  let remainingHidden = [...hiddenNodes];
  /** Previously accepted layer (starts at inputs). */
  let previousLayer = inputNodes;
  /** Accumulated layers (excluding final output which is appended later). */
  const layerAccumulator: any[][] = [];
  while (remainingHidden.length) {
    /** Hidden nodes whose inbound connections originate only from previousLayer. */
    const currentLayer = remainingHidden.filter((hidden) =>
      hidden.connections.in.every((conn: any) =>
        previousLayer.includes(conn.from)
      )
    );
    if (!currentLayer.length)
      throw new Error(
        'Invalid network structure for ONNX export: cannot resolve layered ordering.'
      );
    layerAccumulator.push(previousLayer);
    previousLayer = currentLayer;
    remainingHidden = remainingHidden.filter((h) => !currentLayer.includes(h));
  }
  // Append the last hidden layer and output layer.
  layerAccumulator.push(previousLayer);
  layerAccumulator.push(outputNodes);
  return layerAccumulator;
}

/** Validate layer connectivity and (optionally) homogeneity; mixed activations allowed with per-neuron decomposition. */
function validateLayerHomogeneityAndConnectivity(
  layers: any[][],
  network: Network,
  options: OnnxExportOptions
): void {
  for (let layerIndex = 1; layerIndex < layers.length; layerIndex++) {
    /** Nodes in the source (previous) layer feeding current layer. */
    const previousLayerNodes = layers[layerIndex - 1];
    /** Nodes in the current destination layer being validated. */
    const currentLayerNodes = layers[layerIndex];
    /** Set of activation names encountered. */
    const activationNameSet = new Set(
      currentLayerNodes.map((n: any) => n.squash && n.squash.name)
    );
    if (activationNameSet.size > 1 && !options.allowMixedActivations)
      throw new Error(
        `ONNX export error: Mixed activation functions detected in layer ${layerIndex}. (enable allowMixedActivations to decompose layer)`
      );
    if (activationNameSet.size > 1 && options.allowMixedActivations)
      console.warn(
        `Warning: Mixed activations in layer ${layerIndex}; exporting per-neuron Gemm + Activation (+Concat) baseline.`
      );
    for (const targetNode of currentLayerNodes) {
      for (const sourceNode of previousLayerNodes) {
        const isConnected = targetNode.connections.in.some(
          (conn: any) => conn.from === sourceNode
        );
        if (!isConnected && !options.allowPartialConnectivity)
          throw new Error(
            `ONNX export error: Missing connection from node ${sourceNode.index} to node ${targetNode.index} in layer ${layerIndex}. (enable allowPartialConnectivity)`
          );
      }
    }
  }
}

/** Construct the ONNX model graph (initializers + nodes) given validated layers. */
/**
 * Internal builder: constructs initializers, graph inputs/outputs, and node list from validated layers.
 *
 * Responsibilities:
 *  - Allocate model & (optional) producer metadata.
 *  - Emit per-layer (or per-neuron) Gemm/Activation nodes (legacy or modern ordering).
 *  - When recurrent single-step enabled: inject previous hidden state inputs and diagonal recurrent matrices (Rk),
 *    plus additive fusion (Gemm_in + Gemm_rec -> Add -> Activation).
 *  - When recurrent enabled (experimental heuristics): attempt simplified LSTM/GRU fused node emission by detecting
 *    equal partitions of hidden layer neurons (5-way for LSTM, 4-way for GRU); append initializers LSTM_W/R/B or
 *    GRU_W/R/B without removing the original unfused path yet (future optimization phase).
 *  - Record metadata for layer sizes and recurrent layers when requested.
 *
 * Notes:
 *  - Bias handling for fused recurrent ops is simplified (Rb assumed zero).
 *  - Gate ordering chosen: LSTM [input, forget, cell, output]; GRU [update, reset, candidate].
 *  - Safety: if heuristic shapes mismatch expectations the fused node is skipped silently (metadata still may note fallback).
 */
function buildOnnxModel(
  network: Network,
  layers: any[][],
  options: OnnxExportOptions = {}
): OnnxModel {
  const {
    includeMetadata = false,
    opset = 18,
    batchDimension = false,
    legacyNodeOrdering = false,
    producerName = 'neataptic-ts',
    producerVersion,
    docString,
  } = options;
  /** Input layer nodes (used for input tensor dimension). */
  const inputLayerNodes = layers[0];
  /** Output layer nodes (used for output tensor dimension). */
  const outputLayerNodes = layers[layers.length - 1];
  const batchDims = batchDimension
    ? [{ dim_param: 'N' }, { dim_value: inputLayerNodes.length }]
    : [{ dim_value: inputLayerNodes.length }];
  const outBatchDims = batchDimension
    ? [{ dim_param: 'N' }, { dim_value: outputLayerNodes.length }]
    : [{ dim_value: outputLayerNodes.length }];
  /** Mutable ONNX model under construction (with optional metadata). */
  const model: OnnxModel = {
    graph: {
      inputs: [
        {
          name: 'input',
          type: {
            tensor_type: {
              elem_type: 1,
              shape: { dim: batchDims },
            },
          },
        },
      ],
      outputs: [
        {
          name: 'output',
          type: {
            tensor_type: {
              elem_type: 1,
              shape: { dim: outBatchDims },
            },
          },
        },
      ],
      initializer: [],
      node: [],
    },
  };
  if (includeMetadata) {
    const pkgVersion = (() => {
      try {
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        return require('../../../package.json').version;
      } catch {
        return '0.0.0';
      }
    })();
    model.ir_version = 9; // conservative default
    model.opset_import = [{ version: opset, domain: '' }];
    model.producer_name = producerName;
    model.producer_version = producerVersion || pkgVersion;
    model.doc_string =
      docString ||
      'Exported from NeatapticTS ONNX exporter (phases 1-2 baseline)';
  }
  /** Name of the tensor that feeds into the current Gemm. */
  let previousOutputName = 'input';
  // Detect per-hidden-layer self recurrence support (multi-layer extension of Phase 3 baseline)
  const recurrentLayerIndices: number[] = [];
  if (options.allowRecurrent && options.recurrentSingleStep) {
    for (let layerIndex = 1; layerIndex < layers.length - 1; layerIndex++) {
      const hiddenLayerNodes = layers[layerIndex];
      if (hiddenLayerNodes.some((n: any) => n.connections.self.length > 0)) {
        recurrentLayerIndices.push(layerIndex);
        // Add a graph input representing previous hidden state (same length as this hidden layer)
        const prevName =
          layerIndex === 1 ? 'hidden_prev' : `hidden_prev_l${layerIndex}`;
        model.graph.inputs.push({
          name: prevName,
          type: {
            tensor_type: {
              elem_type: 1,
              shape: {
                dim: batchDimension
                  ? [{ dim_param: 'N' }, { dim_value: hiddenLayerNodes.length }]
                  : [{ dim_value: hiddenLayerNodes.length }],
              },
            },
          },
        });
      }
    }
  }
  const hiddenSizesMetadata: number[] = [];
  for (let layerIndex = 1; layerIndex < layers.length; layerIndex++) {
    const previousLayerNodes = layers[layerIndex - 1];
    const currentLayerNodes = layers[layerIndex];
    const isOutputLayer = layerIndex === layers.length - 1;
    if (!isOutputLayer) hiddenSizesMetadata.push(currentLayerNodes.length);

    // Phase 4 groundwork: check if this layer is declared as a Conv2D mapping.
    const convSpec = options.conv2dMappings?.find(
      (m) => m.layerIndex === layerIndex
    );
    if (convSpec) {
      // Validate dimensional consistency.
      const prevWidthExpected =
        convSpec.inHeight * convSpec.inWidth * convSpec.inChannels;
      const prevWidthActual = previousLayerNodes.length;
      const thisWidthExpected =
        convSpec.outChannels * convSpec.outHeight * convSpec.outWidth;
      const thisWidthActual = currentLayerNodes.length;
      const pads = [
        convSpec.padTop || 0,
        convSpec.padLeft || 0,
        convSpec.padBottom || 0,
        convSpec.padRight || 0,
      ];
      const shapeValid =
        prevWidthExpected === prevWidthActual &&
        thisWidthExpected === thisWidthActual;
      if (!shapeValid) {
        console.warn(
          `Conv2D mapping for layer ${layerIndex} skipped: dimension mismatch (expected prev=${prevWidthExpected} got ${prevWidthActual}; expected this=${thisWidthExpected} got ${thisWidthActual}).`
        );
      } else {
        // Build kernel weights: For each output channel, for each input channel, for each kernel element (kH,kW), derive weight by sampling representative spatial position
        // Heuristic: map neuron ordering row-major over (outChannels, outHeight, outWidth). Representative neuron index for (oc) chosen at spatial (0,0): idx = oc*outHeight*outWidth.
        const W: number[] = [];
        const B: number[] = [];
        for (let oc = 0; oc < convSpec.outChannels; oc++) {
          const repIndex = oc * convSpec.outHeight * convSpec.outWidth; // first spatial location
          const repNeuron = currentLayerNodes[repIndex];
          B.push(repNeuron.bias);
          for (let ic = 0; ic < convSpec.inChannels; ic++) {
            for (let kh = 0; kh < convSpec.kernelHeight; kh++) {
              for (let kw = 0; kw < convSpec.kernelWidth; kw++) {
                // Map (ic, kh, kw) to dense weight index. We approximate by finding inbound connection from input feature corresponding to (ic, hStart+kh, wStart+kw) for hStart=wStart=0.
                const inputFeatureIndex =
                  ic * (convSpec.inHeight * convSpec.inWidth) +
                  kh * convSpec.inWidth +
                  kw;
                const sourceNode = previousLayerNodes[inputFeatureIndex];
                const conn = repNeuron.connections.in.find(
                  (cc: any) => cc.from === sourceNode
                );
                W.push(conn ? conn.weight : 0);
              }
            }
          }
        }
        const convWName = `ConvW${layerIndex - 1}`;
        const convBName = `ConvB${layerIndex - 1}`;
        model.graph.initializer.push({
          name: convWName,
          data_type: 1,
          dims: [
            convSpec.outChannels,
            convSpec.inChannels,
            convSpec.kernelHeight,
            convSpec.kernelWidth,
          ],
          float_data: W,
        });
        model.graph.initializer.push({
          name: convBName,
          data_type: 1,
          dims: [convSpec.outChannels],
          float_data: B,
        });
        const convOut = `Conv_${layerIndex}`;
        model.graph.node.push({
          op_type: 'Conv',
          input: [previousOutputName, convWName, convBName],
          output: [convOut],
          name: `conv_l${layerIndex}`,
          attributes: [
            {
              name: 'kernel_shape',
              type: 'INTS',
              ints: [convSpec.kernelHeight, convSpec.kernelWidth],
            },
            {
              name: 'strides',
              type: 'INTS',
              ints: [convSpec.strideHeight, convSpec.strideWidth],
            },
            { name: 'pads', type: 'INTS', ints: pads },
          ],
        });
        const actOp =
          convSpec.activation ||
          mapActivationToOnnx(currentLayerNodes[0].squash);
        const activationOutputName = `Layer_${layerIndex}`;
        model.graph.node.push({
          op_type: actOp,
          input: [convOut],
          output: [activationOutputName],
          name: `act_conv_l${layerIndex}`,
        });
        previousOutputName = activationOutputName;
        // Optional pooling insertion after conv or recurrent layer
        const poolSpecPostConv = options.pool2dMappings?.find(
          (p) => p.afterLayerIndex === layerIndex
        );
        if (poolSpecPostConv) {
          const kernel = [
            poolSpecPostConv.kernelHeight,
            poolSpecPostConv.kernelWidth,
          ];
          const strides = [
            poolSpecPostConv.strideHeight,
            poolSpecPostConv.strideWidth,
          ];
          const pads = [
            poolSpecPostConv.padTop || 0,
            poolSpecPostConv.padLeft || 0,
            poolSpecPostConv.padBottom || 0,
            poolSpecPostConv.padRight || 0,
          ];
          const poolOut = `Pool_${layerIndex}`;
          model.graph.node.push({
            op_type: poolSpecPostConv.type,
            input: [previousOutputName],
            output: [poolOut],
            name: `pool_after_l${layerIndex}`,
            attributes: [
              { name: 'kernel_shape', type: 'INTS', ints: kernel },
              { name: 'strides', type: 'INTS', ints: strides },
              { name: 'pads', type: 'INTS', ints: pads },
            ],
          });
          previousOutputName = poolOut;
          // Optional flatten bridging (Phase 4 extension)
          if (options.flattenAfterPooling) {
            const flatOut = `PoolFlat_${layerIndex}`;
            model.graph.node.push({
              op_type: 'Flatten',
              input: [previousOutputName],
              output: [flatOut],
              name: `flatten_after_l${layerIndex}`,
              attributes: [{ name: 'axis', type: 'INT', i: 1 }],
            });
            previousOutputName = flatOut;
            model.metadata_props = model.metadata_props || [];
            const flMeta = model.metadata_props.find(
              (m) => m.key === 'flatten_layers'
            );
            if (flMeta) {
              try {
                const arr = JSON.parse(flMeta.value);
                if (Array.isArray(arr) && !arr.includes(layerIndex)) {
                  arr.push(layerIndex);
                  flMeta.value = JSON.stringify(arr);
                }
              } catch {
                flMeta.value = JSON.stringify([layerIndex]);
              }
            } else {
              model.metadata_props.push({
                key: 'flatten_layers',
                value: JSON.stringify([layerIndex]),
              });
            }
          }
          model.metadata_props = model.metadata_props || [];
          const poolLayersMeta = model.metadata_props.find(
            (m) => m.key === 'pool2d_layers'
          );
          if (poolLayersMeta) {
            try {
              const arr = JSON.parse(poolLayersMeta.value);
              if (Array.isArray(arr) && !arr.includes(layerIndex)) {
                arr.push(layerIndex);
                poolLayersMeta.value = JSON.stringify(arr);
              }
            } catch {
              poolLayersMeta.value = JSON.stringify([layerIndex]);
            }
          } else {
            model.metadata_props.push({
              key: 'pool2d_layers',
              value: JSON.stringify([layerIndex]),
            });
          }
          const poolSpecsMeta = model.metadata_props.find(
            (m) => m.key === 'pool2d_specs'
          );
          if (poolSpecsMeta) {
            try {
              const arr = JSON.parse(poolSpecsMeta.value);
              if (Array.isArray(arr)) {
                arr.push({ ...poolSpecPostConv });
                poolSpecsMeta.value = JSON.stringify(arr);
              }
            } catch {
              poolSpecsMeta.value = JSON.stringify([poolSpecPostConv]);
            }
          } else {
            model.metadata_props.push({
              key: 'pool2d_specs',
              value: JSON.stringify([poolSpecPostConv]),
            });
          }
        }
        // Record metadata
        model.metadata_props = model.metadata_props || [];
        const convLayersMeta = model.metadata_props.find(
          (m) => m.key === 'conv2d_layers'
        );
        if (convLayersMeta) {
          try {
            const arr = JSON.parse(convLayersMeta.value);
            if (Array.isArray(arr) && !arr.includes(layerIndex)) {
              arr.push(layerIndex);
              convLayersMeta.value = JSON.stringify(arr);
            }
          } catch {
            convLayersMeta.value = JSON.stringify([layerIndex]);
          }
        } else {
          model.metadata_props.push({
            key: 'conv2d_layers',
            value: JSON.stringify([layerIndex]),
          });
        }
        const convSpecsMeta = model.metadata_props.find(
          (m) => m.key === 'conv2d_specs'
        );
        if (convSpecsMeta) {
          try {
            const arr = JSON.parse(convSpecsMeta.value);
            if (Array.isArray(arr)) {
              arr.push({ ...convSpec });
              convSpecsMeta.value = JSON.stringify(arr);
            }
          } catch {
            convSpecsMeta.value = JSON.stringify([convSpec]);
          }
        } else {
          model.metadata_props.push({
            key: 'conv2d_specs',
            value: JSON.stringify([convSpec]),
          });
        }
        continue; // move to next layer
      }
    }
    const mixed =
      options.allowMixedActivations &&
      new Set(currentLayerNodes.map((n: any) => n.squash && n.squash.name))
        .size > 1;
    if (recurrentLayerIndices.includes(layerIndex) && !isOutputLayer) {
      // Recurrent single-step path for this layer (only supports homogeneous activations)
      if (mixed)
        throw new Error(
          `Recurrent export does not yet support mixed activations in hidden layer ${layerIndex}.`
        );
      // Build feedforward weights W{layerIndex-1} / B{layerIndex-1}
      const weightMatrixValues: number[] = [];
      const biasVector: number[] = new Array(currentLayerNodes.length).fill(0);
      for (let r = 0; r < currentLayerNodes.length; r++) {
        const targetNode: any = currentLayerNodes[r];
        biasVector[r] = targetNode.bias;
        for (let c = 0; c < previousLayerNodes.length; c++) {
          const sourceNode = previousLayerNodes[c];
          const inboundConn = targetNode.connections.in.find(
            (conn: any) => conn.from === sourceNode
          );
          weightMatrixValues.push(inboundConn ? inboundConn.weight : 0);
        }
      }
      const weightTensorName = `W${layerIndex - 1}`;
      const biasTensorName = `B${layerIndex - 1}`;
      model.graph.initializer.push({
        name: weightTensorName,
        data_type: 1,
        dims: [currentLayerNodes.length, previousLayerNodes.length],
        float_data: weightMatrixValues,
      });
      model.graph.initializer.push({
        name: biasTensorName,
        data_type: 1,
        dims: [currentLayerNodes.length],
        float_data: biasVector,
      });
      // Recurrent weight matrix R{layerIndex-1} (self connections only currently; extension point for full intra-layer recurrence)
      const recurrentWeights: number[] = [];
      for (let r = 0; r < currentLayerNodes.length; r++) {
        for (let c = 0; c < currentLayerNodes.length; c++) {
          if (r === c) {
            const selfConn = currentLayerNodes[r].connections.self[0];
            recurrentWeights.push(selfConn ? selfConn.weight : 0);
          } else {
            recurrentWeights.push(0);
          }
        }
      }
      const rName = `R${layerIndex - 1}`;
      model.graph.initializer.push({
        name: rName,
        data_type: 1,
        dims: [currentLayerNodes.length, currentLayerNodes.length],
        float_data: recurrentWeights,
      });
      // Input Gemm (from previous layer output -> current hidden pre-activation)
      (model.graph.node as any).push({
        op_type: 'Gemm',
        input: [previousOutputName, weightTensorName, biasTensorName],
        output: [`Gemm_in_${layerIndex}`],
        name: `gemm_in_l${layerIndex}`,
        attributes: [
          { name: 'alpha', type: 'FLOAT', f: 1 },
          { name: 'beta', type: 'FLOAT', f: 1 },
          { name: 'transB', type: 'INT', i: 1 },
        ],
      });
      // Recurrent Gemm (previous hidden state * Rk)
      const prevHiddenInputName =
        layerIndex === 1 ? 'hidden_prev' : `hidden_prev_l${layerIndex}`;
      (model.graph.node as any).push({
        op_type: 'Gemm',
        input: [prevHiddenInputName, rName],
        output: [`Gemm_rec_${layerIndex}`],
        name: `gemm_rec_l${layerIndex}`,
        attributes: [
          { name: 'alpha', type: 'FLOAT', f: 1 },
          { name: 'beta', type: 'FLOAT', f: 1 },
          { name: 'transB', type: 'INT', i: 1 },
        ],
      });
      // Add fused input + recurrent
      model.graph.node.push({
        op_type: 'Add',
        input: [`Gemm_in_${layerIndex}`, `Gemm_rec_${layerIndex}`],
        output: [`RecurrentSum_${layerIndex}`],
        name: `add_recurrent_l${layerIndex}`,
      });
      // Activation
      model.graph.node.push({
        op_type: mapActivationToOnnx(currentLayerNodes[0].squash),
        input: [`RecurrentSum_${layerIndex}`],
        output: [`Layer_${layerIndex}`],
        name: `act_l${layerIndex}`,
      });
      previousOutputName = `Layer_${layerIndex}`;
    } else if (!mixed) {
      // Unified representation (fast path): single weight & bias tensors.
      const weightMatrixValues: number[] = [];
      const biasVector: number[] = new Array(currentLayerNodes.length).fill(0);
      for (let r = 0; r < currentLayerNodes.length; r++) {
        const targetNode: any = currentLayerNodes[r];
        biasVector[r] = targetNode.bias;
        for (let c = 0; c < previousLayerNodes.length; c++) {
          const sourceNode = previousLayerNodes[c];
          const inboundConn = targetNode.connections.in.find(
            (conn: any) => conn.from === sourceNode
          );
          weightMatrixValues.push(inboundConn ? inboundConn.weight : 0);
        }
      }
      const weightTensorName = `W${layerIndex - 1}`;
      const biasTensorName = `B${layerIndex - 1}`;
      const gemmOutputName = `Gemm_${layerIndex}`;
      const activationOutputName = `Layer_${layerIndex}`;
      model.graph.initializer.push({
        name: weightTensorName,
        data_type: 1,
        dims: [currentLayerNodes.length, previousLayerNodes.length],
        float_data: weightMatrixValues,
      });
      model.graph.initializer.push({
        name: biasTensorName,
        data_type: 1,
        dims: [currentLayerNodes.length],
        float_data: biasVector,
      });
      if (!legacyNodeOrdering) {
        (model.graph.node as any).push({
          op_type: 'Gemm',
          input: [previousOutputName, weightTensorName, biasTensorName],
          output: [gemmOutputName],
          name: `gemm_l${layerIndex}`,
          attributes: [
            { name: 'alpha', type: 'FLOAT', f: 1 },
            { name: 'beta', type: 'FLOAT', f: 1 },
            { name: 'transB', type: 'INT', i: 1 },
          ],
        });
        model.graph.node.push({
          op_type: mapActivationToOnnx(currentLayerNodes[0].squash),
          input: [gemmOutputName],
          output: [activationOutputName],
          name: `act_l${layerIndex}`,
        });
      } else {
        model.graph.node.push({
          op_type: mapActivationToOnnx(currentLayerNodes[0].squash),
          input: [gemmOutputName],
          output: [activationOutputName],
          name: `act_l${layerIndex}`,
        });
        (model.graph.node as any).push({
          op_type: 'Gemm',
          input: [previousOutputName, weightTensorName, biasTensorName],
          output: [gemmOutputName],
          name: `gemm_l${layerIndex}`,
          attributes: [
            { name: 'alpha', type: 'FLOAT', f: 1 },
            { name: 'beta', type: 'FLOAT', f: 1 },
            { name: 'transB', type: 'INT', i: 1 },
          ],
        });
      }
      previousOutputName = activationOutputName;
      // Optional pooling insertion after standard dense layer
      const poolSpecDense = options.pool2dMappings?.find(
        (p) => p.afterLayerIndex === layerIndex
      );
      if (poolSpecDense) {
        const kernel = [poolSpecDense.kernelHeight, poolSpecDense.kernelWidth];
        const strides = [poolSpecDense.strideHeight, poolSpecDense.strideWidth];
        const pads = [
          poolSpecDense.padTop || 0,
          poolSpecDense.padLeft || 0,
          poolSpecDense.padBottom || 0,
          poolSpecDense.padRight || 0,
        ];
        const poolOut = `Pool_${layerIndex}`;
        model.graph.node.push({
          op_type: poolSpecDense.type,
          input: [previousOutputName],
          output: [poolOut],
          name: `pool_after_l${layerIndex}`,
          attributes: [
            { name: 'kernel_shape', type: 'INTS', ints: kernel },
            { name: 'strides', type: 'INTS', ints: strides },
            { name: 'pads', type: 'INTS', ints: pads },
          ],
        });
        previousOutputName = poolOut;
        if (options.flattenAfterPooling) {
          const flatOut = `PoolFlat_${layerIndex}`;
          model.graph.node.push({
            op_type: 'Flatten',
            input: [previousOutputName],
            output: [flatOut],
            name: `flatten_after_l${layerIndex}`,
            attributes: [{ name: 'axis', type: 'INT', i: 1 }],
          });
          previousOutputName = flatOut;
          model.metadata_props = model.metadata_props || [];
          const flMeta = model.metadata_props.find(
            (m) => m.key === 'flatten_layers'
          );
          if (flMeta) {
            try {
              const arr = JSON.parse(flMeta.value);
              if (Array.isArray(arr) && !arr.includes(layerIndex)) {
                arr.push(layerIndex);
                flMeta.value = JSON.stringify(arr);
              }
            } catch {
              flMeta.value = JSON.stringify([layerIndex]);
            }
          } else {
            model.metadata_props.push({
              key: 'flatten_layers',
              value: JSON.stringify([layerIndex]),
            });
          }
        }
        model.metadata_props = model.metadata_props || [];
        const poolLayersMeta = model.metadata_props.find(
          (m) => m.key === 'pool2d_layers'
        );
        if (poolLayersMeta) {
          try {
            const arr = JSON.parse(poolLayersMeta.value);
            if (Array.isArray(arr) && !arr.includes(layerIndex)) {
              arr.push(layerIndex);
              poolLayersMeta.value = JSON.stringify(arr);
            }
          } catch {
            poolLayersMeta.value = JSON.stringify([layerIndex]);
          }
        } else {
          model.metadata_props.push({
            key: 'pool2d_layers',
            value: JSON.stringify([layerIndex]),
          });
        }
        const poolSpecsMeta = model.metadata_props.find(
          (m) => m.key === 'pool2d_specs'
        );
        if (poolSpecsMeta) {
          try {
            const arr = JSON.parse(poolSpecsMeta.value);
            if (Array.isArray(arr)) {
              arr.push({ ...poolSpecDense });
              poolSpecsMeta.value = JSON.stringify(arr);
            }
          } catch {
            poolSpecsMeta.value = JSON.stringify([poolSpecDense]);
          }
        } else {
          model.metadata_props.push({
            key: 'pool2d_specs',
            value: JSON.stringify([poolSpecDense]),
          });
        }
      }
    } else {
      // Per-neuron decomposition: Gemm + Activation per neuron, then Concat.
      const perNeuronActivationOutputs: string[] = [];
      currentLayerNodes.forEach((targetNode: any, idx: number) => {
        // Build single-row weight matrix for neuron idx.
        const weightRow: number[] = [];
        for (let c = 0; c < previousLayerNodes.length; c++) {
          const sourceNode = previousLayerNodes[c];
          const inboundConn = targetNode.connections.in.find(
            (conn: any) => conn.from === sourceNode
          );
          weightRow.push(inboundConn ? inboundConn.weight : 0);
        }
        const weightTensorName = `W${layerIndex - 1}_n${idx}`;
        const biasTensorName = `B${layerIndex - 1}_n${idx}`;
        const gemmOutputName = `Gemm_${layerIndex}_n${idx}`;
        const actOutputName = `Layer_${layerIndex}_n${idx}`;
        model.graph.initializer.push({
          name: weightTensorName,
          data_type: 1,
          dims: [1, previousLayerNodes.length],
          float_data: weightRow,
        });
        model.graph.initializer.push({
          name: biasTensorName,
          data_type: 1,
          dims: [1],
          float_data: [targetNode.bias],
        });
        (model.graph.node as any).push({
          op_type: 'Gemm',
          input: [previousOutputName, weightTensorName, biasTensorName],
          output: [gemmOutputName],
          name: `gemm_l${layerIndex}_n${idx}`,
          attributes: [
            { name: 'alpha', type: 'FLOAT', f: 1 },
            { name: 'beta', type: 'FLOAT', f: 1 },
            { name: 'transB', type: 'INT', i: 1 },
          ],
        });
        model.graph.node.push({
          op_type: mapActivationToOnnx(targetNode.squash),
          input: [gemmOutputName],
          output: [actOutputName],
          name: `act_l${layerIndex}_n${idx}`,
        });
        perNeuronActivationOutputs.push(actOutputName);
      });
      const activationOutputName = `Layer_${layerIndex}`;
      model.graph.node.push({
        op_type: 'Concat',
        input: perNeuronActivationOutputs,
        output: [activationOutputName],
        name: `concat_l${layerIndex}`,
        attributes: [{ name: 'axis', type: 'INT', i: batchDimension ? 1 : 0 }],
      });
      previousOutputName = activationOutputName;
      const poolSpecPerNeuron = options.pool2dMappings?.find(
        (p) => p.afterLayerIndex === layerIndex
      );
      if (poolSpecPerNeuron) {
        const kernel = [
          poolSpecPerNeuron.kernelHeight,
          poolSpecPerNeuron.kernelWidth,
        ];
        const strides = [
          poolSpecPerNeuron.strideHeight,
          poolSpecPerNeuron.strideWidth,
        ];
        const pads = [
          poolSpecPerNeuron.padTop || 0,
          poolSpecPerNeuron.padLeft || 0,
          poolSpecPerNeuron.padBottom || 0,
          poolSpecPerNeuron.padRight || 0,
        ];
        const poolOut = `Pool_${layerIndex}`;
        model.graph.node.push({
          op_type: poolSpecPerNeuron.type,
          input: [previousOutputName],
          output: [poolOut],
          name: `pool_after_l${layerIndex}`,
          attributes: [
            { name: 'kernel_shape', type: 'INTS', ints: kernel },
            { name: 'strides', type: 'INTS', ints: strides },
            { name: 'pads', type: 'INTS', ints: pads },
          ],
        });
        previousOutputName = poolOut;
        if (options.flattenAfterPooling) {
          const flatOut = `PoolFlat_${layerIndex}`;
          model.graph.node.push({
            op_type: 'Flatten',
            input: [previousOutputName],
            output: [flatOut],
            name: `flatten_after_l${layerIndex}`,
            attributes: [{ name: 'axis', type: 'INT', i: 1 }],
          });
          previousOutputName = flatOut;
          model.metadata_props = model.metadata_props || [];
          const flMeta = model.metadata_props.find(
            (m) => m.key === 'flatten_layers'
          );
          if (flMeta) {
            try {
              const arr = JSON.parse(flMeta.value);
              if (Array.isArray(arr) && !arr.includes(layerIndex)) {
                arr.push(layerIndex);
                flMeta.value = JSON.stringify(arr);
              }
            } catch {
              flMeta.value = JSON.stringify([layerIndex]);
            }
          } else {
            model.metadata_props.push({
              key: 'flatten_layers',
              value: JSON.stringify([layerIndex]),
            });
          }
        }
        model.metadata_props = model.metadata_props || [];
        const poolLayersMeta = model.metadata_props.find(
          (m) => m.key === 'pool2d_layers'
        );
        if (poolLayersMeta) {
          try {
            const arr = JSON.parse(poolLayersMeta.value);
            if (Array.isArray(arr) && !arr.includes(layerIndex)) {
              arr.push(layerIndex);
              poolLayersMeta.value = JSON.stringify(arr);
            }
          } catch {
            poolLayersMeta.value = JSON.stringify([layerIndex]);
          }
        } else {
          model.metadata_props.push({
            key: 'pool2d_layers',
            value: JSON.stringify([layerIndex]),
          });
        }
        const poolSpecsMeta = model.metadata_props.find(
          (m) => m.key === 'pool2d_specs'
        );
        if (poolSpecsMeta) {
          try {
            const arr = JSON.parse(poolSpecsMeta.value);
            if (Array.isArray(arr)) {
              arr.push({ ...poolSpecPerNeuron });
              poolSpecsMeta.value = JSON.stringify(arr);
            }
          } catch {
            poolSpecsMeta.value = JSON.stringify([poolSpecPerNeuron]);
          }
        } else {
          model.metadata_props.push({
            key: 'pool2d_specs',
            value: JSON.stringify([poolSpecPerNeuron]),
          });
        }
      }
    }
  }
  // Experimental: Emit fused LSTM nodes for layers matching 5-way partition heuristic (input, forget, cell, output, block)
  // Only if no mixed activations and recurrence allowed; we reuse existing weight matrices by concatenating.
  if (options.allowRecurrent) {
    for (let layerIndex = 1; layerIndex < layers.length - 1; layerIndex++) {
      const current = layers[layerIndex];
      const size = current.length;
      // Fallback markers: record if near pattern but not exact partition (heuristic)
      if (!model.metadata_props) model.metadata_props = [];
      if (size >= 8 && size < 10) {
        model.metadata_props.push({
          key: 'rnn_pattern_fallback',
          value: JSON.stringify({
            layer: layerIndex,
            reason: 'size_between_gru_lstm_thresholds',
          }),
        });
      }
      if (size >= 10 && size % 5 === 0) {
        const unit = size / 5;
        // Build flattened weight segments: treat previousOutputName at detection time (approximation: recompute source)
        const prevLayerNodes = layers[layerIndex - 1];
        const inputGate = current.slice(0, unit);
        const forgetGate = current.slice(unit, unit * 2);
        const cell = current.slice(unit * 2, unit * 3);
        const outputGate = current.slice(unit * 3, unit * 4);
        const outputBlock = current.slice(unit * 4, unit * 5);
        // Compose W and R following ONNX ordering: [i, o, f, c] (we'll pick a stable ordering; here i,f,c,o typical for some frameworks, but we document chosen ordering)
        const gateOrder = [inputGate, forgetGate, cell, outputGate];
        const numGates = gateOrder.length;
        const prevSize = prevLayerNodes.length;
        const W: number[] = []; // shape [numGates*unit, prevSize]
        const R: number[] = []; // shape [numGates*unit, unit]
        const B: number[] = []; // (optional) combine bias: Wb || Rb (we'll just duplicate biases, Rb zeros)
        for (let g = 0; g < numGates; g++) {
          const gate = gateOrder[g];
          for (let r = 0; r < unit; r++) {
            const neuron = gate[r];
            // Input weights
            for (let c = 0; c < prevSize; c++) {
              const source = prevLayerNodes[c];
              const conn = neuron.connections.in.find(
                (cc: any) => cc.from === source
              );
              W.push(conn ? conn.weight : 0);
            }
            // Recurrent (from cell outputBlock considered as hidden state proxy) – we approximate using self connections if exist else 0
            for (let c = 0; c < unit; c++) {
              // Map recurrence only for memory cell group currently (others 0) – simplistic placeholder
              if (gate === cell && c === r) {
                const selfConn = neuron.connections.self[0];
                R.push(selfConn ? selfConn.weight : 0);
              } else R.push(0);
            }
            // Bias (use neuron.bias as input bias; recurrent bias zero)
            B.push(neuron.bias);
          }
        }
        // Add initializers
        model.graph.initializer.push({
          name: `LSTM_W${layerIndex - 1}`,
          data_type: 1,
          dims: [numGates * unit, prevSize],
          float_data: W,
        });
        model.graph.initializer.push({
          name: `LSTM_R${layerIndex - 1}`,
          data_type: 1,
          dims: [numGates * unit, unit],
          float_data: R,
        });
        model.graph.initializer.push({
          name: `LSTM_B${layerIndex - 1}`,
          data_type: 1,
          dims: [numGates * unit],
          float_data: B,
        });
        // Emit pseudo LSTM node (non-spec; uses op_type 'LSTM' with minimal attributes). Input sequence length assumed 1 (no sequence dimension).
        model.graph.node.push({
          op_type: 'LSTM',
          input: [
            previousOutputName,
            `LSTM_W${layerIndex - 1}`,
            `LSTM_R${layerIndex - 1}`,
            `LSTM_B${layerIndex - 1}`,
          ],
          output: [`Layer_${layerIndex}_lstm_hidden`],
          name: `lstm_l${layerIndex}`,
          attributes: [
            { name: 'hidden_size', type: 'INT', i: unit },
            { name: 'layout', type: 'INT', i: 0 },
          ],
        });
        // NOTE: For now we do not replace earlier Gemm/Activation nodes; future pass could prune redundant nodes.
        model.metadata_props = model.metadata_props || [];
        // Aggregate LSTM emitted layer indices (avoid multiple single-element entries)
        const lstmMetaIdx = model.metadata_props.findIndex(
          (m) => m.key === 'lstm_emitted_layers'
        );
        if (lstmMetaIdx >= 0) {
          try {
            const arr = JSON.parse(model.metadata_props[lstmMetaIdx].value);
            if (Array.isArray(arr) && !arr.includes(layerIndex)) {
              arr.push(layerIndex);
              model.metadata_props[lstmMetaIdx].value = JSON.stringify(arr);
            }
          } catch {
            model.metadata_props[lstmMetaIdx].value = JSON.stringify([
              layerIndex,
            ]);
          }
        } else {
          model.metadata_props.push({
            key: 'lstm_emitted_layers',
            value: JSON.stringify([layerIndex]),
          });
        }
      }
      // GRU heuristic: 4-way equal partition (update, reset, candidate, output block)
      if (size >= 8 && size % 4 === 0) {
        const unitG = size / 4;
        const prevLayerNodes = layers[layerIndex - 1];
        const updateGate = current.slice(0, unitG);
        const resetGate = current.slice(unitG, unitG * 2);
        const candidate = current.slice(unitG * 2, unitG * 3);
        const outputBlock = current.slice(unitG * 3, unitG * 4);
        const gateOrderGRU = [updateGate, resetGate, candidate]; // ONNX uses [z, r, h]
        const numGatesGRU = gateOrderGRU.length;
        const prevSizeGRU = prevLayerNodes.length;
        const Wg: number[] = []; // [numGates*H, input]
        const Rg: number[] = []; // [numGates*H, H]
        const Bg: number[] = [];
        for (let g = 0; g < numGatesGRU; g++) {
          const gate = gateOrderGRU[g];
          for (let r = 0; r < unitG; r++) {
            const neuron = gate[r];
            for (let c = 0; c < prevSizeGRU; c++) {
              const src = prevLayerNodes[c];
              const conn = neuron.connections.in.find(
                (cc: any) => cc.from === src
              );
              Wg.push(conn ? conn.weight : 0);
            }
            // Recurrent weights: approximate using self-connection diagonal for candidate group only
            for (let c = 0; c < unitG; c++) {
              if (gate === candidate && c === r) {
                const selfConn = neuron.connections.self[0];
                Rg.push(selfConn ? selfConn.weight : 0);
              } else Rg.push(0);
            }
            Bg.push(neuron.bias);
          }
        }
        model.graph.initializer.push({
          name: `GRU_W${layerIndex - 1}`,
          data_type: 1,
          dims: [numGatesGRU * unitG, prevSizeGRU],
          float_data: Wg,
        });
        model.graph.initializer.push({
          name: `GRU_R${layerIndex - 1}`,
          data_type: 1,
          dims: [numGatesGRU * unitG, unitG],
          float_data: Rg,
        });
        model.graph.initializer.push({
          name: `GRU_B${layerIndex - 1}`,
          data_type: 1,
          dims: [numGatesGRU * unitG],
          float_data: Bg,
        });
        const prevOutName =
          layerIndex === 1 ? 'input' : `Layer_${layerIndex - 1}`;
        model.graph.node.push({
          op_type: 'GRU',
          input: [
            prevOutName,
            `GRU_W${layerIndex - 1}`,
            `GRU_R${layerIndex - 1}`,
            `GRU_B${layerIndex - 1}`,
          ],
          output: [`Layer_${layerIndex}_gru_hidden`],
          name: `gru_l${layerIndex}`,
          attributes: [
            { name: 'hidden_size', type: 'INT', i: unitG },
            { name: 'layout', type: 'INT', i: 0 },
          ],
        });
        model.metadata_props = model.metadata_props || [];
        const gruMetaIdx = model.metadata_props.findIndex(
          (m) => m.key === 'gru_emitted_layers'
        );
        if (gruMetaIdx >= 0) {
          try {
            const arr = JSON.parse(model.metadata_props[gruMetaIdx].value);
            if (Array.isArray(arr) && !arr.includes(layerIndex)) {
              arr.push(layerIndex);
              model.metadata_props[gruMetaIdx].value = JSON.stringify(arr);
            }
          } catch {
            model.metadata_props[gruMetaIdx].value = JSON.stringify([
              layerIndex,
            ]);
          }
        } else {
          model.metadata_props.push({
            key: 'gru_emitted_layers',
            value: JSON.stringify([layerIndex]),
          });
        }
      }
    }
  }
  if (includeMetadata) {
    model.metadata_props = model.metadata_props || [];
    model.metadata_props.push({
      key: 'layer_sizes',
      value: JSON.stringify(hiddenSizesMetadata),
    });
    if (recurrentLayerIndices.length) {
      model.metadata_props.push({
        key: 'recurrent_single_step',
        value: JSON.stringify(recurrentLayerIndices),
      });
    }
    // Optional: Conv weight sharing validation (Phase 4)
    if (
      options.validateConvSharing &&
      options.conv2dMappings &&
      options.conv2dMappings.length
    ) {
      const verified: number[] = [];
      const mismatched: number[] = [];
      for (const spec of options.conv2dMappings) {
        const layerIdx = spec.layerIndex;
        const prevLayerNodes = layers[layerIdx - 1];
        const layerNodes = layers[layerIdx];
        // Only validate if mapping actually emitted (metadata conv2d_layers already recorded earlier). Quick dimension sanity.
        if (!layerNodes || !prevLayerNodes) continue;
        const repPerChannel: number[][] = []; // flattened kernel per outChannel
        let allOk = true;
        for (let oc = 0; oc < spec.outChannels; oc++) {
          // Representative neuron (0,0)
          const repIndex = oc * (spec.outHeight * spec.outWidth);
          const repNeuron = layerNodes[repIndex];
          const kernel: number[] = [];
          for (let ic = 0; ic < spec.inChannels; ic++) {
            for (let kh = 0; kh < spec.kernelHeight; kh++) {
              for (let kw = 0; kw < spec.kernelWidth; kw++) {
                const inputFeatureIndex =
                  ic * (spec.inHeight * spec.inWidth) + kh * spec.inWidth + kw;
                const sourceNode = prevLayerNodes[inputFeatureIndex];
                const conn = repNeuron.connections.in.find(
                  (cc: any) => cc.from === sourceNode
                );
                kernel.push(conn ? conn.weight : 0);
              }
            }
          }
          repPerChannel.push(kernel);
        }
        // Compare each spatial position's kernel to representative
        const tol = 1e-9;
        for (let oc = 0; oc < spec.outChannels && allOk; oc++) {
          for (let oh = 0; oh < spec.outHeight && allOk; oh++) {
            for (let ow = 0; ow < spec.outWidth && allOk; ow++) {
              const idx =
                oc * (spec.outHeight * spec.outWidth) + oh * spec.outWidth + ow;
              const neuron = layerNodes[idx];
              if (!neuron) continue;
              let kPtr = 0;
              for (let ic = 0; ic < spec.inChannels && allOk; ic++) {
                const hBase = oh * spec.strideHeight - (spec.padTop || 0);
                const wBase = ow * spec.strideWidth - (spec.padLeft || 0);
                for (let kh = 0; kh < spec.kernelHeight && allOk; kh++) {
                  for (let kw = 0; kw < spec.kernelWidth && allOk; kw++) {
                    const ih = hBase + kh;
                    const iw = wBase + kw;
                    if (
                      ih < 0 ||
                      ih >= spec.inHeight ||
                      iw < 0 ||
                      iw >= spec.inWidth
                    ) {
                      kPtr++;
                      continue;
                    }
                    const inputFeatureIndex =
                      ic * (spec.inHeight * spec.inWidth) +
                      ih * spec.inWidth +
                      iw;
                    const srcNode = prevLayerNodes[inputFeatureIndex];
                    const conn = neuron.connections.in.find(
                      (cc: any) => cc.from === srcNode
                    );
                    const wVal = conn ? conn.weight : 0;
                    if (Math.abs(wVal - repPerChannel[oc][kPtr]) > tol) {
                      allOk = false;
                    }
                    kPtr++;
                  }
                }
              }
              if (!allOk) break;
            }
          }
        }
        if (allOk) verified.push(layerIdx);
        else {
          mismatched.push(layerIdx);
          console.warn(
            `Conv2D weight sharing mismatch detected in layer ${layerIdx}`
          );
        }
      }
      if (verified.length)
        model.metadata_props.push({
          key: 'conv2d_sharing_verified',
          value: JSON.stringify(verified),
        });
      if (mismatched.length)
        model.metadata_props.push({
          key: 'conv2d_sharing_mismatch',
          value: JSON.stringify(mismatched),
        });
    }
  }
  return model;
}

/** Extract hidden layer sizes from ONNX initializers (weight tensors). */
function deriveHiddenLayerSizes(
  initializers: OnnxTensor[],
  metadataProps?: { key: string; value: string }[]
): number[] {
  // Prefer metadata-provided ordering if available.
  const meta = metadataProps?.find((p) => p.key === 'layer_sizes');
  if (meta) {
    try {
      const parsed = JSON.parse(meta.value);
      if (Array.isArray(parsed)) return parsed;
    } catch {
      /* ignore parse error */
    }
  }
  // Fallback: infer by grouped weight tensor prefixes.
  const layerMap: Record<
    string,
    { aggregated?: OnnxTensor; perNeuron: OnnxTensor[] }
  > = {};
  initializers
    .filter((t) => t.name.startsWith('W'))
    .forEach((t) => {
      const m = /^W(\d+)(?:_n(\d+))?$/i.exec(t.name);
      if (!m) return;
      const layerIdx = m[1];
      layerMap[layerIdx] = layerMap[layerIdx] || { perNeuron: [] };
      if (m[2] !== undefined) layerMap[layerIdx].perNeuron.push(t);
      else layerMap[layerIdx].aggregated = t;
    });
  const sorted = Object.keys(layerMap)
    .map(Number)
    .sort((a, b) => a - b);
  if (!sorted.length) return [];
  const hidden: number[] = [];
  for (let i = 0; i < sorted.length - 1; i++) {
    const entry = layerMap[String(sorted[i])];
    if (entry.aggregated) hidden.push(entry.aggregated.dims[0]);
    else hidden.push(entry.perNeuron.length);
  }
  return hidden;
}

/** Apply weights & biases from ONNX initializers onto the newly created network. */
/**
 * Assign weights & biases to the freshly instantiated layered MLP.
 *
 * Responsibilities:
 *  - Standard dense (Gemm) layers: consume aggregated (Wk/Bk) or per-neuron (Wk_nX/Bk_nX) initializers.
 *  - Mixed activation or partial connectivity decompositions are handled transparently via per-neuron tensors.
 *  - Phase 4 (Conv2D groundwork): when metadata declares a layer as convolutional (`conv2d_layers` + `conv2d_specs`) and
 *    corresponding Conv initializers (ConvWk / ConvBk) are present, expand the convolution weights into the equivalent
 *    dense connection matrix assuming classical sliding window semantics (NCHW, single example, no dilation).
 *
 * Convolution Expansion Notes:
 *  - Layer indexing here uses export-layer indices: hidden layers are 1..H, output layer would be H+1 (Conv mapping currently only applied to hidden layers).
 *  - Conv weight tensor shape: [outChannels, inChannels, kH, kW]. Bias: [outChannels].
 *  - Input feature ordering assumed (channel-major): ic * (H*W) + ih * W + iw.
 *  - Output neuron ordering assumed: oc * (outH*outW) + oh * outW + ow.
 *  - For each output spatial position (oh, ow), receptive field origin = (oh*strideH - padTop, ow*strideW - padLeft).
 *  - If a kernel position maps outside input spatial bounds, it's treated as zero-padding; connection weight contribution omitted (dense connection retains its existing value or is set to 0 if we choose). Here we set weight to 0 for clarity.
 *  - Existing random initialization is overwritten deterministically.
 *  - This expansion is a lossy inverse only if the original dense layer did not strictly represent a convolution (weight sharing broken). We do not validate sharing yet (deferred per plan); we simply impose the convolutional structure.
 */
function assignWeightsAndBiases(
  network: Network,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  metadataProps?: { key: string; value: string }[]
): void {
  // Build map for quick initializer lookup.
  const initMap: Record<string, OnnxTensor> = {};
  onnx.graph.initializer.forEach((t: OnnxTensor) => (initMap[t.name] = t));
  const layerIndices = new Set<number>();
  Object.keys(initMap).forEach((name) => {
    const m = /^W(\d+)(?:_n(\d+))?$/i.exec(name);
    if (m) layerIndices.add(Number(m[1]));
  });
  const sorted = Array.from(layerIndices).sort((a, b) => a - b);
  sorted.forEach((layerIdx, sequentialIdx) => {
    const isHidden = sequentialIdx < hiddenLayerSizes.length;
    const currentLayerNodes = isHidden
      ? network.nodes
          .filter((n: any) => n.type === 'hidden')
          .slice(
            hiddenLayerSizes.slice(0, sequentialIdx).reduce((a, b) => a + b, 0),
            hiddenLayerSizes
              .slice(0, sequentialIdx + 1)
              .reduce((a, b) => a + b, 0)
          )
      : network.nodes.filter((n: any) => n.type === 'output');
    const previousLayerNodes =
      sequentialIdx === 0
        ? network.nodes.filter((n: any) => n.type === 'input')
        : network.nodes
            .filter((n: any) => n.type === 'hidden')
            .slice(
              hiddenLayerSizes
                .slice(0, sequentialIdx - 1)
                .reduce((a, b) => a + b, 0),
              hiddenLayerSizes
                .slice(0, sequentialIdx)
                .reduce((a, b) => a + b, 0)
            );
    const aggregated = initMap[`W${layerIdx}`];
    if (aggregated) {
      const bias = initMap[`B${layerIdx}`];
      for (let r = 0; r < currentLayerNodes.length; r++) {
        for (let c = 0; c < previousLayerNodes.length; c++) {
          const conn = previousLayerNodes[c].connections.out.find(
            (cc: any) => cc.to === currentLayerNodes[r]
          );
          if (conn)
            conn.weight =
              aggregated.float_data[r * previousLayerNodes.length + c];
        }
        currentLayerNodes[r].bias = bias.float_data[r];
      }
    } else {
      currentLayerNodes.forEach((node: any, neuronIdx: number) => {
        const w = initMap[`W${layerIdx}_n${neuronIdx}`];
        const b = initMap[`B${layerIdx}_n${neuronIdx}`];
        if (!w || !b) return;
        for (let c = 0; c < previousLayerNodes.length; c++) {
          const conn = previousLayerNodes[c].connections.out.find(
            (cc: any) => cc.to === node
          );
          if (conn) conn.weight = w.float_data[c];
        }
        node.bias = b.float_data[0];
      });
    }
  });

  // Phase 4: Convolutional layer expansion after standard dense assignment so Conv weights take precedence.
  try {
    const meta = metadataProps || [];
    const convLayersMeta = meta.find((m) => m.key === 'conv2d_layers');
    const convSpecsMeta = meta.find((m) => m.key === 'conv2d_specs');
    if (convLayersMeta && convSpecsMeta) {
      const convLayers: number[] = JSON.parse(convLayersMeta.value);
      const convSpecs: Conv2DMapping[] = JSON.parse(convSpecsMeta.value);
      convLayers.forEach((layerExportIndex) => {
        const spec = convSpecs.find((s) => s.layerIndex === layerExportIndex);
        if (!spec) return;
        // Hidden layer index (0-based among hidden layers)
        const hiddenIndex = layerExportIndex - 1;
        if (hiddenIndex < 0 || hiddenIndex >= hiddenLayerSizes.length) return; // only hidden supported
        const hiddenNodes = network.nodes.filter(
          (n: any) => n.type === 'hidden'
        );
        const start = hiddenLayerSizes
          .slice(0, hiddenIndex)
          .reduce((a, b) => a + b, 0);
        const end = start + hiddenLayerSizes[hiddenIndex];
        const layerNodes = hiddenNodes.slice(start, end);
        // Previous layer nodes (inputs to this conv layer)
        const prevLayerNodes =
          hiddenIndex === 0
            ? network.nodes.filter((n: any) => n.type === 'input')
            : hiddenNodes.slice(
                hiddenLayerSizes
                  .slice(0, hiddenIndex - 1)
                  .reduce((a, b) => a + b, 0),
                hiddenLayerSizes
                  .slice(0, hiddenIndex)
                  .reduce((a, b) => a + b, 0)
              );
        const Wt = onnx.graph.initializer.find(
          (t) => t.name === `ConvW${layerExportIndex - 1}`
        );
        const Bt = onnx.graph.initializer.find(
          (t) => t.name === `ConvB${layerExportIndex - 1}`
        );
        if (!Wt || !Bt) return; // type guard
        const [outChannels, inChannels, kH, kW] = Wt.dims as [
          number,
          number,
          number,
          number
        ];
        // Sanity check vs spec
        if (
          outChannels !== spec.outChannels ||
          inChannels !== spec.inChannels ||
          kH !== spec.kernelHeight ||
          kW !== spec.kernelWidth
        )
          return;
        const strideH = spec.strideHeight;
        const strideW = spec.strideWidth;
        const padTop = spec.padTop || 0;
        const padLeft = spec.padLeft || 0;
        const inH = spec.inHeight;
        const inW = spec.inWidth;
        const outH = spec.outHeight;
        const outW = spec.outWidth;
        // Helper to index weight tensor
        function kernelWeight(
          oc: number,
          ic: number,
          kh: number,
          kw: number
        ): number {
          const idx = ((oc * inChannels + ic) * kH + kh) * kW + kw;
          return Wt!.float_data[idx];
        }
        // Overwrite each neuron's bias & inbound weights according to convolution formula
        for (let oc = 0; oc < outChannels; oc++) {
          for (let oh = 0; oh < outH; oh++) {
            for (let ow = 0; ow < outW; ow++) {
              const neuronLinearIndex = oc * (outH * outW) + oh * outW + ow;
              const neuron = layerNodes[neuronLinearIndex];
              if (!neuron) continue;
              neuron.bias = Bt.float_data[oc];
              // Clear existing inbound weights first (retain connection objects)
              // Build map for quick lookup
              const inConnMap = new Map<any, any>();
              neuron.connections.in.forEach((c: any) =>
                inConnMap.set(c.from, c)
              );
              for (let ic = 0; ic < inChannels; ic++) {
                const ihBase = oh * strideH - padTop;
                const iwBase = ow * strideW - padLeft;
                for (let kh = 0; kh < kH; kh++) {
                  for (let kw = 0; kw < kW; kw++) {
                    const ih = ihBase + kh;
                    const iw = iwBase + kw;
                    if (ih < 0 || ih >= inH || iw < 0 || iw >= inW) continue; // outside bounds -> zero contribution
                    const inputFeatureIndex = ic * (inH * inW) + ih * inW + iw;
                    const srcNode = prevLayerNodes[inputFeatureIndex];
                    if (!srcNode) continue;
                    const conn = inConnMap.get(srcNode);
                    if (conn) conn.weight = kernelWeight(oc, ic, kh, kw);
                  }
                }
              }
            }
          }
        }
      });
    }
  } catch {
    // Swallow conv reconstruction errors (experimental)
  }
}

/** Map activation op_types from ONNX nodes back to internal activation functions. */
function assignActivationFunctions(
  network: Network,
  onnx: OnnxModel,
  hiddenLayerSizes: number[]
): void {
  const hiddenNodes = network.nodes.filter((n: any) => n.type === 'hidden');
  let hiddenOffset = 0;
  // Build map layer->array of per-neuron activation op_types.
  const perLayer: Record<number, string[]> = {};
  onnx.graph.node.forEach((n) => {
    if (
      !['Tanh', 'Sigmoid', 'Logistic', 'Relu', 'Identity'].includes(n.op_type)
    )
      return;
    const m = /^act_l(\d+)(?:_n(\d+))?$/i.exec(n.name || '');
    if (!m) return;
    const layerIdx = Number(m[1]);
    perLayer[layerIdx] = perLayer[layerIdx] || [];
    perLayer[layerIdx].push(n.op_type);
  });
  // Hidden layers (export layer index = hidden layer index + 1)
  for (let hl = 0; hl < hiddenLayerSizes.length; hl++) {
    const exportIdx = hl + 1;
    const ops = perLayer[exportIdx] || [];
    for (let i = 0; i < hiddenLayerSizes[hl]; i++) {
      const op = ops[i] || ops[0];
      let fn = methods.Activation.identity;
      switch (op) {
        case 'Tanh':
          fn = methods.Activation.tanh;
          break;
        case 'Sigmoid':
        case 'Logistic':
          fn = methods.Activation.sigmoid;
          break;
        case 'Relu':
          fn = methods.Activation.relu;
          break;
      }
      if (hiddenNodes[hiddenOffset + i])
        hiddenNodes[hiddenOffset + i].squash = fn;
    }
    hiddenOffset += hiddenLayerSizes[hl];
  }
  // Output layer (export index = hidden count + 1)
  const outputExportIndex = hiddenLayerSizes.length + 1;
  const outOps = perLayer[outputExportIndex] || [];
  const outputFnOp = outOps[0];
  let outputFn = methods.Activation.identity;
  switch (outputFnOp) {
    case 'Tanh':
      outputFn = methods.Activation.tanh;
      break;
    case 'Sigmoid':
    case 'Logistic':
      outputFn = methods.Activation.sigmoid;
      break;
    case 'Relu':
      outputFn = methods.Activation.relu;
      break;
  }
  network.nodes
    .filter((n: any) => n.type === 'output')
    .forEach((n: any) => (n.squash = outputFn));
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Export a minimal multilayer perceptron Network to a lightweight ONNX JSON object.
 *
 * Steps:
 *  1. Rebuild connection cache ensuring up-to-date adjacency.
 *  2. Index nodes for error messaging.
 *  3. Infer strict layer ordering (throws if structure unsupported).
 *  4. Validate homogeneity & full connectivity layer-to-layer.
 *  5. Build initializer tensors (weights + biases) and node list (Gemm + activation pairs).
 *
 * Constraints: See module doc. Throws descriptive errors when assumptions violated.
 */
export function exportToONNX(
  network: Network,
  options: OnnxExportOptions = {}
): OnnxModel {
  rebuildConnectionsLocal(network as any);
  network.nodes.forEach((node: any, idx: number) => (node.index = idx));
  if (!network.connections || network.connections.length === 0)
    throw new Error('ONNX export currently only supports simple MLPs');
  /** Layered node arrays (input, hidden..., output) inferred for export. */
  const layers = inferLayerOrdering(network);
  // Phase 3 extended: preliminary pattern scan for LSTM cell groupings.
  const lstmPatternStubs: { layerIndex: number; unitSize: number }[] = [];
  if (options.allowRecurrent) {
    try {
      for (let li = 1; li < layers.length - 1; li++) {
        const hiddenLayer = layers[li];
        const total = hiddenLayer.length;
        // Heuristic: equal 5-way partition (inputGate, forgetGate, memoryCell, outputGate, outputBlock)
        if (total >= 10 && total % 5 === 0) {
          const seg = total / 5;
          const memorySlice = hiddenLayer.slice(seg * 2, seg * 3);
          const allSelf = memorySlice.every(
            (n: any) => n.connections.self.length === 1
          );
          if (allSelf) {
            lstmPatternStubs.push({ layerIndex: li, unitSize: seg });
          }
        }
      }
    } catch {
      /* ignore heuristic errors */
    }
  }
  validateLayerHomogeneityAndConnectivity(layers, network, options);
  const model = buildOnnxModel(network, layers, options);
  // Phase 4 heuristic conv inference (non-intrusive): if metadata requested and no explicit conv2dMappings for a layer
  // attempt to infer simple single-channel square image + 2x2 or 3x3 kernel patterns. Does NOT alter graph; only metadata.
  if (options.includeMetadata) {
    const inferredSpecs: any[] = [];
    const inferredLayers: number[] = [];
    for (let li = 1; li < layers.length - 1; li++) {
      const prevWidth = layers[li - 1].length;
      const currWidth = layers[li].length;
      // Single-channel square assumption
      const s = Math.sqrt(prevWidth);
      if (Math.abs(s - Math.round(s)) > 1e-9) continue;
      const sInt = Math.round(s);
      // Try kernel sizes 2 or 3 with stride 1, outChannels 1
      for (const k of [3, 2]) {
        if (k >= sInt) continue;
        const outSpatial = sInt - k + 1;
        if (outSpatial * outSpatial === currWidth) {
          // Avoid duplicating explicit specs
          const alreadyDeclared = options.conv2dMappings?.some(
            (m) => m.layerIndex === li
          );
          if (alreadyDeclared) break;
          inferredLayers.push(li);
          inferredSpecs.push({
            layerIndex: li,
            inHeight: sInt,
            inWidth: sInt,
            inChannels: 1,
            kernelHeight: k,
            kernelWidth: k,
            strideHeight: 1,
            strideWidth: 1,
            outHeight: outSpatial,
            outWidth: outSpatial,
            outChannels: 1,
            note: 'heuristic_inferred_no_export_applied',
          });
          break;
        }
      }
    }
    if (inferredLayers.length) {
      model.metadata_props = model.metadata_props || [];
      model.metadata_props.push({
        key: 'conv2d_inferred_layers',
        value: JSON.stringify(inferredLayers),
      });
      model.metadata_props.push({
        key: 'conv2d_inferred_specs',
        value: JSON.stringify(inferredSpecs),
      });
    }
  }
  if (lstmPatternStubs.length) {
    model.metadata_props = model.metadata_props || [];
    model.metadata_props.push({
      key: 'lstm_groups_stub',
      value: JSON.stringify(lstmPatternStubs),
    });
  }
  return model;
}

/**
 * Import a model previously produced by {@link exportToONNX} into a fresh Network instance.
 *
 * Core Steps:
 *  1. Parse input/output tensor shapes (supports optional symbolic batch dim).
 *  2. Derive hidden layer sizes (prefer `layer_sizes` metadata; fallback to weight tensor grouping heuristic).
 *  3. Instantiate matching layered MLP (inputs -> hidden[] -> outputs); remove placeholder hidden nodes for single layer perceptrons.
 *  4. Assign weights & biases (aggregated or per-neuron) from W/B initializers.
 *  5. Reconstruct activation functions from Activation node op_types (layer or per-neuron).
 *  6. Restore recurrent self connections from recorded diagonal Rk matrices if `recurrent_single_step` metadata present.
 *  7. Experimental: Reconstruct LSTM / GRU layers when fused initializers & metadata (`lstm_emitted_layers`, `gru_emitted_layers`) detected
 *     by replacing the corresponding hidden node block with a freshly constructed Layer.lstm / Layer.gru instance and remapping weights.
 *  8. Rebuild flat connection array for downstream invariants.
 *
 * Experimental Behavior:
 *  - LSTM/GRU reconstruction is best-effort; inconsistencies in tensor shapes or gate counts result in silent skip (import still succeeds).
 *  - Recurrent biases (Rb) absent; self-connection diagonal only restored for cell/candidate groups.
 *
 * Limitations:
 *  - Only guaranteed for self-produced models; arbitrary ONNX graphs or differing op orderings are unsupported.
 *  - Fused recurrent node emission currently leaves original unfused Gemm/Activation path in exported model (import ignores duplicates).
 */
export function importFromONNX(onnx: OnnxModel): Network {
  const { default: NetworkVal } = require('../network'); // dynamic import to avoid circular reference at module load
  const { default: Layer } = require('../layer');
  /** Number of input features (dimension of input tensor). */
  const inputShapeDims = onnx.graph.inputs[0].type.tensor_type.shape.dim;
  const inputCount = (inputShapeDims[inputShapeDims.length - 1] as any)
    .dim_value;
  /** Number of output neurons (dimension of output tensor). */
  const outputShapeDims = onnx.graph.outputs[0].type.tensor_type.shape.dim;
  const outputCount = (outputShapeDims[outputShapeDims.length - 1] as any)
    .dim_value;
  /** Hidden layer sizes derived from weight tensor shapes. */
  const hiddenLayerSizes = deriveHiddenLayerSizes(
    onnx.graph.initializer,
    (onnx as any).metadata_props
  );
  /** Newly constructed network mirroring the ONNX architecture. */
  const network: Network = NetworkVal.createMLP(
    inputCount,
    hiddenLayerSizes,
    outputCount
  );
  if (hiddenLayerSizes.length === 0) {
    // Edge case: single-layer perceptron (inputs -> outputs); prune hidden placeholders if any.
    network.nodes = [
      ...network.nodes.filter((n: any) => n.type === 'input'),
      ...network.nodes.filter((n: any) => n.type === 'output'),
    ];
    rebuildConnectionsLocal(network as any);
  }
  assignWeightsAndBiases(
    network,
    onnx,
    hiddenLayerSizes,
    (onnx as any).metadata_props
  );
  assignActivationFunctions(network, onnx, hiddenLayerSizes);
  // Phase 3: restore self-recurrent weights if present
  const meta = (onnx as any).metadata_props || [];
  const recurrentMeta = meta.find(
    (p: any) => p.key === 'recurrent_single_step'
  );
  if (recurrentMeta) {
    let layerIndices: number[] = [];
    try {
      const parsed = JSON.parse(recurrentMeta.value);
      if (Array.isArray(parsed)) layerIndices = parsed;
      else layerIndices = [0];
    } catch {
      layerIndices = [0];
    }
    // For each recorded recurrent layer index, map to hidden layer offset.
    // hiddenLayerSizes reflect each hidden layer sequentially.
    let hiddenStart = 0;
    for (let h = 0; h < hiddenLayerSizes.length; h++) {
      const size = hiddenLayerSizes[h];
      const layerNumber = h + 1; // original export layer numbering (1-based across hidden layers)
      if (layerIndices.includes(layerNumber)) {
        const rName = `R${layerNumber - 1}`;
        const rInit = onnx.graph.initializer.find((t: any) => t.name === rName);
        if (rInit) {
          for (let i = 0; i < size; i++) {
            const node = network.nodes.filter((n: any) => n.type === 'hidden')[
              hiddenStart + i
            ];
            const weight = rInit.float_data[i * size + i];
            let selfConn = node.connections.self[0];
            if (!selfConn) {
              selfConn = Connection.acquire(node as any, node as any, weight);
              node.connections.self.push(selfConn);
              node.connections.in.push(selfConn);
              node.connections.out.push(selfConn);
            } else {
              selfConn.weight = weight;
            }
          }
        }
      }
      hiddenStart += size;
    }
  }
  // Placeholder: detect presence of LSTM grouping metadata (no reconstruction yet, reserved for future mapping)
  const lstmStubMeta = meta.find((p: any) => p.key === 'lstm_groups_stub');
  if (lstmStubMeta) {
    // Intentionally no action currently; future implementation will repartition hidden nodes into gate groups.
  }
  const lstmEmitMeta = meta.find((p: any) => p.key === 'lstm_emitted_layers');
  const gruEmitMeta = meta.find((p: any) => p.key === 'gru_emitted_layers');
  const rnnFallbackMeta = meta.filter(
    (p: any) => p.key === 'rnn_pattern_fallback'
  );
  if (lstmEmitMeta || gruEmitMeta || rnnFallbackMeta.length) {
    // Placeholder: could attach flags on network for introspection; for now, silent.
  }
  // Step 5: Reconstruct LSTM / GRU layers if emitted metadata present (experimental)
  try {
    if (lstmEmitMeta) {
      const layersEmitted: number[] = JSON.parse(lstmEmitMeta.value);
      layersEmitted.forEach((exportLayerIndex) => {
        // Hidden layer index (0-based among hidden layers)
        const hiddenIndex = exportLayerIndex - 1;
        if (hiddenIndex < 0 || hiddenIndex >= hiddenLayerSizes.length) return;
        // Locate LSTM initializer tensors
        const W = onnx.graph.initializer.find(
          (t: any) => t.name === `LSTM_W${hiddenIndex}`
        );
        const R = onnx.graph.initializer.find(
          (t: any) => t.name === `LSTM_R${hiddenIndex}`
        );
        const B = onnx.graph.initializer.find(
          (t: any) => t.name === `LSTM_B${hiddenIndex}`
        );
        if (!W || !R || !B) return; // incomplete
        // Determine unit size (rows = gates*unit, gates assumed 4)
        const rows = W.dims[0];
        const prevSize = W.dims[1];
        const gates = 4;
        if (rows % gates !== 0) return;
        const unit = rows / gates;
        // Calculate offsets into hidden node list for replacement
        const hiddenNodes = network.nodes.filter(
          (n: any) => n.type === 'hidden'
        );
        const start = hiddenLayerSizes
          .slice(0, hiddenIndex)
          .reduce((a, b) => a + b, 0);
        const end = start + hiddenLayerSizes[hiddenIndex];
        const oldLayerNodes = hiddenNodes.slice(start, end);
        // Previous layer output nodes
        const prevLayerNodes =
          hiddenIndex === 0
            ? network.nodes.filter((n: any) => n.type === 'input')
            : hiddenNodes.slice(
                hiddenLayerSizes
                  .slice(0, hiddenIndex - 1)
                  .reduce((a, b) => a + b, 0),
                hiddenLayerSizes
                  .slice(0, hiddenIndex)
                  .reduce((a, b) => a + b, 0)
              );
        const nextLayerIsOutput = hiddenIndex === hiddenLayerSizes.length - 1;
        const nextLayerNodes = nextLayerIsOutput
          ? network.nodes.filter((n: any) => n.type === 'output')
          : hiddenNodes.slice(end, end + hiddenLayerSizes[hiddenIndex + 1]);
        // Remove connections linked to old layer nodes
        network.connections = network.connections.filter(
          (c: any) =>
            !oldLayerNodes.includes(c.from) && !oldLayerNodes.includes(c.to)
        );
        prevLayerNodes.forEach((p: any) => {
          p.connections.out = p.connections.out.filter(
            (c: any) => !oldLayerNodes.includes(c.to)
          );
        });
        nextLayerNodes.forEach((nxt: any) => {
          nxt.connections.in = nxt.connections.in.filter(
            (c: any) => !oldLayerNodes.includes(c.from)
          );
        });
        oldLayerNodes.forEach((n: any) => {
          n.connections.in = [];
          n.connections.out = [];
        });
        // Create new LSTM layer
        const lstmLayer = Layer.lstm(unit);
        // Insert its nodes in place of old hidden nodes (maintain ordering)
        const newHiddenNodes = [...hiddenNodes];
        newHiddenNodes.splice(start, oldLayerNodes.length, ...lstmLayer.nodes);
        // Replace network hidden nodes ordering
        const inputNodes = network.nodes.filter((n: any) => n.type === 'input');
        const outputNodes = network.nodes.filter(
          (n: any) => n.type === 'output'
        );
        network.nodes = [...inputNodes, ...newHiddenNodes, ...outputNodes];
        // Connect previous layer to LSTM layer using its input method
        lstmLayer.input({ output: { nodes: prevLayerNodes } } as any);
        // Connect LSTM output block to next layer nodes
        lstmLayer.output.nodes.forEach((outNode: any) => {
          nextLayerNodes.forEach((nxt: any) => outNode.connect(nxt));
        });
        // Assign weights & biases from canonical W matrix (gate order: input, forget, cell, output)
        const gateOrder = ['input', 'forget', 'cell', 'output'];
        const groupMap: Record<string, any[]> = {
          input: lstmLayer.nodes.slice(0, unit),
          forget: lstmLayer.nodes.slice(unit, unit * 2),
          cell: lstmLayer.nodes.slice(unit * 2, unit * 3),
          output: lstmLayer.nodes.slice(unit * 3, unit * 4),
        };
        for (let g = 0; g < gateOrder.length; g++) {
          for (let r = 0; r < unit; r++) {
            const rowOffset = g * unit + r;
            const neuron = groupMap[gateOrder[g]][r];
            neuron.bias = B.float_data[rowOffset];
            for (let c = 0; c < prevSize; c++) {
              const weight = W.float_data[rowOffset * prevSize + c];
              const src = prevLayerNodes[c];
              const conn = neuron.connections.in.find(
                (cc: any) => cc.from === src
              );
              if (conn) conn.weight = weight;
            }
            if (gateOrder[g] === 'cell') {
              const selfConn = neuron.connections.self[0];
              if (selfConn) {
                const rWeight = R.float_data[rowOffset * unit + r];
                selfConn.weight = rWeight;
              }
            }
          }
        }
      });
    }
    if (gruEmitMeta) {
      const layersEmitted: number[] = JSON.parse(gruEmitMeta.value);
      layersEmitted.forEach((exportLayerIndex) => {
        const hiddenIndex = exportLayerIndex - 1;
        if (hiddenIndex < 0 || hiddenIndex >= hiddenLayerSizes.length) return;
        const W = onnx.graph.initializer.find(
          (t: any) => t.name === `GRU_W${hiddenIndex}`
        );
        const R = onnx.graph.initializer.find(
          (t: any) => t.name === `GRU_R${hiddenIndex}`
        );
        const B = onnx.graph.initializer.find(
          (t: any) => t.name === `GRU_B${hiddenIndex}`
        );
        if (!W || !R || !B) return;
        const rows = W.dims[0];
        const prevSize = W.dims[1];
        const gates = 3; // update, reset, candidate
        if (rows % gates !== 0) return;
        const unit = rows / gates;
        const hiddenNodes = network.nodes.filter(
          (n: any) => n.type === 'hidden'
        );
        const start = hiddenLayerSizes
          .slice(0, hiddenIndex)
          .reduce((a, b) => a + b, 0);
        const end = start + hiddenLayerSizes[hiddenIndex];
        const oldLayerNodes = hiddenNodes.slice(start, end);
        const prevLayerNodes =
          hiddenIndex === 0
            ? network.nodes.filter((n: any) => n.type === 'input')
            : hiddenNodes.slice(
                hiddenLayerSizes
                  .slice(0, hiddenIndex - 1)
                  .reduce((a, b) => a + b, 0),
                hiddenLayerSizes
                  .slice(0, hiddenIndex)
                  .reduce((a, b) => a + b, 0)
              );
        const nextLayerIsOutput = hiddenIndex === hiddenLayerSizes.length - 1;
        const nextLayerNodes = nextLayerIsOutput
          ? network.nodes.filter((n: any) => n.type === 'output')
          : hiddenNodes.slice(end, end + hiddenLayerSizes[hiddenIndex + 1]);
        network.connections = network.connections.filter(
          (c: any) =>
            !oldLayerNodes.includes(c.from) && !oldLayerNodes.includes(c.to)
        );
        prevLayerNodes.forEach((p: any) => {
          p.connections.out = p.connections.out.filter(
            (c: any) => !oldLayerNodes.includes(c.to)
          );
        });
        nextLayerNodes.forEach((nxt: any) => {
          nxt.connections.in = nxt.connections.in.filter(
            (c: any) => !oldLayerNodes.includes(c.from)
          );
        });
        oldLayerNodes.forEach((n: any) => {
          n.connections.in = [];
          n.connections.out = [];
        });
        const gruLayer = Layer.gru(unit);
        const newHiddenNodes = [...hiddenNodes];
        newHiddenNodes.splice(start, oldLayerNodes.length, ...gruLayer.nodes);
        const inputNodes = network.nodes.filter((n: any) => n.type === 'input');
        const outputNodes = network.nodes.filter(
          (n: any) => n.type === 'output'
        );
        network.nodes = [...inputNodes, ...newHiddenNodes, ...outputNodes];
        gruLayer.input({ output: { nodes: prevLayerNodes } } as any);
        gruLayer.output.nodes.forEach((outNode: any) => {
          nextLayerNodes.forEach((nxt: any) => outNode.connect(nxt));
        });
        const gateOrder = ['update', 'reset', 'candidate'];
        const groupMap: Record<string, any[]> = {
          update: gruLayer.nodes.slice(0, unit),
          reset: gruLayer.nodes.slice(unit, unit * 2),
          candidate: gruLayer.nodes.slice(unit * 2, unit * 3),
        };
        for (let g = 0; g < gateOrder.length; g++) {
          for (let r = 0; r < unit; r++) {
            const rowOffset = g * unit + r;
            const neuron = groupMap[gateOrder[g]][r];
            neuron.bias = B.float_data[rowOffset];
            for (let c = 0; c < prevSize; c++) {
              const weight = W.float_data[rowOffset * prevSize + c];
              const src = prevLayerNodes[c];
              const conn = neuron.connections.in.find(
                (cc: any) => cc.from === src
              );
              if (conn) conn.weight = weight;
            }
            if (gateOrder[g] === 'candidate') {
              const selfConn = neuron.connections.self[0];
              if (selfConn) {
                const rWeight = R.float_data[rowOffset * unit + r];
                selfConn.weight = rWeight;
              }
            }
          }
        }
      });
    }
  } catch {
    /* swallow experimental import errors */
  }
  rebuildConnectionsLocal(network as any);
  // Attach pooling metadata (pass-through) for downstream tooling / potential shape simulation.
  try {
    const poolLayersMeta = meta.find((p: any) => p.key === 'pool2d_layers');
    const poolSpecsMeta = meta.find((p: any) => p.key === 'pool2d_specs');
    if (poolLayersMeta) {
      (network as any)._onnxPooling = {
        layers: JSON.parse(poolLayersMeta.value),
        specs: poolSpecsMeta ? JSON.parse(poolSpecsMeta.value) : [],
      };
    }
  } catch {
    /* ignore pooling attachment errors */
  }
  return network;
}

export default { exportToONNX, importFromONNX };
