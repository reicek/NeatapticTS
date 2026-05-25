import type NeatapticNode from '../../../node';
import type { NodeInternals } from '../network.onnx.utils.types';

/** Supported fused recurrent operator families recognized during ONNX import, currently limited to LSTM and GRU. */
export type OnnxFusedRecurrentKind = 'LSTM' | 'GRU';

/**
 * Runtime interface of a reconstructed fused recurrent layer instance.
 *
 * The importer only relies on a narrow runtime contract: access to the
 * reconstructed nodes, an input wiring hook, and an optional output group that
 * can be reconnected to the next restored layer.
 */
export interface OnnxFusedLayerRuntime {
  nodes: NeatapticNode[];
  input: (groupLike: unknown) => void;
  output: { nodes: NeatapticNode[] } | null;
}

/**
 * Fused recurrent family specification used during import reconstruction.
 *
 * This tells the importer how to interpret one emitted ONNX recurrent family:
 * how many gates to expect, what order those gates were serialized in, and
 * which gate owns the self-recurrent diagonal replay.
 */
export type OnnxFusedRecurrentSpec = {
  kind: OnnxFusedRecurrentKind;
  gateCount: number;
  gateOrder: string[];
  recurrentGateName: string;
  metadataKey: string;
};

/** Hidden-layer neighborhood slices around a reconstructed fused layer, including old, previous, and next node lists. */
export type OnnxFusedLayerNeighborhood = {
  hiddenNodes: NeatapticNode[];
  oldLayerNodes: NeatapticNode[];
  previousLayerNodes: NeatapticNode[];
  nextLayerNodes: NeatapticNode[];
  start: number;
  end: number;
};

/**
 * Fused recurrent tensor payload read from ONNX initializers.
 *
 * The importer resolves the three recurrent tensor families up front so the
 * reconstruction pass can focus on wiring and row assignment instead of
 * repeatedly re-looking up initializers.
 */
export type OnnxFusedTensorPayload = {
  inputWeights: number[];
  recurrentWeights: number[];
  biases: number[];
  rows: number;
  previousLayerWidth: number;
};

/** Execution context for one fused recurrent layer reconstruction, carrying spec, export index, and hidden layer index. */
export type OnnxFusedLayerReconstructionContext = {
  spec: OnnxFusedRecurrentSpec;
  exportLayerIndex: number;
  hiddenLayerIndex: number;
};

/** Gate-weight application context for one reconstructed fused layer, carrying spec, unit size, and weight arrays. */
export type OnnxFusedGateApplicationContext = {
  fusedLayer: OnnxFusedLayerRuntime;
  spec: OnnxFusedRecurrentSpec;
  unitSize: number;
  previousLayerWidth: number;
  biases: number[];
  inputWeights: number[];
  recurrentWeights: number[];
  previousLayerNodes: NeatapticNode[];
};

/** Context for assigning one gate-neuron row from flattened ONNX tensors. */
export type OnnxFusedGateRowAssignmentContext = {
  fusedKind: OnnxFusedRecurrentKind;
  gateNeuronInternal: NodeInternals;
  gateName: string;
  recurrentSourceNodes: NeatapticNode[];
  recurrentGateName: string;
  rowOffset: number;
  rowIndex: number;
  unitSize: number;
  previousLayerWidth: number;
  biases: number[];
  inputWeights: number[];
  recurrentWeights: number[];
  previousLayerNodes: NeatapticNode[];
};

/** Context for assigning dense incoming weights for one gate-neuron row. */
export type OnnxIncomingWeightAssignmentContext = {
  gateNeuronInternal: NodeInternals;
  rowOffset: number;
  previousLayerWidth: number;
  inputWeights: number[];
  previousLayerNodes: NeatapticNode[];
};
