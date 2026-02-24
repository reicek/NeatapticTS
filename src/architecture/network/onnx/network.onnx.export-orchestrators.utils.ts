import type Network from '../../network';
import type NeatapticNode from '../../node';
import type {
  Conv2DMapping,
  ConvInferenceEvaluationContext,
  ConvInferenceKernelEvaluationContext,
  ConvInferenceResult,
  ConvInferenceTraversalContext,
  ExportNodeIndexAssignmentContext,
  LstmCandidateContext,
  LstmLayerTraversalContext,
  LstmPatternStub,
  NodeInternals,
  NodeInternalsWithExportIndex,
  OnnxExportOptions,
  OnnxMetadataProperty,
  OnnxModel,
} from './network.onnx.utils.types';

const LSTM_GATE_GROUP_COUNT = 5;
const MIN_LSTM_LAYER_WIDTH = 10;
const REQUIRED_SELF_CONNECTION_COUNT = 1;
const SQUARE_NUMBER_TOLERANCE = 1e-9;
const CONV_KERNEL_CANDIDATES: readonly number[] = [3, 2];

/**
 * Assign stable index values to nodes for export diagnostics.
 *
 * @param network Source network.
 * @returns Nothing.
 */
export function assignExportNodeIndices(network: Network): void {
  const assignmentContexts = createExportNodeIndexAssignmentContexts(network);
  applyExportNodeIndexAssignments(assignmentContexts);
}

/**
 * Collect heuristic LSTM grouping stubs from hidden layers.
 *
 * @param layers Layered network nodes.
 * @param allowRecurrent Whether recurrent export heuristics are enabled.
 * @returns Candidate LSTM pattern stubs.
 */
export function collectLstmPatternStubs(
  layers: NeatapticNode[][],
  allowRecurrent: boolean | undefined,
): LstmPatternStub[] {
  if (!allowRecurrent) return [];
  return safelyCollectLstmPatternStubs(layers);
}

/**
 * Append heuristic conv inference metadata when requested.
 *
 * @param model Target ONNX model.
 * @param layers Layered network nodes.
 * @param options Export options.
 * @returns Nothing.
 */
export function appendConvInferenceMetadata(
  model: OnnxModel,
  layers: NeatapticNode[][],
  options: OnnxExportOptions,
): void {
  if (!options.includeMetadata) return;
  const inferredConvResult = collectInferredConvMetadata({
    layers,
    declaredMappings: options.conv2dMappings,
  });
  if (!hasInferredConvMetadata(inferredConvResult)) return;
  appendMetadataProperties(model, [
    {
      key: 'conv2d_inferred_layers',
      value: JSON.stringify(inferredConvResult.inferredLayers),
    },
    {
      key: 'conv2d_inferred_specs',
      value: JSON.stringify(inferredConvResult.inferredSpecs),
    },
  ]);
}

/**
 * Append LSTM pattern stub metadata.
 *
 * @param model Target ONNX model.
 * @param lstmPatternStubs Pattern stubs.
 * @returns Nothing.
 */
export function appendLstmPatternStubMetadata(
  model: OnnxModel,
  lstmPatternStubs: LstmPatternStub[],
): void {
  if (!lstmPatternStubs.length) return;
  appendMetadataProperties(model, [
    {
      key: 'lstm_groups_stub',
      value: JSON.stringify(lstmPatternStubs),
    },
  ]);
}

/**
 * Create node/index assignment contexts for export diagnostics.
 *
 * @param network Source network.
 * @returns Assignment contexts.
 */
function createExportNodeIndexAssignmentContexts(
  network: Network,
): ExportNodeIndexAssignmentContext[] {
  return network.nodes.map((node, exportIndex) => ({ node, exportIndex }));
}

/**
 * Apply prepared node/index assignment contexts.
 *
 * @param assignmentContexts Prepared contexts.
 * @returns Nothing.
 */
function applyExportNodeIndexAssignments(
  assignmentContexts: ExportNodeIndexAssignmentContext[],
): void {
  assignmentContexts.forEach(applySingleExportNodeIndexAssignment);
}

/**
 * Apply one export index assignment.
 *
 * @param assignmentContext Assignment context.
 * @returns Nothing.
 */
function applySingleExportNodeIndexAssignment(
  assignmentContext: ExportNodeIndexAssignmentContext,
): void {
  const nodeInternal = assignmentContext.node as NodeInternalsWithExportIndex;
  nodeInternal.index = assignmentContext.exportIndex;
}

/**
 * Collect LSTM pattern stubs with heuristic error isolation.
 *
 * @param layers Layered network nodes.
 * @returns LSTM pattern stubs.
 */
function safelyCollectLstmPatternStubs(
  layers: NeatapticNode[][],
): LstmPatternStub[] {
  try {
    return collectLstmPatternStubsFromLayers(layers);
  } catch {
    return [];
  }
}

/**
 * Collect LSTM pattern stubs from hidden layers.
 *
 * @param layers Layered network nodes.
 * @returns LSTM pattern stubs.
 */
function collectLstmPatternStubsFromLayers(
  layers: NeatapticNode[][],
): LstmPatternStub[] {
  const hiddenLayerContexts = createHiddenLayerTraversalContexts(layers);
  const lstmCandidateContexts = hiddenLayerContexts.map(
    createLstmCandidateContext,
  );
  return lstmCandidateContexts
    .filter(isValidLstmCandidateContext)
    .map(mapLstmCandidateToStub);
}

/**
 * Create traversal contexts for hidden layers only.
 *
 * @param layers Layered network nodes.
 * @returns Hidden layer contexts.
 */
function createHiddenLayerTraversalContexts(
  layers: NeatapticNode[][],
): LstmLayerTraversalContext[] {
  return layers.slice(1, -1).map((hiddenLayerNodes, hiddenLayerOffset) => ({
    layerIndex: hiddenLayerOffset + 1,
    hiddenLayerNodes,
  }));
}

/**
 * Build LSTM candidate context for one hidden layer.
 *
 * @param hiddenLayerContext Hidden layer context.
 * @returns LSTM candidate context.
 */
function createLstmCandidateContext(
  hiddenLayerContext: LstmLayerTraversalContext,
): LstmCandidateContext {
  const totalNodes = hiddenLayerContext.hiddenLayerNodes.length;
  const unitSize = totalNodes / LSTM_GATE_GROUP_COUNT;
  const memoryStart = unitSize * 2;
  const memoryEnd = unitSize * 3;
  return {
    layerIndex: hiddenLayerContext.layerIndex,
    totalNodes,
    unitSize,
    memorySliceNodes: hiddenLayerContext.hiddenLayerNodes.slice(
      memoryStart,
      memoryEnd,
    ),
  };
}

/**
 * Determine whether a candidate context satisfies heuristic LSTM conditions.
 *
 * @param candidateContext Candidate context.
 * @returns True when the candidate is a valid LSTM stub.
 */
function isValidLstmCandidateContext(
  candidateContext: LstmCandidateContext,
): boolean {
  const hasMinimumWidth = candidateContext.totalNodes >= MIN_LSTM_LAYER_WIDTH;
  const isExactGateSplit =
    candidateContext.totalNodes % LSTM_GATE_GROUP_COUNT === 0;
  const hasSelfConnectedMemorySlice = candidateContext.memorySliceNodes.every(
    hasRequiredSelfConnectionCount,
  );
  return hasMinimumWidth && isExactGateSplit && hasSelfConnectedMemorySlice;
}

/**
 * Map a valid candidate context to metadata stub.
 *
 * @param candidateContext Valid candidate context.
 * @returns LSTM pattern stub.
 */
function mapLstmCandidateToStub(
  candidateContext: LstmCandidateContext,
): LstmPatternStub {
  return {
    layerIndex: candidateContext.layerIndex,
    unitSize: candidateContext.unitSize,
  };
}

/**
 * Check whether one node has the required self-connection count.
 *
 * @param nodeItem Node to inspect.
 * @returns True when self-connection count matches requirement.
 */
function hasRequiredSelfConnectionCount(nodeItem: NeatapticNode): boolean {
  const nodeInternal = nodeItem as NodeInternals;
  return (
    nodeInternal.connections.self.length === REQUIRED_SELF_CONNECTION_COUNT
  );
}

/**
 * Collect inferred Conv metadata from hidden-layer traversals.
 *
 * @param context Conv traversal context.
 * @returns Inferred Conv metadata result.
 */
function collectInferredConvMetadata(context: {
  layers: NeatapticNode[][];
  declaredMappings: Conv2DMapping[] | undefined;
}): ConvInferenceResult {
  const traversalContexts = createConvTraversalContexts(context);
  const inferredSpecs = traversalContexts
    .map(resolveConvInferenceForLayer)
    .filter(isInferredConvSpec);
  const inferredLayers = inferredSpecs.map(
    (specification) => specification.layerIndex,
  );
  return { inferredLayers, inferredSpecs };
}

/**
 * Create Conv traversal contexts for hidden layers.
 *
 * @param context Conv traversal source context.
 * @returns Conv traversal contexts.
 */
function createConvTraversalContexts(context: {
  layers: NeatapticNode[][];
  declaredMappings: Conv2DMapping[] | undefined;
}): ConvInferenceTraversalContext[] {
  return context.layers
    .slice(1, -1)
    .map((currentLayerNodes, hiddenLayerOffset) => ({
      layerIndex: hiddenLayerOffset + 1,
      previousLayerNodes: context.layers[hiddenLayerOffset],
      currentLayerNodes,
      declaredMappings: context.declaredMappings,
    }));
}

/**
 * Resolve inferred Conv specification for one hidden layer.
 *
 * @param traversalContext Conv traversal context.
 * @returns Inferred Conv specification when matched.
 */
function resolveConvInferenceForLayer(
  traversalContext: ConvInferenceTraversalContext,
): (Conv2DMapping & { note?: string }) | undefined {
  if (isDeclaredConvLayer(traversalContext)) return undefined;
  const evaluationContext =
    createConvInferenceEvaluationContext(traversalContext);
  if (!evaluationContext.isSquareInputWidth) return undefined;
  return resolveConvSpecFromKernelCandidates(evaluationContext);
}

/**
 * Create width/square-evaluation context for Conv inference.
 *
 * @param traversalContext Conv traversal context.
 * @returns Conv evaluation context.
 */
function createConvInferenceEvaluationContext(
  traversalContext: ConvInferenceTraversalContext,
): ConvInferenceEvaluationContext {
  const previousWidth = traversalContext.previousLayerNodes.length;
  const currentWidth = traversalContext.currentLayerNodes.length;
  const squareRootWidth = Math.sqrt(previousWidth);
  const roundedSquareWidth = Math.round(squareRootWidth);
  const squareDelta = Math.abs(squareRootWidth - roundedSquareWidth);
  return {
    layerIndex: traversalContext.layerIndex,
    currentWidth,
    squareWidth: roundedSquareWidth,
    isSquareInputWidth: squareDelta <= SQUARE_NUMBER_TOLERANCE,
  };
}

/**
 * Resolve Conv specification using ordered kernel candidates.
 *
 * @param evaluationContext Conv evaluation context.
 * @returns Inferred Conv specification when matched.
 */
function resolveConvSpecFromKernelCandidates(
  evaluationContext: ConvInferenceEvaluationContext,
): (Conv2DMapping & { note?: string }) | undefined {
  const kernelEvaluationContexts = CONV_KERNEL_CANDIDATES.map((kernelSize) => ({
    evaluationContext,
    kernelSize,
  }));
  return kernelEvaluationContexts
    .map(resolveConvSpecForKernel)
    .find(isInferredConvSpec);
}

/**
 * Resolve Conv specification for one kernel candidate.
 *
 * @param kernelContext Kernel-evaluation context.
 * @returns Inferred Conv specification when matched.
 */
function resolveConvSpecForKernel(
  kernelContext: ConvInferenceKernelEvaluationContext,
): (Conv2DMapping & { note?: string }) | undefined {
  const isKernelTooLarge =
    kernelContext.kernelSize >= kernelContext.evaluationContext.squareWidth;
  if (isKernelTooLarge) return undefined;
  const outputSpatialSize =
    kernelContext.evaluationContext.squareWidth - kernelContext.kernelSize + 1;
  const outputWidth = outputSpatialSize * outputSpatialSize;
  const matchesCurrentWidth =
    outputWidth === kernelContext.evaluationContext.currentWidth;
  if (!matchesCurrentWidth) return undefined;
  return {
    layerIndex: kernelContext.evaluationContext.layerIndex,
    inHeight: kernelContext.evaluationContext.squareWidth,
    inWidth: kernelContext.evaluationContext.squareWidth,
    inChannels: 1,
    kernelHeight: kernelContext.kernelSize,
    kernelWidth: kernelContext.kernelSize,
    strideHeight: 1,
    strideWidth: 1,
    outHeight: outputSpatialSize,
    outWidth: outputSpatialSize,
    outChannels: 1,
    note: 'heuristic_inferred_no_export_applied',
  };
}

/**
 * Check whether a traversal layer already has declared Conv mapping.
 *
 * @param traversalContext Conv traversal context.
 * @returns True when mapping is already declared.
 */
function isDeclaredConvLayer(
  traversalContext: ConvInferenceTraversalContext,
): boolean {
  return (
    traversalContext.declaredMappings?.some(
      (mapping) => mapping.layerIndex === traversalContext.layerIndex,
    ) ?? false
  );
}

/**
 * Type guard for inferred Conv specifications.
 *
 * @param specification Conv specification candidate.
 * @returns True when specification is defined.
 */
function isInferredConvSpec(
  specification: (Conv2DMapping & { note?: string }) | undefined,
): specification is Conv2DMapping & { note?: string } {
  return specification !== undefined;
}

/**
 * Check whether inferred Conv metadata exists.
 *
 * @param inferenceResult Inferred Conv result.
 * @returns True when inferred metadata exists.
 */
function hasInferredConvMetadata(
  inferenceResult: ConvInferenceResult,
): boolean {
  return inferenceResult.inferredLayers.length > 0;
}

/**
 * Append metadata properties in a single, normalized path.
 *
 * @param model Target ONNX model.
 * @param metadataProperties Metadata properties to append.
 * @returns Nothing.
 */
function appendMetadataProperties(
  model: OnnxModel,
  metadataProperties: OnnxMetadataProperty[],
): void {
  model.metadata_props = [
    ...(model.metadata_props ?? []),
    ...metadataProperties,
  ];
}
