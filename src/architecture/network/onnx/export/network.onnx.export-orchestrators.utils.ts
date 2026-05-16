import type Network from '../../network';
import type NeatapticNode from '../../../node';
import type {
  Conv2DMapping,
  OnnxMetadataProperty,
  OnnxModel,
  Pool2DMapping,
} from '../schema/network.onnx.schema.types';
import type {
  ConvInferenceEvaluationContext,
  ConvInferenceKernelEvaluationContext,
  ConvInferenceResult,
  ConvInferenceTraversalContext,
  ExportNodeIndexAssignmentContext,
  LstmCandidateContext,
  LstmLayerTraversalContext,
  LstmPatternStub,
  OnnxExportOptions,
} from './network.onnx.export.types';
import type {
  NodeInternals,
  NodeInternalsWithExportIndex,
} from '../network.onnx.utils.types';
import { isConvMappingWeightShared } from './network.onnx.export-postprocess.utils';

const LSTM_GATE_GROUP_COUNT = 5;
const MIN_LSTM_LAYER_WIDTH = 10;
const REQUIRED_SELF_CONNECTION_COUNT = 1;
const SQUARE_NUMBER_TOLERANCE = 1e-9;
const CONV_KERNEL_CANDIDATES: readonly number[] = [3, 2];
const ZERO_LENGTH = 0;
const MINIMUM_SPATIAL_OUTPUT_SIZE = 1;

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
    flattenAfterPooling: options.flattenAfterPooling,
    poolMappings: options.pool2dMappings,
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
 * Resolve the effective Conv mapping list after optional heuristic promotion.
 *
 * @param layers Layered network nodes.
 * @param options Export options.
 * @returns Declared mappings plus any safety-gated promoted inferred mappings.
 */
export function resolveEffectiveConvMappings(
  layers: NeatapticNode[][],
  options: OnnxExportOptions,
): Conv2DMapping[] | undefined {
  const declaredMappings = options.conv2dMappings ?? [];
  if (!options.autoPromoteInferredConv) {
    return declaredMappings.length ? declaredMappings : undefined;
  }

  const inferredConvResult = collectInferredConvMetadata({
    layers,
    declaredMappings: options.conv2dMappings,
    flattenAfterPooling: options.flattenAfterPooling,
    poolMappings: options.pool2dMappings,
  });

  const availableConvSpecs = [
    ...declaredMappings,
    ...inferredConvResult.inferredSpecs,
  ].map(stripInferredConvNote);

  const promotedMappings = inferredConvResult.inferredSpecs
    .filter((convSpec) =>
      canPromoteInferredConvSpec(layers, options, convSpec, availableConvSpecs),
    )
    .map(stripInferredConvNote);

  const effectiveMappings = [...declaredMappings, ...promotedMappings];
  return effectiveMappings.length ? effectiveMappings : undefined;
}

/**
 * Decide whether an inferred Conv specification is still safe to promote.
 *
 * Promotion eligibility only checks the shared-kernel safety gate.
 * Pooling and flatten boundaries are resolved earlier during inference so
 * only spatially valid candidates reach this step.
 *
 * @param layers Layered network nodes.
 * @param options Export options.
 * @param convSpec Inferred Conv specification candidate.
 * @returns True when the inferred Conv can be promoted safely.
 */
function canPromoteInferredConvSpec(
  layers: NeatapticNode[][],
  options: OnnxExportOptions,
  convSpec: Conv2DMapping & { note?: string },
  availableConvSpecs: Conv2DMapping[],
): boolean {
  return isConvMappingWeightShared(layers, convSpec, {
    ...options,
    conv2dMappings: availableConvSpecs,
  });
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
 * Remove inference-only note fields before promoted specs become real mappings.
 *
 * @param convSpec Inferred Conv specification.
 * @returns Clean Conv mapping suitable for real Conv emission.
 */
function stripInferredConvNote(
  convSpec: Conv2DMapping & { note?: string },
): Conv2DMapping {
  const { note: _ignoredNote, ...cleanConvSpec } = convSpec;
  return cleanConvSpec;
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
  flattenAfterPooling: boolean | undefined;
  poolMappings: Pool2DMapping[] | undefined;
}): ConvInferenceResult {
  const traversalContexts = createConvTraversalContexts(context);
  const inferredSpecs: (Conv2DMapping & { note?: string })[] = [];

  traversalContexts.forEach((traversalContext) => {
    const inferredSpec = resolveConvInferenceForLayer(traversalContext);
    if (!inferredSpec) {
      return;
    }

    traversalContext.availableConvSpecsByLayerIndex.set(
      inferredSpec.layerIndex,
      inferredSpec,
    );
    inferredSpecs.push(inferredSpec);
  });

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
  flattenAfterPooling: boolean | undefined;
  poolMappings: Pool2DMapping[] | undefined;
}): ConvInferenceTraversalContext[] {
  const availableConvSpecsByLayerIndex = new Map(
    (context.declaredMappings ?? []).map((mapping) => [mapping.layerIndex, mapping]),
  );
  const poolMappingsByAfterLayerIndex = new Map(
    (context.poolMappings ?? []).map((poolingMapping) => [
      poolingMapping.afterLayerIndex,
      poolingMapping,
    ]),
  );

  return context.layers
    .slice(1, -1)
    .map((currentLayerNodes, hiddenLayerOffset) => ({
      layerIndex: hiddenLayerOffset + 1,
      previousLayerNodes: context.layers[hiddenLayerOffset],
      currentLayerNodes,
      declaredMappings: context.declaredMappings,
      availableConvSpecsByLayerIndex,
      poolMappingsByAfterLayerIndex,
      flattenAfterPooling: context.flattenAfterPooling,
      totalHiddenLayerCount: context.layers.length - 2,
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
  const evaluationContexts = createConvInferenceEvaluationContexts(
    traversalContext,
  );
  const inferredSpecs = evaluationContexts
    .map(resolveConvSpecFromKernelCandidates)
    .filter(isInferredConvSpec);
  return resolveSingleInferredConvSpec(inferredSpecs);
}

/**
 * Create width/square-evaluation contexts for Conv inference.
 *
 * @param traversalContext Conv traversal context.
 * @returns Conv evaluation contexts.
 */
function createConvInferenceEvaluationContexts(
  traversalContext: ConvInferenceTraversalContext,
): ConvInferenceEvaluationContext[] {
  const pooledEvaluationContext = createPooledConvInferenceEvaluationContext(
    traversalContext,
  );
  if (pooledEvaluationContext) {
    return [pooledEvaluationContext];
  }
  if (hasUpstreamPoolingBoundary(traversalContext)) {
    return [];
  }

  const previousWidth = traversalContext.previousLayerNodes.length;
  const currentWidth = traversalContext.currentLayerNodes.length;
  return collectCandidateInputChannelCounts(previousWidth)
    .map((inputChannels) =>
      createConvInferenceEvaluationContext({
        allowsExactFitKernel: false,
        layerIndex: traversalContext.layerIndex,
        currentWidth,
        inputChannels,
        inputHeight: resolveSquareSpatialWidth(previousWidth, inputChannels),
        inputWidth: resolveSquareSpatialWidth(previousWidth, inputChannels),
      }),
    )
    .filter(isConvInferenceEvaluationContext);
}

/**
 * Resolve one pooled previous-layer evaluation context when pooling keeps the graph spatial.
 *
 * @param traversalContext Conv traversal context.
 * @returns Evaluation context anchored to the derived pooled shape, if usable.
 */
function createPooledConvInferenceEvaluationContext(
  traversalContext: ConvInferenceTraversalContext,
): ConvInferenceEvaluationContext | undefined {
  if (!hasUpstreamPoolingBoundary(traversalContext)) {
    return undefined;
  }

  const previousConvSpec = traversalContext.availableConvSpecsByLayerIndex.get(
    traversalContext.layerIndex - 1,
  );
  const poolingSpec = traversalContext.poolMappingsByAfterLayerIndex.get(
    traversalContext.layerIndex - 1,
  );
  if (!previousConvSpec || !poolingSpec) {
    return undefined;
  }

  const inputHeight = calculateSpatialOutputSize(
    previousConvSpec.outHeight,
    poolingSpec.kernelHeight,
    poolingSpec.strideHeight,
    poolingSpec.padTop ?? ZERO_LENGTH,
    poolingSpec.padBottom ?? ZERO_LENGTH,
  );
  const inputWidth = calculateSpatialOutputSize(
    previousConvSpec.outWidth,
    poolingSpec.kernelWidth,
    poolingSpec.strideWidth,
    poolingSpec.padLeft ?? ZERO_LENGTH,
    poolingSpec.padRight ?? ZERO_LENGTH,
  );
  if (
    inputHeight < MINIMUM_SPATIAL_OUTPUT_SIZE ||
    inputWidth < MINIMUM_SPATIAL_OUTPUT_SIZE
  ) {
    return undefined;
  }

  if (
    traversalContext.flattenAfterPooling &&
    !supportsFlattenedPostPoolConvSubset(
      traversalContext,
      previousConvSpec,
      inputHeight,
      inputWidth,
    )
  ) {
    return undefined;
  }

  return createConvInferenceEvaluationContext({
    allowsExactFitKernel: true,
    layerIndex: traversalContext.layerIndex,
    currentWidth: traversalContext.currentLayerNodes.length,
    inputChannels: previousConvSpec.outChannels,
    inputHeight,
    inputWidth,
  });
}

/**
 * Check whether the current flatten-after-pool bridge fits the narrow supported subset.
 *
 * @param traversalContext Conv traversal context.
 * @param previousConvSpec Previously resolved Conv spec.
 * @param inputHeight Derived pooled input height.
 * @param inputWidth Derived pooled input width.
 * @returns True when the flattened pooled bridge can still feed one final later Conv-like stage.
 */
function supportsFlattenedPostPoolConvSubset(
  traversalContext: ConvInferenceTraversalContext,
  previousConvSpec: Conv2DMapping,
  inputHeight: number,
  inputWidth: number,
): boolean {
  const hasEarlierPoolingBoundary = traversalContext.poolMappingsByAfterLayerIndex.has(
    traversalContext.layerIndex - 2,
  );

  return (
    previousConvSpec.outChannels > ZERO_LENGTH &&
    traversalContext.layerIndex === traversalContext.totalHiddenLayerCount &&
    !hasEarlierPoolingBoundary &&
    inputHeight > ZERO_LENGTH &&
    inputWidth > ZERO_LENGTH
  );
}

/**
 * Check whether the immediately previous layer has pooling configured.
 *
 * @param traversalContext Conv traversal context.
 * @returns True when the previous layer changes spatial shape through pooling.
 */
function hasUpstreamPoolingBoundary(
  traversalContext: ConvInferenceTraversalContext,
): boolean {
  return traversalContext.poolMappingsByAfterLayerIndex.has(
    traversalContext.layerIndex - 1,
  );
}

/**
 * Collect candidate input-channel counts that evenly partition the previous width.
 *
 * @param previousWidth Previous-layer width.
 * @returns Candidate input-channel counts.
 */
function collectCandidateInputChannelCounts(previousWidth: number): number[] {
  return Array.from({ length: previousWidth }, (_unused, offset) => offset + 1)
    .filter(
      (inputChannelCount) => previousWidth % inputChannelCount === 0,
    );
}

/**
 * Create one width/square-evaluation context for a specific channel partition.
 *
 * @param params Evaluation parameters.
 * @returns Conv evaluation context when the per-channel input width is square.
 */
function createConvInferenceEvaluationContext(params: {
  allowsExactFitKernel: boolean;
  layerIndex: number;
  currentWidth: number;
  inputChannels: number;
  inputHeight: number;
  inputWidth: number;
}): ConvInferenceEvaluationContext | undefined {
  if (params.inputHeight <= ZERO_LENGTH || params.inputWidth <= ZERO_LENGTH) {
    return undefined;
  }
  return {
    allowsExactFitKernel: params.allowsExactFitKernel,
    layerIndex: params.layerIndex,
    currentWidth: params.currentWidth,
    inputChannels: params.inputChannels,
    inputHeight: params.inputHeight,
    inputWidth: params.inputWidth,
  };
}

/**
 * Resolve the per-channel square width when a dense width can be partitioned evenly.
 *
 * @param previousWidth Previous-layer dense width.
 * @param inputChannels Candidate channel count.
 * @returns Resolved square spatial width, or zero when the partition is not square.
 */
function resolveSquareSpatialWidth(
  previousWidth: number,
  inputChannels: number,
): number {
  const channelSpatialWidth = previousWidth / inputChannels;
  const squareRootWidth = Math.sqrt(channelSpatialWidth);
  const roundedSquareWidth = Math.round(squareRootWidth);
  const squareDelta = Math.abs(squareRootWidth - roundedSquareWidth);
  return squareDelta <= SQUARE_NUMBER_TOLERANCE
    ? roundedSquareWidth
    : ZERO_LENGTH;
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
  const standardKernelSpec = kernelEvaluationContexts
    .map((kernelContext) => resolveConvSpecForKernel(kernelContext, false))
    .find(isInferredConvSpec);
  if (standardKernelSpec) {
    return standardKernelSpec;
  }
  if (!evaluationContext.allowsExactFitKernel) {
    return undefined;
  }

  return kernelEvaluationContexts
    .map((kernelContext) => resolveConvSpecForKernel(kernelContext, true))
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
  allowExactFitKernel: boolean,
): (Conv2DMapping & { note?: string }) | undefined {
  const isKernelTooLarge =
    allowExactFitKernel
      ? kernelContext.kernelSize > kernelContext.evaluationContext.inputHeight ||
        kernelContext.kernelSize > kernelContext.evaluationContext.inputWidth
      : kernelContext.kernelSize >= kernelContext.evaluationContext.inputHeight ||
        kernelContext.kernelSize >= kernelContext.evaluationContext.inputWidth;
  if (isKernelTooLarge) return undefined;
  const outputHeight =
    kernelContext.evaluationContext.inputHeight - kernelContext.kernelSize + 1;
  const outputWidth =
    kernelContext.evaluationContext.inputWidth - kernelContext.kernelSize + 1;
  const outputFeatureWidth = outputHeight * outputWidth;
  const hasCompatibleOutputChannels =
    kernelContext.evaluationContext.currentWidth % outputFeatureWidth === 0;
  if (!hasCompatibleOutputChannels) return undefined;
  const outputChannels =
    kernelContext.evaluationContext.currentWidth / outputFeatureWidth;
  return {
    layerIndex: kernelContext.evaluationContext.layerIndex,
    inHeight: kernelContext.evaluationContext.inputHeight,
    inWidth: kernelContext.evaluationContext.inputWidth,
    inChannels: kernelContext.evaluationContext.inputChannels,
    kernelHeight: kernelContext.kernelSize,
    kernelWidth: kernelContext.kernelSize,
    strideHeight: 1,
    strideWidth: 1,
    outHeight: outputHeight,
    outWidth: outputWidth,
    outChannels: outputChannels,
    note: 'heuristic_inferred_no_export_applied',
  };
}

/**
 * Calculate one spatial output size from kernel, stride, and padding metadata.
 *
 * @param inputSize Pre-op spatial size.
 * @param kernelSize Kernel size.
 * @param strideSize Stride size.
 * @param leadingPadding Leading padding value.
 * @param trailingPadding Trailing padding value.
 * @returns Derived spatial output size.
 */
function calculateSpatialOutputSize(
  inputSize: number,
  kernelSize: number,
  strideSize: number,
  leadingPadding: number,
  trailingPadding: number,
): number {
  if (
    inputSize <= ZERO_LENGTH ||
    kernelSize <= ZERO_LENGTH ||
    strideSize <= ZERO_LENGTH
  ) {
    return ZERO_LENGTH;
  }

  return (
    Math.floor(
      (inputSize + leadingPadding + trailingPadding - kernelSize) /
        strideSize,
    ) + MINIMUM_SPATIAL_OUTPUT_SIZE
  );
}

/**
 * Keep multi-channel Conv inference conservative when multiple layouts fit.
 *
 * @param inferredSpecs All inferred Conv specs for the layer.
 * @returns The single usable spec, otherwise undefined.
 */
function resolveSingleInferredConvSpec(
  inferredSpecs: (Conv2DMapping & { note?: string })[],
): (Conv2DMapping & { note?: string }) | undefined {
  return inferredSpecs.length === 1 ? inferredSpecs[0] : undefined;
}

/**
 * Type guard for defined Conv inference evaluation contexts.
 *
 * @param evaluationContext Candidate evaluation context.
 * @returns True when the context is defined.
 */
function isConvInferenceEvaluationContext(
  evaluationContext: ConvInferenceEvaluationContext | undefined,
): evaluationContext is ConvInferenceEvaluationContext {
  return evaluationContext !== undefined;
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
