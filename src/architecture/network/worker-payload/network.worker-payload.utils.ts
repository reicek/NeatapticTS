import type Connection from '../../connection';
import type Node from '../../node';
import type {
  NodeWithIndex,
  StandaloneGenerationContext,
} from '../network.types';
import type { ActivationFn } from '../../../multithreading/types';
import { ACTIVATION_FUNCTIONS } from '../../../multithreading/multi.utils';
import type { ActivationFunction } from '../../../methods/activation/activation.utils';
import { resolveActivationKey } from '../serialize/network.serialize.activation.utils';
import {
  asStandaloneProps,
  createGenerationContext,
  ensureOutputNodesExist,
  resolveStandaloneExecutionMetadata,
  seedNodeIndexesAndState,
} from '../standalone/network.standalone.utils.setup';
import type Network from '../network';
import type { ActivationSchedule } from '../network.types';
import type {
  InferencePredictor,
  NetworkInferenceIR,
  PortableInferencePayload,
  PortableInferencePayloadEdge,
  PortableInferencePayloadNode,
  TransferableInferencePayload,
  TransferableInferencePayloadOptions,
} from './network.worker-payload.types';

const NO_GATER_INDEX = -1;
const DEFAULT_RECURRENT_ITERATIONS = 1;
const PORTABLE_PAYLOAD_VERSION = 1;
const PORTABLE_PREDICTOR_STRATEGY = 'portable';
const TRANSFERABLE_PAYLOAD_VERSION = 1;
const TRANSFERABLE_PREDICTOR_STRATEGY = 'transferable';
const DEFAULT_TRANSFERABLE_NUMERIC_PRECISION = 'full';
const SUPPORTED_INFERENCE_ACTIVATION_KEYS = [
  'logistic',
  'tanh',
  'identity',
  'step',
  'relu',
  'softsign',
  'sinusoid',
  'gaussian',
  'bentIdentity',
  'bipolar',
  'bipolarSigmoid',
  'hardTanh',
  'absolute',
  'inverse',
  'selu',
  'softplus',
  'swish',
  'gelu',
  'mish',
] as const;

const INFERENCE_ACTIVATION_IDS_BY_NAME = createInferenceActivationIdsByName();

/**
 * Stable activation-function table used by worker inference transport.
 *
 * Each entry index is part of the public inference payload contract. New worker
 * payloads should reference activations by these numeric ids rather than by
 * serializing function bodies or relying on runtime function identity.
 *
 * @example
 * ```ts
 * const activationFunction = INFERENCE_ACTIVATION_TABLE[0];
 * const outputValue = activationFunction(0.5);
 * ```
 */
export const INFERENCE_ACTIVATION_TABLE: ReadonlyArray<ActivationFn> =
  ACTIVATION_FUNCTIONS;

/**
 * Export one structured-clone-safe inference payload from a live runtime network.
 * The resulting payload keeps deterministic activation ordering, stable node or edge indexing, and canonical activation names so a worker can replay inference semantics without shipping live graph objects.
 *
 * @param network Runtime network to serialize for worker transport.
 * @returns Portable inference payload.
 * @example
 * ```ts
 * const payload = exportPortableInferencePayload(network);
 * console.log(payload.nodes[0]?.activation);
 * ```
 */
export function exportPortableInferencePayload(
  network: Network,
): PortableInferencePayload {
  // Step 1: Reuse the deterministic Phase 0 IR as the payload substrate.
  const inferenceIr = extractNetworkInferenceIR(network);

  // Step 2: Rehydrate the portable payload with canonical activation names.
  return {
    version: PORTABLE_PAYLOAD_VERSION,
    strategy: PORTABLE_PREDICTOR_STRATEGY,
    inputCount: inferenceIr.inputCount,
    outputCount: inferenceIr.outputCount,
    activationSteps: inferenceIr.activationSteps.map((activationStep) => [
      ...activationStep,
    ]),
    nodes: inferenceIr.nodes.map((inferenceNode) =>
      buildPortableInferencePayloadNode(inferenceNode),
    ),
    edges: inferenceIr.edges.map((inferenceEdge) =>
      buildPortableInferencePayloadEdge(inferenceEdge),
    ),
    outputNodeIndices: [...inferenceIr.outputNodeIndices],
    activationTable: [...SUPPORTED_INFERENCE_ACTIVATION_KEYS],
  };
}

/**
 * Export one typed-array inference payload optimized for low-copy worker transport.
 * This variant preserves the same deterministic IR semantics as the portable payload while flattening fields into transfer-friendly typed shelves that can be moved across worker boundaries efficiently.
 *
 * @param network Runtime network to serialize for worker transport.
 * @param options Transferable export configuration.
 * @returns Transferable inference payload.
 * @example
 * ```ts
 * const payload = exportTransferableInferencePayload(network, {
 *   numericPrecision: 'f32',
 * });
 * console.log(payload.edgeWeights.length);
 * ```
 */
export function exportTransferableInferencePayload(
  network: Network,
  options: TransferableInferencePayloadOptions = {},
): TransferableInferencePayload {
  // Step 1: Reuse the deterministic inference IR as the transferable source of truth.
  const inferenceIr = extractNetworkInferenceIR(network);
  const numericPrecision =
    options.numericPrecision ?? DEFAULT_TRANSFERABLE_NUMERIC_PRECISION;
  const flattenedActivationSteps = flattenActivationStepsForTransferablePayload(
    inferenceIr.activationSteps,
  );

  // Step 2: Pack the IR into transfer-friendly typed arrays.
  return {
    version: TRANSFERABLE_PAYLOAD_VERSION,
    strategy: TRANSFERABLE_PREDICTOR_STRATEGY,
    inputCount: inferenceIr.inputCount,
    outputCount: inferenceIr.outputCount,
    activationStepsIndex: new Int32Array(
      flattenedActivationSteps.activationStepsIndex,
    ),
    activationStepsData: new Int32Array(
      flattenedActivationSteps.activationStepsData,
    ),
    nodeIds: new Int32Array(
      inferenceIr.nodes.map((inferenceNode) => inferenceNode.index),
    ),
    nodeBiases: createTransferableNumericArray(
      inferenceIr.nodes.map((inferenceNode) => inferenceNode.bias),
      numericPrecision,
    ),
    nodeResponses: createTransferableNumericArray(
      inferenceIr.nodes.map((inferenceNode) => inferenceNode.response),
      numericPrecision,
    ),
    nodeMasks: createTransferableNumericArray(
      inferenceIr.nodes.map((inferenceNode) => inferenceNode.mask),
      numericPrecision,
    ),
    nodeActivationIds: new Int32Array(
      inferenceIr.nodes.map((inferenceNode) => inferenceNode.activationId),
    ),
    nodeSelfWeights: createTransferableNumericArray(
      inferenceIr.nodes.map((inferenceNode) => inferenceNode.selfWeight),
      numericPrecision,
    ),
    nodeSelfGaterIndices: new Int32Array(
      inferenceIr.nodes.map((inferenceNode) => inferenceNode.selfGaterIndex),
    ),
    edgeFrom: new Int32Array(
      inferenceIr.edges.map((inferenceEdge) => inferenceEdge.from),
    ),
    edgeTo: new Int32Array(
      inferenceIr.edges.map((inferenceEdge) => inferenceEdge.to),
    ),
    edgeWeights: createTransferableNumericArray(
      inferenceIr.edges.map((inferenceEdge) => inferenceEdge.weight),
      numericPrecision,
    ),
    edgeGaterIndices: new Int32Array(
      inferenceIr.edges.map((inferenceEdge) => inferenceEdge.gaterIndex),
    ),
    outputNodeIndices: new Int32Array(inferenceIr.outputNodeIndices),
    activationTableLength: INFERENCE_ACTIVATION_TABLE.length,
  };
}

/**
 * Collect every transferable buffer from one typed-array inference payload.
 *
 * Transfer these buffers exactly once when posting the payload to a worker.
 * After the transfer completes, the sending-side typed arrays are neutered and
 * must not be read again.
 *
 * @param payload Transferable inference payload whose buffers should move.
 * @returns ArrayBuffer transfer list aligned to the payload's typed shelves.
 * @example
 * ```ts
 * const payload = exportTransferableInferencePayload(network);
 * worker.postMessage(payload, getTransferList(payload));
 * ```
 */
export function getTransferList(
  payload: TransferableInferencePayload,
): ArrayBuffer[] {
  return [
    payload.activationStepsIndex,
    payload.activationStepsData,
    payload.nodeIds,
    payload.nodeBiases,
    payload.nodeResponses,
    payload.nodeMasks,
    payload.nodeActivationIds,
    payload.nodeSelfWeights,
    payload.nodeSelfGaterIndices,
    payload.edgeFrom,
    payload.edgeTo,
    payload.edgeWeights,
    payload.edgeGaterIndices,
    payload.outputNodeIndices,
  ].map((typedArray) => typedArray.buffer as ArrayBuffer);
}

/**
 * Create a reusable local predictor from either portable or transferable inference payload contracts.
 * The factory normalizes both transport strategies into one inference interface so callers can benchmark or run fallback execution paths without special-case runtime branching.
 *
 * @param payload Portable or transferable inference payload.
 * @returns Predictor that mirrors runtime no-trace activation semantics.
 * @example
 * ```ts
 * const payload = exportPortableInferencePayload(network);
 * const predictor = createInferencePredictor(payload);
 * const outputValues = predictor.predict([0.25, 0.75]);
 * ```
 */
export function createInferencePredictor(
  payload: PortableInferencePayload,
): InferencePredictor;
export function createInferencePredictor(
  payload: TransferableInferencePayload,
): InferencePredictor;
export function createInferencePredictor(
  payload: PortableInferencePayload | TransferableInferencePayload,
): InferencePredictor {
  if (payload.strategy === TRANSFERABLE_PREDICTOR_STRATEGY) {
    return createTransferableInferencePredictor(payload);
  }

  return createPortableInferencePredictor(payload);
}

/**
 * Create a reusable local predictor from a portable inference payload.
 *
 * @param payload Portable inference payload.
 * @returns Predictor that mirrors runtime no-trace activation semantics.
 */
function createPortableInferencePredictor(
  payload: PortableInferencePayload,
): InferencePredictor {
  // Step 1: Precompute immutable lookup tables for the hot prediction path.
  const portableNodes = resolvePortableNodesByIndex(payload);
  const activationFunctions = portableNodes.map((portableNode) =>
    resolvePortableActivationFunction(portableNode, payload.activationTable),
  );
  const incomingEdgesByTargetIndex = createIncomingEdgesByTargetIndex(
    payload.edges,
    portableNodes.length,
  );
  const edgeIndexesByGaterIndex = createEdgeIndexesByGaterIndex(payload.edges);
  const selfNodeIndexesByGaterIndex =
    createSelfNodeIndexesByGaterIndex(portableNodes);
  const activationValues = new Float64Array(portableNodes.length);
  const stateValues = new Float64Array(portableNodes.length);
  const edgeGains = new Float64Array(payload.edges.length).fill(1);
  const selfConnectionGains = new Float64Array(portableNodes.length).fill(1);

  return {
    strategy: PORTABLE_PREDICTOR_STRATEGY,
    predict(inputValues: ReadonlyArray<number>): number[] {
      validatePredictorInputSize(inputValues, payload.inputCount);
      seedPredictorInputActivations(
        activationValues,
        inputValues,
        payload.inputCount,
      );

      for (const activationStep of payload.activationSteps) {
        for (const nodeIndex of activationStep) {
          activatePortableNode({
            activationFunctions,
            activationValues,
            edgeGains,
            edgeIndexesByGaterIndex,
            incomingEdgesByTargetIndex,
            nodeIndex,
            portableEdges: payload.edges,
            portableNodes,
            selfConnectionGains,
            selfNodeIndexesByGaterIndex,
            stateValues,
          });
        }
      }

      return payload.outputNodeIndices.map(
        (nodeIndex) => activationValues[nodeIndex] ?? 0,
      );
    },
    reset(): void {
      activationValues.fill(0);
      stateValues.fill(0);
      edgeGains.fill(1);
      selfConnectionGains.fill(1);
    },
  };
}

/**
 * Create a reusable local predictor from a transferable inference payload.
 *
 * @param payload Transferable inference payload.
 * @returns Predictor that mirrors runtime no-trace activation semantics.
 */
function createTransferableInferencePredictor(
  payload: TransferableInferencePayload,
): InferencePredictor {
  // Step 1: Validate the typed shelves before building the hot prediction path.
  validateTransferableActivationTableLength(payload.activationTableLength);
  validateTransferableNodeIds(payload.nodeIds);
  validateTransferableFieldLengths(payload);

  // Step 2: Reuse the typed shelves directly for predictor lookup state.
  const activationFunctions = resolveTransferableActivationFunctions(payload);
  const incomingEdgesByTargetIndex =
    createIncomingTransferableEdgesByTargetIndex(
      payload.edgeTo,
      payload.nodeIds.length,
    );
  const edgeIndexesByGaterIndex = createTransferableEdgeIndexesByGaterIndex(
    payload.edgeGaterIndices,
  );
  const selfNodeIndexesByGaterIndex =
    createTransferableSelfNodeIndexesByGaterIndex(payload.nodeSelfGaterIndices);
  const activationValues = new Float64Array(payload.nodeIds.length);
  const stateValues = new Float64Array(payload.nodeIds.length);
  const edgeGains = new Float64Array(payload.edgeFrom.length).fill(1);
  const selfConnectionGains = new Float64Array(payload.nodeIds.length).fill(1);

  return {
    strategy: TRANSFERABLE_PREDICTOR_STRATEGY,
    predict(inputValues: ReadonlyArray<number>): number[] {
      validatePredictorInputSize(inputValues, payload.inputCount);
      seedPredictorInputActivations(
        activationValues,
        inputValues,
        payload.inputCount,
      );

      for (
        let activationStepIndex = 0;
        activationStepIndex < payload.activationStepsIndex.length;
        activationStepIndex += 1
      ) {
        const activationStepStart =
          payload.activationStepsIndex[activationStepIndex];
        const activationStepEnd =
          payload.activationStepsIndex[activationStepIndex + 1] ??
          payload.activationStepsData.length;

        for (
          let activationDataIndex = activationStepStart;
          activationDataIndex < activationStepEnd;
          activationDataIndex += 1
        ) {
          const nodeIndex = resolveTransferableActivationStepNodeIndex(
            payload.activationStepsData,
            activationDataIndex,
          );

          activateTransferableNode({
            activationFunctions,
            activationValues,
            edgeFrom: payload.edgeFrom,
            edgeGains,
            edgeIndexesByGaterIndex,
            edgeWeights: payload.edgeWeights,
            incomingEdgesByTargetIndex,
            nodeBiases: payload.nodeBiases,
            nodeIndex,
            nodeMasks: payload.nodeMasks,
            nodeResponses: payload.nodeResponses,
            nodeSelfWeights: payload.nodeSelfWeights,
            selfConnectionGains,
            selfNodeIndexesByGaterIndex,
            stateValues,
          });
        }
      }

      return Array.from(
        payload.outputNodeIndices,
        (nodeIndex) => activationValues[nodeIndex] ?? 0,
      );
    },
    reset(): void {
      activationValues.fill(0);
      stateValues.fill(0);
      edgeGains.fill(1);
      selfConnectionGains.fill(1);
    },
  };
}

/**
 * Extract a deterministic inference IR from one live network snapshot.
 * This extraction pass captures exactly the runtime data required for worker-side forward execution, including node scalars, filtered forward edges, grouped activation steps, and stable output indexing.
 *
 * @param network Runtime network to snapshot.
 * @returns Worker-friendly inference IR.
 * @example
 * ```ts
 * const inferenceIr = extractNetworkInferenceIR(network);
 * console.log(inferenceIr.outputNodeIndices);
 * ```
 */
export function extractNetworkInferenceIR(
  network: Network,
): NetworkInferenceIR {
  // Step 1: Seed stable indexes from the standalone setup seam.
  const standaloneProps = asStandaloneProps(network);
  ensureOutputNodesExist(standaloneProps);
  const generationContext = createGenerationContext(standaloneProps);
  seedNodeIndexesAndState(generationContext);
  resolveStandaloneExecutionMetadata(network, generationContext);

  // Step 2: Build deterministic node and gene-id lookups.
  const nodesByGeneId = createNodesByGeneId(standaloneProps.nodes);

  // Step 3: Snapshot node scalars, edge list, and grouped activation steps.
  return {
    inputCount: network.input,
    outputCount: network.output,
    nodes: standaloneProps.nodes.map((nodeReference) =>
      buildInferenceIrNode(nodeReference),
    ),
    edges: network.connections
      .filter((connectionReference) =>
        shouldIncludeForwardEdge(connectionReference),
      )
      .map((connectionReference) => buildInferenceIrEdge(connectionReference)),
    activationSteps: resolveActivationSteps(
      network,
      generationContext,
      nodesByGeneId,
    ),
    outputNodeIndices: [...generationContext.outputNodeIndexes],
  };
}

/**
 * Build the stable activation-name lookup used by inference payloads.
 *
 * @returns Activation-id lookup by canonical key or function name.
 */
function createInferenceActivationIdsByName(): Map<string, number> {
  const activationIdsByName = new Map<string, number>();

  SUPPORTED_INFERENCE_ACTIVATION_KEYS.forEach((activationKey, activationId) => {
    activationIdsByName.set(activationKey, activationId);
    activationIdsByName.set(
      ACTIVATION_FUNCTIONS[activationId].name,
      activationId,
    );
  });

  activationIdsByName.set('sigmoid', 0);
  activationIdsByName.set('sigmoidActivation', 0);
  return activationIdsByName;
}

type TransferableNumericPrecision = NonNullable<
  TransferableInferencePayloadOptions['numericPrecision']
>;

/**
 * Build the transferable activation-step packing from grouped IR steps.
 *
 * @param activationSteps Grouped activation steps from the inference IR.
 * @returns Flat activation-step data plus per-step start offsets.
 */
function flattenActivationStepsForTransferablePayload(
  activationSteps: ReadonlyArray<ReadonlyArray<number>>,
): {
  activationStepsData: number[];
  activationStepsIndex: number[];
} {
  const activationStepsData: number[] = [];
  const activationStepsIndex: number[] = [];
  let nextActivationStepOffset = 0;

  for (const activationStep of activationSteps) {
    activationStepsIndex.push(nextActivationStepOffset);
    activationStepsData.push(...activationStep);
    nextActivationStepOffset += activationStep.length;
  }

  return {
    activationStepsData,
    activationStepsIndex,
  };
}

/**
 * Create one numeric typed array using the requested transferable precision.
 *
 * @param values Numeric values to pack.
 * @param numericPrecision Requested precision mode.
 * @returns Float32Array or Float64Array depending on the selected mode.
 */
function createTransferableNumericArray(
  values: readonly number[],
  numericPrecision: TransferableNumericPrecision,
): Float32Array | Float64Array {
  if (numericPrecision === 'f32') {
    return new Float32Array(values);
  }

  return new Float64Array(values);
}

/**
 * Build one deterministic node snapshot for the inference IR.
 *
 * @param nodeReference Runtime node.
 * @returns Node IR snapshot.
 */
function buildInferenceIrNode(nodeReference: Node) {
  const nodeIndex = resolveNodeIndexOrThrow(nodeReference);
  const selfConnection = resolveActiveSelfConnection(nodeReference);

  return {
    index: nodeIndex,
    bias: nodeReference.bias,
    response: nodeReference.response,
    mask: nodeReference.mask,
    activationId: resolveInferenceActivationId(nodeReference),
    selfWeight: selfConnection?.weight ?? 0,
    selfGaterIndex: resolveOptionalNodeIndex(selfConnection?.gater ?? null),
  };
}

/**
 * Build one deterministic non-self edge snapshot for the inference IR.
 *
 * @param connectionReference Runtime connection.
 * @returns Edge IR snapshot.
 */
function buildInferenceIrEdge(connectionReference: Connection) {
  return {
    from: resolveNodeIndexOrThrow(connectionReference.from),
    to: resolveNodeIndexOrThrow(connectionReference.to),
    weight: connectionReference.weight,
    gaterIndex: resolveOptionalNodeIndex(connectionReference.gater),
  };
}

/**
 * Resolve grouped activation steps from the compiled runtime schedule when available.
 *
 * @param network Runtime network.
 * @param generationContext Standalone generation context with fallback traversal order.
 * @param nodesByGeneId Stable node lookup by gene id.
 * @returns Grouped activation steps.
 */
function resolveActivationSteps(
  network: Network,
  generationContext: StandaloneGenerationContext,
  nodesByGeneId: ReadonlyMap<number, Node>,
): ReadonlyArray<ReadonlyArray<number>> {
  const compiledActivationSteps = resolveCompiledActivationSteps(
    network,
    nodesByGeneId,
  );

  if (compiledActivationSteps) {
    return compiledActivationSteps;
  }

  return generationContext.activationNodeIndexes.map((nodeIndex) => [
    nodeIndex,
  ]);
}

/**
 * Resolve grouped activation steps from the compiled activation schedule.
 *
 * @param network Runtime network.
 * @param nodesByGeneId Stable node lookup by gene id.
 * @returns Grouped activation steps or null when the schedule is unavailable.
 */
function resolveCompiledActivationSteps(
  network: Network,
  nodesByGeneId: ReadonlyMap<number, Node>,
): ReadonlyArray<ReadonlyArray<number>> | null {
  const activationSchedule = Reflect.get(network, '_activationSchedule') as
    | ActivationSchedule
    | null
    | undefined;

  if (!activationSchedule || activationSchedule.steps.length === 0) {
    return null;
  }

  const activationSteps: number[][] = [];

  for (const activationStep of activationSchedule.steps) {
    const stepNodeIndexes = activationStep.nodeIds.flatMap(
      (nodeGeneId: number) => {
        const nodeReference = nodesByGeneId.get(nodeGeneId);

        if (!nodeReference || nodeReference.type === 'input') {
          return [];
        }

        return [resolveNodeIndexOrThrow(nodeReference)];
      },
    );

    if (stepNodeIndexes.length === 0) {
      continue;
    }

    const iterationCount =
      activationStep.kind === 'recurrent-component'
        ? (activationStep.iterations ?? DEFAULT_RECURRENT_ITERATIONS)
        : DEFAULT_RECURRENT_ITERATIONS;

    for (
      let iterationIndex = 0;
      iterationIndex < iterationCount;
      iterationIndex += 1
    ) {
      activationSteps.push([...stepNodeIndexes]);
    }
  }

  return activationSteps;
}

/**
 * Build a stable node lookup keyed by runtime gene id.
 *
 * @param nodes Runtime nodes in deterministic index order.
 * @returns Node lookup keyed by gene id.
 */
function createNodesByGeneId(nodes: readonly Node[]): Map<number, Node> {
  const nodesByGeneId = new Map<number, Node>();

  for (const nodeReference of nodes) {
    nodesByGeneId.set(nodeReference.geneId, nodeReference);
  }

  return nodesByGeneId;
}

/**
 * Resolve the supported activation id for one runtime node.
 *
 * @param nodeReference Runtime node.
 * @returns Stable activation id.
 */
function resolveInferenceActivationId(nodeReference: Node): number {
  const activationFunction = nodeReference.squash as ActivationFunction;
  const activationCandidates = [
    resolveActivationKey(activationFunction),
    activationFunction.name,
  ];

  for (const activationCandidate of activationCandidates) {
    const activationId =
      INFERENCE_ACTIVATION_IDS_BY_NAME.get(activationCandidate);

    if (typeof activationId === 'number') {
      return activationId;
    }
  }

  throw new Error(
    `Unsupported worker inference activation '${String(activationFunction.name)}' on node ${resolveNodeIndexOrThrow(nodeReference)}.`,
  );
}

/**
 * Resolve the active self-connection for one runtime node.
 *
 * @param nodeReference Runtime node.
 * @returns Active self-connection or undefined.
 */
function resolveActiveSelfConnection(
  nodeReference: Node,
): Connection | undefined {
  return nodeReference.connections.self.find((connectionReference) =>
    shouldIncludeSelfConnection(connectionReference),
  );
}

/**
 * Determine whether a connection contributes to inference.
 *
 * @param connectionReference Runtime connection.
 * @returns True when the connection should appear in the IR.
 */
function shouldIncludeForwardEdge(connectionReference: Connection): boolean {
  return shouldIncludeConnection(connectionReference);
}

/**
 * Determine whether a self-connection contributes to inference.
 *
 * @param connectionReference Runtime self-connection.
 * @returns True when the self-connection should appear in the IR node snapshot.
 */
function shouldIncludeSelfConnection(connectionReference: Connection): boolean {
  return shouldIncludeConnection(connectionReference);
}

/**
 * Determine whether a connection contributes to inference regardless of shape.
 *
 * @param connectionReference Runtime connection.
 * @returns True when the connection is enabled for inference.
 */
function shouldIncludeConnection(connectionReference: Connection): boolean {
  if (connectionReference.enabled === false) {
    return false;
  }

  if (connectionReference.dcMask === 0) {
    return false;
  }

  return true;
}

/**
 * Resolve a required node index from the seeded standalone surface.
 *
 * @param nodeReference Runtime node.
 * @returns Stable node index.
 */
function resolveNodeIndexOrThrow(nodeReference: Node): number {
  const indexedNode = nodeReference as Partial<NodeWithIndex>;

  if (typeof indexedNode.index === 'number') {
    return indexedNode.index;
  }

  throw new Error(
    `Expected a seeded node index for gene id ${nodeReference.geneId}.`,
  );
}

/**
 * Resolve an optional node index for gater references.
 *
 * @param nodeReference Optional runtime node.
 * @returns Stable node index or `-1` when absent.
 */
function resolveOptionalNodeIndex(nodeReference: Node | null): number {
  if (!nodeReference) {
    return NO_GATER_INDEX;
  }

  return resolveNodeIndexOrThrow(nodeReference);
}

/**
 * Build one portable node payload record from the inference IR.
 *
 * @param inferenceNode Inference IR node snapshot.
 * @returns Structured-clone-safe node payload record.
 */
function buildPortableInferencePayloadNode(
  inferenceNode: NetworkInferenceIR['nodes'][number],
): PortableInferencePayloadNode {
  return {
    id: inferenceNode.index,
    bias: inferenceNode.bias,
    response: inferenceNode.response,
    mask: inferenceNode.mask,
    activation: SUPPORTED_INFERENCE_ACTIVATION_KEYS[inferenceNode.activationId],
    selfWeight: inferenceNode.selfWeight,
    selfGaterIndex: inferenceNode.selfGaterIndex,
  };
}

/**
 * Build one portable edge payload record from the inference IR.
 *
 * @param inferenceEdge Inference IR edge snapshot.
 * @returns Structured-clone-safe edge payload record.
 */
function buildPortableInferencePayloadEdge(
  inferenceEdge: NetworkInferenceIR['edges'][number],
): PortableInferencePayloadEdge {
  return {
    from: inferenceEdge.from,
    to: inferenceEdge.to,
    weight: inferenceEdge.weight,
    gaterIndex: inferenceEdge.gaterIndex,
  };
}

type PortableActivationContext = {
  activationFunctions: readonly ActivationFn[];
  activationValues: Float64Array;
  edgeGains: Float64Array;
  edgeIndexesByGaterIndex: ReadonlyMap<number, readonly number[]>;
  incomingEdgesByTargetIndex: ReadonlyArray<readonly number[]>;
  nodeIndex: number;
  portableEdges: readonly PortableInferencePayloadEdge[];
  portableNodes: readonly PortableInferencePayloadNode[];
  selfConnectionGains: Float64Array;
  selfNodeIndexesByGaterIndex: ReadonlyMap<number, readonly number[]>;
  stateValues: Float64Array;
};

type TransferableActivationContext = {
  activationFunctions: readonly ActivationFn[];
  activationValues: Float64Array;
  edgeFrom: Int32Array;
  edgeGains: Float64Array;
  edgeIndexesByGaterIndex: ReadonlyMap<number, readonly number[]>;
  edgeWeights: TransferableInferencePayload['edgeWeights'];
  incomingEdgesByTargetIndex: ReadonlyArray<readonly number[]>;
  nodeBiases: TransferableInferencePayload['nodeBiases'];
  nodeIndex: number;
  nodeMasks: TransferableInferencePayload['nodeMasks'];
  nodeResponses: TransferableInferencePayload['nodeResponses'];
  nodeSelfWeights: TransferableInferencePayload['nodeSelfWeights'];
  selfConnectionGains: Float64Array;
  selfNodeIndexesByGaterIndex: ReadonlyMap<number, readonly number[]>;
  stateValues: Float64Array;
};

/**
 * Activate one portable node using no-trace runtime semantics.
 *
 * @param activationContext Predictor context for this activation step.
 * @returns Nothing.
 */
function activatePortableNode(
  activationContext: PortableActivationContext,
): void {
  const {
    activationFunctions,
    activationValues,
    edgeGains,
    edgeIndexesByGaterIndex,
    incomingEdgesByTargetIndex,
    nodeIndex,
    portableEdges,
    portableNodes,
    selfConnectionGains,
    selfNodeIndexesByGaterIndex,
    stateValues,
  } = activationContext;
  const portableNode = portableNodes[nodeIndex];

  if (!portableNode) {
    throw new Error(`Expected portable node ${nodeIndex}.`);
  }

  const previousState = stateValues[nodeIndex];
  let nextState = portableNode.bias;

  if (portableNode.selfWeight !== 0) {
    nextState +=
      portableNode.selfWeight * selfConnectionGains[nodeIndex] * previousState;
  }

  for (const edgeIndex of incomingEdgesByTargetIndex[nodeIndex]) {
    const portableEdge = portableEdges[edgeIndex];

    if (!portableEdge) {
      throw new Error(`Expected portable edge ${edgeIndex}.`);
    }

    nextState +=
      activationValues[portableEdge.from] *
      portableEdge.weight *
      edgeGains[edgeIndex];
  }

  stateValues[nodeIndex] = nextState;

  const effectiveState = nextState * portableNode.response;
  const nextActivation =
    activationFunctions[nodeIndex](effectiveState) * portableNode.mask;
  activationValues[nodeIndex] = nextActivation;

  for (const edgeIndex of edgeIndexesByGaterIndex.get(nodeIndex) ?? []) {
    edgeGains[edgeIndex] = nextActivation;
  }

  for (const selfNodeIndex of selfNodeIndexesByGaterIndex.get(nodeIndex) ??
    []) {
    selfConnectionGains[selfNodeIndex] = nextActivation;
  }
}

/**
 * Activate one transferable node using no-trace runtime semantics.
 *
 * @param activationContext Predictor context for this activation step.
 * @returns Nothing.
 */
function activateTransferableNode(
  activationContext: TransferableActivationContext,
): void {
  const {
    activationFunctions,
    activationValues,
    edgeFrom,
    edgeGains,
    edgeIndexesByGaterIndex,
    edgeWeights,
    incomingEdgesByTargetIndex,
    nodeBiases,
    nodeIndex,
    nodeMasks,
    nodeResponses,
    nodeSelfWeights,
    selfConnectionGains,
    selfNodeIndexesByGaterIndex,
    stateValues,
  } = activationContext;

  if (nodeIndex < 0 || nodeIndex >= nodeBiases.length) {
    throw new Error(`Expected transferable node ${nodeIndex}.`);
  }

  const previousState = stateValues[nodeIndex];
  let nextState = nodeBiases[nodeIndex];

  if (nodeSelfWeights[nodeIndex] !== 0) {
    nextState +=
      nodeSelfWeights[nodeIndex] *
      selfConnectionGains[nodeIndex] *
      previousState;
  }

  for (const edgeIndex of incomingEdgesByTargetIndex[nodeIndex]) {
    const edgeSourceNodeIndex = edgeFrom[edgeIndex];
    const edgeWeight = edgeWeights[edgeIndex];

    nextState +=
      activationValues[edgeSourceNodeIndex] * edgeWeight * edgeGains[edgeIndex];
  }

  stateValues[nodeIndex] = nextState;

  const effectiveState = nextState * nodeResponses[nodeIndex];
  const nextActivation =
    activationFunctions[nodeIndex](effectiveState) * nodeMasks[nodeIndex];
  activationValues[nodeIndex] = nextActivation;

  for (const edgeIndex of edgeIndexesByGaterIndex.get(nodeIndex) ?? []) {
    edgeGains[edgeIndex] = nextActivation;
  }

  for (const selfNodeIndex of selfNodeIndexesByGaterIndex.get(nodeIndex) ??
    []) {
    selfConnectionGains[selfNodeIndex] = nextActivation;
  }
}

/**
 * Resolve portable nodes in strict node-index order.
 *
 * @param payload Portable inference payload.
 * @returns Portable nodes aligned to index order.
 */
function resolvePortableNodesByIndex(
  payload: PortableInferencePayload,
): PortableInferencePayloadNode[] {
  const portableNodes = payload.nodes.toSorted((leftNode, rightNode) => {
    return leftNode.id - rightNode.id;
  });

  portableNodes.forEach((portableNode, nodeIndex) => {
    if (portableNode.id !== nodeIndex) {
      throw new Error(
        `Expected portable node id ${nodeIndex}, received ${portableNode.id}.`,
      );
    }
  });

  return portableNodes;
}

/**
 * Resolve the activation function for one portable node.
 *
 * @param portableNode Portable node payload record.
 * @param activationTable Portable activation table.
 * @returns Activation function from the stable inference registry.
 */
function resolvePortableActivationFunction(
  portableNode: PortableInferencePayloadNode,
  activationTable: readonly string[],
): ActivationFn {
  const activationIndex = activationTable.indexOf(portableNode.activation);

  if (activationIndex === -1 || !INFERENCE_ACTIVATION_TABLE[activationIndex]) {
    throw new Error(
      `Unsupported portable activation '${portableNode.activation}' on node ${portableNode.id}.`,
    );
  }

  return INFERENCE_ACTIVATION_TABLE[activationIndex];
}

/**
 * Validate that the transferable activation shelf matches the runtime registry.
 *
 * @param activationTableLength Activation shelf length encoded in the payload.
 * @returns Nothing.
 */
function validateTransferableActivationTableLength(
  activationTableLength: number,
): void {
  if (activationTableLength === INFERENCE_ACTIVATION_TABLE.length) {
    return;
  }

  throw new Error(
    `Expected activation table length ${INFERENCE_ACTIVATION_TABLE.length}, received ${activationTableLength}.`,
  );
}

/**
 * Validate that transferable node ids remain aligned to typed-shelf order.
 *
 * @param nodeIds Transferable node-id shelf.
 * @returns Nothing.
 */
function validateTransferableNodeIds(nodeIds: Int32Array): void {
  for (let nodeIndex = 0; nodeIndex < nodeIds.length; nodeIndex += 1) {
    const nodeId = nodeIds[nodeIndex];

    if (nodeId !== nodeIndex) {
      throw new Error(
        `Expected transferable node id ${nodeIndex}, received ${nodeId}.`,
      );
    }
  }
}

/**
 * Validate that every transferable typed shelf stays aligned to the payload contract.
 *
 * @param payload Transferable inference payload.
 * @returns Nothing.
 */
function validateTransferableFieldLengths(
  payload: TransferableInferencePayload,
): void {
  const expectedNodeCount = payload.nodeIds.length;
  const expectedEdgeCount = payload.edgeFrom.length;

  validateTransferableFieldLength(
    'nodeBiases',
    payload.nodeBiases.length,
    expectedNodeCount,
  );
  validateTransferableFieldLength(
    'nodeResponses',
    payload.nodeResponses.length,
    expectedNodeCount,
  );
  validateTransferableFieldLength(
    'nodeMasks',
    payload.nodeMasks.length,
    expectedNodeCount,
  );
  validateTransferableFieldLength(
    'nodeActivationIds',
    payload.nodeActivationIds.length,
    expectedNodeCount,
  );
  validateTransferableFieldLength(
    'nodeSelfWeights',
    payload.nodeSelfWeights.length,
    expectedNodeCount,
  );
  validateTransferableFieldLength(
    'nodeSelfGaterIndices',
    payload.nodeSelfGaterIndices.length,
    expectedNodeCount,
  );
  validateTransferableFieldLength(
    'edgeTo',
    payload.edgeTo.length,
    expectedEdgeCount,
  );
  validateTransferableFieldLength(
    'edgeWeights',
    payload.edgeWeights.length,
    expectedEdgeCount,
  );
  validateTransferableFieldLength(
    'edgeGaterIndices',
    payload.edgeGaterIndices.length,
    expectedEdgeCount,
  );
}

/**
 * Validate one transferable shelf length against the expected payload count.
 *
 * @param fieldName Payload field name.
 * @param actualLength Actual typed-array length.
 * @param expectedLength Expected aligned length.
 * @returns Nothing.
 */
function validateTransferableFieldLength(
  fieldName: string,
  actualLength: number,
  expectedLength: number,
): void {
  if (actualLength === expectedLength) {
    return;
  }

  throw new Error(
    `Expected transferable field '${fieldName}' to have length ${expectedLength}, received ${actualLength}.`,
  );
}

/**
 * Resolve one transferable activation-step node index or throw when the step data is truncated.
 *
 * @param activationStepsData Flat activation-step node shelf.
 * @param activationDataIndex Index inside the flat activation-step shelf.
 * @returns Transferable node index for one activation step entry.
 */
function resolveTransferableActivationStepNodeIndex(
  activationStepsData: Int32Array,
  activationDataIndex: number,
): number {
  const nodeIndex = activationStepsData[activationDataIndex];

  if (typeof nodeIndex !== 'number') {
    throw new Error(
      `Expected transferable activation step node ${activationDataIndex}.`,
    );
  }

  return nodeIndex;
}

/**
 * Resolve activation functions directly from the transferable activation-id shelf.
 *
 * @param payload Transferable inference payload.
 * @returns Activation functions aligned to the transferable node order.
 */
function resolveTransferableActivationFunctions(
  payload: TransferableInferencePayload,
): ActivationFn[] {
  return Array.from(payload.nodeActivationIds, (activationId, nodeIndex) => {
    const activationFunction = INFERENCE_ACTIVATION_TABLE[activationId];

    if (!activationFunction) {
      throw new Error(
        `Unsupported transferable activation id '${activationId}' on node ${payload.nodeIds[nodeIndex]}.`,
      );
    }

    return activationFunction;
  });
}

/**
 * Build incoming-edge indexes keyed by target node index.
 *
 * @param portableEdges Portable payload edges.
 * @param nodeCount Portable node count.
 * @returns Incoming edge indexes keyed by target node index.
 */
function createIncomingEdgesByTargetIndex(
  portableEdges: readonly PortableInferencePayloadEdge[],
  nodeCount: number,
): ReadonlyArray<readonly number[]> {
  const incomingEdgesByTargetIndex = Array.from(
    { length: nodeCount },
    () => [] as number[],
  );

  portableEdges.forEach((portableEdge, edgeIndex) => {
    incomingEdgesByTargetIndex[portableEdge.to]?.push(edgeIndex);
  });

  return incomingEdgesByTargetIndex;
}

/**
 * Build incoming transferable-edge indexes keyed by target node index.
 *
 * @param edgeTo Transferable edge target-node shelf.
 * @param nodeCount Transferable node count.
 * @returns Incoming edge indexes keyed by target node index.
 */
function createIncomingTransferableEdgesByTargetIndex(
  edgeTo: Int32Array,
  nodeCount: number,
): ReadonlyArray<readonly number[]> {
  const incomingEdgesByTargetIndex = Array.from(
    { length: nodeCount },
    () => [] as number[],
  );

  edgeTo.forEach((targetNodeIndex, edgeIndex) => {
    incomingEdgesByTargetIndex[targetNodeIndex]?.push(edgeIndex);
  });

  return incomingEdgesByTargetIndex;
}

/**
 * Build gated forward-edge indexes keyed by gater node index.
 *
 * @param portableEdges Portable payload edges.
 * @returns Gated edge indexes keyed by gater node index.
 */
function createEdgeIndexesByGaterIndex(
  portableEdges: readonly PortableInferencePayloadEdge[],
): Map<number, readonly number[]> {
  const edgeIndexesByGaterIndex = new Map<number, number[]>();

  portableEdges.forEach((portableEdge, edgeIndex) => {
    if (portableEdge.gaterIndex === NO_GATER_INDEX) {
      return;
    }

    const edgeIndexes =
      edgeIndexesByGaterIndex.get(portableEdge.gaterIndex) ?? [];
    edgeIndexes.push(edgeIndex);
    edgeIndexesByGaterIndex.set(portableEdge.gaterIndex, edgeIndexes);
  });

  return edgeIndexesByGaterIndex;
}

/**
 * Build gated transferable forward-edge indexes keyed by gater node index.
 *
 * @param edgeGaterIndices Transferable edge gater shelf.
 * @returns Gated edge indexes keyed by gater node index.
 */
function createTransferableEdgeIndexesByGaterIndex(
  edgeGaterIndices: Int32Array,
): Map<number, readonly number[]> {
  const edgeIndexesByGaterIndex = new Map<number, number[]>();

  edgeGaterIndices.forEach((gaterNodeIndex, edgeIndex) => {
    if (gaterNodeIndex === NO_GATER_INDEX) {
      return;
    }

    const edgeIndexes = edgeIndexesByGaterIndex.get(gaterNodeIndex) ?? [];
    edgeIndexes.push(edgeIndex);
    edgeIndexesByGaterIndex.set(gaterNodeIndex, edgeIndexes);
  });

  return edgeIndexesByGaterIndex;
}

/**
 * Build gated self-connection indexes keyed by gater node index.
 *
 * @param portableNodes Portable payload nodes.
 * @returns Self-connection node indexes keyed by gater node index.
 */
function createSelfNodeIndexesByGaterIndex(
  portableNodes: readonly PortableInferencePayloadNode[],
): Map<number, readonly number[]> {
  const selfNodeIndexesByGaterIndex = new Map<number, number[]>();

  portableNodes.forEach((portableNode) => {
    if (portableNode.selfGaterIndex === NO_GATER_INDEX) {
      return;
    }

    const selfNodeIndexes =
      selfNodeIndexesByGaterIndex.get(portableNode.selfGaterIndex) ?? [];
    selfNodeIndexes.push(portableNode.id);
    selfNodeIndexesByGaterIndex.set(
      portableNode.selfGaterIndex,
      selfNodeIndexes,
    );
  });

  return selfNodeIndexesByGaterIndex;
}

/**
 * Build gated transferable self-connection indexes keyed by gater node index.
 *
 * @param nodeSelfGaterIndices Transferable self-gater shelf.
 * @returns Self-connection node indexes keyed by gater node index.
 */
function createTransferableSelfNodeIndexesByGaterIndex(
  nodeSelfGaterIndices: Int32Array,
): Map<number, readonly number[]> {
  const selfNodeIndexesByGaterIndex = new Map<number, number[]>();

  nodeSelfGaterIndices.forEach((gaterNodeIndex, nodeIndex) => {
    if (gaterNodeIndex === NO_GATER_INDEX) {
      return;
    }

    const selfNodeIndexes =
      selfNodeIndexesByGaterIndex.get(gaterNodeIndex) ?? [];
    selfNodeIndexes.push(nodeIndex);
    selfNodeIndexesByGaterIndex.set(gaterNodeIndex, selfNodeIndexes);
  });

  return selfNodeIndexesByGaterIndex;
}

/**
 * Validate input vector width for one prediction call.
 *
 * @param inputValues Caller-provided input vector.
 * @param expectedInputCount Required input width.
 * @returns Nothing.
 */
function validatePredictorInputSize(
  inputValues: ReadonlyArray<number>,
  expectedInputCount: number,
): void {
  if (inputValues.length === expectedInputCount) {
    return;
  }

  throw new Error(
    `Expected ${expectedInputCount} input values, received ${inputValues.length}.`,
  );
}

/**
 * Seed the public input activations for one prediction pass.
 *
 * @param activationValues Mutable activation buffer.
 * @param inputValues Caller-provided input vector.
 * @param inputCount Public input width.
 * @returns Nothing.
 */
function seedPredictorInputActivations(
  activationValues: Float64Array,
  inputValues: ReadonlyArray<number>,
  inputCount: number,
): void {
  for (let inputIndex = 0; inputIndex < inputCount; inputIndex += 1) {
    activationValues[inputIndex] = inputValues[inputIndex] ?? 0;
  }
}
