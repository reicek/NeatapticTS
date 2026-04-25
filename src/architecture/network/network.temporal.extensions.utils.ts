import type Connection from '../connection';
import type Node from '../node';
import type Network from './network';
import type { NetworkJSONExtensions } from './network.types';
import type {
  NetworkTemporalGatedBlockDescriptor,
  NetworkTemporalRecurrentModuleDescriptor,
  NetworkTemporalRecurrentModuleKind,
  NetworkTemporalStructureDescriptor,
} from './network.types';

const TEMPORAL_EXTENSION_VERSION = 1;
const SUPPORTED_RECURRENT_MODULE_KINDS = new Set([
  'lstm',
  'gru',
  'narx-memory',
] as const);

type TemporalRecurrentModuleKind = NetworkTemporalRecurrentModuleKind;
type TemporalRecurrentModuleDescriptor =
  NetworkTemporalRecurrentModuleDescriptor;
type TemporalGatedBlockDescriptor = NetworkTemporalGatedBlockDescriptor;

type TemporalDescriptorSet = {
  recurrentModules?: TemporalRecurrentModuleDescriptor[];
  gatedBlocks?: TemporalGatedBlockDescriptor[];
};

type RuntimeNetworkWithSerializedExtensions = Network & {
  nodes: Node[];
  _serializedExtensions?: NetworkJSONExtensions;
};

type LstmRoleNodes = {
  inputGate: Node[];
  forgetGate: Node[];
  memoryCell: Node[];
  outputGate: Node[];
  outputBlock: Node[];
};

type GruRoleNodes = {
  updateGate: Node[];
  inverseUpdateGate: Node[];
  resetGate: Node[];
  memoryCell: Node[];
  output: Node[];
  previousOutput: Node[];
};

/**
 * Split one LSTM layer node list into its canonical role groups.
 *
 * @param layerNodes Flat LSTM layer node list in factory order.
 * @param blockSize Number of nodes allocated per role group.
 * @returns Role-group partition when the shape matches one LSTM block.
 */
export function splitLstmLayerNodes(
  layerNodes: readonly Node[],
  blockSize: number,
): LstmRoleNodes | undefined {
  if (blockSize <= 0 || layerNodes.length < blockSize * 5) {
    return undefined;
  }

  return {
    inputGate: layerNodes.slice(0, blockSize),
    forgetGate: layerNodes.slice(blockSize, blockSize * 2),
    memoryCell: layerNodes.slice(blockSize * 2, blockSize * 3),
    outputGate: layerNodes.slice(blockSize * 3, blockSize * 4),
    outputBlock: layerNodes.slice(blockSize * 4, blockSize * 5),
  };
}

/**
 * Split one GRU layer node list into its canonical role groups.
 *
 * @param layerNodes Flat GRU layer node list in factory order.
 * @param blockSize Number of nodes allocated per role group.
 * @returns Role-group partition when the shape matches one GRU block.
 */
export function splitGruLayerNodes(
  layerNodes: readonly Node[],
  blockSize: number,
): GruRoleNodes | undefined {
  if (blockSize <= 0 || layerNodes.length < blockSize * 6) {
    return undefined;
  }

  return {
    updateGate: layerNodes.slice(0, blockSize),
    inverseUpdateGate: layerNodes.slice(blockSize, blockSize * 2),
    resetGate: layerNodes.slice(blockSize * 2, blockSize * 3),
    memoryCell: layerNodes.slice(blockSize * 3, blockSize * 4),
    output: layerNodes.slice(blockSize * 4, blockSize * 5),
    previousOutput: layerNodes.slice(blockSize * 5, blockSize * 6),
  };
}

/**
 * Build the explicit Step 7.4 descriptor set for one runtime LSTM block.
 *
 * @param network Runtime network carrying the block.
 * @param roleNodes Canonical LSTM role groups.
 * @returns Descriptor set suitable for the network extension bag.
 */
export function buildLstmTemporalDescriptorSet(
  network: Network,
  roleNodes: LstmRoleNodes,
): TemporalDescriptorSet | undefined {
  const moduleNodes = [
    ...roleNodes.inputGate,
    ...roleNodes.forgetGate,
    ...roleNodes.memoryCell,
    ...roleNodes.outputGate,
    ...roleNodes.outputBlock,
  ];
  const recurrentModule = createRecurrentModuleDescriptor(
    network,
    'lstm',
    {
      inputGate: roleNodes.inputGate,
      forgetGate: roleNodes.forgetGate,
      memoryCell: roleNodes.memoryCell,
      outputGate: roleNodes.outputGate,
      outputBlock: roleNodes.outputBlock,
    },
    moduleNodes,
    collectGatedConnectionInnovations(network, [
      ...roleNodes.inputGate,
      ...roleNodes.forgetGate,
      ...roleNodes.outputGate,
    ]),
  );
  const gatedBlock = createGatedBlockDescriptor(network, 'lstm', [
    ...roleNodes.inputGate,
    ...roleNodes.forgetGate,
    ...roleNodes.outputGate,
  ]);

  return createTemporalDescriptorSet(recurrentModule, gatedBlock);
}

/**
 * Build the explicit Step 7.4 descriptor set for one runtime GRU block.
 *
 * @param network Runtime network carrying the block.
 * @param roleNodes Canonical GRU role groups.
 * @returns Descriptor set suitable for the network extension bag.
 */
export function buildGruTemporalDescriptorSet(
  network: Network,
  roleNodes: GruRoleNodes,
): TemporalDescriptorSet | undefined {
  const moduleNodes = [
    ...roleNodes.updateGate,
    ...roleNodes.inverseUpdateGate,
    ...roleNodes.resetGate,
    ...roleNodes.memoryCell,
    ...roleNodes.output,
    ...roleNodes.previousOutput,
  ];
  const recurrentModule = createRecurrentModuleDescriptor(
    network,
    'gru',
    {
      updateGate: roleNodes.updateGate,
      inverseUpdateGate: roleNodes.inverseUpdateGate,
      resetGate: roleNodes.resetGate,
      memoryCell: roleNodes.memoryCell,
      output: roleNodes.output,
      previousOutput: roleNodes.previousOutput,
    },
    moduleNodes,
    collectGatedConnectionInnovations(network, [
      ...roleNodes.updateGate,
      ...roleNodes.inverseUpdateGate,
      ...roleNodes.resetGate,
    ]),
  );
  const gatedBlock = createGatedBlockDescriptor(network, 'gru', [
    ...roleNodes.updateGate,
    ...roleNodes.inverseUpdateGate,
    ...roleNodes.resetGate,
  ]);

  return createTemporalDescriptorSet(recurrentModule, gatedBlock);
}

/**
 * Build one explicit Step 7.4 descriptor set for a NARX delay line.
 *
 * @param network Runtime network carrying the delay line.
 * @param moduleLabel Stable label distinguishing multiple delay-line modules.
 * @param memoryBlocks Ordered memory blocks grouped by delay step.
 * @returns Descriptor set suitable for the network extension bag.
 */
export function buildNarxMemoryTemporalDescriptorSet(
  network: Network,
  moduleLabel: string,
  memoryBlocks: readonly (readonly Node[])[],
): TemporalDescriptorSet | undefined {
  const roleNodes = Object.fromEntries(
    memoryBlocks
      .map((memoryBlock, blockIndex) => [
        `delayStep${blockIndex}`,
        [...memoryBlock],
      ])
      .filter(([, roleNodeEntries]) => roleNodeEntries.length > 0),
  );
  const moduleNodes = memoryBlocks.flatMap((memoryBlock) => memoryBlock);
  const recurrentModule = createRecurrentModuleDescriptor(
    network,
    'narx-memory',
    roleNodes,
    moduleNodes,
    collectBoundaryConnectionInnovations(network, moduleNodes),
    moduleLabel,
  );

  return createTemporalDescriptorSet(recurrentModule);
}

/**
 * Append one or more temporal descriptors to the runtime extension bag.
 *
 * @param network Runtime network that should retain explicit temporal metadata.
 * @param descriptorSet Descriptor set to merge into the hydrated extension bag.
 * @returns Nothing.
 */
export function appendTemporalDescriptorSet(
  network: Network,
  descriptorSet: TemporalDescriptorSet | undefined,
): void {
  if (!descriptorSetHasContent(descriptorSet)) {
    return;
  }

  synchronizeTemporalDescriptorExtensions(network);

  const runtimeNetwork = network as RuntimeNetworkWithSerializedExtensions;
  const nextRecurrentModules = dedupeRecurrentModules([
    ...readRecurrentModules(runtimeNetwork._serializedExtensions),
    ...(descriptorSet?.recurrentModules ?? []),
  ]).filter((descriptor) =>
    isRecurrentModuleDescriptorValidOnRuntimeNetwork(network, descriptor),
  );
  const nextGatedBlocks = dedupeGatedBlocks([
    ...readGatedBlocks(runtimeNetwork._serializedExtensions),
    ...(descriptorSet?.gatedBlocks ?? []),
  ]).filter((descriptor) =>
    isGatedBlockDescriptorValidOnRuntimeNetwork(network, descriptor),
  );

  writeTemporalDescriptorExtensions(
    runtimeNetwork,
    nextRecurrentModules,
    nextGatedBlocks,
  );
}

/**
 * Remove stale temporal descriptors that no longer match the runtime graph.
 *
 * Disabled connections still count as historically present genes for the Step
 * 7.4 lane. Synchronization therefore retires descriptors only when their
 * referenced nodes, connection innovations, or gating ownership disappear from
 * the runtime graph, not when one of those genes is merely toggled inactive.
 *
 * @param network Runtime network whose hydrated extension bag should be normalized.
 * @returns Nothing.
 */
export function synchronizeTemporalDescriptorExtensions(network: Network): void {
  const runtimeNetwork = network as RuntimeNetworkWithSerializedExtensions;
  if (!runtimeNetwork._serializedExtensions) {
    return;
  }

  const nextRecurrentModules = readRecurrentModules(
    runtimeNetwork._serializedExtensions,
  ).filter((descriptor) =>
    isRecurrentModuleDescriptorValidOnRuntimeNetwork(network, descriptor),
  );
  const nextGatedBlocks = readGatedBlocks(
    runtimeNetwork._serializedExtensions,
  ).filter((descriptor) =>
    isGatedBlockDescriptorValidOnRuntimeNetwork(network, descriptor),
  );

  writeTemporalDescriptorExtensions(
    runtimeNetwork,
    nextRecurrentModules,
    nextGatedBlocks,
  );
}

/**
 * Collect gene ids currently owned by validated recurrent module descriptors.
 *
 * Mutation-repair helpers sometimes need to distinguish ordinary hidden nodes
 * from hidden nodes that are internal parts of one explicit recurrent module.
 * Those module-owned nodes can look locally stranded even when the module as a
 * whole is valid, so repair code should consult this helper before rewiring
 * them like generic hidden neurons.
 *
 * @param network Runtime network whose temporal descriptor ownership should be read.
 * @returns Gene-id set for hidden nodes protected by live recurrent descriptors.
 */
export function resolveTemporalRecurrentModuleNodeGeneIds(
  network: Network,
): Set<number> {
  synchronizeTemporalDescriptorExtensions(network);

  const runtimeNetwork = network as RuntimeNetworkWithSerializedExtensions;
  return new Set(
    readRecurrentModules(runtimeNetwork._serializedExtensions)
      .flatMap((recurrentModule) =>
        Object.values(recurrentModule.nodeGeneIdsByRole),
      )
      .flat()
      .filter((geneId): geneId is number => Number.isFinite(geneId)),
  );
}

/**
 * Describe the validated temporal structure currently attached to a runtime network.
 *
 * This accessor is the public read seam for recurrent-aware diagnostics and
 * visualization work. It synchronizes the hydrated extension bag against the
 * live graph first, then returns a cloned snapshot so consumers never need to
 * inspect private runtime properties directly.
 *
 * @param network Runtime network whose temporal structure should be described.
 * @returns Read-only recurrent-module and gated-block snapshot.
 */
export function describeTemporalStructure(
  network: Network,
): NetworkTemporalStructureDescriptor {
  synchronizeTemporalDescriptorExtensions(network);

  const runtimeNetwork = network as RuntimeNetworkWithSerializedExtensions;
  return {
    recurrentModules: readRecurrentModules(runtimeNetwork._serializedExtensions).map(
      (recurrentModule) => ({
        ...recurrentModule,
        ...(resolveDerivedModuleLabel(recurrentModule)
          ? { moduleLabel: resolveDerivedModuleLabel(recurrentModule) }
          : {}),
      }),
    ),
    gatedBlocks: readGatedBlocks(runtimeNetwork._serializedExtensions),
  };
}

/**
 * Preserve parent temporal descriptors that remain structurally valid on one offspring.
 *
 * @param offspring Offspring runtime network produced by crossover.
 * @param parents Parent runtime networks that may carry temporal descriptors.
 * @returns Nothing.
 */
export function inheritTemporalDescriptorExtensions(
  offspring: Network,
  parents: readonly Network[],
): void {
  const inheritedRecurrentModules: TemporalRecurrentModuleDescriptor[] = [];
  const inheritedGatedBlocks: TemporalGatedBlockDescriptor[] = [];

  for (const parent of parents) {
    synchronizeTemporalDescriptorExtensions(parent);

    const runtimeParent = parent as RuntimeNetworkWithSerializedExtensions;
    inheritedRecurrentModules.push(
      ...readRecurrentModules(runtimeParent._serializedExtensions),
    );
    inheritedGatedBlocks.push(
      ...readGatedBlocks(runtimeParent._serializedExtensions),
    );
  }

  const runtimeOffspring = offspring as RuntimeNetworkWithSerializedExtensions;
  const nextRecurrentModules = dedupeRecurrentModules(
    inheritedRecurrentModules,
  ).filter((descriptor) =>
    isRecurrentModuleDescriptorValidOnRuntimeNetwork(offspring, descriptor),
  );
  const nextGatedBlocks = dedupeGatedBlocks(inheritedGatedBlocks).filter(
    (descriptor) =>
      isGatedBlockDescriptorValidOnRuntimeNetwork(offspring, descriptor),
  );

  writeTemporalDescriptorExtensions(
    runtimeOffspring,
    nextRecurrentModules,
    nextGatedBlocks,
  );
}

function descriptorSetHasContent(
  descriptorSet: TemporalDescriptorSet | undefined,
): boolean {
  return Boolean(
    descriptorSet?.recurrentModules?.length || descriptorSet?.gatedBlocks?.length,
  );
}

function createTemporalDescriptorSet(
  recurrentModule?: TemporalRecurrentModuleDescriptor,
  gatedBlock?: TemporalGatedBlockDescriptor,
): TemporalDescriptorSet | undefined {
  if (!recurrentModule) {
    return undefined;
  }

  return {
    recurrentModules: [recurrentModule],
    ...(gatedBlock ? { gatedBlocks: [gatedBlock] } : {}),
  };
}

function createRecurrentModuleDescriptor(
  network: Network,
  kind: TemporalRecurrentModuleKind,
  roleNodes: Record<string, readonly Node[]>,
  moduleNodes: readonly Node[],
  additionalConnectionInnovations: readonly number[],
  moduleLabel?: string,
): TemporalRecurrentModuleDescriptor | undefined {
  const nodeGeneIdsByRole = Object.fromEntries(
    Object.entries(roleNodes)
      .map(([roleName, roleNodeEntries]) => [
        roleName,
        collectFiniteNodeGeneIds(roleNodeEntries),
      ])
      .filter(([, nodeGeneIds]) => nodeGeneIds.length > 0),
  );
  const connectionInnovations = mergeConnectionInnovationSets(
    collectInternalConnectionInnovations(network, moduleNodes),
    additionalConnectionInnovations,
  );

  if (
    !SUPPORTED_RECURRENT_MODULE_KINDS.has(kind) ||
    Object.keys(nodeGeneIdsByRole).length === 0 ||
    connectionInnovations.length === 0
  ) {
    return undefined;
  }

  return {
    moduleId: createDescriptorId('module', kind, moduleNodes, moduleLabel),
    kind,
    nodeGeneIdsByRole,
    connectionInnovations,
  };
}

function createGatedBlockDescriptor(
  network: Network,
  kind: Exclude<TemporalRecurrentModuleKind, 'narx-memory'>,
  gaterNodes: readonly Node[],
): TemporalGatedBlockDescriptor | undefined {
  const gaterNodeSet = new Set(gaterNodes);
  const gatedConnections = collectRegisteredConnections(network).filter(
    (connection) =>
      connection.gater != null &&
      gaterNodeSet.has(connection.gater) &&
      Number.isFinite(connection.innovation),
  );
  const gaterGeneIds = [...new Set(
    gatedConnections
      .map((connection) => connection.gater?.geneId)
      .filter((geneId): geneId is number => Number.isFinite(geneId)),
  )].toSorted((leftGeneId, rightGeneId) => leftGeneId - rightGeneId);
  const connectionInnovations = gatedConnections
    .map((connection) => connection.innovation)
    .filter((innovation): innovation is number => Number.isFinite(innovation))
    .toSorted((leftInnovation, rightInnovation) => leftInnovation - rightInnovation);

  if (gaterGeneIds.length === 0 || connectionInnovations.length === 0) {
    return undefined;
  }

  return {
    blockId: createDescriptorId('gated:block', kind, gaterNodes),
    gaterGeneIds,
    connectionInnovations,
  };
}

function collectFiniteNodeGeneIds(nodes: readonly Node[]): number[] {
  return [...new Set(
    nodes
      .map((node) => node.geneId)
      .filter((geneId): geneId is number => Number.isFinite(geneId)),
  )].toSorted((leftGeneId, rightGeneId) => leftGeneId - rightGeneId);
}

function collectInternalConnectionInnovations(
  network: Network,
  moduleNodes: readonly Node[],
): number[] {
  const moduleNodeSet = new Set(moduleNodes);

  return [...new Set(
    collectRegisteredConnections(network)
      .filter(
        (connection) =>
          moduleNodeSet.has(connection.from) && moduleNodeSet.has(connection.to),
      )
      .map((connection) => connection.innovation)
      .filter((innovation): innovation is number => Number.isFinite(innovation)),
  )].toSorted((leftInnovation, rightInnovation) => leftInnovation - rightInnovation);
}

function collectGatedConnectionInnovations(
  network: Network,
  gaterNodes: readonly Node[],
): number[] {
  const gaterNodeSet = new Set(gaterNodes);

  return [...new Set(
    collectRegisteredConnections(network)
      .filter(
        (connection) =>
          connection.gater != null && gaterNodeSet.has(connection.gater),
      )
      .map((connection) => connection.innovation)
      .filter((innovation): innovation is number => Number.isFinite(innovation)),
  )].toSorted((leftInnovation, rightInnovation) => leftInnovation - rightInnovation);
}

function collectBoundaryConnectionInnovations(
  network: Network,
  moduleNodes: readonly Node[],
): number[] {
  const moduleNodeSet = new Set(moduleNodes);

  return [...new Set(
    collectRegisteredConnections(network)
      .filter(
        (connection) =>
          moduleNodeSet.has(connection.from) || moduleNodeSet.has(connection.to),
      )
      .map((connection) => connection.innovation)
      .filter((innovation): innovation is number => Number.isFinite(innovation)),
  )].toSorted((leftInnovation, rightInnovation) => leftInnovation - rightInnovation);
}

function mergeConnectionInnovationSets(
  ...innovationSets: ReadonlyArray<readonly number[]>
): number[] {
  return [...new Set(
    innovationSets
      .flatMap((innovationSet) => innovationSet)
      .filter((innovation): innovation is number => Number.isFinite(innovation)),
  )].toSorted((leftInnovation, rightInnovation) => leftInnovation - rightInnovation);
}

/**
 * Collect every registered runtime connection, including disabled genes.
 *
 * Temporal descriptors treat disabled edges as dormant structure rather than
 * deletion, so descriptor validation must index every connection that still
 * belongs to the runtime graph.
 *
 * @param network Runtime network whose registered connections should be read.
 * @returns Registered connections across forward and self-edge shelves.
 */
function collectRegisteredConnections(network: Network): Connection[] {
  const runtimeNetwork = network as RuntimeNetworkWithSerializedExtensions & {
    nodes: Array<{
      connections: {
        out: Connection[];
        self: Connection[];
      };
    }>;
  };
  const seenConnections = new Set<Connection>();
  const registeredConnections: Connection[] = [];

  runtimeNetwork.nodes.forEach((node) => {
    [...node.connections.out, ...node.connections.self].forEach((connection) => {
      if (seenConnections.has(connection)) {
        return;
      }

      seenConnections.add(connection);
      registeredConnections.push(connection);
    });
  });

  return registeredConnections;
}

function createDescriptorId(
  prefix: string,
  kind: TemporalRecurrentModuleKind | Exclude<TemporalRecurrentModuleKind, 'narx-memory'>,
  nodes: readonly Node[],
  label?: string,
): string {
  const identitySegment = String(resolveDescriptorIdentity(nodes));
  return typeof label === 'string' && label.length > 0
    ? `${prefix}:${kind}:${label}:${identitySegment}`
    : `${prefix}:${kind}:${identitySegment}`;
}

function resolveDescriptorIdentity(nodes: readonly Node[]): number {
  const nodeGeneIds = collectFiniteNodeGeneIds(nodes);
  return nodeGeneIds[0]!;
}

function resolveDerivedModuleLabel(
  recurrentModule: TemporalRecurrentModuleDescriptor,
): string | undefined {
  if (recurrentModule.kind !== 'narx-memory') {
    return undefined;
  }

  const moduleIdSegments = recurrentModule.moduleId.split(':');
  const derivedModuleLabel = moduleIdSegments.at(-2);
  return typeof derivedModuleLabel === 'string' && derivedModuleLabel.length > 0
    ? derivedModuleLabel
    : undefined;
}

function dedupeRecurrentModules(
  recurrentModules: readonly TemporalRecurrentModuleDescriptor[],
): TemporalRecurrentModuleDescriptor[] {
  return Array.from(
    new Map(
      recurrentModules.map((recurrentModule) => [
        recurrentModule.moduleId,
        recurrentModule,
      ]),
    ).values(),
  ).toSorted((leftModule, rightModule) =>
    leftModule.moduleId.localeCompare(rightModule.moduleId),
  );
}

function dedupeGatedBlocks(
  gatedBlocks: readonly TemporalGatedBlockDescriptor[],
): TemporalGatedBlockDescriptor[] {
  return Array.from(
    new Map(gatedBlocks.map((gatedBlock) => [gatedBlock.blockId, gatedBlock])).values(),
  ).toSorted((leftBlock, rightBlock) =>
    leftBlock.blockId.localeCompare(rightBlock.blockId),
  );
}

function readRecurrentModules(
  extensions: NetworkJSONExtensions | undefined,
): TemporalRecurrentModuleDescriptor[] {
  const recurrentModules = (extensions?.values as TemporalDescriptorSet | undefined)
    ?.recurrentModules;
  return Array.isArray(recurrentModules)
    ? recurrentModules.map((recurrentModule) => structuredClone(recurrentModule))
    : [];
}

function readGatedBlocks(
  extensions: NetworkJSONExtensions | undefined,
): TemporalGatedBlockDescriptor[] {
  const gatedBlocks = (extensions?.values as TemporalDescriptorSet | undefined)
    ?.gatedBlocks;
  return Array.isArray(gatedBlocks)
    ? gatedBlocks.map((gatedBlock) => structuredClone(gatedBlock))
    : [];
}

function writeTemporalDescriptorExtensions(
  runtimeNetwork: RuntimeNetworkWithSerializedExtensions,
  recurrentModules: readonly TemporalRecurrentModuleDescriptor[],
  gatedBlocks: readonly TemporalGatedBlockDescriptor[],
): void {
  const nonTemporalValues = readNonTemporalValues(runtimeNetwork._serializedExtensions);
  const nextValues: Record<string, unknown> = {
    ...nonTemporalValues,
    ...(recurrentModules.length > 0
      ? { recurrentModules: structuredClone(recurrentModules) }
      : {}),
    ...(gatedBlocks.length > 0
      ? { gatedBlocks: structuredClone(gatedBlocks) }
      : {}),
  };

  if (Object.keys(nextValues).length === 0) {
    Reflect.deleteProperty(runtimeNetwork, '_serializedExtensions');
    return;
  }

  runtimeNetwork._serializedExtensions = {
    version: resolveExtensionVersion(runtimeNetwork._serializedExtensions),
    values: nextValues,
  };
}

function readNonTemporalValues(
  extensions: NetworkJSONExtensions | undefined,
): Record<string, unknown> {
  if (!isPlainObjectRecord(extensions?.values)) {
    return {};
  }

  return Object.fromEntries(
    Object.entries(extensions.values).filter(
      ([valueKey]) => valueKey !== 'recurrentModules' && valueKey !== 'gatedBlocks',
    ),
  );
}

function resolveExtensionVersion(
  _extensions: NetworkJSONExtensions | undefined,
): number {
  return TEMPORAL_EXTENSION_VERSION;
}

function isRecurrentModuleDescriptorValidOnRuntimeNetwork(
  network: Network,
  recurrentModule: TemporalRecurrentModuleDescriptor,
): boolean {
  if (
    typeof recurrentModule?.moduleId !== 'string' ||
    recurrentModule.moduleId.length === 0 ||
    !SUPPORTED_RECURRENT_MODULE_KINDS.has(recurrentModule.kind)
  ) {
    return false;
  }

  if (!isPlainObjectRecord(recurrentModule.nodeGeneIdsByRole)) {
    return false;
  }

  const liveGeneIds = new Set(collectFiniteNodeGeneIds((network as RuntimeNetworkWithSerializedExtensions).nodes));
  const liveConnectionsByInnovation = createConnectionLookupByInnovation(network);
  const roleNodeGeneIdGroups = Object.values(recurrentModule.nodeGeneIdsByRole);

  if (
    roleNodeGeneIdGroups.length === 0 ||
    roleNodeGeneIdGroups.some(
      (roleNodeGeneIds) =>
        !Array.isArray(roleNodeGeneIds) ||
        roleNodeGeneIds.length === 0 ||
        roleNodeGeneIds.some(
          (geneId) => !Number.isFinite(geneId) || !liveGeneIds.has(geneId),
        ),
    )
  ) {
    return false;
  }

  return isKnownNonEmptyNumberArray(
    recurrentModule.connectionInnovations,
    liveConnectionsByInnovation,
  );
}

function isGatedBlockDescriptorValidOnRuntimeNetwork(
  network: Network,
  gatedBlock: TemporalGatedBlockDescriptor,
): boolean {
  if (typeof gatedBlock?.blockId !== 'string' || gatedBlock.blockId.length === 0) {
    return false;
  }

  const liveGeneIds = new Set(collectFiniteNodeGeneIds((network as RuntimeNetworkWithSerializedExtensions).nodes));
  const liveConnectionsByInnovation = createConnectionLookupByInnovation(network);

  if (
    !Array.isArray(gatedBlock.gaterGeneIds) ||
    gatedBlock.gaterGeneIds.length === 0 ||
    gatedBlock.gaterGeneIds.some(
      (geneId) => !Number.isFinite(geneId) || !liveGeneIds.has(geneId),
    )
  ) {
    return false;
  }

  if (!isKnownNonEmptyNumberArray(gatedBlock.connectionInnovations, liveConnectionsByInnovation)) {
    return false;
  }

  return gatedBlock.connectionInnovations.every((connectionInnovation) => {
    const liveConnection = liveConnectionsByInnovation.get(connectionInnovation);
    return (
      typeof liveConnection?.gater?.geneId === 'number' &&
      gatedBlock.gaterGeneIds.includes(liveConnection.gater.geneId)
    );
  });
}

function createConnectionLookupByInnovation(
  network: Network,
): Map<number, Connection> {
  return new Map(
    collectRegisteredConnections(network)
      .map((connection) => [connection.innovation, connection] as const)
      .filter(([innovation]) => Number.isFinite(innovation)),
  );
}

function isKnownNonEmptyNumberArray<T>(
  values: unknown,
  lookup: Map<number, T>,
): values is number[] {
  return (
    Array.isArray(values) &&
    values.length > 0 &&
    values.every(
      (value) => Number.isFinite(value) && lookup.has(value as number),
    )
  );
}

function isPlainObjectRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}