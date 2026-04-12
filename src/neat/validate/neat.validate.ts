import type Connection from '../../architecture/connection/connection';
import type Network from '../../architecture/network/network';
import type { NetworkTopologyIntent } from '../../architecture/network/network.types';
import type Node from '../../architecture/node/node';
import { GENOME_CACHE_FIELD_KEYS } from '../cache/cache';
import {
  assertValidGenomeContract as assertValidGenomeContractImpl,
  validateGenomeContract as validateGenomeContractImpl,
} from '../genome/genome';
import type {
  NeatGenome,
  NeatGenomeValidationReport,
} from '../genome/genome';
import { NeatNativeGenomeValidationError } from './neat.validate.errors';
import type {
  NativeGenomeValidationIssue,
  NativeGenomeValidationReport,
} from './neat.validate.types';

/**
 * Native-genome validation for the proper-NEAT contract.
 *
 * This validator is the controller's "trust but verify" seam.
 *
 * Most internal algorithms assume a *native* genome has already committed to
 * explicit identity fields:
 *
 * - every node has a finite `geneId`
 * - every connection has a finite `innovation`
 * - endpoints and gaters resolve to runtime nodes
 * - topology intent agrees with the actual edge orientation
 *
 * When those invariants are violated, downstream failures can be extremely
 * indirect (compatibility walks, crossover materialization, speciation, etc.).
 * `validateNativeGenome()` pushes that failure back toward the write path.
 *
 * This is also an export/replay safety boundary: checkpoints should not
 * normalize malformed native state into a snapshot that appears portable.
 */

type RuntimeConnectionListName = 'connections' | 'selfconns';

type NativeGenomeRuntime = {
  nodes: Node[];
  connections: Connection[];
  selfconns: Connection[];
  gates: Connection[];
  _compatCache?: Array<[number, number]>;
  _outputCache?: unknown;
  _traceCache?: unknown;
  _enforceAcyclic?: boolean;
  _id?: number;
  getTopologyIntent: () => NetworkTopologyIntent;
};

interface NodeEntry {
  node: Node;
  path: string;
}

interface ConnectionEntry {
  connection: Connection;
  listName: RuntimeConnectionListName;
  listIndex: number;
  path: string;
}

/**
 * Validate one native NEAT genome before compatibility, speciation, or
 * crossover code consumes it.
 *
 * Native genomes are expected to carry explicit node gene ids, explicit
 * connection innovations, resolvable endpoint and gater references, and a
 * topology-intent contract that agrees with the low-level runtime flags and the
 * actual edge orientation in memory.
 *
 * Use this helper in dev and test flows when you want malformed native genomes
 * to fail near the write path that created them rather than much later inside a
 * compatibility walk or crossover materialization pass.
 *
 * @param genome Network-shaped native genome candidate to inspect.
 * @returns Structured report listing all discovered invariant violations.
 * @example
 * ```ts
 * const report = validateNativeGenome(genome);
 *
 * if (!report.isValid) {
 *   console.log(report.issues);
 * }
 * ```
 */
export function validateNativeGenome(
  genome: Network,
): NativeGenomeValidationReport {
  const runtimeGenome = genome as unknown as NativeGenomeRuntime;
  const topologyIntent = runtimeGenome.getTopologyIntent();
  const issues: NativeGenomeValidationIssue[] = [];
  const nodeEntries = createNodeEntries(runtimeGenome.nodes);
  const connectionEntries = createConnectionEntries(runtimeGenome);

  // Step 1: Validate stable node identity before inspecting connection history.
  validateNodeGeneIds(nodeEntries, issues);

  // Step 2: Validate connection identity, endpoints, and gater bookkeeping.
  validateConnections(runtimeGenome, connectionEntries, issues);

  // Step 3: Validate topology intent and runtime acyclic semantics.
  validateTopologyIntent(runtimeGenome, connectionEntries, topologyIntent, issues);

  // Step 4: Validate compatibility-cache coherence and stale derived-cache residue.
  validateCompatibilityCache(runtimeGenome, connectionEntries, issues);
  validateDerivedCaches(runtimeGenome, issues);

  // Step 5: Return one compact report for tests and tooling.
  return {
    isValid: issues.length === 0,
    genomeId: runtimeGenome._id,
    topologyIntent,
    nodeCount: runtimeGenome.nodes.length,
    connectionCount: connectionEntries.length,
    issues,
  };
}

/**
 * Assert that a native NEAT genome satisfies the proper-NEAT validation rules.
 *
 * This is the fail-fast convenience wrapper for dev/test callers that prefer an
 * exception over manually checking `report.isValid`.
 *
 * @param genome Network-shaped native genome candidate.
 * @returns Nothing.
 * @throws {NeatNativeGenomeValidationError} When the genome violates at least
 * one native-genome invariant.
 * @example
 * ```ts
 * assertValidNativeGenome(genome);
 * ```
 */
export function assertValidNativeGenome(genome: Network): void {
  const validationReport = validateNativeGenome(genome);
  if (validationReport.isValid) {
    return;
  }

  throw new NeatNativeGenomeValidationError(
    buildValidationFailureMessage(validationReport),
    validationReport.issues,
  );
}

/**
 * Validate one strict genome contract without requiring a live `Network`
 * phenotype.
 *
 * @param genome - Strict structural genome contract.
 * @returns Structured genome validation report.
 */
export function validateGenomeContract(
  genome: NeatGenome,
): NeatGenomeValidationReport {
  return validateGenomeContractImpl(genome);
}

/**
 * Assert that one strict genome contract satisfies the Step 7.1 structural
 * identity rules.
 *
 * @param genome - Strict structural genome contract.
 * @returns Nothing.
 */
export function assertValidGenomeContract(genome: NeatGenome): void {
  assertValidGenomeContractImpl(genome);
}

function createNodeEntries(nodes: Node[]): NodeEntry[] {
  return nodes.map((node, nodeIndex) => ({
    node,
    path: `nodes[${nodeIndex}]`,
  }));
}

function createConnectionEntries(
  runtimeGenome: NativeGenomeRuntime,
): ConnectionEntry[] {
  const forwardConnections: ConnectionEntry[] = runtimeGenome.connections.map(
    (connection, listIndex) => ({
      connection,
      listName: 'connections' as const,
      listIndex,
      path: `connections[${listIndex}]`,
    }),
  );
  const selfConnections: ConnectionEntry[] = runtimeGenome.selfconns.map(
    (connection, listIndex) => ({
      connection,
      listName: 'selfconns' as const,
      listIndex,
      path: `selfconns[${listIndex}]`,
    }),
  );

  return [...forwardConnections, ...selfConnections];
}

function validateNodeGeneIds(
  nodeEntries: NodeEntry[],
  issues: NativeGenomeValidationIssue[],
): void {
  const nodePathsByGeneId = new Map<number, string>();

  for (const nodeEntry of nodeEntries) {
    const geneId = nodeEntry.node.geneId;

    if (!Number.isFinite(geneId)) {
      issues.push(
        createIssue(
          'missing-node-gene-id',
          `${nodeEntry.path}.geneId`,
          'Native genomes must assign a finite geneId to every runtime node.',
        ),
      );
      continue;
    }

    const firstPath = nodePathsByGeneId.get(geneId);
    if (firstPath) {
      issues.push(
        createIssue(
          'duplicate-node-gene-id',
          `${nodeEntry.path}.geneId`,
          'Node gene ids must stay unique across one native genome.',
          {
            duplicateGeneId: geneId,
            firstPath,
          },
        ),
      );
      continue;
    }

    nodePathsByGeneId.set(geneId, `${nodeEntry.path}.geneId`);
  }
}

function validateConnections(
  runtimeGenome: NativeGenomeRuntime,
  connectionEntries: ConnectionEntry[],
  issues: NativeGenomeValidationIssue[],
): void {
  const innovationPaths = new Map<number, string>();
  const allConnections = new Set(connectionEntries.map(({ connection }) => connection));
  const gatedConnections = new Set(runtimeGenome.gates);

  for (const connectionEntry of connectionEntries) {
    validateConnectionInnovation(
      connectionEntry,
      innovationPaths,
      issues,
    );
    validateResolvedNode(
      runtimeGenome,
      connectionEntry.path,
      connectionEntry.connection.from,
      'from',
      issues,
    );
    validateResolvedNode(
      runtimeGenome,
      connectionEntry.path,
      connectionEntry.connection.to,
      'to',
      issues,
    );
    validateGaterResolution(
      runtimeGenome,
      connectionEntry,
      gatedConnections,
      issues,
    );
  }

  for (let gateIndex = 0; gateIndex < runtimeGenome.gates.length; gateIndex++) {
    const gatedConnection = runtimeGenome.gates[gateIndex];
    if (allConnections.has(gatedConnection) && gatedConnection.gater) {
      continue;
    }

    issues.push(
      createIssue(
        'gated-connection-registration-mismatch',
        `gates[${gateIndex}]`,
        'Every registered gated connection must also exist in the runtime edge lists and resolve a gater node.',
      ),
    );
  }
}

function validateConnectionInnovation(
  connectionEntry: ConnectionEntry,
  innovationPaths: Map<number, string>,
  issues: NativeGenomeValidationIssue[],
): void {
  const innovation = connectionEntry.connection.innovation;
  const innovationPath = `${connectionEntry.path}.innovation`;

  if (!Number.isFinite(innovation)) {
    issues.push(
      createIssue(
        'missing-connection-innovation',
        innovationPath,
        'Native genomes must assign an explicit finite innovation number to every connection gene.',
      ),
    );
    return;
  }

  const firstPath = innovationPaths.get(innovation);
  if (firstPath) {
    issues.push(
      createIssue(
        'duplicate-connection-innovation',
        innovationPath,
        'Connection innovations must stay unique across one native genome.',
        {
          duplicateInnovation: innovation,
          firstPath,
        },
      ),
    );
    return;
  }

  innovationPaths.set(innovation, innovationPath);
}

function validateResolvedNode(
  runtimeGenome: NativeGenomeRuntime,
  connectionPath: string,
  node: Node,
  endpointLabel: 'from' | 'to' | 'gater',
  issues: NativeGenomeValidationIssue[],
): number | undefined {
  if (!node || typeof node.index !== 'number' || !Number.isInteger(node.index)) {
    issues.push(
      createIssue(
        endpointLabel === 'gater'
          ? 'gater-resolution-failed'
          : 'endpoint-resolution-failed',
        `${connectionPath}.${endpointLabel}`,
        `Connection ${endpointLabel} references must resolve to a runtime node with an integer index.`,
      ),
    );
    return undefined;
  }

  const runtimeNodePosition = runtimeGenome.nodes.indexOf(node);
  if (runtimeNodePosition === -1) {
    issues.push(
      createIssue(
        endpointLabel === 'gater'
          ? 'gater-resolution-failed'
          : 'endpoint-resolution-failed',
        `${connectionPath}.${endpointLabel}`,
        `Connection ${endpointLabel} references must point at a node that is still registered in the genome node array.`,
        {
          nodeIndex: node.index,
          nodeGeneId: node.geneId,
        },
      ),
    );
    return undefined;
  }

  return runtimeNodePosition;
}

function validateGaterResolution(
  runtimeGenome: NativeGenomeRuntime,
  connectionEntry: ConnectionEntry,
  gatedConnections: Set<Connection>,
  issues: NativeGenomeValidationIssue[],
): void {
  const { connection, path } = connectionEntry;
  if (!connection.gater) {
    return;
  }

  validateResolvedNode(runtimeGenome, path, connection.gater, 'gater', issues);

  if (gatedConnections.has(connection)) {
    return;
  }

  issues.push(
    createIssue(
      'gated-connection-registration-mismatch',
      `${path}.gater`,
      'Connections with an attached gater must also be registered in network.gates.',
    ),
  );
}

function validateTopologyIntent(
  runtimeGenome: NativeGenomeRuntime,
  connectionEntries: ConnectionEntry[],
  topologyIntent: NetworkTopologyIntent,
  issues: NativeGenomeValidationIssue[],
): void {
  const enforceAcyclic = runtimeGenome._enforceAcyclic;
  const expectsAcyclic = topologyIntent === 'feed-forward';

  if (expectsAcyclic !== !!enforceAcyclic) {
    issues.push(
      createIssue(
        'topology-intent-mismatch',
        '_enforceAcyclic',
        'The runtime acyclic flag must stay synchronized with the public topology intent contract.',
        {
          topologyIntent,
          enforceAcyclic: !!enforceAcyclic,
        },
      ),
    );
  }

  if (!expectsAcyclic) {
    return;
  }

  for (const connectionEntry of connectionEntries) {
    const fromIndex = validateResolvedNode(
      runtimeGenome,
      connectionEntry.path,
      connectionEntry.connection.from,
      'from',
      issues,
    );
    const toIndex = validateResolvedNode(
      runtimeGenome,
      connectionEntry.path,
      connectionEntry.connection.to,
      'to',
      issues,
    );

    if (
      typeof fromIndex === 'number' &&
      typeof toIndex === 'number' &&
      fromIndex >= toIndex
    ) {
      issues.push(
        createIssue(
          'feed-forward-recurrent-connection',
          connectionEntry.path,
          'Feed-forward native genomes must not contain self or backward edges.',
          {
            fromIndex,
            toIndex,
            listName: connectionEntry.listName,
            listIndex: connectionEntry.listIndex,
          },
        ),
      );
    }
  }
}

function validateCompatibilityCache(
  runtimeGenome: NativeGenomeRuntime,
  connectionEntries: ConnectionEntry[],
  issues: NativeGenomeValidationIssue[],
): void {
  const compatCache = runtimeGenome._compatCache;
  if (compatCache === undefined) {
    return;
  }

  if (!Array.isArray(compatCache) || !compatCache.every(isCompatCacheEntry)) {
    issues.push(
      createIssue(
        'compat-cache-mismatch',
        '_compatCache',
        'Compatibility cache entries must stay in `[innovation, weight]` tuple form.',
      ),
    );
    return;
  }

  const expectedPairs = connectionEntries
    .filter(({ connection }) => Number.isFinite(connection.innovation))
    .map(({ connection }) => [connection.innovation, connection.weight] as [number, number])
    .toSorted(([leftInnovation], [rightInnovation]) => leftInnovation - rightInnovation);

  const cacheMatches =
    compatCache.length === expectedPairs.length &&
    compatCache.every(
      ([cachedInnovation, cachedWeight], cacheIndex) =>
        cachedInnovation === expectedPairs[cacheIndex][0] &&
        Object.is(cachedWeight, expectedPairs[cacheIndex][1]),
    );

  if (cacheMatches) {
    return;
  }

  issues.push(
    createIssue(
      'compat-cache-mismatch',
      '_compatCache',
      'Compatibility cache contents must match the current sorted innovation-weight view of the genome.',
      {
        cacheLength: compatCache.length,
        expectedLength: expectedPairs.length,
      },
    ),
  );
}

function validateDerivedCaches(
  runtimeGenome: NativeGenomeRuntime,
  issues: NativeGenomeValidationIssue[],
): void {
  for (const cacheFieldKey of GENOME_CACHE_FIELD_KEYS) {
    if (cacheFieldKey === '_compatCache') {
      continue;
    }

    if (Object.prototype.hasOwnProperty.call(runtimeGenome, cacheFieldKey)) {
      issues.push(
        createIssue(
          'stale-derived-cache',
          cacheFieldKey,
          'Derived genome caches should be invalidated before native-genome validation or structural NEAT work continues.',
        ),
      );
    }
  }
}

function isCompatCacheEntry(value: unknown): value is [number, number] {
  return (
    Array.isArray(value) &&
    value.length === 2 &&
    Number.isFinite(value[0]) &&
    Number.isFinite(value[1])
  );
}

function buildValidationFailureMessage(
  validationReport: NativeGenomeValidationReport,
): string {
  const issuePreview = validationReport.issues
    .slice(0, 3)
    .map((issue) => `${issue.code} at ${issue.path}`)
    .join('; ');

  return `Native genome validation failed with ${validationReport.issues.length} issue(s): ${issuePreview}`;
}

function createIssue(
  code: NativeGenomeValidationIssue['code'],
  path: string,
  message: string,
  details?: NativeGenomeValidationIssue['details'],
): NativeGenomeValidationIssue {
  return {
    code,
    path,
    message,
    details,
  };
}

export type {
  NativeGenomeValidationIssue,
  NativeGenomeValidationIssueCode,
  NativeGenomeValidationReport,
} from './neat.validate.types';
export { NeatNativeGenomeValidationError } from './neat.validate.errors';