import type Network from '../../architecture/network/network';
import { validateNativeGenome } from '../validate/neat.validate';
import { NeatExportPopulationValidationError } from './neat.export.errors';
import {
  LEGACY_CHECKPOINT_FORMAT_VERSION,
  type GenomeControllerCarrier,
  type NeatMetaJSON,
  type NeatStateJSON,
} from './neat.export.types';

/**
 * Export checkpoint guard utilities.
 *
 * These helpers exist because the broader `Network.toJSON()` / `Network.fromJSON()`
 * surface is intentionally permissive: it can serialize and restore a network
 * even when some controller-owned identity or metadata is missing.
 *
 * A proper-NEAT checkpoint boundary is stricter.
 *
 * - Population snapshots must be validator-clean native genomes.
 * - Serialized nodes must carry explicit `geneId` fields.
 * - Serialized connections must carry explicit `innovation`, `fromGeneId`, and
 *   `toGeneId` fields (plus `gaterGeneId` when gated).
 *
 * If any of those are missing, import should fail fast rather than silently
 * normalizing into a "valid looking" but replay-divergent controller state.
 *
 * @example
 * ```ts
 * // During import, validate the payload before calling `Network.fromJSON()`.
 * assertSerializedGenomeCarriesCheckpointIdentity(payload, 0);
 * const genome = Network.fromJSON(payload);
 * ```
 */

/**
 * Assert that one live genome satisfies the native proper-NEAT contract.
 *
 * Export and import both depend on validator-clean native genomes so checkpoint
 * payloads do not normalize malformed controller state into a seemingly valid
 * snapshot.
 *
 * @param genome - Live genome being exported or rehydrated.
 * @param genomeIndex - Stable population index used in error messages.
 * @param operation - Current checkpoint operation for diagnostics.
 * @returns Nothing.
 * @throws {NeatExportPopulationValidationError} When native validation fails.
 */
export function assertCheckpointGenomeIsNative(
  genome: GenomeControllerCarrier,
  genomeIndex: number,
  operation: 'export' | 'import',
): void {
  const validationReport = validateNativeGenome(genome as unknown as Network);

  if (validationReport.isValid) {
    return;
  }

  const firstIssue = validationReport.issues[0];
  throw new NeatExportPopulationValidationError(
    `Cannot ${operation} population genome ${genomeIndex}: ${firstIssue?.message ?? 'native genome validation failed'}.`,
  );
}

/**
 * Assert that a serialized checkpoint payload carries explicit historical identity.
 *
 * `Network.fromJSON()` intentionally supports more permissive restore flows, so
 * the strict proper-NEAT checkpoint boundary must validate the serialized node
 * and connection identity fields before runtime rehydration can synthesize any
 * replacement structure.
 *
 * @param networkPayload - Raw network JSON payload from persistence.
 * @param genomeIndex - Stable population index used in diagnostics.
 * @returns Nothing.
 * @throws {NeatExportPopulationValidationError} When identity fields are missing.
 */
export function assertSerializedGenomeCarriesCheckpointIdentity(
  networkPayload: Record<string, unknown>,
  genomeIndex: number,
): void {
  const serializedNodes = readSerializedCheckpointEntries(networkPayload.nodes);
  const serializedConnections = readSerializedCheckpointEntries(
    networkPayload.connections,
  );

  assertSerializedNodesCarryCheckpointIdentity(serializedNodes, genomeIndex);
  assertSerializedConnectionsCarryCheckpointIdentity(
    serializedConnections,
    genomeIndex,
  );
}

function readSerializedCheckpointEntries(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function assertSerializedNodesCarryCheckpointIdentity(
  serializedNodes: unknown[],
  genomeIndex: number,
): void {
  for (const [nodeIndex, serializedNode] of serializedNodes.entries()) {
    assertSerializedNodeCarriesCheckpointIdentity(
      serializedNode,
      genomeIndex,
      nodeIndex,
    );
  }
}

function assertSerializedNodeCarriesCheckpointIdentity(
  serializedNode: unknown,
  genomeIndex: number,
  nodeIndex: number,
): void {
  if (
    !serializedNode ||
    typeof serializedNode !== 'object' ||
    Array.isArray(serializedNode) ||
    typeof (serializedNode as { geneId?: unknown }).geneId !== 'number'
  ) {
    throw new NeatExportPopulationValidationError(
      `Cannot import population genome ${genomeIndex}: serialized node ${nodeIndex} is missing an explicit geneId.`,
    );
  }
}

function assertSerializedConnectionsCarryCheckpointIdentity(
  serializedConnections: unknown[],
  genomeIndex: number,
): void {
  for (const [
    connectionIndex,
    serializedConnection,
  ] of serializedConnections.entries()) {
    const connectionIdentity = readSerializedConnectionIdentity(
      serializedConnection,
      genomeIndex,
      connectionIndex,
    );

    assertSerializedConnectionNumberField(
      connectionIdentity.innovation,
      genomeIndex,
      connectionIndex,
      'is missing an explicit innovation id.',
    );
    assertSerializedConnectionNumberField(
      connectionIdentity.fromGeneId,
      genomeIndex,
      connectionIndex,
      'is missing fromGeneId.',
    );
    assertSerializedConnectionNumberField(
      connectionIdentity.toGeneId,
      genomeIndex,
      connectionIndex,
      'is missing toGeneId.',
    );

    if (connectionIdentity.gater != null) {
      assertSerializedConnectionNumberField(
        connectionIdentity.gaterGeneId,
        genomeIndex,
        connectionIndex,
        'is missing gaterGeneId.',
      );
    }
  }
}

function readSerializedConnectionIdentity(
  serializedConnection: unknown,
  genomeIndex: number,
  connectionIndex: number,
): {
  innovation?: unknown;
  fromGeneId?: unknown;
  toGeneId?: unknown;
  gater?: unknown;
  gaterGeneId?: unknown;
} {
  if (
    !serializedConnection ||
    typeof serializedConnection !== 'object' ||
    Array.isArray(serializedConnection)
  ) {
    throw new NeatExportPopulationValidationError(
      `Cannot import population genome ${genomeIndex}: serialized connection ${connectionIndex} must be an object payload.`,
    );
  }

  return serializedConnection as {
    innovation?: unknown;
    fromGeneId?: unknown;
    toGeneId?: unknown;
    gater?: unknown;
    gaterGeneId?: unknown;
  };
}

function assertSerializedConnectionNumberField(
  value: unknown,
  genomeIndex: number,
  connectionIndex: number,
  messageSuffix: string,
): void {
  if (typeof value !== 'number') {
    throw new NeatExportPopulationValidationError(
      `Cannot import population genome ${genomeIndex}: serialized connection ${connectionIndex} ${messageSuffix}`,
    );
  }
}

/**
 * Compute the next genome-id floor implied by one population.
 *
 * Import paths use this after population restore so future offspring ids remain
 * above every stable genome id already present in memory.
 *
 * @param population - Restored controller population.
 * @returns Next safe genome id after the maximum observed id.
 */
export function findNextGenomeIdFloor(
  population: GenomeControllerCarrier[],
): number {
  const maxObservedGenomeId = population.reduce(
    (currentMaxGenomeId, genome) =>
      typeof genome._id === 'number'
        ? Math.max(currentMaxGenomeId, genome._id)
        : currentMaxGenomeId,
    0,
  );

  return maxObservedGenomeId + 1;
}

/**
 * Resolve the effective meta-checkpoint format version.
 *
 * Missing version tags are treated as the legacy pre-versioned format so the
 * export boundary can decide whether to restore or reject older payloads.
 *
 * @param neatJSON - Serialized controller meta payload.
 * @returns Effective meta format version.
 */
export function resolveMetaFormatVersion(neatJSON: NeatMetaJSON): number {
  return typeof neatJSON.formatVersion === 'number'
    ? neatJSON.formatVersion
    : LEGACY_CHECKPOINT_FORMAT_VERSION;
}

/**
 * Resolve the effective full-checkpoint format version.
 *
 * Missing version tags are treated as the legacy pre-versioned format so the
 * full restore path can branch cleanly between older payloads and the current
 * strict checkpoint contract.
 *
 * @param stateBundle - Serialized full-checkpoint payload.
 * @returns Effective full-checkpoint format version.
 */
export function resolveStateFormatVersion(stateBundle: NeatStateJSON): number {
  return typeof stateBundle.formatVersion === 'number'
    ? stateBundle.formatVersion
    : LEGACY_CHECKPOINT_FORMAT_VERSION;
}
