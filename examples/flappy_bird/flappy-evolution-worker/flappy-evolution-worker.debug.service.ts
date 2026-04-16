import type Network from '../../../src/architecture/network';
import { createGenomeFromNetwork } from '../../../src/neat/genome/genome';
import type { ExampleArchitectureProfileId } from '../../architectureProfiles';
import { FLAPPY_ENABLE_RECURRENT_DEBUG_LOGS } from '../constants/constants';

const RECURRENT_ARCHITECTURE_PROFILE_IDS = new Set<ExampleArchitectureProfileId>([
  'narx',
  'gru',
  'lstm',
]);

interface RecurrentDebugMarkerOptions {
  architectureProfileId: ExampleArchitectureProfileId;
  phase: string;
  generation?: number;
  extra?: Record<string, unknown>;
}

interface RecurrentDebugPopulationSnapshotOptions {
  architectureProfileId: ExampleArchitectureProfileId;
  phase: string;
  generation?: number;
  population: Network[];
  extra?: Record<string, unknown>;
}

interface RecurrentDebugPlaybackReferenceOptions {
  architectureProfileId: ExampleArchitectureProfileId;
  phase: string;
  generation?: number;
  population: Network[];
  playbackNetworks: Network[];
  extra?: Record<string, unknown>;
}

type RecurrentDebugNetworkSummary = {
  populationIndex: number;
  nodeCount: number;
  forwardConnectionCount: number;
  selfConnectionCount: number;
  gateCount: number;
  maxInnovation: number;
  strictGenomeOk: boolean;
  strictGenomeError?: string;
  duplicateInnovations?: Array<{
    innovation: number;
    count: number;
    endpoints: string[];
  }>;
};

/**
 * Emits one compact recurrent debug marker when the selected profile is stateful.
 *
 * The logging pass is intentionally restricted to the recurrent profiles so the
 * browser console stays readable while we isolate where the live GRU, LSTM, and
 * NARX path first diverges from the passing helper-only tests.
 *
 * @param options - Marker phase, profile id, and optional debug metadata.
 * @returns Nothing.
 */
export function logRecurrentDebugMarker(
  options: RecurrentDebugMarkerOptions,
): void {
  if (!shouldLogRecurrentDebug(options.architectureProfileId)) {
    return;
  }

  postRecurrentDebugPayload({
    kind: 'marker',
    phase: options.phase,
    architectureProfileId: options.architectureProfileId,
    generation: options.generation,
    ...(options.extra ? { extra: options.extra } : {}),
  });
}

/**
 * Emits structural health summaries for the current recurrent population.
 *
 * Each network summary records whether the strict-genome adapter accepts the
 * live runtime phenotype and, when it does not, reports the first duplicate
 * innovation groups visible in JSON form. This is the fastest way to tell
 * whether corruption happens before playback, during playback, or after the
 * winner-clone handoff.
 *
 * @param options - Profile id, phase label, and population snapshot inputs.
 * @returns Nothing.
 */
export function logRecurrentPopulationSnapshot(
  options: RecurrentDebugPopulationSnapshotOptions,
): void {
  if (!shouldLogRecurrentDebug(options.architectureProfileId)) {
    return;
  }

  postRecurrentDebugPayload({
    kind: 'population-snapshot',
    phase: options.phase,
    architectureProfileId: options.architectureProfileId,
    generation: options.generation,
    populationSize: options.population.length,
    population: options.population.map((network, populationIndex) =>
      resolveRecurrentDebugNetworkSummary(network, populationIndex),
    ),
    ...(options.extra ? { extra: options.extra } : {}),
  });
}

/**
 * Emits whether playback birds are reusing the same network objects as the live population.
 *
 * This matters because the browser-only playback path is the largest remaining
 * difference between the failing live recurrent flow and the helper-level tests
 * that already pass.
 *
 * @param options - Profile id plus live-population and playback-network references.
 * @returns Nothing.
 */
export function logRecurrentPlaybackReferenceSnapshot(
  options: RecurrentDebugPlaybackReferenceOptions,
): void {
  if (!shouldLogRecurrentDebug(options.architectureProfileId)) {
    return;
  }

  const sharedReferenceCount = options.playbackNetworks.reduce(
    (matchedReferenceCount, playbackNetwork, populationIndex) =>
      matchedReferenceCount +
      Number(playbackNetwork === options.population[populationIndex]),
    0,
  );

  postRecurrentDebugPayload({
    kind: 'playback-reference-snapshot',
    phase: options.phase,
    architectureProfileId: options.architectureProfileId,
    generation: options.generation,
    sharedReferenceCount,
    playbackNetworkCount: options.playbackNetworks.length,
    populationSize: options.population.length,
    ...(options.extra ? { extra: options.extra } : {}),
  });
}

function shouldLogRecurrentDebug(
  architectureProfileId: ExampleArchitectureProfileId,
): boolean {
  return (
    FLAPPY_ENABLE_RECURRENT_DEBUG_LOGS &&
    RECURRENT_ARCHITECTURE_PROFILE_IDS.has(architectureProfileId)
  );
}

function resolveRecurrentDebugNetworkSummary(
  network: Network,
  populationIndex: number,
): RecurrentDebugNetworkSummary {
  const runtimeNetwork = network as Network & {
    nodes?: unknown[];
    connections?: Array<{ innovation?: number }>;
    selfconns?: Array<{ innovation?: number }>;
    gates?: unknown[];
  };
  const forwardConnections = Array.isArray(runtimeNetwork.connections)
    ? runtimeNetwork.connections
    : [];
  const selfConnections = Array.isArray(runtimeNetwork.selfconns)
    ? runtimeNetwork.selfconns
    : [];
  const allInnovationIds = forwardConnections
    .concat(selfConnections)
    .map((connectionReference) => connectionReference.innovation)
    .filter((innovationId): innovationId is number =>
      typeof innovationId === 'number' && Number.isFinite(innovationId),
    );

  try {
    createGenomeFromNetwork(network);

    return {
      populationIndex,
      nodeCount: Array.isArray(runtimeNetwork.nodes)
        ? runtimeNetwork.nodes.length
        : 0,
      forwardConnectionCount: forwardConnections.length,
      selfConnectionCount: selfConnections.length,
      gateCount: Array.isArray(runtimeNetwork.gates)
        ? runtimeNetwork.gates.length
        : 0,
      maxInnovation:
        allInnovationIds.length > 0 ? Math.max(...allInnovationIds) : -1,
      strictGenomeOk: true,
    };
  } catch (error) {
    return {
      populationIndex,
      nodeCount: Array.isArray(runtimeNetwork.nodes)
        ? runtimeNetwork.nodes.length
        : 0,
      forwardConnectionCount: forwardConnections.length,
      selfConnectionCount: selfConnections.length,
      gateCount: Array.isArray(runtimeNetwork.gates)
        ? runtimeNetwork.gates.length
        : 0,
      maxInnovation:
        allInnovationIds.length > 0 ? Math.max(...allInnovationIds) : -1,
      strictGenomeOk: false,
      strictGenomeError: resolveUnknownErrorMessage(error),
      duplicateInnovations: resolveDuplicateInnovationGroups(network),
    };
  }
}

function resolveDuplicateInnovationGroups(network: Network): Array<{
  innovation: number;
  count: number;
  endpoints: string[];
}> {
  const serializedNetwork = network.toJSON() as {
    connections?: Array<Record<string, unknown>>;
  };
  const connections = Array.isArray(serializedNetwork.connections)
    ? serializedNetwork.connections
    : [];
  const connectionsByInnovation = new Map<number, Array<Record<string, unknown>>>();

  for (const serializedConnection of connections) {
    const innovation = serializedConnection.innovation;
    if (typeof innovation !== 'number' || !Number.isFinite(innovation)) {
      continue;
    }

    const existingConnections = connectionsByInnovation.get(innovation) ?? [];
    existingConnections.push(serializedConnection);
    connectionsByInnovation.set(innovation, existingConnections);
  }

  return [...connectionsByInnovation.entries()]
    .filter(([, duplicatedConnections]) => duplicatedConnections.length > 1)
    .toSorted(
      ([leftInnovation], [rightInnovation]) => leftInnovation - rightInnovation,
    )
    .slice(0, 3)
    .map(([innovation, duplicatedConnections]) => ({
      innovation,
      count: duplicatedConnections.length,
      endpoints: duplicatedConnections.slice(0, 4).map((duplicatedConnection) =>
        `${String(duplicatedConnection.from)}->${String(duplicatedConnection.to)}${duplicatedConnection.gater !== undefined ? `@${String(duplicatedConnection.gater)}` : ''}`,
      ),
    }));
}

function resolveUnknownErrorMessage(error: unknown): string {
  return String((error as Error)?.message ?? error);
}

function postRecurrentDebugPayload(payload: Record<string, unknown>): void {
  try {
    console.log(`[flappy-recurrent-debug] ${JSON.stringify(payload)}`);
  } catch {
    // Ignore debug logging failures so instrumentation never changes runtime behavior.
  }
}