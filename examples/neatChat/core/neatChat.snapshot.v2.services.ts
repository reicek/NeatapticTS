import Network from '../../../src/architecture/network/network';
import {
  exportPortableInferencePayload,
  fromParameterVector,
  toParameterVector,
  type ParameterLayoutEntry,
  type ParameterVector,
  type PortableInferencePayload,
} from '../../../src/neataptic.ts';
import {
  NEATCHAT_REPLAY_BUFFER_MAX_EXCHANGES,
  NEATCHAT_SPECIAL_TOKENS,
} from './neatChat.constants';
import {
  NeatChatNonNegativeIntegerValidationError,
  NeatChatSnapshotShapeError,
  NeatChatSnapshotVersionError,
} from './neatChat.errors';
import { createNeatChatEpisodicMemoryBank } from './neatChat.memory.services';
import { buildNeatChatVocabulary } from './neatChat.session.services';
import { resolvePositiveInteger } from './neatChat.tokenization.utils';
import type {
  NeatChatExchangeRecord,
  NeatChatSession,
  NeatChatSessionSnapshotV2,
} from './neatChat.types';
import type { NeatChatEpisodicMemoryBank } from './neatChat.memory.types';

export type { NeatChatSessionSnapshotV2 };

type StructuredParameterVectorSnapshot = {
  readonly values: readonly number[];
  readonly layoutVersion: 1;
  readonly descriptorHash: string;
  readonly layoutEntries: readonly ParameterLayoutEntry[];
};

type NormalizedNeatChatSessionSnapshotV2 = Omit<
  NeatChatSessionSnapshotV2,
  'parameterVector'
> & {
  readonly parameterVector: StructuredParameterVectorSnapshot;
};

const SUPPORTED_PARAMETER_VECTOR_LAYOUT_VERSION = 1;
const INITIAL_FNV1A_HASH = 2_166_136_261;
const FNV1A_PRIME = 16_777_619;

/**
 * Serializes a live NEATchat session into the v2 JSON-safe snapshot envelope.
 *
 * Version 2 keeps the v1 `networkJson` topology payload and adds a normalized
 * parameter-vector summary. The summary captures every trainable weight,
 * including the recurrent self-connection weights that encode LSTM gate states,
 * GRU reset and update dynamics, and NARX memory-tap coupling. This allows
 * `importNeatChatSessionV2` to rebuild the graph topology first and then replay
 * exact weights through `fromParameterVector(...)` for deterministic continued
 * training.
 *
 * @param session - Current live session to persist.
 * @returns JSON-safe v2 snapshot that can restore topology, vocabulary, and exact weights.
 *
 * @example
 * ```ts
 * // Export and persist a live session across browser reloads.
 * const snapshot = exportNeatChatSessionV2(session);
 * localStorage.setItem('neatchat-session', JSON.stringify(snapshot));
 *
 * // Restore in a new page load.
 * const raw = localStorage.getItem('neatchat-session');
 * const restoredSession = importNeatChatSessionV2(JSON.parse(raw));
 * ```
 */
export function exportNeatChatSessionV2(
  session: NeatChatSession,
): NeatChatSessionSnapshotV2 {
  // Step 1: Capture the live network in both topology and parameter-vector forms.
  const networkJson = structuredClone(session.network.toJSON());
  const parameterVectorSnapshot = createParameterVectorSnapshot(
    toParameterVector(session.network),
  );

  // Step 2: Fold session-owned vocabulary, counters, and exchanges into the snapshot.
  return buildNeatChatV2SnapshotFromSession(
    session,
    networkJson,
    parameterVectorSnapshot,
  );
}

/**
 * Restores a live NEATchat session from a v2 snapshot bundle.
 *
 * The import path is intentionally two-phase: it first rebuilds the vocabulary
 * and network topology shell from `networkJson`, then replays the validated
 * parameter vector through `fromParameterVector(...)`. The two-phase approach
 * ensures that recurrent self-connection weights — LSTM gate states, GRU reset
 * and update coupling, and NARX memory taps — are restored to the exact scalar
 * values captured at export time, not re-initialized by the topology builder.
 *
 * The bundle is validated strictly before any runtime mutation. Shape and version
 * errors surface as typed exceptions so callers can distinguish version mismatches
 * from malformed payloads without inspecting raw properties.
 *
 * @param bundle - Parsed v2 snapshot payload created by `exportNeatChatSessionV2`.
 * @returns Restored live session with matching vocabulary, topology, and weights.
 * @throws {NeatChatSnapshotVersionError} When `bundle.formatVersion` is not `2`.
 * @throws {NeatChatSnapshotShapeError} When required payload fields are missing or malformed.
 * @throws {NeatChatPositiveIntegerValidationError} When a positive counter is zero or negative.
 * @throws {NeatChatNonNegativeIntegerValidationError} When a non-negative counter is negative.
 *
 * @example
 * ```ts
 * // Restore from a persisted JSON string, catching version and shape errors.
 * const raw = localStorage.getItem('neatchat-session');
 * try {
 *   const restoredSession = importNeatChatSessionV2(JSON.parse(raw));
 *   console.log('Restored', restoredSession.learnedExchangeCount, 'exchanges.');
 * } catch (err) {
 *   if (err instanceof NeatChatSnapshotVersionError) {
 *     console.warn('Unsupported snapshot version — start a new session.');
 *   } else if (err instanceof NeatChatSnapshotShapeError) {
 *     console.error('Snapshot payload is malformed:', err.message);
 *   }
 * }
 * ```
 */
export function importNeatChatSessionV2(
  bundle: NeatChatSessionSnapshotV2,
): NeatChatSession;
export function importNeatChatSessionV2(bundle: unknown): NeatChatSession;
export function importNeatChatSessionV2(bundle: unknown): NeatChatSession {
  // Step 1: Validate and normalize the external bundle before any runtime mutation.
  const validatedBundle = validateNeatChatSessionSnapshotV2(bundle);

  // Step 2: Rebuild the vocabulary and topology-owned network shell.
  const restoredVocabulary = buildNeatChatVocabulary(
    validatedBundle.retainedTerms,
  );
  const restoredNetwork = Network.fromJSON(validatedBundle.networkJson);

  // Step 3: Replay exact scalar weights, then fold the restored session envelope.
  fromParameterVector(
    restoredNetwork,
    materializeParameterVector(validatedBundle.parameterVector),
  );

  return createRestoredNeatChatSession(
    validatedBundle,
    restoredVocabulary,
    restoredNetwork,
  );
}

/**
 * Exports the live session network as a portable worker-friendly inference payload.
 *
 * This keeps the NEATchat session boundary aligned with the public
 * `exportPortableInferencePayload(...)` seam without forcing callers to reach
 * through the session object manually. The returned payload is structured-clone
 * safe and can be transferred to a `Worker` via `postMessage`.
 *
 * When worker threads are unavailable (for example in a restricted browser
 * environment or during tests), callers can fall back to activating the live
 * session network directly. The payload format is identical in both paths, so
 * no output normalization is needed when switching between worker and
 * single-thread activation.
 *
 * @param session - Live session whose network should be exported for worker inference.
 * @returns Portable inference payload for deterministic worker-backed or single-thread activation.
 *
 * @example
 * ```ts
 * // Transfer the session network to a worker for background candidate scoring.
 * const payload = exportNeatChatPortablePayload(session);
 * worker.postMessage({ type: 'infer', payload }, []);
 * ```
 */
export function exportNeatChatPortablePayload(
  session: NeatChatSession,
): PortableInferencePayload {
  return exportPortableInferencePayload(session.network);
}

function buildNeatChatV2SnapshotFromSession(
  session: NeatChatSession,
  networkJson: Record<string, unknown>,
  parameterVectorSnapshot: StructuredParameterVectorSnapshot,
): NeatChatSessionSnapshotV2 {
  return {
    formatVersion: 2,
    retainedTerms: collectRetainedTerms(session),
    networkJson,
    parameterVector: parameterVectorSnapshot,
    exchanges: cloneExchangeRecords(session.exchanges),
    learnedExchangeCount: session.learnedExchangeCount,
    learnedTokenPairCount: session.learnedTokenPairCount,
    seededTokenPairCount: session.seededTokenPairCount,
    contextWindowTokenCount: session.contextWindowTokenCount,
    extensions: createSnapshotExtensions(session),
  };
}

function collectRetainedTerms(session: NeatChatSession): string[] {
  return session.vocabulary.indexToTerm.slice(NEATCHAT_SPECIAL_TOKENS.length);
}

function cloneExchangeRecords(
  exchangeRecords: readonly NeatChatExchangeRecord[],
): NeatChatExchangeRecord[] {
  return exchangeRecords.map((exchangeRecord) => ({
    userMessage: exchangeRecord.userMessage,
    response: exchangeRecord.response,
    trainedTokenPairCount: exchangeRecord.trainedTokenPairCount,
    userTokens: [...exchangeRecord.userTokens],
    responseTokens: [...exchangeRecord.responseTokens],
  }));
}

function createSnapshotExtensions(
  session: NeatChatSession,
): NeatChatSessionSnapshotV2['extensions'] {
  return {
    neatchat: {
      vocabularySize: session.vocabulary.size,
      memoryBank: cloneMemoryBank(session.memoryBank),
      candidateLog: cloneCandidateLogEntries(session.candidateLog),
      routingLog: cloneRoutingLogEntries(session.routingLog),
    },
  };
}

function cloneCandidateLogEntries(
  candidateLog: NeatChatSession['candidateLog'],
): NeatChatSession['candidateLog'] {
  return candidateLog.map((candidateLogEntry) => ({
    status: candidateLogEntry.status,
    trainedOnExchangeCount: candidateLogEntry.trainedOnExchangeCount,
    evaluationScores: { ...candidateLogEntry.evaluationScores },
    decidedAt: candidateLogEntry.decidedAt,
  }));
}

function cloneRoutingLogEntries(
  routingLog: NeatChatSession['routingLog'],
): NeatChatSession['routingLog'] {
  return structuredClone(routingLog);
}

function cloneMemoryBank(
  memoryBank: NeatChatEpisodicMemoryBank,
): NeatChatEpisodicMemoryBank {
  return {
    records: memoryBank.records.map((memoryRecord) => ({
      key: memoryRecord.key,
      value: memoryRecord.value,
      addedAt: memoryRecord.addedAt,
      hitCount: memoryRecord.hitCount,
    })),
    maxRecords: memoryBank.maxRecords,
    lastConsolidatedAt: memoryBank.lastConsolidatedAt ?? null,
  };
}

function createParameterVectorSnapshot(
  parameterVector: ParameterVector,
): StructuredParameterVectorSnapshot {
  const layoutEntries = parameterVector.layout.entries.map(
    cloneParameterLayoutEntry,
  );

  return {
    values: Array.from(parameterVector.values),
    layoutVersion: parameterVector.layout.version,
    descriptorHash: createParameterDescriptorHash(
      layoutEntries,
      parameterVector.layout.version,
    ),
    layoutEntries,
  };
}

function validateNeatChatSessionSnapshotV2(
  bundle: unknown,
): NormalizedNeatChatSessionSnapshotV2 {
  if (!isNonArrayRecord(bundle)) {
    throw new NeatChatSnapshotShapeError(
      'NEATchat session snapshot bundle must be an object.',
    );
  }

  const bundleRecord = bundle;

  if (bundleRecord.formatVersion !== 2) {
    throw new NeatChatSnapshotVersionError(
      `Unsupported NEATchat session snapshot version: ${String(bundleRecord.formatVersion)}.`,
    );
  }

  if (!Array.isArray(bundleRecord.retainedTerms)) {
    throw new NeatChatSnapshotShapeError(
      'NEATchat session snapshot retainedTerms must be an array.',
    );
  }

  if (!isNonArrayRecord(bundleRecord.networkJson)) {
    throw new NeatChatSnapshotShapeError(
      'NEATchat session snapshot networkJson must be an object.',
    );
  }

  if (!Array.isArray(bundleRecord.exchanges)) {
    throw new NeatChatSnapshotShapeError(
      'NEATchat session snapshot exchanges must be an array.',
    );
  }

  const retainedTerms = bundleRecord.retainedTerms;
  const exchanges = bundleRecord.exchanges;
  const networkJson = bundleRecord.networkJson;

  return {
    formatVersion: 2,
    retainedTerms: retainedTerms.map((term) => String(term)),
    networkJson: structuredClone(networkJson),
    parameterVector: normalizeParameterVectorSnapshot(
      bundleRecord.parameterVector,
    ),
    exchanges: normalizeExchangeRecords(exchanges),
    learnedExchangeCount: resolveNonNegativeInteger(
      resolveIntegerField(
        bundleRecord.learnedExchangeCount,
        'learnedExchangeCount',
      ),
      'learnedExchangeCount',
    ),
    learnedTokenPairCount: resolveNonNegativeInteger(
      resolveIntegerField(
        bundleRecord.learnedTokenPairCount,
        'learnedTokenPairCount',
      ),
      'learnedTokenPairCount',
    ),
    seededTokenPairCount: resolveNonNegativeInteger(
      resolveIntegerField(
        bundleRecord.seededTokenPairCount,
        'seededTokenPairCount',
      ),
      'seededTokenPairCount',
    ),
    contextWindowTokenCount: resolvePositiveInteger(
      resolveIntegerField(
        bundleRecord.contextWindowTokenCount,
        'contextWindowTokenCount',
      ),
      'contextWindowTokenCount',
    ),
    extensions: normalizeSnapshotExtensions(
      bundleRecord.extensions,
      retainedTerms.length,
    ),
  };
}

function normalizeExchangeRecords(
  exchangeRecords: readonly unknown[],
): NeatChatExchangeRecord[] {
  return exchangeRecords.map((exchangeRecord, exchangeIndex) =>
    normalizeExchangeRecord(exchangeRecord, exchangeIndex),
  );
}

function normalizeExchangeRecord(
  exchangeRecord: unknown,
  exchangeIndex: number,
): NeatChatExchangeRecord {
  if (!isNonArrayRecord(exchangeRecord)) {
    throw new NeatChatSnapshotShapeError(
      `NEATchat session snapshot exchanges[${exchangeIndex}] must be an object.`,
    );
  }

  if (!Array.isArray(exchangeRecord.userTokens)) {
    throw new NeatChatSnapshotShapeError(
      `NEATchat session snapshot exchanges[${exchangeIndex}].userTokens must be an array.`,
    );
  }

  if (!Array.isArray(exchangeRecord.responseTokens)) {
    throw new NeatChatSnapshotShapeError(
      `NEATchat session snapshot exchanges[${exchangeIndex}].responseTokens must be an array.`,
    );
  }

  return {
    userMessage: String(exchangeRecord.userMessage),
    response: String(exchangeRecord.response),
    trainedTokenPairCount: resolvePositiveInteger(
      resolveIntegerField(
        exchangeRecord.trainedTokenPairCount,
        `exchanges[${exchangeIndex}].trainedTokenPairCount`,
      ),
      'trainedTokenPairCount',
    ),
    userTokens: exchangeRecord.userTokens.map((token) => String(token)),
    responseTokens: exchangeRecord.responseTokens.map((token) => String(token)),
  };
}

function normalizeSnapshotExtensions(
  extensions: unknown,
  retainedTermCount: number,
): NeatChatSessionSnapshotV2['extensions'] {
  if (!isNonArrayRecord(extensions)) {
    throw new NeatChatSnapshotShapeError(
      'NEATchat session snapshot extensions must be an object.',
    );
  }

  const neatchatExtension = extensions.neatchat;

  if (!isNonArrayRecord(neatchatExtension)) {
    throw new NeatChatSnapshotShapeError(
      'NEATchat session snapshot extensions.neatchat must be an object.',
    );
  }

  const vocabularySize = resolvePositiveInteger(
    Number(neatchatExtension.vocabularySize),
    'extensions.neatchat.vocabularySize',
  );
  const memoryBank = normalizeOptionalMemoryBank(neatchatExtension.memoryBank);
  const candidateLog = normalizeOptionalCandidateLog(
    neatchatExtension.candidateLog,
  );
  const routingLog = normalizeOptionalRoutingLog(neatchatExtension.routingLog);
  const remainingNeatchatExtension = { ...neatchatExtension };
  Reflect.deleteProperty(remainingNeatchatExtension, 'memoryBank');
  Reflect.deleteProperty(remainingNeatchatExtension, 'candidateLog');
  Reflect.deleteProperty(remainingNeatchatExtension, 'routingLog');
  const expectedVocabularySize =
    retainedTermCount + NEATCHAT_SPECIAL_TOKENS.length;

  if (vocabularySize !== expectedVocabularySize) {
    throw new NeatChatSnapshotShapeError(
      `NEATchat session snapshot extensions.neatchat.vocabularySize must equal ${String(expectedVocabularySize)}.`,
    );
  }

  return {
    ...extensions,
    neatchat: {
      ...remainingNeatchatExtension,
      vocabularySize,
      ...(memoryBank == null ? {} : { memoryBank }),
      ...(candidateLog == null ? {} : { candidateLog }),
      ...(routingLog == null ? {} : { routingLog }),
    },
  };
}

function normalizeOptionalCandidateLog(
  candidateLog: unknown,
): NeatChatSessionSnapshotV2['extensions']['neatchat']['candidateLog'] {
  if (!Array.isArray(candidateLog)) {
    return undefined;
  }

  const normalizedCandidateLog = candidateLog.map(
    normalizeOptionalCandidateLogEntry,
  );

  if (
    normalizedCandidateLog.some(
      (candidateLogEntry) => candidateLogEntry == null,
    )
  ) {
    return undefined;
  }

  return normalizedCandidateLog.filter(isDefinedCandidateLogEntry);
}

function normalizeOptionalCandidateLogEntry(
  candidateLogEntry: unknown,
): NeatChatSession['candidateLog'][number] | undefined {
  if (!isNonArrayRecord(candidateLogEntry)) {
    return undefined;
  }

  const decidedAt = Number(candidateLogEntry.decidedAt);
  const trainedOnExchangeCount = Number(
    candidateLogEntry.trainedOnExchangeCount,
  );

  if (
    !isCandidateLogStatus(candidateLogEntry.status) ||
    !Number.isFinite(decidedAt) ||
    !Number.isInteger(trainedOnExchangeCount) ||
    trainedOnExchangeCount < 0 ||
    !isNumericRecord(candidateLogEntry.evaluationScores)
  ) {
    return undefined;
  }

  return {
    status: candidateLogEntry.status,
    trainedOnExchangeCount,
    evaluationScores: cloneNumericRecord(candidateLogEntry.evaluationScores),
    decidedAt,
  };
}

function isDefinedCandidateLogEntry(
  candidateLogEntry: NeatChatSession['candidateLog'][number] | undefined,
): candidateLogEntry is NeatChatSession['candidateLog'][number] {
  return candidateLogEntry != null;
}

function normalizeOptionalRoutingLog(
  routingLog: unknown,
): NeatChatSessionSnapshotV2['extensions']['neatchat']['routingLog'] {
  if (!Array.isArray(routingLog)) {
    return undefined;
  }

  const normalizedRoutingLog = routingLog.map(normalizeOptionalRoutingLogEntry);

  if (normalizedRoutingLog.some((routingLogEntry) => routingLogEntry == null)) {
    return undefined;
  }

  return normalizedRoutingLog.filter(isDefinedRoutingLogEntry);
}

function normalizeOptionalRoutingLogEntry(
  routingLogEntry: unknown,
): NeatChatSession['routingLog'][number] | undefined {
  if (!isNonArrayRecord(routingLogEntry)) {
    return undefined;
  }

  const decidedAt = Number(routingLogEntry.decidedAt);
  const candidateCount = Number(routingLogEntry.candidateCount);

  if (
    !Number.isFinite(decidedAt) ||
    !Number.isInteger(candidateCount) ||
    candidateCount < 0 ||
    !Array.isArray(routingLogEntry.comparedCandidatePaths) ||
    !isRoutingPath(routingLogEntry.selectedPath)
  ) {
    return undefined;
  }

  const comparedCandidatePaths = routingLogEntry.comparedCandidatePaths.map(
    (routingPath) => (isRoutingPath(routingPath) ? routingPath : undefined),
  );

  if (
    comparedCandidatePaths.some((routingPath) => routingPath == null) ||
    candidateCount !== comparedCandidatePaths.length ||
    !isRoutingNumericRecord(routingLogEntry.scores)
  ) {
    return undefined;
  }

  const retrievedMemoryKeysByPath = normalizeOptionalRoutingStringArrayRecord(
    routingLogEntry.retrievedMemoryKeysByPath,
  );
  const retrievedMemoryCountByPath = normalizeOptionalRoutingNumericRecord(
    routingLogEntry.retrievedMemoryCountByPath,
  );
  const responsesByPath = normalizeOptionalRoutingStringRecord(
    routingLogEntry.responsesByPath,
  );

  if (
    (routingLogEntry.retrievedMemoryKeysByPath != null &&
      retrievedMemoryKeysByPath == null) ||
    (routingLogEntry.retrievedMemoryCountByPath != null &&
      retrievedMemoryCountByPath == null) ||
    (routingLogEntry.responsesByPath != null && responsesByPath == null)
  ) {
    return undefined;
  }

  return {
    decidedAt,
    selectedPath: routingLogEntry.selectedPath,
    comparedCandidatePaths: comparedCandidatePaths.filter(isDefinedRoutingPath),
    candidateCount,
    scores: cloneNumericRecord(routingLogEntry.scores),
    ...(retrievedMemoryKeysByPath == null ? {} : { retrievedMemoryKeysByPath }),
    ...(retrievedMemoryCountByPath == null
      ? {}
      : { retrievedMemoryCountByPath }),
    ...(responsesByPath == null ? {} : { responsesByPath }),
  };
}

function isDefinedRoutingLogEntry(
  routingLogEntry: NeatChatSession['routingLog'][number] | undefined,
): routingLogEntry is NeatChatSession['routingLog'][number] {
  return routingLogEntry != null;
}

function isDefinedRoutingPath(
  routingPath:
    NeatChatSession['routingLog'][number]['selectedPath'] | undefined,
): routingPath is NeatChatSession['routingLog'][number]['selectedPath'] {
  return routingPath != null;
}

function isRoutingPath(
  value: unknown,
): value is NeatChatSession['routingLog'][number]['selectedPath'] {
  return (
    value === 'base' ||
    value === 'personalized' ||
    value === 'retrieval-grounded'
  );
}

function isRoutingNumericRecord(
  value: unknown,
): value is Record<string, number> {
  if (!isNumericRecord(value)) {
    return false;
  }

  return Object.keys(value).every((recordKey) => isRoutingPath(recordKey));
}

function normalizeOptionalRoutingNumericRecord(
  value: unknown,
):
  | Partial<
      Record<NeatChatSession['routingLog'][number]['selectedPath'], number>
    >
  | undefined {
  if (!isRoutingNumericRecord(value)) {
    return undefined;
  }

  return Object.fromEntries(
    Object.entries(value).map(([recordKey, recordValue]) => [
      recordKey,
      Number(recordValue),
    ]),
  ) as Partial<
    Record<NeatChatSession['routingLog'][number]['selectedPath'], number>
  >;
}

function normalizeOptionalRoutingStringArrayRecord(
  value: unknown,
):
  | Partial<
      Record<
        NeatChatSession['routingLog'][number]['selectedPath'],
        readonly string[]
      >
    >
  | undefined {
  if (!isNonArrayRecord(value)) {
    return undefined;
  }

  const routingEntries = Object.entries(value);

  if (
    routingEntries.some(
      ([recordKey, recordValue]) =>
        !isRoutingPath(recordKey) ||
        !Array.isArray(recordValue) ||
        !recordValue.every((entryValue) => typeof entryValue === 'string'),
    )
  ) {
    return undefined;
  }

  return Object.fromEntries(
    routingEntries.map(([recordKey, recordValue]) => [
      recordKey,
      [...(recordValue as readonly string[])],
    ]),
  ) as Partial<
    Record<
      NeatChatSession['routingLog'][number]['selectedPath'],
      readonly string[]
    >
  >;
}

function normalizeOptionalRoutingStringRecord(
  value: unknown,
):
  | Partial<
      Record<NeatChatSession['routingLog'][number]['selectedPath'], string>
    >
  | undefined {
  if (!isNonArrayRecord(value)) {
    return undefined;
  }

  const routingEntries = Object.entries(value);

  if (
    routingEntries.some(
      ([recordKey, recordValue]) =>
        !isRoutingPath(recordKey) || typeof recordValue !== 'string',
    )
  ) {
    return undefined;
  }

  return Object.fromEntries(routingEntries) as Partial<
    Record<NeatChatSession['routingLog'][number]['selectedPath'], string>
  >;
}

function isCandidateLogStatus(
  value: unknown,
): value is NeatChatSession['candidateLog'][number]['status'] {
  return value === 'promoted' || value === 'rejected';
}

function isNumericRecord(value: unknown): value is Record<string, number> {
  if (!isNonArrayRecord(value)) {
    return false;
  }

  return Object.values(value).every((recordValue) =>
    Number.isFinite(Number(recordValue)),
  );
}

function cloneNumericRecord(
  record: Record<string, number>,
): Record<string, number> {
  return Object.fromEntries(
    Object.entries(record).map(([recordKey, recordValue]) => [
      recordKey,
      Number(recordValue),
    ]),
  );
}

function normalizeOptionalMemoryBank(
  memoryBank: unknown,
): NeatChatEpisodicMemoryBank | undefined {
  if (!isNonArrayRecord(memoryBank) || !Array.isArray(memoryBank.records)) {
    return undefined;
  }

  const maxRecords = Number(memoryBank.maxRecords);
  const lastConsolidatedAt = memoryBank.lastConsolidatedAt;

  if (!Number.isFinite(maxRecords)) {
    return undefined;
  }

  if (
    lastConsolidatedAt != null &&
    !Number.isFinite(Number(lastConsolidatedAt))
  ) {
    return undefined;
  }

  const normalizedRecords = memoryBank.records.map(
    normalizeOptionalMemoryRecord,
  );

  if (normalizedRecords.some((memoryRecord) => memoryRecord == null)) {
    return undefined;
  }

  return {
    records: normalizedRecords.filter(isDefinedMemoryRecord),
    maxRecords,
    lastConsolidatedAt:
      lastConsolidatedAt == null ? null : Number(lastConsolidatedAt),
  };
}

function normalizeOptionalMemoryRecord(
  memoryRecord: unknown,
): NeatChatEpisodicMemoryBank['records'][number] | undefined {
  if (!isNonArrayRecord(memoryRecord)) {
    return undefined;
  }

  const addedAt = Number(memoryRecord.addedAt);
  const hitCount = Number(memoryRecord.hitCount);

  if (!Number.isFinite(addedAt) || !Number.isFinite(hitCount)) {
    return undefined;
  }

  return {
    key: String(memoryRecord.key),
    value: String(memoryRecord.value),
    addedAt,
    hitCount,
  };
}

function isDefinedMemoryRecord(
  memoryRecord: NeatChatEpisodicMemoryBank['records'][number] | undefined,
): memoryRecord is NeatChatEpisodicMemoryBank['records'][number] {
  return memoryRecord != null;
}

function normalizeParameterVectorSnapshot(
  parameterVector: unknown,
): StructuredParameterVectorSnapshot {
  if (looksLikeRuntimeParameterVector(parameterVector)) {
    return normalizeRuntimeParameterVector(parameterVector);
  }

  if (!isNonArrayRecord(parameterVector)) {
    throw new NeatChatSnapshotShapeError(
      'NEATchat session snapshot parameterVector must be an object.',
    );
  }

  const layoutEntries = resolveParameterLayoutEntries(
    parameterVector.layoutEntries,
    'parameterVector.layoutEntries',
  );
  const layoutVersion = resolveParameterLayoutVersion(
    parameterVector.layoutVersion,
    'parameterVector.layoutVersion',
  );
  const values = resolveParameterValues(
    parameterVector.values,
    'parameterVector.values',
  );

  assertParameterVectorLengthMatchesLayout(values, layoutEntries);

  const descriptorHash = resolveDescriptorHash(
    parameterVector.descriptorHash,
    layoutEntries,
    layoutVersion,
  );

  return {
    values,
    layoutVersion,
    descriptorHash,
    layoutEntries,
  };
}

function looksLikeRuntimeParameterVector(
  parameterVector: unknown,
): parameterVector is ParameterVector {
  if (!isNonArrayRecord(parameterVector)) {
    return false;
  }

  const parameterVectorRecord = parameterVector;

  return (
    isNonArrayRecord(parameterVectorRecord.layout) &&
    Array.isArray(parameterVectorRecord.layout.entries) &&
    isParameterValueCollection(parameterVectorRecord.values)
  );
}

function normalizeRuntimeParameterVector(
  parameterVector: ParameterVector,
): StructuredParameterVectorSnapshot {
  const layoutEntries = resolveParameterLayoutEntries(
    parameterVector.layout.entries,
    'parameterVector.layout.entries',
  );
  const layoutVersion = resolveParameterLayoutVersion(
    parameterVector.layout.version,
    'parameterVector.layout.version',
  );
  const values = resolveParameterValues(
    parameterVector.values,
    'parameterVector.values',
  );

  assertParameterVectorLengthMatchesLayout(values, layoutEntries);

  return {
    values,
    layoutVersion,
    descriptorHash: createParameterDescriptorHash(layoutEntries, layoutVersion),
    layoutEntries,
  };
}

function resolveParameterLayoutEntries(
  layoutEntries: unknown,
  fieldName: string,
): ParameterLayoutEntry[] {
  if (!Array.isArray(layoutEntries)) {
    throw new NeatChatSnapshotShapeError(
      `NEATchat session snapshot ${fieldName} must be an array.`,
    );
  }

  return layoutEntries.map((layoutEntry, layoutIndex) =>
    resolveParameterLayoutEntry(layoutEntry, `${fieldName}[${layoutIndex}]`),
  );
}

function resolveParameterLayoutEntry(
  layoutEntry: unknown,
  fieldName: string,
): ParameterLayoutEntry {
  if (!isNonArrayRecord(layoutEntry)) {
    throw new NeatChatSnapshotShapeError(
      `NEATchat session snapshot ${fieldName} must be an object.`,
    );
  }

  if (layoutEntry.kind === 'bias') {
    return {
      kind: 'bias',
      nodeId: resolveIntegerField(layoutEntry.nodeId, `${fieldName}.nodeId`),
    };
  }

  if (layoutEntry.kind === 'weight') {
    const innovation = resolveOptionalIntegerField(
      layoutEntry.innovation,
      `${fieldName}.innovation`,
    );

    return innovation == null
      ? {
          kind: 'weight',
          from: resolveIntegerField(layoutEntry.from, `${fieldName}.from`),
          to: resolveIntegerField(layoutEntry.to, `${fieldName}.to`),
        }
      : {
          kind: 'weight',
          from: resolveIntegerField(layoutEntry.from, `${fieldName}.from`),
          to: resolveIntegerField(layoutEntry.to, `${fieldName}.to`),
          innovation,
        };
  }

  throw new NeatChatSnapshotShapeError(
    `NEATchat session snapshot ${fieldName}.kind must be "bias" or "weight".`,
  );
}

function resolveParameterLayoutVersion(
  layoutVersion: unknown,
  fieldName: string,
): 1 {
  const resolvedLayoutVersion = resolveIntegerField(layoutVersion, fieldName);

  if (resolvedLayoutVersion !== SUPPORTED_PARAMETER_VECTOR_LAYOUT_VERSION) {
    throw new NeatChatSnapshotShapeError(
      `NEATchat session snapshot ${fieldName} must be ${String(SUPPORTED_PARAMETER_VECTOR_LAYOUT_VERSION)}.`,
    );
  }

  return SUPPORTED_PARAMETER_VECTOR_LAYOUT_VERSION;
}

function resolveParameterValues(values: unknown, fieldName: string): number[] {
  if (!isParameterValueCollection(values)) {
    throw new NeatChatSnapshotShapeError(
      `NEATchat session snapshot ${fieldName} must be an array or Float64Array.`,
    );
  }

  return Array.from(values, (parameterValue, parameterIndex) =>
    resolveFiniteNumber(parameterValue, `${fieldName}[${parameterIndex}]`),
  );
}

function isParameterValueCollection(
  values: unknown,
): values is readonly number[] | Float64Array {
  return Array.isArray(values) || values instanceof Float64Array;
}

function assertParameterVectorLengthMatchesLayout(
  values: readonly number[],
  layoutEntries: readonly ParameterLayoutEntry[],
): void {
  if (values.length !== layoutEntries.length) {
    throw new NeatChatSnapshotShapeError(
      'NEATchat session snapshot parameterVector values length must match layoutEntries length.',
    );
  }
}

function resolveDescriptorHash(
  descriptorHash: unknown,
  layoutEntries: readonly ParameterLayoutEntry[],
  layoutVersion: number,
): string {
  if (typeof descriptorHash !== 'string' || descriptorHash.length === 0) {
    throw new NeatChatSnapshotShapeError(
      'NEATchat session snapshot parameterVector.descriptorHash must be a non-empty string.',
    );
  }

  const expectedDescriptorHash = createParameterDescriptorHash(
    layoutEntries,
    layoutVersion,
  );

  if (descriptorHash !== expectedDescriptorHash) {
    throw new NeatChatSnapshotShapeError(
      'NEATchat session snapshot parameterVector.descriptorHash does not match layoutEntries.',
    );
  }

  return descriptorHash;
}

function materializeParameterVector(
  parameterVectorSnapshot: StructuredParameterVectorSnapshot,
): ParameterVector {
  return {
    layout: {
      version: parameterVectorSnapshot.layoutVersion,
      entries: parameterVectorSnapshot.layoutEntries.map(
        cloneParameterLayoutEntry,
      ),
    },
    values: Float64Array.from(parameterVectorSnapshot.values),
  };
}

function cloneParameterLayoutEntry(
  parameterLayoutEntry: ParameterLayoutEntry,
): ParameterLayoutEntry {
  if (parameterLayoutEntry.kind === 'bias') {
    return {
      kind: 'bias',
      nodeId: parameterLayoutEntry.nodeId,
    };
  }

  return parameterLayoutEntry.innovation == null
    ? {
        kind: 'weight',
        from: parameterLayoutEntry.from,
        to: parameterLayoutEntry.to,
      }
    : {
        kind: 'weight',
        from: parameterLayoutEntry.from,
        to: parameterLayoutEntry.to,
        innovation: parameterLayoutEntry.innovation,
      };
}

function createParameterDescriptorHash(
  layoutEntries: readonly ParameterLayoutEntry[],
  layoutVersion: number,
): string {
  const descriptorSummary = [
    `version:${String(layoutVersion)}`,
    ...layoutEntries.map(summarizeParameterLayoutEntry),
  ].join('|');

  let rollingHash = INITIAL_FNV1A_HASH;

  for (const descriptorCharacter of descriptorSummary) {
    rollingHash ^= descriptorCharacter.charCodeAt(0);
    rollingHash = Math.imul(rollingHash, FNV1A_PRIME) >>> 0;
  }

  return rollingHash.toString(16).padStart(8, '0');
}

function summarizeParameterLayoutEntry(
  parameterLayoutEntry: ParameterLayoutEntry,
): string {
  if (parameterLayoutEntry.kind === 'bias') {
    return `bias:${String(parameterLayoutEntry.nodeId)}`;
  }

  const innovationSummary =
    parameterLayoutEntry.innovation == null
      ? 'none'
      : String(parameterLayoutEntry.innovation);

  return `weight:${String(parameterLayoutEntry.from)}->${String(parameterLayoutEntry.to)}:innovation:${innovationSummary}`;
}

function createRestoredNeatChatSession(
  bundle: NormalizedNeatChatSessionSnapshotV2,
  vocabulary: ReturnType<typeof buildNeatChatVocabulary>,
  network: Network,
): NeatChatSession {
  const restoredMemoryBank =
    bundle.extensions.neatchat.memoryBank == null
      ? createNeatChatEpisodicMemoryBank()
      : cloneMemoryBank(bundle.extensions.neatchat.memoryBank);

  return {
    vocabulary,
    network,
    exchanges: cloneExchangeRecords(bundle.exchanges),
    learnedExchangeCount: bundle.learnedExchangeCount,
    learnedTokenPairCount: bundle.learnedTokenPairCount,
    seededTokenPairCount: bundle.seededTokenPairCount,
    contextWindowTokenCount: bundle.contextWindowTokenCount,
    replayBufferExchangeCount: Math.min(
      bundle.exchanges.length,
      NEATCHAT_REPLAY_BUFFER_MAX_EXCHANGES,
    ),
    pendingCandidates: [],
    candidateLog: cloneCandidateLogEntries(
      bundle.extensions.neatchat.candidateLog ?? [],
    ),
    memoryBank: restoredMemoryBank,
    routingLog: cloneRoutingLogEntries(
      bundle.extensions.neatchat.routingLog ?? [],
    ),
  };
}

function isNonArrayRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function resolveIntegerField(value: unknown, fieldName: string): number {
  const resolvedValue = Number(value);

  if (!Number.isInteger(resolvedValue)) {
    throw new NeatChatSnapshotShapeError(
      `NEATchat session snapshot ${fieldName} must be an integer.`,
    );
  }

  return resolvedValue;
}

function resolveOptionalIntegerField(
  value: unknown,
  fieldName: string,
): number | undefined {
  return value == null ? undefined : resolveIntegerField(value, fieldName);
}

function resolveFiniteNumber(value: unknown, fieldName: string): number {
  const resolvedValue = Number(value);

  if (!Number.isFinite(resolvedValue)) {
    throw new NeatChatSnapshotShapeError(
      `NEATchat session snapshot ${fieldName} must be a finite number.`,
    );
  }

  return resolvedValue;
}

function resolveNonNegativeInteger(value: number, fieldName: string): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new NeatChatNonNegativeIntegerValidationError(
      `${fieldName} must be a non-negative integer.`,
    );
  }

  return value;
}
