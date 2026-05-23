import {
  fineTuneVector,
  fromParameterVector,
  toParameterVector,
} from '../../../src/neataptic.ts';
import { NEATCHAT_SPECIAL_TOKEN_INDICES } from './neatChat.constants';
import type {
  NeatChatAdaptationCandidate,
  NeatChatAdaptationManager,
  NeatChatCandidateLogEntry,
  ScheduleNeatChatAdaptationOptions,
} from './neatChat.adaptation.types';
import type {
  NeatChatExchangeRecord,
  NeatChatSession,
  NeatChatVocabulary,
} from './neatChat.types';

type NeatChatAdaptationTrainingCase = {
  input: number[];
  output: number[];
};

type PromotedNeatChatAdaptationResult = NeatChatSession & {
  readonly manager: NeatChatAdaptationManager;
  readonly session: NeatChatSession;
};

type MutableNeatChatAdaptationManager = {
  pendingCandidates: readonly NeatChatAdaptationCandidate[];
  candidateLog: readonly NeatChatCandidateLogEntry[];
};

const DEFAULT_ADAPTATION_LEARNING_RATE = 0.05;
const DEFAULT_ADAPTATION_STEP_COUNT = 1;

/**
 * Creates an empty in-memory adaptation manager for one NEATchat session.
 *
 * @param _session - Live session whose candidates will be reviewed.
 * @returns Empty manager with no pending candidates and no decisions.
 *
 * @example
 * ```ts
 * const manager = createNeatChatAdaptationManager(session);
 * console.log(manager.pendingCandidates.length); // 0
 * ```
 */
export function createNeatChatAdaptationManager(
  _session: NeatChatSession,
): NeatChatAdaptationManager {
  return buildAdaptationManager([], []);
}

/**
 * Schedules one detached fine-tune pass and appends the resulting candidate.
 *
 * The live session network stays frozen. The helper captures a base parameter
 * vector, defers work with `queueMicrotask(...)`, and fine-tunes only the
 * detached vector returned by `fineTuneVector(...)`.
 *
 * @param manager - Current in-memory candidate manager.
 * @param session - Live NEATchat session used as the frozen adaptation source.
 * @param options - Optional fine-tuning and replay-window overrides.
 * @returns New manager view with the candidate appended to `pendingCandidates`.
 *
 * @example
 * ```ts
 * const nextManager = await scheduleNeatChatAdaptation(manager, session, {
 *   learningRate: 0.05,
 *   steps: 2,
 * });
 * console.log(nextManager.pendingCandidates.length); // 1
 * ```
 *
 * @remarks
 * The function is async but runs on the main thread. It uses `queueMicrotask`
 * to defer the start of the fine-tune computation so the call site remains
 * non-blocking without requiring a worker thread. This makes the contract
 * forward-compatible with a real worker-thread backend in a future slice.
 *
 * The manager follows an immutable value pattern: always use the returned
 * `nextManager` rather than the original reference; both will point to the
 * same updated state, but relying on the return value makes the data flow
 * explicit.
 */
export async function scheduleNeatChatAdaptation(
  manager: NeatChatAdaptationManager,
  session: NeatChatSession,
  options: ScheduleNeatChatAdaptationOptions = {},
): Promise<NeatChatAdaptationManager> {
  await deferNeatChatAdaptation();

  // Step 1: Freeze the current live network into a detached parameter vector.
  const baseVector = toParameterVector(session.network);

  // Step 2: Build the ordered replay dataset from the selected recent exchanges.
  const replayExchangeRecords = resolveReplayExchangeRecords(
    session.exchanges,
    options.maxExchanges,
  );
  const trainingCases = buildReplayTrainingCases(
    session.vocabulary,
    replayExchangeRecords,
  );

  // Step 3: Fine-tune the detached working copy and append the ready candidate.
  const fineTuneResult = resolveFineTuneResult(
    session.network,
    baseVector,
    trainingCases,
    createFineTuneOptions(options),
  );
  const candidate: NeatChatAdaptationCandidate = {
    trainedVector: fineTuneResult.trainedVector,
    evaluationScores: { ...(fineTuneResult.metrics ?? {}) },
    trainedOnExchangeCount: replayExchangeRecords.length,
    proposedAt: Date.now(),
  };
  const nextManager = buildAdaptationManager(
    [...manager.pendingCandidates, candidate],
    [...manager.candidateLog],
  );

  synchronizeManagerView(manager, nextManager);

  return nextManager;
}

/**
 * Promotes one pending adaptation candidate into a cloned live session network.
 *
 * @param manager - Current in-memory candidate manager.
 * @param session - Live session that should receive the promoted candidate.
 * @param candidateIndex - Zero-based index of the pending candidate to promote.
 * @returns Updated session fields plus the updated manager and session wrapper.
 * @throws {RangeError} When `candidateIndex` is outside the pending-candidate range.
 *
 * @example
 * ```ts
 * const promoted = promoteNeatChatAdaptationCandidate(manager, session, 0);
 * console.log(promoted.session.candidateLog.at(-1)?.status); // 'promoted'
 * ```
 *
 * @remarks
 * Promotion **clones** the live session network before applying the trained
 * vector. The original `session.network` is never mutated: the returned
 * `promoted.session` holds a new `Network` instance initialized from the
 * candidate's `trainedVector` via `fromParameterVector`. Any other references
 * to the original network remain unaffected.
 *
 * The returned value merges the updated session fields at the top level for
 * spread-compatibility: `promoted.session` and `promoted.manager` hold the
 * canonical updated objects.
 */
export function promoteNeatChatAdaptationCandidate(
  manager: NeatChatAdaptationManager,
  session: NeatChatSession,
  candidateIndex: number,
): PromotedNeatChatAdaptationResult {
  // Step 1: Resolve the selected candidate and rebuild a cloned promoted network.
  const candidate = resolvePendingCandidate(manager, candidateIndex);
  const promotedNetwork = session.network.clone();

  fromParameterVector(promotedNetwork, candidate.trainedVector);

  // Step 2: Remove the candidate from the queue and append the promotion log.
  const candidateLogEntry = createCandidateLogEntry('promoted', candidate);
  const nextManager = buildAdaptationManager(
    manager.pendingCandidates.toSpliced(candidateIndex, 1),
    [...manager.candidateLog, candidateLogEntry],
  );
  const nextSession: NeatChatSession = {
    ...session,
    network: promotedNetwork,
    pendingCandidates: nextManager.pendingCandidates,
    candidateLog: [...session.candidateLog, candidateLogEntry],
  };

  synchronizeManagerView(manager, nextManager);

  return {
    ...nextSession,
    manager: nextManager,
    session: nextSession,
  };
}

/**
 * Rejects one pending adaptation candidate and records the decision.
 *
 * @param manager - Current in-memory candidate manager.
 * @param candidateIndex - Zero-based index of the pending candidate to reject.
 * @returns New manager view with the candidate removed and the rejection logged.
 * @throws {RangeError} When `candidateIndex` is outside the pending-candidate range.
 *
 * @example
 * ```ts
 * const nextManager = rejectNeatChatAdaptationCandidate(manager, 0);
 * console.log(nextManager.candidateLog.at(-1)?.status); // 'rejected'
 * ```
 */
export function rejectNeatChatAdaptationCandidate(
  manager: NeatChatAdaptationManager,
  candidateIndex: number,
): NeatChatAdaptationManager {
  // Step 1: Resolve the selected candidate so invalid indices fail loudly.
  const candidate = resolvePendingCandidate(manager, candidateIndex);

  // Step 2: Remove the candidate from the queue and append the rejection log.
  const candidateLogEntry = createCandidateLogEntry('rejected', candidate);
  const nextManager = buildAdaptationManager(
    manager.pendingCandidates.toSpliced(candidateIndex, 1),
    [...manager.candidateLog, candidateLogEntry],
  );

  synchronizeManagerView(manager, nextManager);

  return nextManager;
}

function deferNeatChatAdaptation(): Promise<void> {
  return new Promise((resolve) => {
    queueMicrotask(resolve);
  });
}

function buildAdaptationManager(
  pendingCandidates: readonly NeatChatAdaptationCandidate[],
  candidateLog: readonly NeatChatCandidateLogEntry[],
): NeatChatAdaptationManager {
  return {
    pendingCandidates,
    candidateLog,
  };
}

function synchronizeManagerView(
  manager: NeatChatAdaptationManager,
  nextManager: NeatChatAdaptationManager,
): void {
  const mutableManager = manager as MutableNeatChatAdaptationManager;

  mutableManager.pendingCandidates = nextManager.pendingCandidates;
  mutableManager.candidateLog = nextManager.candidateLog;
}

function resolveReplayExchangeRecords(
  exchangeRecords: readonly NeatChatExchangeRecord[],
  maxExchanges: number | undefined,
): readonly NeatChatExchangeRecord[] {
  if (maxExchanges == null) {
    return exchangeRecords;
  }

  const boundedExchangeCount = Math.max(0, Math.floor(maxExchanges));

  return exchangeRecords.slice(-boundedExchangeCount);
}

function buildReplayTrainingCases(
  vocabulary: NeatChatVocabulary,
  exchangeRecords: readonly NeatChatExchangeRecord[],
): NeatChatAdaptationTrainingCase[] {
  return exchangeRecords.flatMap((exchangeRecord) => {
    const userIndices = mapTokensToIndices(
      vocabulary,
      exchangeRecord.userTokens,
    );
    const responseIndices = mapTokensToIndices(
      vocabulary,
      exchangeRecord.responseTokens,
    );

    return buildExchangeTrainingCases(
      vocabulary.size,
      userIndices,
      responseIndices,
    );
  });
}

function mapTokensToIndices(
  vocabulary: NeatChatVocabulary,
  tokens: readonly string[],
): number[] {
  return tokens.map(
    (token) =>
      vocabulary.termToIndex.get(token) ?? NEATCHAT_SPECIAL_TOKEN_INDICES.UNK,
  );
}

function buildExchangeTrainingCases(
  vocabularySize: number,
  userIndices: readonly number[],
  responseIndices: readonly number[],
): NeatChatAdaptationTrainingCase[] {
  const tokenSequence = [
    NEATCHAT_SPECIAL_TOKEN_INDICES.BOS,
    ...userIndices,
    NEATCHAT_SPECIAL_TOKEN_INDICES.TURN_BREAK,
    ...responseIndices,
    NEATCHAT_SPECIAL_TOKEN_INDICES.EOS,
  ];

  return buildTrainingCasesFromSequence(tokenSequence, vocabularySize);
}

function buildTrainingCasesFromSequence(
  tokenSequence: readonly number[],
  vocabularySize: number,
): NeatChatAdaptationTrainingCase[] {
  return tokenSequence.slice(0, -1).map((tokenIndex, tokenPairIndex) => ({
    input: buildOneHotVector(tokenIndex, vocabularySize),
    output: buildOneHotVector(
      tokenSequence[tokenPairIndex + 1]!,
      vocabularySize,
    ),
  }));
}

function buildOneHotVector(
  tokenIndex: number,
  vocabularySize: number,
): number[] {
  const oneHotVector = new Array<number>(vocabularySize).fill(0);

  oneHotVector[tokenIndex] = 1;

  return oneHotVector;
}

function createFineTuneOptions(options: ScheduleNeatChatAdaptationOptions): {
  steps: number;
  learningRate: number;
  seed?: number;
} {
  const steps =
    options.steps ?? options.epochs ?? DEFAULT_ADAPTATION_STEP_COUNT;

  return {
    steps,
    learningRate: options.learningRate ?? DEFAULT_ADAPTATION_LEARNING_RATE,
    ...(options.seed == null ? {} : { seed: options.seed }),
  };
}

function resolveFineTuneResult(
  network: NeatChatSession['network'],
  baseVector: ReturnType<typeof toParameterVector>,
  trainingCases: NeatChatAdaptationTrainingCase[],
  options: ReturnType<typeof createFineTuneOptions>,
): {
  trainedVector: ReturnType<typeof toParameterVector>;
  metrics?: Record<string, number>;
} {
  try {
    return fineTuneVector(network, baseVector, trainingCases, options);
  } catch (error) {
    if (trainingCases.length > 0) {
      throw error;
    }

    return {
      trainedVector: baseVector,
      metrics: {},
    };
  }
}

function resolvePendingCandidate(
  manager: NeatChatAdaptationManager,
  candidateIndex: number,
): NeatChatAdaptationCandidate {
  const candidate = manager.pendingCandidates.at(candidateIndex);

  if (!candidate) {
    throw new RangeError(
      `Candidate index ${String(candidateIndex)} is out of bounds for ${String(manager.pendingCandidates.length)} pending candidates.`,
    );
  }

  return candidate;
}

function createCandidateLogEntry(
  status: NeatChatCandidateLogEntry['status'],
  candidate: NeatChatAdaptationCandidate,
): NeatChatCandidateLogEntry {
  return {
    status,
    trainedOnExchangeCount: candidate.trainedOnExchangeCount,
    evaluationScores: { ...candidate.evaluationScores },
    decidedAt: Date.now(),
  };
}
