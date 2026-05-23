import { fromParameterVector } from '../../../src/neataptic.ts';
import {
  NEATCHAT_MAX_RESPONSE_TOKENS,
  NEATCHAT_MINIMUM_RESPONSE_LENGTH,
  NEATCHAT_REPETITION_PENALTY_FACTOR,
  NEATCHAT_REPETITION_WINDOW_SIZE,
  NEATCHAT_SINGLE_WORD_PENALTY_FACTOR,
  NEATCHAT_SPECIAL_TOKEN_INDICES,
  NEATCHAT_SPECIAL_TOKENS,
} from './neatChat.constants';
import { tokenizeNeatChatText } from './neatChat.tokenization.utils';
import type { NeatChatAdaptationCandidate, NeatChatAdaptationManager } from './neatChat.adaptation.types';
import type { NeatChatMemoryRetrievalResult } from './neatChat.memory.types';
import type {
  NeatChatRoutingCandidate,
  NeatChatRoutingDecisionLogEntry,
  NeatChatRoutingPath,
} from './neatChat.routing.types';
import type { NeatChatSession, NeatChatVocabulary } from './neatChat.types';

const DEFAULT_ROUTING_PROMPT_TEXT = 'assistant response';

const ROUTING_PATH_PRIORITY: Readonly<Record<NeatChatRoutingPath, number>> = {
  base: 0,
  personalized: 1,
  'retrieval-grounded': 2,
};

/**
 * Generates deterministic response candidates for the active NEATchat session.
 *
 * The helper always includes the live-session base path, optionally evaluates
 * the newest detached personalization candidate without promoting it, and adds
 * a retrieval-grounded path when ranked memories are available.
 *
 * @param session - Current live session used as the immutable routing source.
 * @param adaptationManager - Detached adaptation-manager view holding reviewable candidates.
 * @param retrievedMemories - Ranked retrieval results to optionally ground one candidate.
 * @param promptText - Optional prompt text used for candidate generation.
 * @returns Ordered routing candidates beginning with the base path.
 *
 * @example
 * ```ts
 * // Session with no pending candidates and no retrieved memories → one base candidate.
 * const candidates = generateNeatChatCandidates(session, adaptationManager, []);
 * // candidates.length === 1
 * // candidates[0].routingPath === 'base'
 *
 * // With a pending adaptation candidate and retrieved memories → up to three candidates.
 * const allCandidates = generateNeatChatCandidates(session, adaptationManager, memories);
 * // allCandidates.map(c => c.routingPath) includes 'base', 'personalized', and
 * // 'retrieval-grounded' when both sources are active.
 * ```
 */
export function generateNeatChatCandidates(
  session: NeatChatSession,
  adaptationManager: Pick<
    NeatChatAdaptationManager,
    'pendingCandidates' | 'candidateLog'
  >,
  retrievedMemories: readonly NeatChatMemoryRetrievalResult[],
  promptText?: string,
): NeatChatRoutingCandidate[] {
  const resolvedPromptText = resolveRoutingPromptText(
    session,
    retrievedMemories,
    promptText,
  );
  const baseCandidate = createBaseRoutingCandidate(session, resolvedPromptText);
  const personalizedCandidate = createPersonalizedRoutingCandidate(
    session,
    resolvedPromptText,
    adaptationManager.pendingCandidates.at(-1),
  );
  const retrievalGroundedCandidate = createRetrievalGroundedCandidate(
    session,
    resolvedPromptText,
    retrievedMemories,
  );

  return [
    baseCandidate,
    ...(personalizedCandidate == null ? [] : [personalizedCandidate]),
    ...(retrievalGroundedCandidate == null ? [] : [retrievalGroundedCandidate]),
  ];
}

/**
 * Selects the highest-scoring routing candidate deterministically.
 *
 * Equal scores break in favor of the base path so the live shipped behavior is
 * the stable fallback whenever two candidates are otherwise indistinguishable.
 *
 * @param candidates - Candidate list produced by `generateNeatChatCandidates`.
 * @returns Highest-ranked candidate.
 * @throws {RangeError} When no candidates are provided.
 *
 * @example
 * ```ts
 * const candidates = generateNeatChatCandidates(session, adaptationManager, memories);
 * const winner = selectNeatChatCandidate(candidates);
 * // winner.routingPath is the path with the highest score.
 * // On a tie, 'base' wins over 'personalized', which wins over 'retrieval-grounded'.
 * ```
 */
export function selectNeatChatCandidate(
  candidates: readonly NeatChatRoutingCandidate[],
): NeatChatRoutingCandidate {
  const selectedCandidate = candidates.toSorted(compareRoutingCandidates)[0];

  if (selectedCandidate == null) {
    throw new RangeError('NEATchat routing requires at least one candidate.');
  }

  return selectedCandidate;
}

/**
 * Appends a durable routing decision entry without mutating the existing log.
 *
 * The new entry records the decision timestamp, the winning path, all paths
 * that participated, per-path scores, and memory provenance for any
 * retrieval-grounded candidate. The returned log is a pure append: the input
 * array is never modified.
 *
 * @param routingLog - Existing routing log.
 * @param selectedCandidate - Candidate selected by the routing policy.
 * @param candidates - Full ordered candidate set that participated in the comparison.
 * @returns New routing log with the appended decision entry.
 *
 * @example
 * ```ts
 * const winner = selectNeatChatCandidate(candidates);
 * const updatedLog = appendNeatChatRoutingDecision(session.routingLog, winner, candidates);
 * // updatedLog.length === session.routingLog.length + 1
 * // updatedLog.at(-1).selectedPath === winner.routingPath
 * // session.routingLog is unchanged (pure append).
 * ```
 */
export function appendNeatChatRoutingDecision(
  routingLog: readonly NeatChatRoutingDecisionLogEntry[],
  selectedCandidate: NeatChatRoutingCandidate,
  candidates: readonly NeatChatRoutingCandidate[],
): NeatChatRoutingDecisionLogEntry[] {
  return [
    ...routingLog,
    {
      decidedAt: Date.now(),
      selectedPath: selectedCandidate.routingPath,
      comparedCandidatePaths: candidates.map(
        (candidate) => candidate.routingPath,
      ),
      candidateCount: candidates.length,
      scores: Object.fromEntries(
        candidates.map((candidate) => [candidate.routingPath, candidate.score]),
      ) as Partial<Record<NeatChatRoutingPath, number>>,
      retrievedMemoryKeysByPath: buildRetrievedMemoryKeysByPath(candidates),
      retrievedMemoryCountByPath: buildRetrievedMemoryCountByPath(candidates),
      responsesByPath: Object.fromEntries(
        candidates.map((candidate) => [candidate.routingPath, candidate.response]),
      ) as Partial<Record<NeatChatRoutingPath, string>>,
    },
  ];
}

function createBaseRoutingCandidate(
  session: NeatChatSession,
  promptText: string,
): NeatChatRoutingCandidate {
  const responseResult = generateCandidateResponse(session, session.network, promptText);

  return {
    routingPath: 'base',
    response: responseResult.response,
    responseTokens: responseResult.responseTokens,
    score: clampToUnitInterval(responseResult.averageSelectedTokenConfidence),
    retrievedMemoryCount: 0,
    retrievedMemoryKeys: [],
  };
}

function createPersonalizedRoutingCandidate(
  session: NeatChatSession,
  promptText: string,
  adaptationCandidate: NeatChatAdaptationCandidate | undefined,
): NeatChatRoutingCandidate | undefined {
  if (adaptationCandidate == null) {
    return undefined;
  }

  const personalizedNetwork = cloneNetworkIfAvailable(session.network);

  if (personalizedNetwork == null) {
    return undefined;
  }

  fromParameterVector(personalizedNetwork, adaptationCandidate.trainedVector);

  const responseResult = generateCandidateResponse(
    session,
    personalizedNetwork,
    promptText,
  );
  const adaptationScore = resolveAdaptationScore(adaptationCandidate);

  return {
    routingPath: 'personalized',
    response: responseResult.response,
    responseTokens: responseResult.responseTokens,
    score: clampToUnitInterval(
      (responseResult.averageSelectedTokenConfidence + adaptationScore) / 2,
    ),
    retrievedMemoryCount: 0,
    retrievedMemoryKeys: [],
  };
}

function createRetrievalGroundedCandidate(
  session: NeatChatSession,
  promptText: string,
  retrievedMemories: readonly NeatChatMemoryRetrievalResult[],
): NeatChatRoutingCandidate | undefined {
  if (retrievedMemories.length === 0) {
    return undefined;
  }

  const retrievedMemoryKeys = retrievedMemories.map((memoryResult) => memoryResult.key);
  const groundingPromptText = `${promptText} ${buildRetrievedMemoryContext(retrievedMemories)}`.trim();
  const responseResult = generateCandidateResponse(
    session,
    session.network,
    groundingPromptText,
  );
  const retrievalSupportScore = resolveRetrievalSupportScore(
    responseResult.responseTokens,
    retrievedMemories,
  );

  return {
    routingPath: 'retrieval-grounded',
    response: responseResult.response,
    responseTokens: responseResult.responseTokens,
    score: clampToUnitInterval(
      (responseResult.averageSelectedTokenConfidence + retrievalSupportScore) / 2,
    ),
    retrievedMemoryCount: retrievedMemories.length,
    retrievedMemoryKeys,
  };
}

function generateCandidateResponse(
  session: NeatChatSession,
  sourceNetwork: NeatChatSession['network'],
  promptText: string,
): {
  readonly response: string;
  readonly responseTokens: readonly string[];
  readonly averageSelectedTokenConfidence: number;
} {
  const promptTokens = tokenizeNeatChatText(
    promptText,
    session.contextWindowTokenCount,
  );
  const promptIndices = promptTokens.map(
    (token) =>
      session.vocabulary.termToIndex.get(token) ?? NEATCHAT_SPECIAL_TOKEN_INDICES.UNK,
  );
  const workingNetwork = cloneNetworkIfAvailable(sourceNetwork) ?? sourceNetwork;

  workingNetwork.clear();

  const responseSelection = inferRoutingResponseTokenSelection(
    workingNetwork,
    session.vocabulary.size,
    promptIndices,
  );
  const responseTokens = resolveResponseTokens(
    session.vocabulary,
    responseSelection.responseIndices,
    promptIndices,
  );

  return {
    response:
      responseTokens.length > 0
        ? responseTokens.join(' ')
        : session.vocabulary.indexToTerm[NEATCHAT_SPECIAL_TOKEN_INDICES.UNK]!,
    responseTokens,
    averageSelectedTokenConfidence:
      responseSelection.averageSelectedTokenConfidence,
  };
}

function resolveRoutingPromptText(
  session: NeatChatSession,
  retrievedMemories: readonly NeatChatMemoryRetrievalResult[],
  promptText: string | undefined,
): string {
  if (promptText != null) {
    return promptText.trim();
  }

  const latestUserMessage = session.exchanges.at(-1)?.userMessage?.trim();

  if (latestUserMessage != null && latestUserMessage.length > 0) {
    return latestUserMessage;
  }

  const retrievedPromptText = retrievedMemories
    .map((memoryResult) => memoryResult.key)
    .join(' ')
    .trim();

  return retrievedPromptText.length > 0
    ? retrievedPromptText
    : DEFAULT_ROUTING_PROMPT_TEXT;
}

function buildRetrievedMemoryContext(
  retrievedMemories: readonly NeatChatMemoryRetrievalResult[],
): string {
  return retrievedMemories
    .map((memoryResult) => `${memoryResult.key} ${memoryResult.value}`)
    .join(' ')
    .trim();
}

function resolveResponseTokens(
  vocabulary: NeatChatVocabulary,
  responseIndices: readonly number[],
  promptIndices: readonly number[],
): readonly string[] {
  const decodedResponseTokens = responseIndices.map(
    (tokenIndex) => vocabulary.indexToTerm[tokenIndex]!,
  );

  if (decodedResponseTokens.length > 0) {
    return decodedResponseTokens;
  }

  if (promptIndices.length === 0) {
    return [];
  }

  const fallbackToken = resolveFallbackToken(
    vocabulary,
    promptIndices.at(-1)!,
  );

  return [fallbackToken];
}

function resolveFallbackToken(
  vocabulary: NeatChatVocabulary,
  requestedTokenIndex: number,
): string {
  const requestedToken = vocabulary.indexToTerm[requestedTokenIndex];

  if (
    requestedToken !== undefined &&
    requestedToken !== vocabulary.indexToTerm[NEATCHAT_SPECIAL_TOKEN_INDICES.UNK] &&
    requestedToken !== 'BOS' &&
    requestedToken !== 'EOS' &&
    requestedToken !== 'TURN_BREAK'
  ) {
    return requestedToken;
  }

  const firstNonSpecialToken = vocabulary.indexToTerm.find(
    (term) =>
      term !== vocabulary.indexToTerm[NEATCHAT_SPECIAL_TOKEN_INDICES.UNK] &&
      term !== 'BOS' &&
      term !== 'EOS' &&
      term !== 'TURN_BREAK',
  );

  return firstNonSpecialToken ?? vocabulary.indexToTerm[0]!;
}

function inferRoutingResponseTokenSelection(
  network: NeatChatSession['network'],
  vocabularySize: number,
  userIndices: readonly number[],
): {
  readonly responseIndices: number[];
  readonly averageSelectedTokenConfidence: number;
} {
  network.activate(buildOneHotVector(NEATCHAT_SPECIAL_TOKEN_INDICES.BOS, vocabularySize));

  for (const tokenIndex of userIndices) {
    network.activate(buildOneHotVector(tokenIndex, vocabularySize));
  }

  network.activate(
    buildOneHotVector(NEATCHAT_SPECIAL_TOKEN_INDICES.TURN_BREAK, vocabularySize),
  );

  const responseIndices: number[] = [];
  let currentIndex: number = NEATCHAT_SPECIAL_TOKEN_INDICES.TURN_BREAK;
  let selectedTokenConfidenceSum = 0;

  for (let step = 0; step < NEATCHAT_MAX_RESPONSE_TOKENS; step++) {
    const output = network.activate(buildOneHotVector(currentIndex, vocabularySize));
    const canTerminate = responseIndices.length >= NEATCHAT_MINIMUM_RESPONSE_LENGTH;
    const nextIndex = selectNextResponseTokenIndex(
      output,
      vocabularySize,
      responseIndices,
      canTerminate,
    );

    if (nextIndex === NEATCHAT_SPECIAL_TOKEN_INDICES.EOS) {
      break;
    }

    const selectedTokenConfidence = output[nextIndex]!;

    responseIndices.push(nextIndex);
    selectedTokenConfidenceSum += Math.max(0, selectedTokenConfidence);
    currentIndex = nextIndex;
  }

  return {
    responseIndices,
    averageSelectedTokenConfidence:
      responseIndices.length === 0
        ? 0
        : selectedTokenConfidenceSum / responseIndices.length,
  };
}

function buildOneHotVector(index: number, size: number): number[] {
  const vector = new Array<number>(size).fill(0);

  vector[index] = 1;

  return vector;
}

function cloneNetworkIfAvailable(
  network: NeatChatSession['network'],
): NeatChatSession['network'] | undefined {
  return typeof network.clone === 'function' ? network.clone() : undefined;
}

function selectNextResponseTokenIndex(
  tokenScores: number[],
  vocabularySize: number,
  recentResponseIndices: readonly number[],
  canTerminate: boolean,
): number {
  let bestTokenIndex: number = NEATCHAT_SPECIAL_TOKEN_INDICES.EOS;
  let bestTokenScore = -Infinity;
  const recentWindow = recentResponseIndices.slice(-NEATCHAT_REPETITION_WINDOW_SIZE);
  const recentIndexSet = new Set(recentWindow);

  for (let tokenIndex = 0; tokenIndex < vocabularySize; tokenIndex++) {
    if (isDisallowedResponseTokenIndex(tokenIndex, vocabularySize)) {
      continue;
    }

    if (tokenIndex === NEATCHAT_SPECIAL_TOKEN_INDICES.EOS && !canTerminate) {
      continue;
    }

    let rawScore = tokenScores[tokenIndex] ?? -Infinity;

    if (recentIndexSet.has(tokenIndex)) {
      rawScore = rawScore / NEATCHAT_REPETITION_PENALTY_FACTOR;
    }

    if (
      tokenIndex === NEATCHAT_SPECIAL_TOKEN_INDICES.EOS &&
      recentResponseIndices.length < 4
    ) {
      rawScore = rawScore / NEATCHAT_SINGLE_WORD_PENALTY_FACTOR;
    }

    if (rawScore > bestTokenScore) {
      bestTokenScore = rawScore;
      bestTokenIndex = tokenIndex;
    }
  }

  return bestTokenIndex;
}

function isDisallowedResponseTokenIndex(
  tokenIndex: number,
  vocabularySize: number,
): boolean {
  if (
    tokenIndex === NEATCHAT_SPECIAL_TOKEN_INDICES.BOS ||
    tokenIndex === NEATCHAT_SPECIAL_TOKEN_INDICES.TURN_BREAK
  ) {
    return true;
  }

  if (
    vocabularySize > NEATCHAT_SPECIAL_TOKENS.length &&
    tokenIndex === NEATCHAT_SPECIAL_TOKEN_INDICES.UNK
  ) {
    return true;
  }

  return false;
}

function resolveAdaptationScore(
  adaptationCandidate: NeatChatAdaptationCandidate,
): number {
  const heldOutAccuracy = adaptationCandidate.evaluationScores.heldOutAccuracy;

  if (typeof heldOutAccuracy === 'number') {
    return clampToUnitInterval(heldOutAccuracy);
  }

  const firstEvaluationScore = Object.values(
    adaptationCandidate.evaluationScores,
  )[0];

  return clampToUnitInterval(firstEvaluationScore ?? 0);
}

function resolveRetrievalSupportScore(
  responseTokens: readonly string[],
  retrievedMemories: readonly NeatChatMemoryRetrievalResult[],
): number {
  const retrievedMemoryTokens = tokenizeNeatChatText(
    buildRetrievedMemoryContext(retrievedMemories),
    Number.MAX_SAFE_INTEGER,
  );

  if (retrievedMemoryTokens.length === 0 || responseTokens.length === 0) {
    return clampToUnitInterval(retrievedMemories.length / 3);
  }

  const retrievedMemoryTokenSet = new Set(retrievedMemoryTokens);
  const matchedTokenCount = responseTokens.reduce(
    (matchCount, responseToken) =>
      matchCount + (retrievedMemoryTokenSet.has(responseToken) ? 1 : 0),
    0,
  );
  const overlapRatio = matchedTokenCount / responseTokens.length;
  const memoryCoverageScore = Math.min(retrievedMemories.length / 3, 1);

  return clampToUnitInterval((overlapRatio + memoryCoverageScore) / 2);
}

function compareRoutingCandidates(
  leftCandidate: NeatChatRoutingCandidate,
  rightCandidate: NeatChatRoutingCandidate,
): number {
  if (leftCandidate.score !== rightCandidate.score) {
    return rightCandidate.score - leftCandidate.score;
  }

  return (
    ROUTING_PATH_PRIORITY[leftCandidate.routingPath] -
    ROUTING_PATH_PRIORITY[rightCandidate.routingPath]
  );
}

function buildRetrievedMemoryKeysByPath(
  candidates: readonly NeatChatRoutingCandidate[],
): NeatChatRoutingDecisionLogEntry['retrievedMemoryKeysByPath'] {
  const entries = candidates
    .filter((candidate) => candidate.retrievedMemoryKeys.length > 0)
    .map((candidate) => [candidate.routingPath, [...candidate.retrievedMemoryKeys]] as const);

  return entries.length === 0
    ? undefined
    : (Object.fromEntries(entries) as Partial<
        Record<NeatChatRoutingPath, readonly string[]>
      >);
}

function buildRetrievedMemoryCountByPath(
  candidates: readonly NeatChatRoutingCandidate[],
): NeatChatRoutingDecisionLogEntry['retrievedMemoryCountByPath'] {
  const entries = candidates
    .filter((candidate) => candidate.retrievedMemoryCount > 0)
    .map((candidate) => [candidate.routingPath, candidate.retrievedMemoryCount] as const);

  return entries.length === 0
    ? undefined
    : (Object.fromEntries(entries) as Partial<Record<NeatChatRoutingPath, number>>);
}

function clampToUnitInterval(score: number): number {
  if (!Number.isFinite(score)) {
    return 0;
  }

  return Math.min(Math.max(score, 0), 1);
}