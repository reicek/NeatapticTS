/**
 * @module neatChat.session.services
 *
 * Core session lifecycle and exchange-loop services for NEATchat.
 *
 * Provides three public surface categories:
 * - **Vocabulary** — {@link buildNeatChatVocabulary}
 * - **Session creation / pretraining** — {@link createNeatChatSession},
 *   {@link pretrainNeatChatSessionWithConversationLines},
 *   {@link updateNeatChatSessionContextWindowTokenCount}
 * - **Exchange loop** — {@link runNeatChatExchange}
 *
 * ### Exchange loop flow (`runNeatChatExchange`)
 *
 * ```mermaid
 * sequenceDiagram
 *     participant U as User
 *     participant E as runNeatChatExchange
 *     participant M as Memory bank
 *     participant R as Routing / candidates
 *     participant S as Safety gate
 *     participant L as Online learning
 *
 *     U->>E: userMessage
 *     E->>M: retrieveNeatChatMemories (top-3)
 *     M-->>E: retrievedMemories
 *     E->>R: generateNeatChatCandidates
 *     R-->>E: surfacedCandidates (placeholder tokens stripped)
 *     loop resolveSelectedCandidate
 *         E->>S: checkSafety(candidate)
 *         S-->>E: safetyResult
 *         E->>E: recentDuplicateCheck
 *     end
 *     E->>L: applyOnlineLearningUpdate (observed + replay + anchor cases)
 *     L-->>E: updatedNetwork + committedTokenPairCount
 *     E-->>U: response + updatedSession
 * ```
 *
 * The online-learning step (Step 2 inside `runNeatChatExchange`) commits weight
 * updates unconditionally after the routing decision. Candidate generation runs
 * _before_ any weight mutation so all scoring is based on the pre-update network
 * state, keeping routing and learning decoupled within a single exchange.
 */
import { Architect, Network, methods } from '../../../src/browser-entry.ts';
import {
  NEATCHAT_DEFAULT_ARCHITECTURE_FAMILY,
  NEATCHAT_DEFAULT_CONTEXT_WINDOW_TOKEN_COUNT,
  NEATCHAT_DEFAULT_NARX_INPUT_MEMORY,
  NEATCHAT_DEFAULT_NARX_OUTPUT_MEMORY,
  NEATCHAT_DEFAULT_RECURRENT_BLOCK_SIZE,
  NEATCHAT_DEFAULT_SESSION_BOOTSTRAP_TOP_WORD_LIMIT,
  NEATCHAT_DEFAULT_UNKNOWN_TOKEN,
  NEATCHAT_LIVE_CHAT_MAX_VOCAB_TERMS,
  NEATCHAT_MAX_RESPONSE_TOKENS,
  NEATCHAT_MAX_SEED_CONVERSATION_LINES,
  NEATCHAT_MINIMUM_RESPONSE_LENGTH,
  NEATCHAT_ONLINE_LEARNING_ITERATIONS,
  NEATCHAT_ONLINE_ANCHOR_MAX_TRAINING_CASES,
  NEATCHAT_ONLINE_ANCHOR_SET_LINE_COUNT,
  NEATCHAT_ONLINE_LOSS_GATE_MAX_TRAINING_CASES,
  NEATCHAT_ONLINE_LOSS_GATE_SAMPLE_CASES,
  NEATCHAT_ONLINE_LEARNING_MIN_CONFIDENCE,
  NEATCHAT_ONLINE_LEARNING_MIN_LOSS_IMPROVEMENT,
  NEATCHAT_ONLINE_LEARNING_MOMENTUM,
  NEATCHAT_ONLINE_LEARNING_RATE,
  NEATCHAT_REPLAY_BUFFER_MAX_EXCHANGES,
  NEATCHAT_REPETITION_PENALTY_FACTOR,
  NEATCHAT_REPETITION_WINDOW_SIZE,
  NEATCHAT_SEED_LINE_TRAINING_ITERATIONS,
  NEATCHAT_SEED_STREAM_TRAINING_ITERATIONS,
  NEATCHAT_SEED_TRAINING_MOMENTUM,
  NEATCHAT_SEED_TRAINING_RATE,
  NEATCHAT_SEED_TRIGRAM_TRAINING_ITERATIONS,
  NEATCHAT_SINGLE_WORD_PENALTY_FACTOR,
  NEATCHAT_SPECIAL_TOKEN_INDICES,
  NEATCHAT_SPECIAL_TOKENS,
} from './neatChat.constants';
import {
  tokenizeNeatChatText,
  createNeatChatPretrainingPreview,
  resolvePositiveInteger,
} from './neatChat.tokenization.utils';
import { createNeatChatEpisodicMemoryBank } from './neatChat.memory.services';
import { retrieveNeatChatMemories } from './neatChat.memory.services';
import {
  appendNeatChatRoutingDecision,
  generateNeatChatCandidates,
  selectNeatChatCandidate,
} from './neatChat.routing.services';
import { checkSafety } from './neatChat.safety.services';
import type { NeatChatAdaptationManager } from './neatChat.adaptation.types';
import type { NeatChatRoutingCandidate } from './neatChat.routing.types';
import type { SafetyViolation } from './neatChat.safety.types';
import type {
  CreateNeatChatSessionOptions,
  NeatChatExchangeRecord,
  NeatChatExchangeResult,
  NeatChatSession,
  NeatChatVocabulary,
} from './neatChat.types';
import { NEATCHAT_SAMPLE_CONVERSATION_LINES } from '../sample-conversation';

const NEATCHAT_DEFAULT_SESSION_BOOTSTRAP_TERMS =
  buildDefaultSessionBootstrapTerms();
const NEATCHAT_ONLINE_ANCHOR_CONVERSATION_LINES =
  NEATCHAT_SAMPLE_CONVERSATION_LINES.slice(
    0,
    NEATCHAT_ONLINE_ANCHOR_SET_LINE_COUNT,
  );
const NEATCHAT_ROUTING_MAX_RETRIEVED_MEMORIES = 3;
const NEATCHAT_RESPONSE_PUNCTUATION_PLACEHOLDER_PREFIX = 'PUNC_';
const NEATCHAT_RECENT_RESPONSE_DUPLICATE_WINDOW = 3;
const NEATCHAT_NO_SAFE_CANDIDATE_FALLBACK_RESPONSE_TOKENS = [
  'can',
  'you',
  'rephrase',
  'that',
] as const;
const NEATCHAT_ADDITIONAL_NO_SAFE_CANDIDATE_FALLBACK_RESPONSE_TOKENS = [
  ['i', 'see'] as const,
  ['sounds', 'great'] as const,
  ['tell', 'me', 'about', 'that'] as const,
  ['i', 'am', 'good'] as const,
  ['i', 'will', 'stay', 'home'] as const,
  ['steady', 'answer'] as const,
] as const;
const NEATCHAT_PROMPT_AWARE_NO_SAFE_CANDIDATE_FALLBACKS = [
  {
    promptPattern: /\bhow was your day\b/i,
    responseTokens: ['i', 'am', 'feeling', 'good'] as const,
  },
  {
    promptPattern: /\bweekend\b/i,
    responseTokens: ['i', 'have', 'plans'] as const,
  },
] as const;
const NEATCHAT_SHORT_PROMPT_FRAGMENT_OPENERS = new Set([
  'how',
  'what',
  'when',
  'where',
  'why',
  'who',
]);
const NEATCHAT_SHORT_PROMPT_FRAGMENT_LINKING_VERBS = new Set([
  'are',
  'is',
  'was',
  'were',
]);
const NEATCHAT_SYNTHESIZED_FALLBACK_VIOLATIONS = new Set<SafetyViolation>([
  'incomplete-fragment',
]);

/**
 * Builds a token vocabulary from retained corpus terms plus stable special tokens.
 *
 * @param retainedTerms - Ordered retained terms from the corpus, excluding special tokens.
 * @returns Vocabulary with bidirectional term-to-index mapping.
 */
export function buildNeatChatVocabulary(
  retainedTerms: readonly string[] = [],
): NeatChatVocabulary {
  const indexToTerm: string[] = [
    NEATCHAT_DEFAULT_UNKNOWN_TOKEN,
    'BOS',
    'EOS',
    'TURN_BREAK',
    ...retainedTerms,
  ];
  const termToIndex = new Map(
    indexToTerm.map((term, termIndex) => [term, termIndex] as const),
  );

  return { size: indexToTerm.length, termToIndex, indexToTerm };
}

/**
 * Creates a live chat session with a seed network and optional retained vocabulary.
 *
 * @param options - Corpus terms and optional builder overrides.
 * @returns Fresh session with an untrained seed network and empty exchange history.
 */
export function createNeatChatSession(
  options: CreateNeatChatSessionOptions = {},
): NeatChatSession {
  const liveChatVocabLimit =
    options.liveChatVocabLimit ?? NEATCHAT_LIVE_CHAT_MAX_VOCAB_TERMS;
  const sourceRetainedTerms =
    options.corpusRetainedTerms && options.corpusRetainedTerms.length > 0
      ? options.corpusRetainedTerms
      : NEATCHAT_DEFAULT_SESSION_BOOTSTRAP_TERMS;
  const cappedRetainedTerms = sourceRetainedTerms.slice(0, liveChatVocabLimit);

  const vocabulary = buildNeatChatVocabulary(cappedRetainedTerms);

  const architectureFamily =
    options.architectureFamily ?? NEATCHAT_DEFAULT_ARCHITECTURE_FAMILY;
  const recurrentBlockSize = resolvePositiveInteger(
    options.recurrentBlockSize ?? NEATCHAT_DEFAULT_RECURRENT_BLOCK_SIZE,
    'recurrentBlockSize',
  );
  const contextWindowTokenCount = resolvePositiveInteger(
    options.contextWindowTokenCount ??
      NEATCHAT_DEFAULT_CONTEXT_WINDOW_TOKEN_COUNT,
    'contextWindowTokenCount',
  );
  const ioSize = vocabulary.size;
  let sessionNetwork: Network;

  if (architectureFamily === 'gru') {
    sessionNetwork = Architect.gru(ioSize, recurrentBlockSize, ioSize, {
      inputToOutput: true,
    });
  } else if (architectureFamily === 'narx') {
    sessionNetwork = Architect.narx(
      ioSize,
      [recurrentBlockSize],
      ioSize,
      NEATCHAT_DEFAULT_NARX_INPUT_MEMORY,
      NEATCHAT_DEFAULT_NARX_OUTPUT_MEMORY,
    );
  } else {
    sessionNetwork = Architect.lstm(ioSize, recurrentBlockSize, ioSize, {
      inputToOutput: true,
    });
  }

  const seedConversationLines = (options.seedConversationLines ?? []).slice(
    0,
    NEATCHAT_MAX_SEED_CONVERSATION_LINES,
  );
  const seededTokenPairCount = applySessionSeedConversationTraining(
    sessionNetwork,
    vocabulary,
    seedConversationLines,
  );

  return {
    vocabulary,
    network: sessionNetwork,
    exchanges: [],
    learnedExchangeCount: 0,
    learnedTokenPairCount: 0,
    seededTokenPairCount,
    contextWindowTokenCount,
    replayBufferExchangeCount: 0,
    pendingCandidates: [],
    candidateLog: [],
    memoryBank: createNeatChatEpisodicMemoryBank(),
    routingLog: [],
  };
}

/**
 * Updates the live-session context-window token count without resetting weights.
 *
 * @param session - Existing session to update.
 * @param contextWindowTokenCount - New positive token-window size.
 * @returns Updated session with the new context-window size.
 */
export function updateNeatChatSessionContextWindowTokenCount(
  session: NeatChatSession,
  contextWindowTokenCount: number,
): NeatChatSession {
  const resolvedContextWindowTokenCount = resolvePositiveInteger(
    contextWindowTokenCount,
    'contextWindowTokenCount',
  );

  if (resolvedContextWindowTokenCount === session.contextWindowTokenCount) {
    return session;
  }

  return {
    ...session,
    contextWindowTokenCount: resolvedContextWindowTokenCount,
  };
}

/**
 * Applies additional seed-conversation training to an existing live session.
 *
 * @param session - Existing live session to update.
 * @param seedConversationLines - Additional conversation lines for warm-up.
 * @returns Updated session with accumulated seeded token-pair count.
 */
export function pretrainNeatChatSessionWithConversationLines(
  session: NeatChatSession,
  seedConversationLines: readonly string[],
): NeatChatSession {
  const boundedSeedConversationLines = [...seedConversationLines]
    .map((conversationLine) => conversationLine.trim())
    .filter((conversationLine) => conversationLine.length > 0)
    .slice(0, NEATCHAT_MAX_SEED_CONVERSATION_LINES);
  const additionalSeededTokenPairCount = applySessionSeedConversationTraining(
    session.network,
    session.vocabulary,
    boundedSeedConversationLines,
  );

  if (additionalSeededTokenPairCount === 0) {
    return session;
  }

  return {
    ...session,
    seededTokenPairCount:
      session.seededTokenPairCount + additionalSeededTokenPairCount,
  };
}

/**
 * Runs one user-bot exchange and applies a narrow supervised update afterward.
 *
 * Before committing a response, placeholder tokens are stripped from surfaced
 * routing candidates and every candidate is screened through `checkSafety`.
 * Recent-response duplicates from the last three exchanges are skipped so
 * consecutive turns stay varied. When every candidate fails the safety gate,
 * a bounded vocab-filtered fallback floor is returned instead of the top unsafe
 * fragment.
 *
 * @param session - Current live session holding the network and vocabulary.
 * @param userMessage - Raw user message text.
 * @returns Exchange result including the bot reply and the updated session.
 */
export function runNeatChatExchange(
  session: NeatChatSession,
  userMessage: string,
): NeatChatExchangeResult {
  const contextWindow = session.contextWindowTokenCount;

  const userTokens = tokenizeNeatChatText(userMessage, contextWindow);

  // Step 1: Rank memories and compare response-path candidates without mutating weights.
  const retrievedMemories = retrieveNeatChatMemories(session, userMessage, {
    maxResults: NEATCHAT_ROUTING_MAX_RETRIEVED_MEMORIES,
  });
  const routingCandidates = generateNeatChatCandidates(
    session,
    createRoutingAdaptationManager(session),
    retrievedMemories,
    userMessage,
  );
  const surfacedRoutingCandidates = routingCandidates.map(
    stripPunctuationPlaceholderTokensFromCandidate,
  );
  const selectedCandidate = resolveSelectedCandidate(
    session,
    surfacedRoutingCandidates,
    userMessage,
  );
  const responseTokens = [...selectedCandidate.responseTokens];
  const response = selectedCandidate.response;

  // Step 2: Reuse the existing online-learning path after the routing decision is made.
  const trainingCases = buildObservedUserHistoryTrainingCases(
    session.vocabulary,
    session.exchanges,
    userTokens,
  );
  const replayTrainingCases = buildReplayExchangeTrainingCases(
    session.vocabulary,
    session.exchanges,
  );
  const anchorTrainingCases = buildAnchorReplayTrainingCases(
    session.vocabulary,
  );
  const onlineLearningOutcome = applyOnlineLearningUpdate(
    session.network,
    [...trainingCases, ...replayTrainingCases, ...anchorTrainingCases],
    selectedCandidate.score,
  );
  const trainedTokenPairCount = onlineLearningOutcome.committedTokenPairCount;

  // Step 3: Fold the selected response and durable routing decision into the updated session.
  const exchangeRecord: NeatChatExchangeRecord = {
    userMessage,
    response,
    trainedTokenPairCount,
    userTokens,
    responseTokens,
  };

  return {
    response,
    responseTokens,
    userTokens,
    trainedTokenPairCount,
    updatedSession: {
      ...session,
      network: onlineLearningOutcome.updatedNetwork,
      exchanges: [...session.exchanges, exchangeRecord],
      learnedExchangeCount: session.learnedExchangeCount + 1,
      learnedTokenPairCount:
        session.learnedTokenPairCount + trainedTokenPairCount,
      seededTokenPairCount: session.seededTokenPairCount,
      replayBufferExchangeCount: resolveReplayBufferExchangeCount(
        session.exchanges.length + 1,
      ),
      routingLog: appendNeatChatRoutingDecision(
        session.routingLog,
        selectedCandidate,
        surfacedRoutingCandidates,
      ),
    },
  };

  function resolveSelectedCandidate(
    chatSession: NeatChatSession,
    candidates: readonly NeatChatRoutingCandidate[],
    promptText: string,
  ): NeatChatRoutingCandidate {
    const topCandidate = selectNeatChatCandidate(candidates);
    const recentResponseSignatures =
      resolveRecentResponseSignatures(chatSession);
    let remainingCandidates = [...candidates];
    const rejectedViolations: SafetyViolation[] = [];

    // Step 1: Walk the current local candidate set in the existing routing order.
    while (remainingCandidates.length > 0) {
      const currentCandidate = selectNeatChatCandidate(remainingCandidates);
      const safetyResult = checkSafety(chatSession, currentCandidate.response);
      const shortPromptEchoFragment =
        safetyResult.ok &&
        isShortPromptEchoFragment(currentCandidate.responseTokens);

      if (
        safetyResult.ok &&
        !shortPromptEchoFragment &&
        !isRecentResponseDuplicate(
          currentCandidate.responseTokens,
          recentResponseSignatures,
        )
      ) {
        return currentCandidate;
      }

      if (shortPromptEchoFragment) {
        rejectedViolations.push('incomplete-fragment');
      } else if (safetyResult.violation !== null) {
        rejectedViolations.push(safetyResult.violation);
      }

      const currentCandidateIndex =
        remainingCandidates.indexOf(currentCandidate);
      remainingCandidates = remainingCandidates.toSpliced(
        currentCandidateIndex,
        1,
      );
    }

    // Step 2: Replace an all-unsafe candidate set with a bounded floor.
    const shouldCreateSynthesizedFallback =
      rejectedViolations.length > 0 &&
      rejectedViolations.every((violation) =>
        NEATCHAT_SYNTHESIZED_FALLBACK_VIOLATIONS.has(violation),
      );

    if (shouldCreateSynthesizedFallback) {
      return createNoSafeCandidateFallback(
        topCandidate,
        chatSession,
        promptText,
        recentResponseSignatures,
      );
    }

    return topCandidate;
  }

  function createNoSafeCandidateFallback(
    rejectedTopCandidate: NeatChatRoutingCandidate,
    chatSession: NeatChatSession,
    promptText: string,
    recentResponseSignatures: ReadonlySet<string>,
  ): NeatChatRoutingCandidate {
    const responseTokens = resolveNoSafeCandidateFallbackResponseTokens(
      chatSession,
      promptText,
      recentResponseSignatures,
    );

    return {
      ...rejectedTopCandidate,
      response: responseTokens.join(' '),
      responseTokens,
      score: 0,
    };
  }

  function resolveNoSafeCandidateFallbackResponseTokens(
    chatSession: NeatChatSession,
    promptText: string,
    recentResponseSignatures: ReadonlySet<string>,
  ): readonly string[] {
    const orderedFallbackResponseTokens = [
      resolvePromptAwareNoSafeCandidateFallbackResponseTokens(
        chatSession,
        promptText,
      ),
      NEATCHAT_NO_SAFE_CANDIDATE_FALLBACK_RESPONSE_TOKENS,
      ...NEATCHAT_ADDITIONAL_NO_SAFE_CANDIDATE_FALLBACK_RESPONSE_TOKENS.filter(
        (responseTokens) =>
          responseTokens.every((token) =>
            chatSession.vocabulary.termToIndex.has(token),
          ),
      ),
    ].filter(
      (responseTokens): responseTokens is readonly string[] =>
        responseTokens !== undefined &&
        responseTokens.every((token) =>
          chatSession.vocabulary.termToIndex.has(token),
        ),
    );
    const freshFallbackResponseTokens = orderedFallbackResponseTokens.find(
      (responseTokens) =>
        !isRecentResponseDuplicate(responseTokens, recentResponseSignatures),
    );

    if (freshFallbackResponseTokens !== undefined) {
      return freshFallbackResponseTokens;
    }

    return (
      orderedFallbackResponseTokens[0] ??
      NEATCHAT_NO_SAFE_CANDIDATE_FALLBACK_RESPONSE_TOKENS
    );
  }

  function resolvePromptAwareNoSafeCandidateFallbackResponseTokens(
    chatSession: NeatChatSession,
    promptText: string,
  ): readonly string[] | undefined {
    return NEATCHAT_PROMPT_AWARE_NO_SAFE_CANDIDATE_FALLBACKS.find(
      ({ promptPattern, responseTokens }) =>
        promptPattern.test(promptText) &&
        responseTokens.every((token) =>
          chatSession.vocabulary.termToIndex.has(token),
        ),
    )?.responseTokens;
  }

  function isRecentResponseDuplicate(
    responseTokens: readonly string[],
    recentResponseSignatures: ReadonlySet<string>,
  ): boolean {
    return recentResponseSignatures.has(
      createResponseSignature(responseTokens),
    );
  }

  function resolveRecentResponseSignatures(
    chatSession: NeatChatSession,
  ): ReadonlySet<string> {
    return new Set(
      chatSession.exchanges
        .slice(-NEATCHAT_RECENT_RESPONSE_DUPLICATE_WINDOW)
        .map((exchangeRecord) =>
          createResponseSignature(exchangeRecord.responseTokens),
        ),
    );
  }

  function createResponseSignature(responseTokens: readonly string[]): string {
    return responseTokens.join(' ').trim().toLowerCase();
  }

  function isShortPromptEchoFragment(
    responseTokens: readonly string[],
  ): boolean {
    if (responseTokens.length < 2 || responseTokens.length > 3) {
      return false;
    }

    const normalizedTokens = responseTokens.map((token) =>
      token.trim().toLowerCase(),
    );
    const firstToken = normalizedTokens[0];
    const secondToken = normalizedTokens[1];
    const thirdToken = normalizedTokens[2];

    if (
      firstToken === undefined ||
      secondToken === undefined ||
      !NEATCHAT_SHORT_PROMPT_FRAGMENT_OPENERS.has(firstToken) ||
      !NEATCHAT_SHORT_PROMPT_FRAGMENT_LINKING_VERBS.has(secondToken)
    ) {
      return false;
    }

    return normalizedTokens.length === 2 || thirdToken === firstToken;
  }
}

function stripPunctuationPlaceholderTokensFromCandidate(
  candidate: NeatChatRoutingCandidate,
): NeatChatRoutingCandidate {
  const surfacedResponseTokens = candidate.responseTokens.filter(
    (token) => !isPunctuationPlaceholderToken(token),
  );

  if (surfacedResponseTokens.length === candidate.responseTokens.length) {
    return candidate;
  }

  return {
    ...candidate,
    response: surfacedResponseTokens.join(' ').trim(),
    responseTokens: surfacedResponseTokens,
  };
}

function isPunctuationPlaceholderToken(token: string): boolean {
  return token.startsWith(NEATCHAT_RESPONSE_PUNCTUATION_PLACEHOLDER_PREFIX);
}

function createRoutingAdaptationManager(
  session: NeatChatSession,
): Pick<NeatChatAdaptationManager, 'pendingCandidates' | 'candidateLog'> {
  return {
    pendingCandidates: session.pendingCandidates,
    candidateLog: session.candidateLog,
  };
}

/**
 * Generates response token indices by feeding user tokens through the network.
 *
 * @param network - Session network to activate (state is already reset by caller).
 * @param vocabSize - Total vocabulary size for one-hot encoding.
 * @param userIndices - Vocabulary indices of the user message tokens.
 * @returns Decoded response token indices (not including EOS).
 */
export function inferResponseTokenIndices(
  network: Network,
  vocabSize: number,
  userIndices: readonly number[],
): number[] {
  return inferResponseTokenSelection(network, vocabSize, userIndices)
    .responseIndices;
}

function inferResponseTokenSelection(
  network: Network,
  vocabSize: number,
  userIndices: readonly number[],
): {
  readonly responseIndices: number[];
  readonly averageSelectedTokenConfidence: number;
} {
  network.activate(
    buildOneHotVector(NEATCHAT_SPECIAL_TOKEN_INDICES.BOS, vocabSize),
  );

  for (const tokenIndex of userIndices) {
    network.activate(buildOneHotVector(tokenIndex, vocabSize));
  }

  network.activate(
    buildOneHotVector(NEATCHAT_SPECIAL_TOKEN_INDICES.TURN_BREAK, vocabSize),
  );

  const responseIndices: number[] = [];
  let currentIndex: number = NEATCHAT_SPECIAL_TOKEN_INDICES.TURN_BREAK;
  let selectedTokenConfidenceSum = 0;

  for (let step = 0; step < NEATCHAT_MAX_RESPONSE_TOKENS; step++) {
    const output = network.activate(buildOneHotVector(currentIndex, vocabSize));
    // Allow termination only after minimum response length is met
    const canTerminate =
      responseIndices.length >= NEATCHAT_MINIMUM_RESPONSE_LENGTH;
    const nextIndex = selectNextResponseTokenIndex(
      output,
      vocabSize,
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

/**
 * Builds seed-training cases from one full conversation stream.
 *
 * @param vocabulary - Session vocabulary.
 * @param seedConversationLines - Ordered conversation lines.
 * @returns Full-stream next-token training cases.
 */
export function buildSeedFullStreamTrainingCases(
  vocabulary: NeatChatVocabulary,
  seedConversationLines: readonly string[],
): Array<{ input: number[]; output: number[] }> {
  const tokenIndices: number[] = [NEATCHAT_SPECIAL_TOKEN_INDICES.BOS];

  for (const conversationLine of seedConversationLines) {
    const lineIndices = mapTextToVocabularyIndices(
      vocabulary,
      conversationLine,
    );

    if (lineIndices.length > 0) {
      tokenIndices.push(...lineIndices);
      tokenIndices.push(NEATCHAT_SPECIAL_TOKEN_INDICES.TURN_BREAK);
    }
  }

  tokenIndices.push(NEATCHAT_SPECIAL_TOKEN_INDICES.EOS);

  return buildTrainingCasesFromSequence(tokenIndices, vocabulary.size);
}

/**
 * Maps a text line to vocabulary indices within the default context window.
 *
 * @param vocabulary - Session vocabulary.
 * @param text - Text line to map.
 * @returns Vocabulary indices for the tokenized line.
 */
export function mapTextToVocabularyIndices(
  vocabulary: NeatChatVocabulary,
  text: string,
): number[] {
  const lineTokens = tokenizeNeatChatText(
    text,
    NEATCHAT_DEFAULT_CONTEXT_WINDOW_TOKEN_COUNT,
  );

  return lineTokens.map(
    (token) =>
      vocabulary.termToIndex.get(token) ?? NEATCHAT_SPECIAL_TOKEN_INDICES.UNK,
  );
}

function buildOneHotVector(index: number, size: number): number[] {
  const vector = new Array<number>(size).fill(0);

  vector[index] = 1;

  return vector;
}

function buildDefaultSessionBootstrapTerms(): readonly string[] {
  const sampleConversationText = NEATCHAT_SAMPLE_CONVERSATION_LINES.join('\n');
  const pretrainingPreview = createNeatChatPretrainingPreview({
    corpusText: sampleConversationText,
    topWordLimit: NEATCHAT_DEFAULT_SESSION_BOOTSTRAP_TOP_WORD_LIMIT,
  });

  return pretrainingPreview.retainedTerms;
}

function buildExchangeTrainingCases(
  vocabSize: number,
  userIndices: readonly number[],
  responseIndices: readonly number[],
): Array<{ input: number[]; output: number[] }> {
  const fullSequence = [
    NEATCHAT_SPECIAL_TOKEN_INDICES.BOS,
    ...userIndices,
    NEATCHAT_SPECIAL_TOKEN_INDICES.TURN_BREAK,
    ...responseIndices,
    NEATCHAT_SPECIAL_TOKEN_INDICES.EOS,
  ];

  const trainingCases: Array<{ input: number[]; output: number[] }> = [];

  for (
    let sequenceIndex = 0;
    sequenceIndex < fullSequence.length - 1;
    sequenceIndex++
  ) {
    trainingCases.push({
      input: buildOneHotVector(fullSequence[sequenceIndex]!, vocabSize),
      output: buildOneHotVector(fullSequence[sequenceIndex + 1]!, vocabSize),
    });
  }

  return trainingCases;
}

function buildObservedUserHistoryTrainingCases(
  vocabulary: NeatChatVocabulary,
  previousExchanges: readonly NeatChatExchangeRecord[],
  currentUserTokens: readonly string[],
): Array<{ input: number[]; output: number[] }> {
  const observedUserTurns = [
    ...previousExchanges.map((exchangeRecord) => exchangeRecord.userTokens),
    currentUserTokens,
  ];
  const historySequence: number[] = [NEATCHAT_SPECIAL_TOKEN_INDICES.BOS];

  for (const observedUserTurn of observedUserTurns) {
    const userTurnIndices = observedUserTurn.map(
      (token) =>
        vocabulary.termToIndex.get(token) ?? NEATCHAT_SPECIAL_TOKEN_INDICES.UNK,
    );

    if (userTurnIndices.length > 0) {
      historySequence.push(...userTurnIndices);
      historySequence.push(NEATCHAT_SPECIAL_TOKEN_INDICES.TURN_BREAK);
    }
  }

  historySequence.push(NEATCHAT_SPECIAL_TOKEN_INDICES.EOS);

  return buildTrainingCasesFromSequence(historySequence, vocabulary.size);
}

function buildReplayExchangeTrainingCases(
  vocabulary: NeatChatVocabulary,
  previousExchanges: readonly NeatChatExchangeRecord[],
): Array<{ input: number[]; output: number[] }> {
  const replayExchangeRecords = previousExchanges.slice(
    -NEATCHAT_REPLAY_BUFFER_MAX_EXCHANGES,
  );

  return replayExchangeRecords.flatMap((exchangeRecord) => {
    const userIndices = exchangeRecord.userTokens.map(
      (token) =>
        vocabulary.termToIndex.get(token) ?? NEATCHAT_SPECIAL_TOKEN_INDICES.UNK,
    );
    const responseIndices = exchangeRecord.responseTokens.map(
      (token) =>
        vocabulary.termToIndex.get(token) ?? NEATCHAT_SPECIAL_TOKEN_INDICES.UNK,
    );

    return buildExchangeTrainingCases(
      vocabulary.size,
      userIndices,
      responseIndices,
    );
  });
}

function buildAnchorReplayTrainingCases(
  vocabulary: NeatChatVocabulary,
): Array<{ input: number[]; output: number[] }> {
  const anchorTrainingCases = buildSeedAdjacentPairTrainingCases(
    vocabulary,
    NEATCHAT_ONLINE_ANCHOR_CONVERSATION_LINES,
  );

  return anchorTrainingCases.slice(
    0,
    NEATCHAT_ONLINE_ANCHOR_MAX_TRAINING_CASES,
  );
}

function resolveReplayBufferExchangeCount(exchangeCount: number): number {
  return Math.min(exchangeCount, NEATCHAT_REPLAY_BUFFER_MAX_EXCHANGES);
}

function applyOnlineLearningUpdate(
  network: Network,
  trainingCases: Array<{ input: number[]; output: number[] }>,
  averageSelectedTokenConfidence: number,
): {
  readonly updatedNetwork: Network;
  readonly committedTokenPairCount: number;
} {
  const confidenceGatePassed =
    averageSelectedTokenConfidence >= NEATCHAT_ONLINE_LEARNING_MIN_CONFIDENCE;

  if (confidenceGatePassed) {
    network.train(trainingCases, {
      iterations: NEATCHAT_ONLINE_LEARNING_ITERATIONS,
      rate: NEATCHAT_ONLINE_LEARNING_RATE,
      momentum: NEATCHAT_ONLINE_LEARNING_MOMENTUM,
      batchSize: 1,
      allowRecurrent: true,
      cost: methods.Cost.softmaxCrossEntropy,
    });

    return {
      updatedNetwork: clearSessionNetworkRuntimeState(network),
      committedTokenPairCount: trainingCases.length,
    };
  }

  if (trainingCases.length > NEATCHAT_ONLINE_LOSS_GATE_MAX_TRAINING_CASES) {
    return {
      updatedNetwork: clearSessionNetworkRuntimeState(network),
      committedTokenPairCount: 0,
    };
  }

  const sampledTrainingCases = trainingCases.slice(
    0,
    NEATCHAT_ONLINE_LOSS_GATE_SAMPLE_CASES,
  );
  const baselineError = evaluateNetworkError(network, sampledTrainingCases);
  const candidateNetwork = Network.fromJSON(
    network.toJSON() as Record<string, unknown>,
  );

  candidateNetwork.train(trainingCases, {
    iterations: NEATCHAT_ONLINE_LEARNING_ITERATIONS,
    rate: NEATCHAT_ONLINE_LEARNING_RATE,
    momentum: NEATCHAT_ONLINE_LEARNING_MOMENTUM,
    batchSize: 1,
    allowRecurrent: true,
    cost: methods.Cost.softmaxCrossEntropy,
  });
  const updatedError = evaluateNetworkError(
    candidateNetwork,
    sampledTrainingCases,
  );
  const lossImprovement = baselineError - updatedError;
  const lossGatePassed =
    lossImprovement >= NEATCHAT_ONLINE_LEARNING_MIN_LOSS_IMPROVEMENT;
  const shouldCommitUpdate = confidenceGatePassed || lossGatePassed;

  if (!shouldCommitUpdate) {
    return {
      updatedNetwork: clearSessionNetworkRuntimeState(network),
      committedTokenPairCount: 0,
    };
  }

  return {
    updatedNetwork: clearSessionNetworkRuntimeState(candidateNetwork),
    committedTokenPairCount: trainingCases.length,
  };
}

function evaluateNetworkError(
  network: Network,
  trainingCases: Array<{ input: number[]; output: number[] }>,
): number {
  const evaluationSummary = network.test(
    trainingCases,
    methods.Cost.softmaxCrossEntropy,
  );

  if (!Number.isFinite(evaluationSummary.error)) {
    return Infinity;
  }

  return evaluationSummary.error;
}

function applySessionSeedConversationTraining(
  network: Network,
  vocabulary: NeatChatVocabulary,
  seedConversationLines: readonly string[],
): number {
  if (seedConversationLines.length < 2) {
    return 0;
  }

  const fullStreamCases = buildSeedFullStreamTrainingCases(
    vocabulary,
    seedConversationLines,
  );
  const adjacentPairCases = buildSeedAdjacentPairTrainingCases(
    vocabulary,
    seedConversationLines,
  );
  const trigramCases = buildSeedTrigramTrainingCases(
    vocabulary,
    seedConversationLines,
  );
  const totalCaseCount =
    fullStreamCases.length + adjacentPairCases.length + trigramCases.length;

  network.train(fullStreamCases, {
    iterations: NEATCHAT_SEED_STREAM_TRAINING_ITERATIONS,
    rate: NEATCHAT_SEED_TRAINING_RATE,
    momentum: NEATCHAT_SEED_TRAINING_MOMENTUM,
    batchSize: 1,
    allowRecurrent: true,
    cost: methods.Cost.softmaxCrossEntropy,
  });

  network.train(adjacentPairCases, {
    iterations: NEATCHAT_SEED_LINE_TRAINING_ITERATIONS,
    rate: NEATCHAT_SEED_TRAINING_RATE,
    momentum: NEATCHAT_SEED_TRAINING_MOMENTUM,
    batchSize: 1,
    allowRecurrent: true,
    cost: methods.Cost.softmaxCrossEntropy,
  });

  network.train(trigramCases, {
    iterations: NEATCHAT_SEED_TRIGRAM_TRAINING_ITERATIONS,
    rate: NEATCHAT_SEED_TRAINING_RATE,
    momentum: NEATCHAT_SEED_TRAINING_MOMENTUM,
    batchSize: 1,
    allowRecurrent: true,
    cost: methods.Cost.softmaxCrossEntropy,
  });

  clearSessionNetworkRuntimeState(network);

  return totalCaseCount;
}

function clearSessionNetworkRuntimeState(network: Network): Network {
  network.clear();
  return network;
}

function buildSeedAdjacentPairTrainingCases(
  vocabulary: NeatChatVocabulary,
  seedConversationLines: readonly string[],
): Array<{ input: number[]; output: number[] }> {
  const trainingCases: Array<{ input: number[]; output: number[] }> = [];

  for (
    let conversationIndex = 0;
    conversationIndex < seedConversationLines.length - 1;
    conversationIndex++
  ) {
    const promptLine = seedConversationLines[conversationIndex]!;
    const replyLine = seedConversationLines[conversationIndex + 1]!;
    const promptIndices = mapTextToVocabularyIndices(vocabulary, promptLine);
    const replyIndices = mapTextToVocabularyIndices(vocabulary, replyLine);

    trainingCases.push(
      ...buildExchangeTrainingCases(
        vocabulary.size,
        promptIndices,
        replyIndices,
      ),
    );
  }

  return trainingCases;
}

/**
 * Builds extended sequence training cases from longer windows to teach better phrase patterns.
 * This helps the network learn common multi-token sequences that appear in the corpus.
 *
 * @param vocabulary - Session vocabulary.
 * @param seedConversationLines - Ordered conversation lines.
 * @returns Extended training cases for better sequence modeling.
 */
function buildSeedTrigramTrainingCases(
  vocabulary: NeatChatVocabulary,
  seedConversationLines: readonly string[],
): Array<{ input: number[]; output: number[] }> {
  const trainingCases: Array<{ input: number[]; output: number[] }> = [];

  // Process each line with extended context (look back 1 more token for better patterns)
  for (
    let conversationIndex = 0;
    conversationIndex < seedConversationLines.length - 1;
    conversationIndex++
  ) {
    const promptLine = seedConversationLines[conversationIndex]!;
    const replyLine = seedConversationLines[conversationIndex + 1]!;
    const promptIndices = mapTextToVocabularyIndices(vocabulary, promptLine);
    const replyIndices = mapTextToVocabularyIndices(vocabulary, replyLine);

    // Build extended sequence: BOS -> prompt tokens -> TURN_BREAK -> reply tokens -> EOS
    const extendedSequence = [
      NEATCHAT_SPECIAL_TOKEN_INDICES.BOS,
      ...promptIndices,
      NEATCHAT_SPECIAL_TOKEN_INDICES.TURN_BREAK,
      ...replyIndices,
      NEATCHAT_SPECIAL_TOKEN_INDICES.EOS,
    ];

    // Extract consecutive pair patterns from the extended sequence
    for (
      let sequenceIndex = 0;
      sequenceIndex < extendedSequence.length - 1;
      sequenceIndex++
    ) {
      const currentTokenIndex = extendedSequence[sequenceIndex]!;
      const nextTokenIndex = extendedSequence[sequenceIndex + 1]!;

      trainingCases.push({
        input: buildOneHotVector(currentTokenIndex, vocabulary.size),
        output: buildOneHotVector(nextTokenIndex, vocabulary.size),
      });
    }
  }

  return trainingCases;
}

function buildTrainingCasesFromSequence(
  tokenIndices: readonly number[],
  vocabSize: number,
): Array<{ input: number[]; output: number[] }> {
  const trainingCases: Array<{ input: number[]; output: number[] }> = [];

  for (
    let sequenceIndex = 0;
    sequenceIndex < tokenIndices.length - 1;
    sequenceIndex++
  ) {
    const currentTokenIndex = tokenIndices[sequenceIndex]!;
    const nextTokenIndex = tokenIndices[sequenceIndex + 1]!;

    trainingCases.push({
      input: buildOneHotVector(currentTokenIndex, vocabSize),
      output: buildOneHotVector(nextTokenIndex, vocabSize),
    });
  }

  return trainingCases;
}

function selectNextResponseTokenIndex(
  tokenScores: number[],
  vocabularySize: number,
  recentResponseIndices: readonly number[],
  canTerminate: boolean,
): number {
  let bestTokenIndex: number = NEATCHAT_SPECIAL_TOKEN_INDICES.EOS;
  let bestTokenScore = -Infinity;

  const recentWindow = recentResponseIndices.slice(
    -NEATCHAT_REPETITION_WINDOW_SIZE,
  );
  const recentIndexSet = new Set(recentWindow);

  for (let tokenIndex = 0; tokenIndex < vocabularySize; tokenIndex++) {
    if (isDisallowedResponseTokenIndex(tokenIndex, vocabularySize)) {
      continue;
    }

    // Strongly penalize early termination (single-word responses)
    if (tokenIndex === NEATCHAT_SPECIAL_TOKEN_INDICES.EOS && !canTerminate) {
      continue; // Force continuation by skipping EOS entirely when too short
    }

    let rawScore = tokenScores[tokenIndex] ?? -Infinity;

    // Apply repetition penalty for recently seen tokens
    if (recentIndexSet.has(tokenIndex)) {
      rawScore = rawScore / NEATCHAT_REPETITION_PENALTY_FACTOR;
    }

    // Apply additional penalty for EOS when response is still short (encourage length)
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
