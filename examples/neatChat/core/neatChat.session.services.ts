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
  NEATCHAT_ONLINE_LEARNING_MOMENTUM,
  NEATCHAT_ONLINE_LEARNING_RATE,
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
  const userIndices = userTokens.map(
    (token) =>
      session.vocabulary.termToIndex.get(token) ??
      NEATCHAT_SPECIAL_TOKEN_INDICES.UNK,
  );

  session.network.clear();
  const responseIndices = inferResponseTokenIndices(
    session.network,
    session.vocabulary.size,
    userIndices,
  );
  const responseTokens = responseIndices.map(
    (tokenIndex) =>
      session.vocabulary.indexToTerm[tokenIndex] ??
      NEATCHAT_DEFAULT_UNKNOWN_TOKEN,
  );
  if (responseTokens.length === 0 && userIndices.length > 0) {
    const fallbackTokenIndex =
      userIndices.at(-1) ?? NEATCHAT_SPECIAL_TOKEN_INDICES.UNK;
    const fallbackToken = resolveResponseFallbackToken(
      session.vocabulary,
      fallbackTokenIndex,
    );

    responseTokens.push(fallbackToken);
  }
  const response =
    responseTokens.length > 0
      ? responseTokens.join(' ')
      : NEATCHAT_DEFAULT_UNKNOWN_TOKEN;

  const trainingCases = buildObservedUserHistoryTrainingCases(
    session.vocabulary,
    session.exchanges,
    userTokens,
  );
  const trainedTokenPairCount = applyOnlineLearningUpdate(
    session.network,
    trainingCases,
  );

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
      exchanges: [...session.exchanges, exchangeRecord],
      learnedExchangeCount: session.learnedExchangeCount + 1,
      learnedTokenPairCount:
        session.learnedTokenPairCount + trainedTokenPairCount,
      seededTokenPairCount: session.seededTokenPairCount,
    },
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

    responseIndices.push(nextIndex);
    currentIndex = nextIndex;
  }

  return responseIndices;
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

  if (tokenIndices.length < 2) {
    return [];
  }

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

  if (index >= 0 && index < size) {
    vector[index] = 1;
  }

  return vector;
}

function resolveResponseFallbackToken(
  vocabulary: NeatChatVocabulary,
  requestedTokenIndex: number,
): string {
  const requestedToken = vocabulary.indexToTerm[requestedTokenIndex];

  if (
    requestedToken !== undefined &&
    requestedToken !== NEATCHAT_DEFAULT_UNKNOWN_TOKEN &&
    requestedToken !== 'BOS' &&
    requestedToken !== 'EOS' &&
    requestedToken !== 'TURN_BREAK'
  ) {
    return requestedToken;
  }

  const firstNonSpecialToken = vocabulary.indexToTerm.find(
    (term) =>
      term !== NEATCHAT_DEFAULT_UNKNOWN_TOKEN &&
      term !== 'BOS' &&
      term !== 'EOS' &&
      term !== 'TURN_BREAK',
  );

  return firstNonSpecialToken ?? NEATCHAT_DEFAULT_UNKNOWN_TOKEN;
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

function applyOnlineLearningUpdate(
  network: Network,
  trainingCases: Array<{ input: number[]; output: number[] }>,
): number {
  if (trainingCases.length === 0) {
    return 0;
  }

  network.train(trainingCases, {
    iterations: NEATCHAT_ONLINE_LEARNING_ITERATIONS,
    rate: NEATCHAT_ONLINE_LEARNING_RATE,
    momentum: NEATCHAT_ONLINE_LEARNING_MOMENTUM,
    batchSize: 1,
    allowRecurrent: true,
    cost: methods.Cost.softmaxCrossEntropy,
  });

  return trainingCases.length;
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

  if (totalCaseCount === 0) {
    return 0;
  }

  if (fullStreamCases.length > 0) {
    network.train(fullStreamCases, {
      iterations: NEATCHAT_SEED_STREAM_TRAINING_ITERATIONS,
      rate: NEATCHAT_SEED_TRAINING_RATE,
      momentum: NEATCHAT_SEED_TRAINING_MOMENTUM,
      batchSize: 1,
      allowRecurrent: true,
      cost: methods.Cost.softmaxCrossEntropy,
    });
  }

  if (adjacentPairCases.length > 0) {
    network.train(adjacentPairCases, {
      iterations: NEATCHAT_SEED_LINE_TRAINING_ITERATIONS,
      rate: NEATCHAT_SEED_TRAINING_RATE,
      momentum: NEATCHAT_SEED_TRAINING_MOMENTUM,
      batchSize: 1,
      allowRecurrent: true,
      cost: methods.Cost.softmaxCrossEntropy,
    });
  }

  if (trigramCases.length > 0) {
    network.train(trigramCases, {
      iterations: NEATCHAT_SEED_TRIGRAM_TRAINING_ITERATIONS,
      rate: NEATCHAT_SEED_TRAINING_RATE,
      momentum: NEATCHAT_SEED_TRAINING_MOMENTUM,
      batchSize: 1,
      allowRecurrent: true,
      cost: methods.Cost.softmaxCrossEntropy,
    });
  }

  return totalCaseCount;
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
    const promptLine = seedConversationLines[conversationIndex] ?? '';
    const replyLine = seedConversationLines[conversationIndex + 1] ?? '';
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
    const promptLine = seedConversationLines[conversationIndex] ?? '';
    const replyLine = seedConversationLines[conversationIndex + 1] ?? '';
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
    for (let i = 0; i < extendedSequence.length - 1; i++) {
      const currentIdx = extendedSequence[i];
      const nextIdx = extendedSequence[i + 1];

      if (currentIdx !== undefined && nextIdx !== undefined) {
        trainingCases.push({
          input: buildOneHotVector(currentIdx, vocabulary.size),
          output: buildOneHotVector(nextIdx, vocabulary.size),
        });
      }
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
    const currentTokenIndex = tokenIndices[sequenceIndex];
    const nextTokenIndex = tokenIndices[sequenceIndex + 1];

    if (currentTokenIndex === undefined || nextTokenIndex === undefined) {
      continue;
    }

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
