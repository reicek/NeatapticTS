import { Network, methods } from '../../../src/browser-entry.ts';
import {
  NEATCHAT_AB_DEFAULT_RECURRENT_BLOCK_SIZE,
  NEATCHAT_AB_DEFAULT_VOCAB_LIMIT,
  NEATCHAT_AB_MAX_SEED_CONVERSATION_LINES,
  NEATCHAT_AB_MAX_VALIDATION_CONVERSATION_LINES,
  NEATCHAT_AB_MAX_WARM_START_CASES,
  NEATCHAT_AB_MAX_WARM_START_LINES,
  NEATCHAT_SEED_TRAINING_MOMENTUM,
  NEATCHAT_SEED_TRAINING_RATE,
  NEATCHAT_SPECIAL_TOKEN_INDICES,
} from './neatChat.constants';
import {
  buildSeedFullStreamTrainingCases,
  createNeatChatSession,
  inferResponseTokenIndices,
  mapTextToVocabularyIndices,
  runNeatChatExchange,
} from './neatChat.session.services';
import type {
  CreateNeatChatAbComparisonOptions,
  NeatChatAbComparisonResult,
  NeatChatLightweightMetrics,
  NeatChatSession,
  NeatChatVocabulary,
} from './neatChat.types';

/**
 * Runs one prompt against blank-start and preseeded sessions and returns metrics.
 *
 * Both variants share the same prompt and vocabulary-construction settings so
 * the comparison isolates the effect of optional preseed training.
 *
 * @param prompt - Prompt evaluated by both A/B variants.
 * @param options - Vocabulary, seed, and held-out validation options.
 * @returns Ordered blank-start and preseeded variant rows with metrics.
 */
export function createNeatChatAbComparison(
  prompt: string,
  options: CreateNeatChatAbComparisonOptions = {},
): NeatChatAbComparisonResult {
  const effectiveRecurrentBlockSize =
    options.recurrentBlockSize ?? NEATCHAT_AB_DEFAULT_RECURRENT_BLOCK_SIZE;
  const effectiveLiveChatVocabLimit = Math.max(
    1,
    Math.min(
      options.liveChatVocabLimit ?? NEATCHAT_AB_DEFAULT_VOCAB_LIMIT,
      NEATCHAT_AB_DEFAULT_VOCAB_LIMIT,
    ),
  );
  const retainedTerms = (options.corpusRetainedTerms ?? []).slice(
    0,
    effectiveLiveChatVocabLimit,
  );
  const seedConversationLines = (options.seedConversationLines ?? []).slice(
    0,
    NEATCHAT_AB_MAX_SEED_CONVERSATION_LINES,
  );
  const validationConversationLines = (
    options.validationConversationLines ?? []
  ).slice(0, NEATCHAT_AB_MAX_VALIDATION_CONVERSATION_LINES);

  const blankStartSession = createNeatChatSession({
    corpusRetainedTerms: retainedTerms,
    liveChatVocabLimit: effectiveLiveChatVocabLimit,
    architectureFamily: options.architectureFamily,
    recurrentBlockSize: effectiveRecurrentBlockSize,
  });
  const preseededSession = createNeatChatSession({
    corpusRetainedTerms: retainedTerms,
    liveChatVocabLimit: effectiveLiveChatVocabLimit,
    architectureFamily: options.architectureFamily,
    recurrentBlockSize: effectiveRecurrentBlockSize,
  });
  applyAbWarmStartTraining(
    preseededSession.network,
    preseededSession.vocabulary,
    seedConversationLines,
  );
  const blankStartResult = runNeatChatExchange(blankStartSession, prompt);
  const preseededResult = runNeatChatExchange(preseededSession, prompt);

  return {
    prompt,
    variants: [
      {
        variant: 'blank-start',
        prompt,
        response: blankStartResult.response,
        responseTokens: blankStartResult.responseTokens,
        trainedTokenPairCount: blankStartResult.trainedTokenPairCount,
        metrics: buildLightweightMetrics(
          blankStartResult.updatedSession,
          blankStartResult.responseTokens,
          validationConversationLines,
        ),
      },
      {
        variant: 'preseeded',
        prompt,
        response: preseededResult.response,
        responseTokens: preseededResult.responseTokens,
        trainedTokenPairCount: preseededResult.trainedTokenPairCount,
        metrics: buildLightweightMetrics(
          preseededResult.updatedSession,
          preseededResult.responseTokens,
          validationConversationLines,
        ),
      },
    ],
  };
}

function argmaxIndex(values: number[]): number {
  let bestIndex = 0;
  let bestValue = values[0] ?? -Infinity;

  for (let valueIndex = 1; valueIndex < values.length; valueIndex++) {
    const currentValue = values[valueIndex] ?? -Infinity;

    if (currentValue > bestValue) {
      bestValue = currentValue;
      bestIndex = valueIndex;
    }
  }

  return bestIndex;
}

function buildLightweightMetrics(
  session: NeatChatSession,
  responseTokens: readonly string[],
  validationConversationLines: readonly string[],
): NeatChatLightweightMetrics {
  return {
    heldOutNextTokenAccuracy: evaluateHeldOutNextTokenAccuracy(
      session,
      validationConversationLines,
    ),
    repetitionRate: evaluateResponseRepetitionRate(responseTokens),
    responseLengthStability: evaluateResponseLengthStability(
      session,
      responseTokens,
      validationConversationLines,
    ),
  };
}

function evaluateHeldOutNextTokenAccuracy(
  session: NeatChatSession,
  validationConversationLines: readonly string[],
): number {
  const validationCases = buildSeedFullStreamTrainingCases(
    session.vocabulary,
    validationConversationLines,
  ).filter((validationCase) => {
    const expectedTokenIndex = argmaxIndex(validationCase.output);

    return !isSpecialTokenIndex(expectedTokenIndex);
  });

  if (validationCases.length === 0) {
    return 0;
  }

  session.network.clear();

  let correctPredictionCount = 0;

  for (const validationCase of validationCases) {
    const prediction = session.network.activate(validationCase.input);
    const predictedIndex = argmaxIndex(prediction);
    const expectedIndex = argmaxIndex(validationCase.output);

    if (predictedIndex === expectedIndex) {
      correctPredictionCount += 1;
    }
  }

  session.network.clear();

  const heldOutAccuracyPercent =
    (correctPredictionCount / validationCases.length) * 100;

  return Number(heldOutAccuracyPercent.toFixed(2));

  function isSpecialTokenIndex(tokenIndex: number): boolean {
    return (
      tokenIndex === NEATCHAT_SPECIAL_TOKEN_INDICES.UNK ||
      tokenIndex === NEATCHAT_SPECIAL_TOKEN_INDICES.BOS ||
      tokenIndex === NEATCHAT_SPECIAL_TOKEN_INDICES.EOS ||
      tokenIndex === NEATCHAT_SPECIAL_TOKEN_INDICES.TURN_BREAK
    );
  }
}

function evaluateResponseRepetitionRate(
  responseTokens: readonly string[],
): number {
  if (responseTokens.length < 2) {
    return 0;
  }

  let repeatedAdjacentPairCount = 0;

  for (
    let responseTokenIndex = 1;
    responseTokenIndex < responseTokens.length;
    responseTokenIndex++
  ) {
    const previousToken = responseTokens[responseTokenIndex - 1];
    const currentToken = responseTokens[responseTokenIndex];

    if (previousToken === currentToken) {
      repeatedAdjacentPairCount += 1;
    }
  }

  const adjacentPairCount = responseTokens.length - 1;

  return Number((repeatedAdjacentPairCount / adjacentPairCount).toFixed(4));
}

function evaluateResponseLengthStability(
  session: NeatChatSession,
  responseTokens: readonly string[],
  validationConversationLines: readonly string[],
): number {
  const probePromptLines = validationConversationLines.slice(0, 3);
  const responseLengths = [responseTokens.length];

  for (const probePromptLine of probePromptLines) {
    const probeInputIndices = mapTextToVocabularyIndices(
      session.vocabulary,
      probePromptLine,
    );

    session.network.clear();

    const probeResponseIndices = inferResponseTokenIndices(
      session.network,
      session.vocabulary.size,
      probeInputIndices,
    );

    responseLengths.push(probeResponseIndices.length);
  }

  session.network.clear();

  const meanResponseLength =
    responseLengths.reduce(
      (total, responseLength) => total + responseLength,
      0,
    ) / responseLengths.length;
  const meanAbsoluteDeviation =
    responseLengths.reduce(
      (total, responseLength) =>
        total + Math.abs(responseLength - meanResponseLength),
      0,
    ) / responseLengths.length;
  const stabilityScore = 1 / (1 + meanAbsoluteDeviation);

  return Number(stabilityScore.toFixed(4));
}

function applyAbWarmStartTraining(
  network: Network,
  vocabulary: NeatChatVocabulary,
  seedConversationLines: readonly string[],
): number {
  const warmStartLines = seedConversationLines.slice(
    0,
    NEATCHAT_AB_MAX_WARM_START_LINES,
  );

  if (warmStartLines.length < 2) {
    return 0;
  }

  const warmStartCases = buildSeedFullStreamTrainingCases(
    vocabulary,
    warmStartLines,
  ).slice(0, NEATCHAT_AB_MAX_WARM_START_CASES);

  if (warmStartCases.length === 0) {
    return 0;
  }

  network.train(warmStartCases, {
    iterations: 1,
    rate: NEATCHAT_SEED_TRAINING_RATE,
    momentum: NEATCHAT_SEED_TRAINING_MOMENTUM,
    batchSize: 1,
    allowRecurrent: true,
    cost: methods.Cost.softmaxCrossEntropy,
  });

  return warmStartCases.length;
}
