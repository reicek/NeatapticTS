import { createNeatChatSeedNetwork } from '../index.ts';
import { createNeatChatEpisodicMemoryBank } from './neatChat.memory.services';
import {
  buildSeedFullStreamTrainingCases,
  buildNeatChatVocabulary,
  createNeatChatSession,
  inferResponseTokenIndices,
  pretrainNeatChatSessionWithConversationLines,
  runNeatChatExchange,
  updateNeatChatSessionContextWindowTokenCount,
} from './neatChat.session.services';

describe('neatChat session services', () => {
  it('buildNeatChatVocabulary keeps only the stable control tokens when no retained terms are supplied', () => {
    // Arrange
    const vocabulary = buildNeatChatVocabulary();

    // Assert
    expect(vocabulary.indexToTerm).toEqual(['UNK', 'BOS', 'EOS', 'TURN_BREAK']);
  });

  it('createNeatChatSession includes bootstrap vocabulary terms by default', () => {
    // Arrange
    const session = createNeatChatSession();

    // Assert
    expect(session.vocabulary.indexToTerm.length).toBeGreaterThan(4);
  });

  it('createNeatChatSession builds GRU and NARX sessions from the architecture branch', () => {
    // Arrange
    const gruSession = createNeatChatSession({
      architectureFamily: 'gru',
      corpusRetainedTerms: ['hello'],
      recurrentBlockSize: 4,
    });
    const narxSession = createNeatChatSession({
      architectureFamily: 'narx',
      corpusRetainedTerms: ['hello'],
      recurrentBlockSize: 4,
    });

    // Assert
    expect({
      gruVocabularySize: gruSession.vocabulary.size,
      narxVocabularySize: narxSession.vocabulary.size,
    }).toEqual({
      gruVocabularySize: 5,
      narxVocabularySize: 5,
    });
  });

  it('runNeatChatExchange falls back to the last in-vocabulary token when decoding yields no response tokens', () => {
    // Arrange
    const session = createCoverageSession({
      retainedTerms: ['hello'],
      network: createStaticScoreNetwork(5),
    });

    // Act
    const result = runNeatChatExchange(session, createRepeatedPrompt('hello', 11));

    // Assert
    expect(result.response).toBe('hello');
  });

  it('runNeatChatExchange can commit a low-confidence update when the baseline error is non-finite', () => {
    // Arrange
    const session = createCoverageSession({
      retainedTerms: ['hello'],
      network: createStaticScoreNetwork(5, { baselineError: Infinity }),
    });

    // Act
    const result = runNeatChatExchange(session, 'hello');

    // Assert
    expect(result.trainedTokenPairCount).toBeGreaterThan(0);
  });

  it('runNeatChatExchange skips low-confidence updates when the baseline error does not improve', () => {
    // Arrange
    const session = createCoverageSession({
      retainedTerms: ['hello'],
      network: createStaticScoreNetwork(5, { baselineError: 0 }),
    });

    // Act
    const result = runNeatChatExchange(session, 'hello');

    // Assert
    expect(result.trainedTokenPairCount).toBe(0);
  });

  it('runNeatChatExchange falls back to UNK when the vocabulary has no non-special terms', () => {
    // Arrange
    const session = createCoverageSession({
      retainedTerms: [],
      network: createStaticScoreNetwork(4),
    });

    // Act
    const result = runNeatChatExchange(
      session,
      createRepeatedPrompt('mystery', 11),
    );

    // Assert
    expect(result.response).toBe('UNK');
  });

  it('runNeatChatExchange returns UNK when the prompt tokenizes to no user tokens and decoding emits nothing', () => {
    // Arrange
    const session = createCoverageSession({
      retainedTerms: ['hello'],
      network: createStaticScoreNetwork(5),
    });

    // Act
    const result = runNeatChatExchange(session, '   ');

    // Assert
    expect(result.response).toBe('UNK');
  });

  it('runNeatChatExchange replays unknown prior responses even when a previous user turn is empty', () => {
    // Arrange
    const session = createCoverageSession({
      retainedTerms: ['hello'],
      network: createStaticScoreNetwork(5),
      exchanges: [
        {
          userMessage: '',
          response: 'mystery',
          trainedTokenPairCount: 0,
          userTokens: [],
          responseTokens: ['mystery'],
        },
      ],
    });

    // Act
    const result = runNeatChatExchange(session, createRepeatedPrompt('hello', 11));

    // Assert
    expect(result.updatedSession.replayBufferExchangeCount).toBe(2);
  });

  it('runNeatChatExchange replays out-of-vocabulary prior user tokens through the UNK slot', () => {
    // Arrange
    const recordingNetwork = createStaticScoreNetwork(5, {
      activationOutput: [0, 0, 0, 0, 1],
    });
    const session = createCoverageSession({
      retainedTerms: ['hello'],
      network: recordingNetwork,
      exchanges: [
        {
          userMessage: 'mystery',
          response: 'hello',
          trainedTokenPairCount: 0,
          userTokens: ['mystery'],
          responseTokens: ['hello'],
        },
      ],
    });

    // Act
    runNeatChatExchange(session, 'hello');

    // Assert
    expect(recordingNetwork.getRecordedTrainingCaseIndices().slice(5, 9)).toEqual([
      { inputTokenIndex: 1, outputTokenIndex: 0 },
      { inputTokenIndex: 0, outputTokenIndex: 3 },
      { inputTokenIndex: 3, outputTokenIndex: 4 },
      { inputTokenIndex: 4, outputTokenIndex: 2 },
    ]);
  });

  it('pretrainNeatChatSessionWithConversationLines returns the same session when no usable line pairs remain', () => {
    // Arrange
    const session = createNeatChatSession({
      corpusRetainedTerms: ['hello'],
      recurrentBlockSize: 8,
    });

    // Act
    const updatedSession = pretrainNeatChatSessionWithConversationLines(
      session,
      ['   '],
    );

    // Assert
    expect(updatedSession).toBe(session);
  });

  it('pretrainNeatChatSessionWithConversationLines increments seededTokenPairCount when usable line pairs are provided', () => {
    // Arrange
    const session = createNeatChatSession({
      corpusRetainedTerms: ['hello', 'there', 'general', 'kenobi'],
      recurrentBlockSize: 8,
    });

    // Act
    const updatedSession = pretrainNeatChatSessionWithConversationLines(
      session,
      ['hello there', 'general kenobi'],
    );

    // Assert
    expect(updatedSession.seededTokenPairCount).toBeGreaterThan(
      session.seededTokenPairCount,
    );
  });

  it('inferResponseTokenIndices returns no tokens when every candidate score is missing', () => {
    // Arrange
    const emptyScoreNetwork = createStaticScoreNetwork(5);

    // Act
    const responseIndices = inferResponseTokenIndices(
      emptyScoreNetwork as never,
      5,
      [4],
    );

    // Assert
    expect(responseIndices).toEqual([]);
  });

  it('buildSeedFullStreamTrainingCases skips blank seed lines instead of adding an empty turn break', () => {
    // Arrange
    const vocabulary = buildNeatChatVocabulary(['hello', 'world']);

    // Act
    const trainingCases = buildSeedFullStreamTrainingCases(vocabulary, [
      'hello',
      '   ',
      'world',
    ]);

    // Assert
    expect(trainingCases).toHaveLength(5);
  });

  it('updateNeatChatSessionContextWindowTokenCount updates the token window without resetting learned counts', () => {
    // Arrange
    const session = createNeatChatSession({
      corpusRetainedTerms: ['hello', 'there'],
      contextWindowTokenCount: 24,
      recurrentBlockSize: 8,
    });
    const exchangeResult = runNeatChatExchange(session, 'hello there');

    // Act
    const updatedSession = updateNeatChatSessionContextWindowTokenCount(
      exchangeResult.updatedSession,
      50,
    );

    // Assert
    expect({
      contextWindowTokenCount: updatedSession.contextWindowTokenCount,
      learnedExchangeCount: updatedSession.learnedExchangeCount,
      learnedTokenPairCount: updatedSession.learnedTokenPairCount,
    }).toEqual({
      contextWindowTokenCount: 50,
      learnedExchangeCount: exchangeResult.updatedSession.learnedExchangeCount,
      learnedTokenPairCount:
        exchangeResult.updatedSession.learnedTokenPairCount,
    });
  });

  it('updateNeatChatSessionContextWindowTokenCount returns the same session when the count is unchanged', () => {
    // Arrange
    const session = createNeatChatSession({
      corpusRetainedTerms: ['hello'],
      contextWindowTokenCount: 24,
      recurrentBlockSize: 8,
    });

    // Act
    const updatedSession = updateNeatChatSessionContextWindowTokenCount(
      session,
      24,
    );

    // Assert
    expect(updatedSession).toBe(session);
  });

  it('createNeatChatSession keeps seededTokenPairCount at 0 when fewer than two seed lines are provided', () => {
    // Arrange
    const session = createNeatChatSession({
      corpusRetainedTerms: ['hello'],
      seedConversationLines: ['hello there'],
    });

    // Assert
    expect(session.seededTokenPairCount).toBe(0);
  });
});

function createCoverageSession({
  retainedTerms,
  network,
  exchanges = [],
}: {
  readonly retainedTerms: readonly string[];
  readonly network: ReturnType<typeof createStaticScoreNetwork>;
  readonly exchanges?: Array<{
    readonly userMessage: string;
    readonly response: string;
    readonly trainedTokenPairCount: number;
    readonly userTokens: readonly string[];
    readonly responseTokens: readonly string[];
  }>;
}): Parameters<typeof runNeatChatExchange>[0] {
  const vocabulary = buildNeatChatVocabulary(retainedTerms);

  return {
    vocabulary,
    network: network as never,
    exchanges,
    learnedExchangeCount: exchanges.length,
    learnedTokenPairCount: exchanges.reduce(
      (totalCount, exchangeRecord) =>
        totalCount + exchangeRecord.trainedTokenPairCount,
      0,
    ),
    seededTokenPairCount: 0,
    contextWindowTokenCount: 24,
    replayBufferExchangeCount: exchanges.length,
    pendingCandidates: [],
    candidateLog: [],
    memoryBank: createNeatChatEpisodicMemoryBank(),
    routingLog: [],
  };
}

function createStaticScoreNetwork(
  vocabularySize: number,
  options: {
    readonly activationOutput?: readonly number[];
    readonly baselineError?: number;
  } = {},
) {
  const jsonSourceNetwork = createNeatChatSeedNetwork({
    recurrentBlockSize: 4,
    vocabularySize: Math.max(1, vocabularySize - 4),
  }).network;
  const activationOutput = options.activationOutput ?? [];
  const baselineError = options.baselineError ?? 0;
  const recordedTrainingCaseIndices: Array<{
    readonly inputTokenIndex: number;
    readonly outputTokenIndex: number;
  }> = [];

  return {
    activate: () => [...activationOutput],
    clear: () => undefined,
    getRecordedTrainingCaseIndices: () => [...recordedTrainingCaseIndices],
    test: () => ({ error: baselineError }),
    toJSON: () => jsonSourceNetwork.toJSON(),
    train: (trainingCases: Array<{ input: number[]; output: number[] }>) => {
      recordedTrainingCaseIndices.splice(
        0,
        recordedTrainingCaseIndices.length,
        ...trainingCases.map((trainingCase) => ({
          inputTokenIndex: trainingCase.input.indexOf(1),
          outputTokenIndex: trainingCase.output.indexOf(1),
        })),
      );

      return undefined;
    },
  };
}

function createRepeatedPrompt(token: string, repeatCount: number): string {
  return Array.from({ length: repeatCount }, () => token).join(' ');
}