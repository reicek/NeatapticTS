import {
  buildNeatChatVocabulary,
  createNeatChatExampleContract,
  createNeatChatAbComparison,
  createNeatChatPretrainingPreview,
  createNeatChatSeedNetwork,
  createNeatChatSession,
  exportNeatChatSession,
  formatNeatChatExampleContract,
  importNeatChatSession,
  pretrainNeatChatSessionWithConversationLines,
  updateNeatChatSessionContextWindowTokenCount,
  extractNeatChatConversationLines,
  getNeatChatSampleConversationLines,
  estimateNeatChatRuntime,
  runNeatChatExchange,
  splitNeatChatSeedAndValidationLines,
  tokenizeNeatChatText,
} from './index';
import {
  buildSeedFullStreamTrainingCases,
  inferResponseTokenIndices,
  mapTextToVocabularyIndices,
} from './core/neatChat.session.services';
import { createNeatChatEpisodicMemoryBank } from './core/neatChat.memory.services';
import {
  NeatChatSnapshotShapeError,
  NeatChatSnapshotVersionError,
} from './core/neatChat.errors';

describe('neatChat public contract behavior', () => {
  it('exposes a bounded sequence-learning contract with Flappy visualizer reuse metadata', () => {
    // Arrange
    const exampleContract = createNeatChatExampleContract();

    // Assert
    expect({
      exampleId: exampleContract.exampleId,
      defaultArchitectureFamily: exampleContract.defaultArchitectureFamily,
      publishedExamplesCategory: exampleContract.publishedExamplesCategory,
      publicBrowserEntrypointModulePath:
        exampleContract.publicBrowserEntrypointModulePath,
      publicBrowserHostPagePath: exampleContract.publicBrowserHostPagePath,
      publishedDocsExamplePath: exampleContract.publishedDocsExamplePath,
      supportedArchitectureFamilies:
        exampleContract.supportedArchitectureFamilies,
      defaultTopWordLimit: exampleContract.pretraining.defaultTopWordLimit,
      recommendedTopWordLimitRange:
        exampleContract.pretraining.recommendedTopWordLimitRange,
      defaultContextWindowTokenCount:
        exampleContract.pretraining.defaultContextWindowTokenCount,
      unknownToken: exampleContract.pretraining.unknownToken,
      tokenizationRule: exampleContract.pretraining.tokenizationRule,
      defaultRuntimeEstimate:
        exampleContract.pretraining.defaultRuntimeEstimate,
      corpusReportFieldKeys: exampleContract.pretraining.corpusReportFields.map(
        (field) => field.key,
      ),
      abVariants: exampleContract.abComparison.variants,
      lightweightMetrics: exampleContract.abComparison.lightweightMetrics,
      visualizationExampleId: exampleContract.visualization.exampleId,
      visualizationOwnerModulePath:
        exampleContract.visualization.ownerModulePath,
      visualizationFrameResolverModulePath:
        exampleContract.visualization.frameResolverModulePath,
    }).toEqual({
      exampleId: 'NEATchat',
      defaultArchitectureFamily: 'lstm',
      publishedExamplesCategory: 'flagship',
      publicBrowserEntrypointModulePath: 'examples/neatChat/browser-entry.ts',
      publicBrowserHostPagePath: 'examples/neatChat/index.html',
      publishedDocsExamplePath: 'docs/examples/neatChat/index.html',
      supportedArchitectureFamilies: ['lstm', 'gru', 'narx'],
      defaultTopWordLimit: 3000,
      recommendedTopWordLimitRange: [300, 5000],
      defaultContextWindowTokenCount: 24,
      unknownToken: 'UNK',
      tokenizationRule:
        "Lowercase text, normalize common contractions (for example, can't to can not), collapse punctuation into compact class tokens, bucket numeric terms by size, truncate each prompt or reply slice to the short context window, and map out-of-vocabulary terms to UNK once the retained vocabulary is known.",
      defaultRuntimeEstimate: {
        topWordLimit: 3000,
        specialTokenCount: 4,
        estimatedRetainedVocabularySize: 3004,
        contextWindowTokenCount: 24,
        expectedPretrainingDurationBucket: 'moderate',
        summary:
          'Retain about 3004 tokens including UNK, BOS, EOS, TURN_BREAK, keep each prompt or reply slice to 24 tokens, and expect a moderate bounded pretraining pass at this scale.',
      },
      corpusReportFieldKeys: [
        'characterCount',
        'tokenCount',
        'uniqueTermCount',
        'retainedTermCount',
        'retainedTokenCoveragePercent',
      ],
      abVariants: ['blank-start', 'preseeded'],
      lightweightMetrics: [
        'held-out-next-token-accuracy',
        'repetition-rate',
        'response-length-stability',
      ],
      visualizationExampleId: 'flappy_bird',
      visualizationOwnerModulePath:
        'examples/flappy_bird/browser-entry/host/host.ts',
      visualizationFrameResolverModulePath:
        'examples/flappy_bird/browser-entry/network-view/network-view.ts',
    });
  });

  it('builds a small default seed network from the public LSTM entrypoint', () => {
    // Arrange
    const seedNetworkResult = createNeatChatSeedNetwork({ vocabularySize: 32 });

    // Assert
    expect({
      architectureFamily: seedNetworkResult.summary.architectureFamily,
      effectiveVocabularySize:
        seedNetworkResult.summary.effectiveVocabularySize,
      inputCount: seedNetworkResult.summary.inputCount,
      outputCount: seedNetworkResult.summary.outputCount,
      includesUnknownToken: seedNetworkResult.summary.includesUnknownToken,
      specialTokens: seedNetworkResult.summary.specialTokens,
      topologyIntent: seedNetworkResult.summary.topologyIntent,
    }).toEqual({
      architectureFamily: 'lstm',
      effectiveVocabularySize: 36,
      inputCount: 36,
      outputCount: 36,
      includesUnknownToken: true,
      specialTokens: ['UNK', 'BOS', 'EOS', 'TURN_BREAK'],
      topologyIntent: 'unconstrained',
    });
  });

  it('builds GRU and NARX seed networks from the public builder entrypoint', () => {
    // Arrange
    const gruSeedNetworkResult = createNeatChatSeedNetwork({
      architectureFamily: 'gru',
      recurrentBlockSize: 4,
      vocabularySize: 8,
    });
    const narxSeedNetworkResult = createNeatChatSeedNetwork({
      architectureFamily: 'narx',
      recurrentBlockSize: 4,
      vocabularySize: 8,
    });

    // Assert
    expect({
      gruArchitectureFamily: gruSeedNetworkResult.summary.architectureFamily,
      narxArchitectureFamily: narxSeedNetworkResult.summary.architectureFamily,
    }).toEqual({
      gruArchitectureFamily: 'gru',
      narxArchitectureFamily: 'narx',
    });
  });

  it('shares tokenizer and runtime estimate across Node and browser previews', () => {
    // Arrange
    const runtimeEstimate = estimateNeatChatRuntime({ topWordLimit: 1200 });
    const tokenizedPrompt = tokenizeNeatChatText(
      'One two three four five six',
      4,
    );

    // Assert
    expect({
      expectedPretrainingDurationBucket:
        runtimeEstimate.expectedPretrainingDurationBucket,
      estimatedRetainedVocabularySize:
        runtimeEstimate.estimatedRetainedVocabularySize,
      contextWindowTokenCount: runtimeEstimate.contextWindowTokenCount,
      tokenizedPrompt,
    }).toEqual({
      expectedPretrainingDurationBucket: 'short',
      estimatedRetainedVocabularySize: 1204,
      contextWindowTokenCount: 24,
      tokenizedPrompt: ['one', 'two', 'three', 'four'],
    });
  });

  it('tokenizeNeatChatText expands common contractions into stable semantic terms', () => {
    // Arrange
    const promptText = "I can't wait";

    // Act
    const tokenizedPrompt = tokenizeNeatChatText(promptText, 12);

    // Assert
    expect(tokenizedPrompt).toEqual(['i', 'can', 'not', 'wait']);
  });

  it('tokenizeNeatChatText emits punctuation class tokens for sentence boundaries', () => {
    // Arrange
    const promptText = 'hello, world!';

    // Act
    const tokenizedPrompt = tokenizeNeatChatText(promptText, 12);

    // Assert
    expect(tokenizedPrompt).toEqual([
      'hello',
      'PUNC_PAUSE',
      'world',
      'PUNC_SENTENCE_END',
    ]);
  });

  it('tokenizeNeatChatText buckets raw numbers into compact semantic ranges', () => {
    // Arrange
    const promptText = '7 42 2026';

    // Act
    const tokenizedPrompt = tokenizeNeatChatText(promptText, 12);

    // Assert
    expect(tokenizedPrompt).toEqual(['NUM_SMALL', 'NUM_MEDIUM', 'NUM_LARGE']);
  });

  it('builds an optional chunked pretraining preview from pasted text', () => {
    // Arrange
    const pretrainingPreview = createNeatChatPretrainingPreview({
      corpusText: 'star sun star moon star sun',
      topWordLimit: 2,
      chunkTokenCount: 2,
      previewTermCount: 2,
    });

    // Assert
    expect({
      hasCorpus: pretrainingPreview.hasCorpus,
      characterCount: pretrainingPreview.characterCount,
      totalTokenCount: pretrainingPreview.totalTokenCount,
      uniqueTermCount: pretrainingPreview.uniqueTermCount,
      processedChunkCount: pretrainingPreview.processedChunkCount,
      retainedTermCount: pretrainingPreview.retainedTermCount,
      retainedTokenCoveragePercent:
        pretrainingPreview.retainedTokenCoveragePercent,
      corpusReport: pretrainingPreview.corpusReport,
      previewTerms: pretrainingPreview.previewTerms,
    }).toEqual({
      hasCorpus: true,
      characterCount: 27,
      totalTokenCount: 6,
      uniqueTermCount: 3,
      processedChunkCount: 3,
      retainedTermCount: 2,
      retainedTokenCoveragePercent: 83.33,
      corpusReport: {
        characterCount: 27,
        tokenCount: 6,
        uniqueTermCount: 3,
        retainedTermCount: 2,
        retainedTokenCoveragePercent: 83.33,
      },
      previewTerms: ['star', 'sun'],
    });
  });

  it('returns retainedTerms alongside previewTerms in the pretraining preview', () => {
    // Arrange
    const pretrainingPreview = createNeatChatPretrainingPreview({
      corpusText: 'star sun star moon star sun',
      topWordLimit: 2,
      previewTermCount: 1,
    });

    // Assert
    expect(pretrainingPreview.retainedTerms).toEqual(['star', 'sun']);
  });

  it('formats the contract preview with corpus labels and delivery progress', () => {
    // Arrange
    const exampleContract = createNeatChatExampleContract();
    const seedNetworkSummary = createNeatChatSeedNetwork({
      vocabularySize: 8,
    }).summary;

    // Act
    const formattedContract = formatNeatChatExampleContract(
      exampleContract,
      seedNetworkSummary,
    );

    // Assert
    expect(
      formattedContract.includes('Character count') &&
        formattedContract.includes(
          'available: Step 1 - Define the example contract',
        ) &&
        formattedContract.includes('flappy_bird'),
    ).toBe(true);
  });
});

const COMPACT_ONLINE_LEARNING_RETAINED_TERMS = [
  'hello',
  'world',
  'how',
  'are',
  'you',
  'there',
] as const;

function createCompactSession(
  sessionOptions: Parameters<typeof createNeatChatSession>[0] = {},
) {
  return createNeatChatSession({
    corpusRetainedTerms: COMPACT_ONLINE_LEARNING_RETAINED_TERMS,
    recurrentBlockSize: 8,
    ...sessionOptions,
  });
}

describe('neatChat online learning behavior', () => {
  describe('vocabulary mapping', () => {
    it('buildNeatChatVocabulary keeps only the stable control tokens when no retained terms are supplied', () => {
      // Arrange
      const vocabulary = buildNeatChatVocabulary();

      // Assert
      expect(vocabulary.indexToTerm).toEqual([
        'UNK',
        'BOS',
        'EOS',
        'TURN_BREAK',
      ]);
    });

    it('buildNeatChatVocabulary places special tokens at stable indices 0–3', () => {
      // Arrange
      const vocabulary = buildNeatChatVocabulary(['hello', 'world']);

      // Assert
      expect(vocabulary.indexToTerm[0]).toBe('UNK');
    });

    it('buildNeatChatVocabulary maps retained terms starting at index 4', () => {
      // Arrange
      const vocabulary = buildNeatChatVocabulary(['hello', 'world']);

      // Assert
      expect(vocabulary.termToIndex.get('hello')).toBe(4);
    });

    it('buildNeatChatVocabulary reports the correct total size', () => {
      // Arrange
      const vocabulary = buildNeatChatVocabulary(['hello', 'world']);

      // Assert
      expect(vocabulary.size).toBe(6);
    });
  });

  describe('session updates and exchange outcomes', () => {
    it('createNeatChatSession builds a session with expected vocabulary size', () => {
      // Arrange
      const session = createNeatChatSession({
        corpusRetainedTerms: ['hello', 'world', 'foo'],
      });

      // Assert
      expect(session.vocabulary.size).toBe(7);
    });

    it('createNeatChatSession starts with zero exchanges', () => {
      // Arrange
      const session = createCompactSession();

      // Assert
      expect(session.exchanges.length).toBe(0);
    });

    it('createNeatChatSession starts with zero learned exchange count', () => {
      // Arrange
      const session = createCompactSession();

      // Assert
      expect(session.learnedExchangeCount).toBe(0);
    });

    it('createNeatChatSession builds GRU and NARX sessions from the session-service builder branch', () => {
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

    it('createNeatChatSession uses provided contextWindowTokenCount', () => {
      // Arrange
      const session = createCompactSession({ contextWindowTokenCount: 50 });

      // Assert
      expect(session.contextWindowTokenCount).toBe(50);
    });

    it('runNeatChatExchange tokenization respects session contextWindowTokenCount', () => {
      // Arrange
      const session = createCompactSession({ contextWindowTokenCount: 2 });

      // Act
      const result = runNeatChatExchange(session, 'one two three four');

      // Assert
      expect(result.userTokens).toEqual(['one', 'two']);
    });

    it('updateNeatChatSessionContextWindowTokenCount updates without resetting learned counts', () => {
      // Arrange
      const session = createCompactSession({ contextWindowTokenCount: 24 });
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
        learnedExchangeCount:
          exchangeResult.updatedSession.learnedExchangeCount,
        learnedTokenPairCount:
          exchangeResult.updatedSession.learnedTokenPairCount,
      });
    });

    it('updateNeatChatSessionContextWindowTokenCount returns the same session when the count is unchanged', () => {
      // Arrange
      const session = createCompactSession({ contextWindowTokenCount: 24 });

      // Act
      const updatedSession = updateNeatChatSessionContextWindowTokenCount(
        session,
        24,
      );

      // Assert
      expect(updatedSession).toBe(session);
    });

    it('createNeatChatSession includes non-special bootstrap terms by default', () => {
      // Arrange
      const session = createNeatChatSession();

      // Assert
      expect(session.vocabulary.size).toBeGreaterThan(4);
    });

    it('runNeatChatExchange does not emit an UNK-only response in default sessions', () => {
      // Arrange
      const session = createNeatChatSession();

      // Act
      const result = runNeatChatExchange(session, 'hello');

      // Assert
      expect(/^UNK(?: UNK)*$/.test(result.response)).toBe(false);
    });

    it('runNeatChatExchange returns a non-empty response string', () => {
      // Arrange
      const session = createCompactSession();

      // Act
      const result = runNeatChatExchange(session, 'hello');

      // Assert
      expect(result.response.length).toBeGreaterThan(0);
    });

    it('runNeatChatExchange increments learnedExchangeCount by 1', () => {
      // Arrange
      const session = createCompactSession();

      // Act
      const result = runNeatChatExchange(session, 'hello');

      // Assert
      expect(result.updatedSession.learnedExchangeCount).toBe(1);
    });

    it('runNeatChatExchange reports a positive trainedTokenPairCount', () => {
      // Arrange
      const session = createNeatChatSession({
        corpusRetainedTerms: ['hello', 'world'],
      });

      // Act
      const result = runNeatChatExchange(session, 'hello world');

      // Assert
      expect(result.trainedTokenPairCount).toBeGreaterThan(0);
    });

    it('runNeatChatExchange appends the exchange record to the session history', () => {
      // Arrange
      const session = createCompactSession();

      // Act
      const firstResult = runNeatChatExchange(session, 'hello');
      const secondResult = runNeatChatExchange(
        firstResult.updatedSession,
        'world',
      );

      // Assert
      expect(secondResult.updatedSession.exchanges.length).toBe(2);
    });

    it('runNeatChatExchange accumulates learnedTokenPairCount across multiple exchanges', () => {
      // Arrange
      const session = createNeatChatSession({
        corpusRetainedTerms: ['hello', 'world'],
      });

      // Act
      const firstResult = runNeatChatExchange(session, 'hello');
      const secondResult = runNeatChatExchange(
        firstResult.updatedSession,
        'world',
      );

      // Assert
      expect(secondResult.updatedSession.learnedTokenPairCount).toBeGreaterThan(
        firstResult.updatedSession.learnedTokenPairCount,
      );
    });

    it('runNeatChatExchange includes observed-turn cases plus the compact anchor replay set', () => {
      // Arrange
      const session = createNeatChatSession({
        corpusRetainedTerms: ['hello', 'world'],
      });

      // Act
      const result = runNeatChatExchange(session, 'hello world');

      // Assert
      expect(result.trainedTokenPairCount).toBe(8);
    });

    it('runNeatChatExchange may skip low-confidence commits while preserving valid non-negative pair counts', () => {
      // Arrange
      const session = createNeatChatSession({
        corpusRetainedTerms: ['hello', 'world', 'how', 'are', 'you'],
      });

      // Act
      const firstResult = runNeatChatExchange(session, 'hello');
      const secondResult = runNeatChatExchange(
        firstResult.updatedSession,
        'how are you',
      );

      // Assert
      expect({
        firstIsNonNegativeInteger:
          Number.isInteger(firstResult.trainedTokenPairCount) &&
          firstResult.trainedTokenPairCount >= 0,
        secondIsNonNegativeInteger:
          Number.isInteger(secondResult.trainedTokenPairCount) &&
          secondResult.trainedTokenPairCount >= 0,
      }).toEqual({
        firstIsNonNegativeInteger: true,
        secondIsNonNegativeInteger: true,
      });
    });

    it('extractNeatChatConversationLines parses JSON arrays into trimmed lines', () => {
      // Arrange
      const conversationDump = '[" hi ", "there", " ", "friend "]';

      // Act
      const lines = extractNeatChatConversationLines(conversationDump);

      // Assert
      expect(lines).toEqual(['hi', 'there', 'friend']);
    });

    it('extractNeatChatConversationLines falls back to non-empty plain text lines', () => {
      // Arrange
      const plainText = 'hello there\n\n general kenobi\n';

      // Act
      const lines = extractNeatChatConversationLines(plainText);

      // Assert
      expect(lines).toEqual(['hello there', 'general kenobi']);
    });

    it('createNeatChatSession records seededTokenPairCount when seeded conversation lines are provided', () => {
      // Arrange
      const session = createNeatChatSession({
        corpusRetainedTerms: ['hello', 'there', 'general', 'kenobi'],
        seedConversationLines: ['hello there', 'general kenobi'],
      });

      // Assert
      expect(session.seededTokenPairCount).toBeGreaterThan(0);
    });

    it('pretrainNeatChatSessionWithConversationLines increments seededTokenPairCount in-place', () => {
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

    it('pretrainNeatChatSessionWithConversationLines preserves learnedExchangeCount', () => {
      // Arrange
      const seededSession = createNeatChatSession({
        corpusRetainedTerms: ['hello', 'there', 'general', 'kenobi'],
        recurrentBlockSize: 8,
      });
      const exchangeResult = runNeatChatExchange(seededSession, 'hello there');

      // Act
      const updatedSession = pretrainNeatChatSessionWithConversationLines(
        exchangeResult.updatedSession,
        ['general kenobi', 'hello there'],
      );

      // Assert
      expect(updatedSession.learnedExchangeCount).toBe(
        exchangeResult.updatedSession.learnedExchangeCount,
      );
    });

    it('pretrainNeatChatSessionWithConversationLines returns the same session when no usable line pairs remain', () => {
      // Arrange
      const session = createCompactSession();

      // Act
      const updatedSession = pretrainNeatChatSessionWithConversationLines(
        session,
        ['   '],
      );

      // Assert
      expect(updatedSession).toBe(session);
    });

    it('runNeatChatExchange does not emit TURN_BREAK in decoded response tokens', () => {
      // Arrange — tiny fixture with small recurrent block keeps seeding under 2 s
      const sampleConversationLines =
        getNeatChatSampleConversationLines().slice(0, 12);
      const sampleConversationText = sampleConversationLines.join('\n');
      const pretrainingPreview = createNeatChatPretrainingPreview({
        corpusText: sampleConversationText,
        topWordLimit: 20,
      });
      const splitResult = splitNeatChatSeedAndValidationLines(
        sampleConversationLines,
        2,
      );
      const session = createNeatChatSession({
        corpusRetainedTerms: pretrainingPreview.retainedTerms,
        seedConversationLines: splitResult.seedConversationLines,
        liveChatVocabLimit: 20,
        recurrentBlockSize: 8,
      });

      // Act
      const result = runNeatChatExchange(session, 'good morning');

      // Assert
      expect(result.responseTokens.includes('TURN_BREAK')).toBe(false);
    });

    it('runNeatChatExchange does not collapse to UNK-only response after scripted pretraining', () => {
      // Arrange — tiny fixture with small recurrent block keeps seeding under 2 s;
      // prompt uses 'hi' which is the first corpus word and always in the retained vocabulary
      const sampleConversationLines =
        getNeatChatSampleConversationLines().slice(0, 12);
      const sampleConversationText = sampleConversationLines.join('\n');
      const pretrainingPreview = createNeatChatPretrainingPreview({
        corpusText: sampleConversationText,
        topWordLimit: 20,
      });
      const splitResult = splitNeatChatSeedAndValidationLines(
        sampleConversationLines,
        2,
      );
      const session = createNeatChatSession({
        corpusRetainedTerms: pretrainingPreview.retainedTerms,
        seedConversationLines: splitResult.seedConversationLines,
        liveChatVocabLimit: 20,
        recurrentBlockSize: 8,
      });

      // Act — use the first retained term so the prompt is unconditionally in-vocab
      const inVocabPrompt = pretrainingPreview.retainedTerms[0] ?? 'hi';
      const result = runNeatChatExchange(session, inVocabPrompt);

      // Assert — with an in-vocab prompt and UNK blocked for non-trivial vocabularies,
      // the response must not be UNK
      expect(result.response).not.toBe('UNK');
    });

    it('runNeatChatExchange falls back to the last in-vocabulary token when decoding yields no response tokens', () => {
      // Arrange
      const session = createCoverageSession({
        retainedTerms: ['hello'],
        network: createStaticScoreNetwork(5),
      });

      // Act
      const result = runNeatChatExchange(
        session,
        createRepeatedPrompt('hello', 11),
      );

      // Assert
      expect(result.response).toBe('hello');
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
      const result = runNeatChatExchange(
        session,
        createRepeatedPrompt('hello', 11),
      );

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
      expect(
        recordingNetwork.getRecordedTrainingCaseIndices().slice(5, 9),
      ).toEqual([
        { inputTokenIndex: 1, outputTokenIndex: 0 },
        { inputTokenIndex: 0, outputTokenIndex: 3 },
        { inputTokenIndex: 3, outputTokenIndex: 4 },
        { inputTokenIndex: 4, outputTokenIndex: 2 },
      ]);
    });
  });
});

describe('neatChat helper edge cases', () => {
  it('splitNeatChatSeedAndValidationLines keeps every trimmed line in validation when the corpus is too small', () => {
    // Arrange
    const smallConversationLines = [' hello ', '', 'there'];

    // Act
    const splitResult = splitNeatChatSeedAndValidationLines(
      smallConversationLines,
      4,
    );

    // Assert
    expect(splitResult).toEqual({
      seedConversationLines: [],
      validationConversationLines: ['hello', 'there'],
    });
  });

  it('mapTextToVocabularyIndices maps out-of-vocabulary tokens to the UNK index', () => {
    // Arrange
    const vocabulary = buildNeatChatVocabulary(['hello']);

    // Act
    const tokenIndices = mapTextToVocabularyIndices(
      vocabulary,
      'hello mystery',
    );

    // Assert
    expect(tokenIndices).toEqual([4, 0]);
  });

  it('buildSeedFullStreamTrainingCases keeps the BOS-to-EOS transition even when no seed lines are supplied', () => {
    // Arrange
    const vocabulary = buildNeatChatVocabulary(['hello']);

    // Act
    const trainingCases = buildSeedFullStreamTrainingCases(vocabulary, []);

    // Assert
    expect(trainingCases).toHaveLength(1);
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
});

describe('neatChat A/B evaluation behavior', () => {
  describe('comparison outputs', () => {
    it('getNeatChatSampleConversationLines exposes a large reusable scripted talk corpus', () => {
      // Arrange
      const sampleConversationLines = getNeatChatSampleConversationLines();

      // Assert
      expect(sampleConversationLines.length).toBeGreaterThan(150);
    });

    it('createNeatChatPretrainingPreview extracts substantial token and term coverage from scripted talk', () => {
      // Arrange
      const sampleConversationLines = getNeatChatSampleConversationLines();
      const sampleConversationText = sampleConversationLines.join('\n');

      // Act
      const pretrainingPreview = createNeatChatPretrainingPreview({
        corpusText: sampleConversationText,
        topWordLimit: 3000,
      });

      // Assert
      expect({
        hasCorpus: pretrainingPreview.hasCorpus,
        tokenCount: pretrainingPreview.totalTokenCount,
        uniqueTermCount: pretrainingPreview.uniqueTermCount,
        retainedTokenCoveragePercent:
          pretrainingPreview.retainedTokenCoveragePercent,
      }).toEqual({
        hasCorpus: true,
        tokenCount: expect.any(Number),
        uniqueTermCount: expect.any(Number),
        retainedTokenCoveragePercent: 100,
      });
    });

    it('splitNeatChatSeedAndValidationLines reserves six held-out lines by default for scripted talk', () => {
      // Arrange
      const sampleConversationLines = getNeatChatSampleConversationLines();

      // Act
      const splitResult = splitNeatChatSeedAndValidationLines(
        sampleConversationLines,
      );

      // Assert
      expect({
        seedLineCount: splitResult.seedConversationLines.length,
        validationLineCount: splitResult.validationConversationLines.length,
      }).toEqual({
        seedLineCount: sampleConversationLines.length - 6,
        validationLineCount: 6,
      });
    });

    it('createNeatChatAbComparison computes metrics using scripted talk-driven seed and validation slices', () => {
      // Arrange
      const sampleConversationLines = getNeatChatSampleConversationLines();
      const sampleConversationText = sampleConversationLines.join('\n');
      const pretrainingPreview = createNeatChatPretrainingPreview({
        corpusText: sampleConversationText,
        topWordLimit: 3000,
      });
      const splitResult = splitNeatChatSeedAndValidationLines(
        sampleConversationLines,
      );

      // Act
      const comparisonResult = createNeatChatAbComparison('how was your day', {
        corpusRetainedTerms: pretrainingPreview.retainedTerms,
        seedConversationLines: splitResult.seedConversationLines,
        validationConversationLines: splitResult.validationConversationLines,
      });

      // Assert
      expect(
        comparisonResult.variants.every(
          (variantResult) =>
            variantResult.metrics.heldOutNextTokenAccuracy >= 0,
        ),
      ).toBe(true);
    });

    it('splitNeatChatSeedAndValidationLines keeps a small held-out validation slice', () => {
      // Arrange
      const conversationLines = [
        'line one',
        'line two',
        'line three',
        'line four',
        'line five',
        'line six',
        'line seven',
        'line eight',
      ];

      // Act
      const splitResult = splitNeatChatSeedAndValidationLines(
        conversationLines,
        3,
      );

      // Assert
      expect(splitResult).toEqual({
        seedConversationLines: [
          'line one',
          'line two',
          'line three',
          'line four',
          'line five',
        ],
        validationConversationLines: ['line six', 'line seven', 'line eight'],
      });
    });

    it('createNeatChatAbComparison returns both blank-start and preseeded variants for the same prompt', () => {
      // Arrange
      const conversationLines = [
        'hi there',
        'hello friend',
        'how are you',
        'doing well today',
        'let us walk',
        'sounds great',
      ];
      const splitResult = splitNeatChatSeedAndValidationLines(
        conversationLines,
        2,
      );

      // Act
      const comparisonResult = createNeatChatAbComparison('hello there', {
        corpusRetainedTerms: [
          'hi',
          'there',
          'hello',
          'friend',
          'how',
          'are',
          'you',
        ],
        seedConversationLines: splitResult.seedConversationLines,
        validationConversationLines: splitResult.validationConversationLines,
      });

      // Assert
      expect(
        comparisonResult.variants.map((variantResult) => variantResult.variant),
      ).toEqual(['blank-start', 'preseeded']);
    });

    it('createNeatChatAbComparison reports held-out next-token accuracy for each variant', () => {
      // Arrange
      const conversationLines = [
        'hi there',
        'hello friend',
        'how are you',
        'doing well today',
        'let us walk',
        'sounds great',
      ];
      const splitResult = splitNeatChatSeedAndValidationLines(
        conversationLines,
        2,
      );

      // Act
      const comparisonResult = createNeatChatAbComparison('hello there', {
        corpusRetainedTerms: [
          'hi',
          'there',
          'hello',
          'friend',
          'how',
          'are',
          'you',
        ],
        seedConversationLines: splitResult.seedConversationLines,
        validationConversationLines: splitResult.validationConversationLines,
      });

      // Assert
      expect(
        comparisonResult.variants.every(
          (variantResult) =>
            variantResult.metrics.heldOutNextTokenAccuracy >= 0,
        ),
      ).toBe(true);
    });

    it('createNeatChatAbComparison reports repetition rate in [0, 1] for each variant', () => {
      // Arrange
      const conversationLines = [
        'hi there',
        'hello friend',
        'how are you',
        'doing well today',
        'let us walk',
        'sounds great',
      ];
      const splitResult = splitNeatChatSeedAndValidationLines(
        conversationLines,
        2,
      );

      // Act
      const comparisonResult = createNeatChatAbComparison('hello there', {
        corpusRetainedTerms: [
          'hi',
          'there',
          'hello',
          'friend',
          'how',
          'are',
          'you',
        ],
        seedConversationLines: splitResult.seedConversationLines,
        validationConversationLines: splitResult.validationConversationLines,
      });

      // Assert
      expect(
        comparisonResult.variants.every(
          (variantResult) =>
            variantResult.metrics.repetitionRate >= 0 &&
            variantResult.metrics.repetitionRate <= 1,
        ),
      ).toBe(true);
    });

    it('createNeatChatAbComparison reports response-length stability in [0, 1] for each variant', () => {
      // Arrange
      const conversationLines = [
        'hi there',
        'hello friend',
        'how are you',
        'doing well today',
        'let us walk',
        'sounds great',
      ];
      const splitResult = splitNeatChatSeedAndValidationLines(
        conversationLines,
        2,
      );

      // Act
      const comparisonResult = createNeatChatAbComparison('hello there', {
        corpusRetainedTerms: [
          'hi',
          'there',
          'hello',
          'friend',
          'how',
          'are',
          'you',
        ],
        seedConversationLines: splitResult.seedConversationLines,
        validationConversationLines: splitResult.validationConversationLines,
      });

      // Assert
      expect(
        comparisonResult.variants.every(
          (variantResult) =>
            variantResult.metrics.responseLengthStability >= 0 &&
            variantResult.metrics.responseLengthStability <= 1,
        ),
      ).toBe(true);
    });
  });
});

describe('exportNeatChatSession / importNeatChatSession (v1 roundtrip)', () => {
  it('exports a bundle with formatVersion 1', () => {
    // Arrange
    const session = createNeatChatSession({
      corpusRetainedTerms: ['hello', 'there', 'general', 'kenobi'],
      recurrentBlockSize: 8,
      seedConversationLines: ['hello there', 'general kenobi'],
    });

    // Act
    const bundle = exportNeatChatSession(session);

    // Assert
    expect(bundle.formatVersion).toBe(1);
  });

  it('imports a round-tripped session with matching vocabulary size', () => {
    // Arrange
    const session = createNeatChatSession({
      corpusRetainedTerms: ['hello', 'there', 'general', 'kenobi'],
      recurrentBlockSize: 8,
      seedConversationLines: ['hello there', 'general kenobi'],
    });
    const bundle = exportNeatChatSession(session);

    // Act
    const importedSession = importNeatChatSession(bundle);

    // Assert
    expect(importedSession.vocabulary.size).toBe(session.vocabulary.size);
  });

  it('throws NeatChatSnapshotVersionError for wrong formatVersion', () => {
    // Arrange
    const seedNetwork = createNeatChatSeedNetwork({
      vocabularySize: 4,
    });
    const invalidSnapshot = {
      contextWindowTokenCount: 4,
      exchanges: [],
      formatVersion: 2,
      learnedExchangeCount: 0,
      learnedTokenPairCount: 0,
      networkJson: seedNetwork.network.toJSON(),
      retainedTerms: ['hello', 'there'],
      seededTokenPairCount: 0,
    };

    // Act
    const importAction = () => importNeatChatSession(invalidSnapshot as never);

    // Assert
    expect(importAction).toThrow(NeatChatSnapshotVersionError);
  });

  it('throws NeatChatSnapshotShapeError for missing retainedTerms', () => {
    // Arrange
    const session = createNeatChatSession({
      corpusRetainedTerms: ['hello', 'there', 'general', 'kenobi'],
      recurrentBlockSize: 8,
      seedConversationLines: ['hello there', 'general kenobi'],
    });
    const invalidSnapshot = {
      ...exportNeatChatSession(session),
      retainedTerms: undefined,
    };

    // Act
    const importAction = () => importNeatChatSession(invalidSnapshot as never);

    // Assert
    expect(importAction).toThrow(NeatChatSnapshotShapeError);
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
