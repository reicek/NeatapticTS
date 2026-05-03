import {
  buildNeatChatVocabulary,
  createNeatChatExampleContract,
  createNeatChatAbComparison,
  createNeatChatPretrainingPreview,
  createNeatChatSeedNetwork,
  createNeatChatSession,
  pretrainNeatChatSessionWithConversationLines,
  updateNeatChatSessionContextWindowTokenCount,
  extractNeatChatConversationLines,
  getNeatChatSampleConversationLines,
  estimateNeatChatRuntime,
  runNeatChatExchange,
  splitNeatChatSeedAndValidationLines,
  tokenizeNeatChatText,
} from './index';

describe('neatChat example contract', () => {
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
        'Lowercase the text, keep Unicode letter and number runs plus apostrophes, truncate each prompt or reply slice to the short context window, and map out-of-vocabulary terms to UNK once the retained vocabulary is known.',
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

  it('shares the Step 2 tokenizer and runtime estimate across Node and browser previews', () => {
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
});

describe('neatChat Step 5 — online learning', () => {
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
    const session = createNeatChatSession();

    // Assert
    expect(session.exchanges.length).toBe(0);
  });

  it('createNeatChatSession starts with zero learned exchange count', () => {
    // Arrange
    const session = createNeatChatSession();

    // Assert
    expect(session.learnedExchangeCount).toBe(0);
  });

  it('createNeatChatSession uses provided contextWindowTokenCount', () => {
    // Arrange
    const session = createNeatChatSession({ contextWindowTokenCount: 50 });

    // Assert
    expect(session.contextWindowTokenCount).toBe(50);
  });

  it('runNeatChatExchange tokenization respects session contextWindowTokenCount', () => {
    // Arrange
    const session = createNeatChatSession({ contextWindowTokenCount: 2 });

    // Act
    const result = runNeatChatExchange(session, 'one two three four');

    // Assert
    expect(result.userTokens).toEqual(['one', 'two']);
  });

  it('updateNeatChatSessionContextWindowTokenCount updates without resetting learned counts', () => {
    // Arrange
    const session = createNeatChatSession({
      corpusRetainedTerms: ['hello', 'there'],
      contextWindowTokenCount: 24,
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
    const session = createNeatChatSession({
      corpusRetainedTerms: ['hello', 'world'],
    });

    // Act
    const result = runNeatChatExchange(session, 'hello');

    // Assert
    expect(result.response.length).toBeGreaterThan(0);
  });

  it('runNeatChatExchange increments learnedExchangeCount by 1', () => {
    // Arrange
    const session = createNeatChatSession();

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
    const session = createNeatChatSession();

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

  it('runNeatChatExchange trains first exchange on observed user tokens only', () => {
    // Arrange
    const session = createNeatChatSession({
      corpusRetainedTerms: ['hello', 'world'],
    });

    // Act
    const result = runNeatChatExchange(session, 'hello world');

    // Assert
    expect(result.trainedTokenPairCount).toBe(4);
  });

  it('runNeatChatExchange grows training pairs with observed user-history turns', () => {
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
    expect(secondResult.trainedTokenPairCount).toBe(7);
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

  it('runNeatChatExchange does not emit TURN_BREAK in decoded response tokens', () => {
    // Arrange — tiny fixture with small recurrent block keeps seeding under 2 s
    const sampleConversationLines = getNeatChatSampleConversationLines().slice(
      0,
      12,
    );
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
    const sampleConversationLines = getNeatChatSampleConversationLines().slice(
      0,
      12,
    );
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
});

describe('neatChat Step 6 — A/B interaction and lightweight evaluation', () => {
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
        (variantResult) => variantResult.metrics.heldOutNextTokenAccuracy >= 0,
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
        (variantResult) => variantResult.metrics.heldOutNextTokenAccuracy >= 0,
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
