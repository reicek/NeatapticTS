/** @jest-environment jsdom */

jest.mock('./default-pretrained-session-snapshot', () => ({
  DEFAULT_NEATCHAT_PRETRAINED_SESSION_SNAPSHOT: {
    formatVersion: 1,
    retainedTerms: ['default', 'snapshot', 'reply'],
    networkJson: { nodes: [] },
    exchanges: [],
    learnedExchangeCount: 0,
    learnedTokenPairCount: 0,
    seededTokenPairCount: 24,
    contextWindowTokenCount: 50,
  },
}));

import { start } from './browser-entry';
import {
  createNeatChatAbComparison,
  createNeatChatAdaptationManager,
  createNeatChatSession,
  createNeatChatPretrainingPreview,
  exportNeatChatSession,
  getNeatChatSampleConversationLines,
  importNeatChatSession,
  pretrainNeatChatSessionWithConversationLines,
  runNeatChatExchange,
  scheduleNeatChatAdaptation,
  updateNeatChatSessionContextWindowTokenCount,
} from './index';

jest.mock('./index', () => {
  const sampleConversationLines = [
    'line 1',
    'line 2',
    'line 3',
    'line 4',
    'line 5',
    'line 6',
    'line 7',
    'line 8',
    'line 9',
    'line 10',
  ];

  return {
    createNeatChatExampleContract: jest.fn(() => ({
      pretraining: {
        defaultTopWordLimit: 20,
        defaultContextWindowTokenCount: 24,
        defaultChunkTokenCount: 64,
        recommendedTopWordLimitRange: [1, 5000],
        unknownToken: 'UNK',
        tokenizationRule: 'tokenization rule',
      },
      visualization: {
        exampleId: 'flappy_bird',
      },
    })),
    createNeatChatSeedNetwork: jest.fn(() => ({
      summary: {
        architectureFamily: 'lstm',
      },
    })),
    estimateNeatChatRuntime: jest.fn(
      (
        options: {
          topWordLimit?: number;
          contextWindowTokenCount?: number;
        } = {},
      ) => ({
        topWordLimit: options.topWordLimit ?? 20,
        contextWindowTokenCount: options.contextWindowTokenCount ?? 24,
        estimatedRetainedVocabularySize: (options.topWordLimit ?? 20) + 4,
        expectedPretrainingDurationBucket: 'short',
        summary: 'runtime summary',
      }),
    ),
    createNeatChatPretrainingPreview: jest.fn(
      (options: { corpusText: string }) => {
        const corpusText = options.corpusText ?? '';
        const normalizedWords =
          corpusText.toLowerCase().match(/[a-z0-9']+/g) ?? [];
        const uniqueTerms = new Set(normalizedWords);
        const previewTerms = [...uniqueTerms].slice(0, 4);
        const hasCorpus = corpusText.trim().length > 0;

        return {
          hasCorpus,
          retainedTermCount: uniqueTerms.size,
          retainedTokenCoveragePercent: hasCorpus ? 100 : 0,
          processedChunkCount: hasCorpus ? 1 : 0,
          previewTerms,
          retainedTerms: [...uniqueTerms],
          corpusReport: {
            characterCount: corpusText.length,
            tokenCount: normalizedWords.length,
            uniqueTermCount: uniqueTerms.size,
            retainedTermCount: uniqueTerms.size,
            retainedTokenCoveragePercent: hasCorpus ? 100 : 0,
          },
        };
      },
    ),
    extractNeatChatConversationLines: jest.fn(() => []),
    splitNeatChatSeedAndValidationLines: jest.fn(() => ({
      seedConversationLines: [],
      validationConversationLines: [],
    })),
    createNeatChatSession: jest.fn(() => ({
      vocabulary: { size: 6 },
      exchanges: [],
      learnedExchangeCount: 0,
      learnedTokenPairCount: 0,
      seededTokenPairCount: 0,
      contextWindowTokenCount: 24,
      pendingCandidates: [],
      candidateLog: [],
      routingLog: [],
      network: {},
    })),
    updateNeatChatSessionContextWindowTokenCount: jest.fn(
      (session, contextWindowTokenCount) => ({
        ...session,
        contextWindowTokenCount,
      }),
    ),
    pretrainNeatChatSessionWithConversationLines: jest.fn(
      (session, seedLines) => ({
        ...session,
        seededTokenPairCount: session.seededTokenPairCount + seedLines.length,
      }),
    ),
    createNeatChatAdaptationManager: jest.fn(() => ({
      pendingCandidates: [],
      candidateLog: [],
    })),
    scheduleNeatChatAdaptation: jest.fn(async (manager, session) => ({
      ...manager,
      pendingCandidates: [
        {
          createdAt: 123,
          sourceExchangeCount: session.learnedExchangeCount,
          trainedVector: [session.learnedExchangeCount],
          evaluationScores: {
            heldOutNextTokenAccuracy: 0.75,
          },
        },
      ],
    })),
    runNeatChatExchange: jest.fn((session, userMessage: string) => {
      const hasPendingCandidate = session.pendingCandidates.length > 0;
      const response = hasPendingCandidate
        ? 'personalized reply'
        : session.seededTokenPairCount > 0
          ? 'learned reply'
          : 'blank reply';
      const responseTokens = response.split(' ');

      return {
        response,
        responseTokens,
      userTokens: ['hello'],
      trainedTokenPairCount: 3,
      updatedSession: {
        ...session,
        exchanges: [
          ...session.exchanges,
          {
            userMessage,
            response,
            trainedTokenPairCount: 3,
            userTokens: ['hello'],
            responseTokens,
          },
        ],
        learnedExchangeCount: session.learnedExchangeCount + 1,
        learnedTokenPairCount: session.learnedTokenPairCount + 3,
      },
      };
    }),
    createNeatChatAbComparison: jest.fn((prompt: string) => ({
      prompt,
      variants: [
        {
          variant: 'blank-start',
          prompt,
          response: 'blank variant reply',
          responseTokens: ['blank', 'variant', 'reply'],
          trainedTokenPairCount: 0,
          metrics: {
            heldOutNextTokenAccuracy: 12.5,
            repetitionRate: 0.25,
            responseLengthStability: 0.5,
          },
        },
        {
          variant: 'preseeded',
          prompt,
          response: 'preseeded variant reply',
          responseTokens: ['preseeded', 'variant', 'reply'],
          trainedTokenPairCount: 0,
          metrics: {
            heldOutNextTokenAccuracy: 87.5,
            repetitionRate: 0,
            responseLengthStability: 1,
          },
        },
      ],
    })),
    exportNeatChatSession: jest.fn((session) => ({
      formatVersion: 1,
      retainedTerms: ['line', 'sample'],
      networkJson: { nodes: [] },
      exchanges: session.exchanges,
      learnedExchangeCount: session.learnedExchangeCount,
      learnedTokenPairCount: session.learnedTokenPairCount,
      seededTokenPairCount: session.seededTokenPairCount,
      contextWindowTokenCount: session.contextWindowTokenCount,
    })),
    importNeatChatSession: jest.fn((snapshot) => ({
      vocabulary: { size: snapshot.retainedTerms.length + 4 },
      exchanges: snapshot.exchanges,
      learnedExchangeCount: snapshot.learnedExchangeCount,
      learnedTokenPairCount: snapshot.learnedTokenPairCount,
      seededTokenPairCount: snapshot.seededTokenPairCount,
      contextWindowTokenCount: snapshot.contextWindowTokenCount,
      pendingCandidates: snapshot.pendingCandidates ?? [],
      candidateLog: snapshot.candidateLog ?? [],
      routingLog: snapshot.routingLog ?? [],
      network: {},
    })),
    tokenizeNeatChatText: jest.fn(() => ['hello']),
    getNeatChatSampleConversationLines: jest.fn(() => sampleConversationLines),
  };
});

const mockedPretrainNeatChatSessionWithConversationLines = jest.mocked(
  pretrainNeatChatSessionWithConversationLines,
);
const mockedCreateNeatChatAdaptationManager = jest.mocked(
  createNeatChatAdaptationManager,
);
const mockedCreateNeatChatSession = jest.mocked(createNeatChatSession);
const mockedExportNeatChatSession = jest.mocked(exportNeatChatSession);
const mockedRunNeatChatExchange = jest.mocked(runNeatChatExchange);
const mockedScheduleNeatChatAdaptation = jest.mocked(
  scheduleNeatChatAdaptation,
);
const mockedUpdateNeatChatSessionContextWindowTokenCount = jest.mocked(
  updateNeatChatSessionContextWindowTokenCount,
);
const mockedCreateNeatChatPretrainingPreview = jest.mocked(
  createNeatChatPretrainingPreview,
);
const mockedGetNeatChatSampleConversationLines = jest.mocked(
  getNeatChatSampleConversationLines,
);
const mockedImportNeatChatSession = jest.mocked(importNeatChatSession);
const mockedCreateNeatChatAbComparison = jest.mocked(
  createNeatChatAbComparison,
);

describe('neatChat browser-entry sample pretraining', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="neat-chat-output"></div>';
    mockedCreateNeatChatAdaptationManager.mockClear();
    mockedCreateNeatChatSession.mockClear();
    mockedExportNeatChatSession.mockClear();
    mockedImportNeatChatSession.mockClear();
    mockedPretrainNeatChatSessionWithConversationLines.mockClear();
    mockedRunNeatChatExchange.mockClear();
    mockedScheduleNeatChatAdaptation.mockClear();
    mockedUpdateNeatChatSessionContextWindowTokenCount.mockClear();
    mockedCreateNeatChatPretrainingPreview.mockClear();
    mockedCreateNeatChatAbComparison.mockClear();
    mockedGetNeatChatSampleConversationLines.mockClear();
    Object.defineProperty(window, 'requestAnimationFrame', {
      configurable: true,
      writable: true,
      value: (callback: FrameRequestCallback) => {
        callback(0);
        return 1;
      },
    });
  });

  it('trains the next four sample lines on each button click', async () => {
    await start('neat-chat-output');
    const pretrainButton = document.querySelector<HTMLButtonElement>(
      '[data-neat-chat-load-sample]',
    );

    pretrainButton?.click();
    pretrainButton?.click();

    expect({
      firstChunk:
        mockedPretrainNeatChatSessionWithConversationLines.mock.calls[0]?.[1],
      secondChunk:
        mockedPretrainNeatChatSessionWithConversationLines.mock.calls[1]?.[1],
      sampleLineCount:
        mockedGetNeatChatSampleConversationLines.mock.results[0]?.value.length,
    }).toEqual({
      firstChunk: ['line 5', 'line 6', 'line 7', 'line 8'],
      secondChunk: ['line 9', 'line 10'],
      sampleLineCount: 10,
    });
  });

  it('shows shipped basepoint status text when loading the default snapshot', async () => {
    await start('neat-chat-output');

    expect(
      document.querySelector<HTMLElement>(
        '[data-neat-chat-session-snapshot-status]',
      )?.textContent,
    ).toBe(
      'Shipped pretrained basepoint active. Live chat now uses the bundled snapshot; the corpus preview report only changes when you paste or sample-train lines.',
    );
  });

  it('loads the shipped pretrained snapshot as the default browser starting point', async () => {
    await start('neat-chat-output');

    expect({
      importCallCount: mockedImportNeatChatSession.mock.calls.length,
      importedSnapshotRetainedTerms:
        mockedImportNeatChatSession.mock.calls[0]?.[0]?.retainedTerms,
      createdFreshSessionCount: mockedCreateNeatChatSession.mock.calls.length,
    }).toEqual({
      importCallCount: 1,
      importedSnapshotRetainedTerms: ['default', 'snapshot', 'reply'],
      createdFreshSessionCount: 0,
    });
  });

  it('syncs the visible context-window control to the shipped snapshot on startup', async () => {
    await start('neat-chat-output');

    expect(
      document.querySelector<HTMLSelectElement>(
        '[data-neat-chat-context-window-token-count]',
      )?.value,
    ).toBe('50');
  });

  it('uses the updated pre-trained session for subsequent live-chat replies', async () => {
    await start('neat-chat-output');
    const liveInput = document.querySelector<HTMLInputElement>(
      '[data-neat-chat-live-input]',
    );
    const liveForm = document.querySelector<HTMLFormElement>(
      '[data-neat-chat-live-form]',
    );
    const pretrainButton = document.querySelector<HTMLButtonElement>(
      '[data-neat-chat-load-sample]',
    );

    if (liveInput && liveForm) {
      liveInput.value = 'hello';
      liveForm.dispatchEvent(
        new Event('submit', { bubbles: true, cancelable: true }),
      );
    }

    pretrainButton?.click();

    if (liveInput && liveForm) {
      liveInput.value = 'hello again';
      liveForm.dispatchEvent(
        new Event('submit', { bubbles: true, cancelable: true }),
      );
    }

    const historyText =
      document.querySelector<HTMLElement>('[data-neat-chat-history]')
        ?.textContent ?? '';

    expect({
      hasBlankReply: historyText.includes('blank reply'),
      hasLearnedReply: historyText.includes('learned reply'),
      hasSystemPretrainMessage: historyText.includes(
        'Pre-trained lines 5-8 of 10.',
      ),
    }).toEqual({
      hasBlankReply: false,
      hasLearnedReply: true,
      hasSystemPretrainMessage: true,
    });
  });

  it('schedules adaptation after turn 1 so turn 2 can see a personalized candidate', async () => {
    await start('neat-chat-output');
    const liveInput = document.querySelector<HTMLInputElement>(
      '[data-neat-chat-live-input]',
    );
    const liveForm = document.querySelector<HTMLFormElement>(
      '[data-neat-chat-live-form]',
    );

    if (liveInput && liveForm) {
      liveInput.value = 'first message';
      liveForm.dispatchEvent(
        new Event('submit', { bubbles: true, cancelable: true }),
      );
    }

    await Promise.resolve();
    await Promise.resolve();

    if (liveInput && liveForm) {
      liveInput.value = 'second message';
      liveForm.dispatchEvent(
        new Event('submit', { bubbles: true, cancelable: true }),
      );
    }

    const historyText =
      document.querySelector<HTMLElement>('[data-neat-chat-history]')
        ?.textContent ?? '';

    expect({
      scheduleCallCount: mockedScheduleNeatChatAdaptation.mock.calls.length,
      scheduledExchangeCount:
        mockedScheduleNeatChatAdaptation.mock.calls[0]?.[1]
          ?.learnedExchangeCount,
      secondTurnPendingCandidateCount:
        mockedRunNeatChatExchange.mock.calls[1]?.[0]?.pendingCandidates.length,
      hasPersonalizedTurnTwoReply: historyText.includes('personalized reply'),
    }).toEqual({
      scheduleCallCount: 1,
      scheduledExchangeCount: 1,
      secondTurnPendingCandidateCount: 1,
      hasPersonalizedTurnTwoReply: true,
    });
  });

  it('updates corpus stats display after sample pretraining clicks', async () => {
    await start('neat-chat-output');
    const pretrainButton = document.querySelector<HTMLButtonElement>(
      '[data-neat-chat-load-sample]',
    );

    pretrainButton?.click();

    const retainedTermCountText =
      document.querySelector<HTMLElement>(
        '[data-neat-chat-corpus-retained-term-count]',
      )?.textContent ?? '';
    const corpusTextareaText =
      document.querySelector<HTMLTextAreaElement>('[data-neat-chat-corpus]')
        ?.value ?? '';

    const firstSampleChunk = ['line 5', 'line 6', 'line 7', 'line 8'].join(
      '\n',
    );

    expect({
      pretrainingPreviewRecomputed:
        mockedCreateNeatChatPretrainingPreview.mock.calls.length >= 2,
      retainedTermCountMovedOffZero: Number(retainedTermCountText) > 0,
      corpusTextareaText,
    }).toEqual({
      pretrainingPreviewRecomputed: true,
      retainedTermCountMovedOffZero: true,
      corpusTextareaText: firstSampleChunk,
    });
  });

  it('renders the full optional pretraining report controls on startup', async () => {
    await start('neat-chat-output');

    expect({
      topWordLimitValue: document.querySelector<HTMLInputElement>(
        '[data-neat-chat-top-word-limit]',
      )?.value,
      characterCount: document.querySelector<HTMLElement>(
        '[data-neat-chat-corpus-character-count]',
      )?.textContent,
      tokenCount: document.querySelector<HTMLElement>(
        '[data-neat-chat-corpus-token-count]',
      )?.textContent,
      uniqueTermCount: document.querySelector<HTMLElement>(
        '[data-neat-chat-corpus-unique-term-count]',
      )?.textContent,
      chunkCount: document.querySelector<HTMLElement>(
        '[data-neat-chat-corpus-chunk-count]',
      )?.textContent,
      pretrainingStatus: document.querySelector<HTMLElement>(
        '[data-neat-chat-pretraining-status]',
      )?.textContent,
      previewVocabulary: document.querySelector<HTMLElement>(
        '[data-neat-chat-pretraining-preview-vocabulary]',
      )?.textContent,
    }).toEqual({
      topWordLimitValue: '20',
      characterCount: '0',
      tokenCount: '0',
      uniqueTermCount: '0',
      chunkCount: '0',
      pretrainingStatus: 'paste corpus text to prepare a preseeded preview',
      previewVocabulary: 'UNK',
    });
  });

  it('rebuilds the preview using the user-provided top-word limit', async () => {
    await start('neat-chat-output');

    const corpusField = document.querySelector<HTMLTextAreaElement>(
      '[data-neat-chat-corpus]',
    );
    const topWordLimitField = document.querySelector<HTMLInputElement>(
      '[data-neat-chat-top-word-limit]',
    );
    const rerenderButton = document.querySelector<HTMLButtonElement>(
      '[data-neat-chat-rerender]',
    );

    if (corpusField && topWordLimitField) {
      corpusField.value = 'star sun star moon';
      topWordLimitField.value = '7';
    }

    rerenderButton?.click();

    const latestPreviewCall =
      mockedCreateNeatChatPretrainingPreview.mock.calls.at(-1)?.[0];

    expect({
      topWordLimit: latestPreviewCall?.topWordLimit,
      corpusText: latestPreviewCall?.corpusText,
    }).toEqual({
      topWordLimit: 7,
      corpusText: 'star sun star moon',
    });
  });

  it('renders a browser A/B evaluation surface for blank-start versus preseeded checks', async () => {
    await start('neat-chat-output');

    expect({
      promptValue: document.querySelector<HTMLInputElement>(
        '[data-neat-chat-ab-prompt]',
      )?.value,
      hasRunButton:
        document.querySelector<HTMLButtonElement>('[data-neat-chat-run-ab]') !==
        null,
      resultsText: document.querySelector<HTMLElement>(
        '[data-neat-chat-ab-results]',
      )?.textContent,
    }).toEqual({
      promptValue: 'hello there',
      hasRunButton: true,
      resultsText:
        'Run a blank-start versus preseeded comparison to inspect held-out next-token accuracy, repetition rate, and response-length stability.',
    });
  });

  it('runs the browser A/B evaluation with the active prompt', async () => {
    await start('neat-chat-output');

    const abPromptField = document.querySelector<HTMLInputElement>(
      '[data-neat-chat-ab-prompt]',
    );
    const runAbButton = document.querySelector<HTMLButtonElement>(
      '[data-neat-chat-run-ab]',
    );

    if (abPromptField) {
      abPromptField.value = 'how was your day';
    }

    runAbButton?.click();

    const abResultsText = document.querySelector<HTMLElement>(
      '[data-neat-chat-ab-results]',
    )?.textContent;

    expect({
      prompt: mockedCreateNeatChatAbComparison.mock.calls.at(-1)?.[0],
      renderedComparison:
        abResultsText?.includes('blank variant reply') === true &&
        abResultsText?.includes('preseeded variant reply') === true &&
        abResultsText?.includes('Held-out next-token accuracy') === true,
    }).toEqual({
      prompt: 'how was your day',
      renderedComparison: true,
    });
  });

  it('updates live context window from the dropdown without resetting session progression', async () => {
    await start('neat-chat-output');
    const liveInput = document.querySelector<HTMLInputElement>(
      '[data-neat-chat-live-input]',
    );
    const liveForm = document.querySelector<HTMLFormElement>(
      '[data-neat-chat-live-form]',
    );
    const contextWindowSelect = document.querySelector<HTMLSelectElement>(
      '[data-neat-chat-context-window-token-count]',
    );

    if (liveInput && liveForm) {
      liveInput.value = 'first message';
      liveForm.dispatchEvent(
        new Event('submit', { bubbles: true, cancelable: true }),
      );
    }

    if (contextWindowSelect) {
      contextWindowSelect.value = '50';
      contextWindowSelect.dispatchEvent(new Event('change', { bubbles: true }));
    }

    if (liveInput && liveForm) {
      liveInput.value = 'second message';
      liveForm.dispatchEvent(
        new Event('submit', { bubbles: true, cancelable: true }),
      );
    }

    const historyText =
      document.querySelector<HTMLElement>('[data-neat-chat-history]')
        ?.textContent ?? '';
    const sessionStatsText =
      document.querySelector<HTMLElement>('[data-neat-chat-session-stats]')
        ?.textContent ?? '';

    expect({
      contextWindowUpdateCalledWith:
        mockedUpdateNeatChatSessionContextWindowTokenCount.mock.calls[0]?.[1],
      hasContextWindowSystemMessage: historyText.includes(
        'Updated live context window to 50 tokens without resetting learned weights.',
      ),
      includesUpdatedContextWindowInStats:
        sessionStatsText.includes('Context Window') &&
        sessionStatsText.includes('50 tokens'),
      includesSecondExchangeInStats:
        sessionStatsText.includes('Exchanges') &&
        sessionStatsText.includes('2'),
      includesPretrainedTermsColumn:
        sessionStatsText.includes('Live Retained Terms') &&
        sessionStatsText.includes('3'),
    }).toEqual({
      contextWindowUpdateCalledWith: 50,
      hasContextWindowSystemMessage: true,
      includesUpdatedContextWindowInStats: true,
      includesSecondExchangeInStats: true,
      includesPretrainedTermsColumn: true,
    });
  });

  it('updates imported snapshot stats to the newly loaded retained-term count', async () => {
    await start('neat-chat-output');

    const importSessionInput = document.querySelector<HTMLInputElement>(
      '[data-neat-chat-import-session-input]',
    );
    const importedSnapshot = {
      text: async () =>
        JSON.stringify({
          formatVersion: 1,
          retainedTerms: ['alpha', 'beta', 'gamma', 'delta', 'epsilon'],
          networkJson: { nodes: [] },
          exchanges: [],
          learnedExchangeCount: 0,
          learnedTokenPairCount: 0,
          seededTokenPairCount: 12,
          contextWindowTokenCount: 32,
        }),
    } as File;

    if (importSessionInput) {
      Object.defineProperty(importSessionInput, 'files', {
        configurable: true,
        value: [importedSnapshot],
      });
      importSessionInput.dispatchEvent(new Event('change', { bubbles: true }));
      await Promise.resolve();
      await Promise.resolve();
    }

    const contextWindowValue = document.querySelector<HTMLSelectElement>(
      '[data-neat-chat-context-window-token-count]',
    )?.value;
    const retainedTermCount = document.querySelector<HTMLElement>(
      '[data-neat-chat-stats-retained-terms]',
    )?.textContent;

    expect({
      contextWindowValue,
      retainedTermCount,
    }).toEqual({
      contextWindowValue: '32',
      retainedTermCount: '5',
    });
  });
});
