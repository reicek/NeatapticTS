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
  createNeatChatSession,
  createNeatChatPretrainingPreview,
  exportNeatChatSession,
  getNeatChatSampleConversationLines,
  importNeatChatSession,
  pretrainNeatChatSessionWithConversationLines,
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
    estimateNeatChatRuntime: jest.fn(() => ({
      topWordLimit: 20,
      contextWindowTokenCount: 24,
      estimatedRetainedVocabularySize: 24,
      expectedPretrainingDurationBucket: 'short',
      summary: 'runtime summary',
    })),
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
    runNeatChatExchange: jest.fn((session) => ({
      response:
        session.seededTokenPairCount > 0 ? 'learned reply' : 'blank reply',
      responseTokens:
        session.seededTokenPairCount > 0
          ? ['learned', 'reply']
          : ['blank', 'reply'],
      userTokens: ['hello'],
      trainedTokenPairCount: 3,
      updatedSession: {
        ...session,
        learnedExchangeCount: session.learnedExchangeCount + 1,
        learnedTokenPairCount: session.learnedTokenPairCount + 3,
      },
    })),
    createNeatChatAbComparison: jest.fn(() => ({
      prompt: 'hello',
      variants: [],
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
      network: {},
    })),
    tokenizeNeatChatText: jest.fn(() => ['hello']),
    getNeatChatSampleConversationLines: jest.fn(() => sampleConversationLines),
  };
});

const mockedPretrainNeatChatSessionWithConversationLines = jest.mocked(
  pretrainNeatChatSessionWithConversationLines,
);
const mockedCreateNeatChatSession = jest.mocked(createNeatChatSession);
const mockedExportNeatChatSession = jest.mocked(exportNeatChatSession);
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

describe('neatChat browser-entry sample pretraining', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="neat-chat-output"></div>';
    mockedCreateNeatChatSession.mockClear();
    mockedExportNeatChatSession.mockClear();
    mockedImportNeatChatSession.mockClear();
    mockedPretrainNeatChatSessionWithConversationLines.mockClear();
    mockedUpdateNeatChatSessionContextWindowTokenCount.mockClear();
    mockedCreateNeatChatPretrainingPreview.mockClear();
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
      'Shipped pretrained basepoint active. Export the session after extra training if it improves.',
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
      includesUpdatedContextWindowInStats: sessionStatsText.includes(
        'Context window: 50 tokens',
      ),
      includesSecondExchangeInStats: sessionStatsText.includes('Exchanges: 2'),
    }).toEqual({
      contextWindowUpdateCalledWith: 50,
      hasContextWindowSystemMessage: true,
      includesUpdatedContextWindowInStats: true,
      includesSecondExchangeInStats: true,
    });
  });
});
