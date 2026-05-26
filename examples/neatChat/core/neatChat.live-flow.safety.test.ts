jest.mock('./neatChat.routing.services', () => {
  const actualRoutingServices = jest.requireActual<
    typeof import('./neatChat.routing.services')
  >('./neatChat.routing.services');

  return {
    ...actualRoutingServices,
    generateNeatChatCandidates: jest.fn(
      actualRoutingServices.generateNeatChatCandidates,
    ),
  };
});

jest.mock('./neatChat.safety.services', () => {
  const actualSafetyServices = jest.requireActual<
    typeof import('./neatChat.safety.services')
  >('./neatChat.safety.services');

  return {
    ...actualSafetyServices,
    checkSafety: jest.fn(actualSafetyServices.checkSafety),
  };
});

import { generateNeatChatCandidates } from './neatChat.routing.services';
import { checkSafety } from './neatChat.safety.services';
import {
  createNeatChatSession,
  runNeatChatExchange,
} from './neatChat.session.services';
import type {
  NeatChatRoutingCandidate,
  NeatChatRoutingPath,
} from './neatChat.routing.types';
import type { NeatChatExchangeRecord, NeatChatSession } from './neatChat.types';

const actualRoutingServices = jest.requireActual<
  typeof import('./neatChat.routing.services')
>('./neatChat.routing.services');
const actualSafetyServices = jest.requireActual<
  typeof import('./neatChat.safety.services')
>('./neatChat.safety.services');

const mockedGenerateNeatChatCandidates = jest.mocked(
  generateNeatChatCandidates,
);
const mockedCheckSafety = jest.mocked(checkSafety);
const NO_SAFE_CANDIDATE_FALLBACK_RESPONSE = 'can you rephrase that';
const NO_SAFE_CANDIDATE_FALLBACK_TOKENS = [
  'can',
  'you',
  'rephrase',
  'that',
] as const;

describe('neatChat live-flow safety contract', () => {
  beforeEach(() => {
    mockedGenerateNeatChatCandidates.mockReset();
    mockedGenerateNeatChatCandidates.mockImplementation(
      actualRoutingServices.generateNeatChatCandidates,
    );
    mockedCheckSafety.mockReset();
    mockedCheckSafety.mockImplementation(actualSafetyServices.checkSafety);
  });

  it('runs the safety gate before accepting the selected live candidate', () => {
    // Arrange
    const session = createPretrainedLiveSession();
    const loopingCandidate = createRoutingCandidate({
      response: 'hello hello hello hello',
      responseTokens: ['hello', 'hello', 'hello', 'hello'],
      routingPath: 'base',
      score: 0.95,
    });

    mockedGenerateNeatChatCandidates.mockReturnValue([loopingCandidate]);

    // Act
    runNeatChatExchange(session, 'hello');

    // Assert
    expect(mockedCheckSafety).toHaveBeenCalledWith(
      expect.objectContaining({ seededTokenPairCount: 7_776 }),
      loopingCandidate.response,
    );
  });

  it('falls back to the next safe local candidate when the top score fails safety', () => {
    // Arrange
    const session = createPretrainedLiveSession();
    const loopingCandidate = createRoutingCandidate({
      response: 'hello hello hello hello',
      responseTokens: ['hello', 'hello', 'hello', 'hello'],
      routingPath: 'base',
      score: 0.95,
    });
    const safeCandidate = createRoutingCandidate({
      response: 'steady answer',
      responseTokens: ['steady', 'answer'],
      routingPath: 'personalized',
      score: 0.6,
    });

    mockedGenerateNeatChatCandidates.mockReturnValue([
      loopingCandidate,
      safeCandidate,
    ]);
    mockedCheckSafety.mockReturnValueOnce({
      ok: false,
      violation: 'repetition-collapse',
      detail:
        'Response bigram repetition fraction exceeds the collapse threshold; the network may be looping.',
    });
    mockedCheckSafety.mockReturnValueOnce({
      ok: true,
      violation: null,
      detail: '',
    });

    // Act
    const result = runNeatChatExchange(session, 'hello');

    // Assert
    expect({
      response: result.response,
      selectedPath: result.updatedSession.routingLog.at(-1)?.selectedPath,
    }).toEqual({
      response: safeCandidate.response,
      selectedPath: safeCandidate.routingPath,
    });
  });

  it('strips placeholder punctuation tokens before accepting a live candidate', () => {
    // Arrange
    const session = createPretrainedLiveSession();
    const placeholderCandidate = createRoutingCandidate({
      response: 'i have PUNC_SENTENCE_END',
      responseTokens: ['i', 'have', 'PUNC_SENTENCE_END'],
      routingPath: 'base',
      score: 0.95,
    });
    const safeCandidate = createRoutingCandidate({
      response: 'i have plans',
      responseTokens: ['i', 'have', 'plans'],
      routingPath: 'personalized',
      score: 0.6,
    });

    mockedGenerateNeatChatCandidates.mockReturnValue([
      placeholderCandidate,
      safeCandidate,
    ]);

    // Act
    const result = runNeatChatExchange(session, 'do you have plans');

    // Assert
    expect({
      response: result.response,
      selectedPath: result.updatedSession.routingLog.at(-1)?.selectedPath,
    }).toEqual({
      response: 'i have',
      selectedPath: placeholderCandidate.routingPath,
    });
  });

  it('strips placeholder punctuation tokens from the selected response when no safe alternative exists', () => {
    // Arrange
    const session = createPretrainedLiveSession();
    const placeholderCandidate = createRoutingCandidate({
      response: 'i have PUNC_SENTENCE_END',
      responseTokens: ['i', 'have', 'PUNC_SENTENCE_END'],
      routingPath: 'base',
      score: 0.95,
    });

    mockedGenerateNeatChatCandidates.mockReturnValue([placeholderCandidate]);

    // Act
    const result = runNeatChatExchange(session, 'do you have plans');

    // Assert
    expect({
      response: result.response,
      responseTokens: result.responseTokens,
      selectedPath: result.updatedSession.routingLog.at(-1)?.selectedPath,
    }).toEqual({
      response: 'i have',
      responseTokens: ['i', 'have'],
      selectedPath: placeholderCandidate.routingPath,
    });
  });

  it('falls back to the next complete local candidate when the top score is a clipped fragment', () => {
    // Arrange
    const session = createPretrainedLiveSession();
    const clippedCandidate = createRoutingCandidate({
      response: 'i will',
      responseTokens: ['i', 'will'],
      routingPath: 'base',
      score: 0.95,
    });
    const completeCandidate = createRoutingCandidate({
      response: 'i will stay home',
      responseTokens: ['i', 'will', 'stay', 'home'],
      routingPath: 'personalized',
      score: 0.6,
    });

    mockedGenerateNeatChatCandidates.mockReturnValue([
      clippedCandidate,
      completeCandidate,
    ]);

    // Act
    const result = runNeatChatExchange(session, 'do you have plans');

    // Assert
    expect({
      response: result.response,
      selectedPath: result.updatedSession.routingLog.at(-1)?.selectedPath,
    }).toEqual({
      response: completeCandidate.response,
      selectedPath: completeCandidate.routingPath,
    });
  });

  it('skips a short prompt-echo fragment when a fresh local alternative exists', () => {
    // Arrange
    const session = createPretrainedLiveSession();
    const promptEchoCandidate = createRoutingCandidate({
      response: 'how was how',
      responseTokens: ['how', 'was', 'how'],
      routingPath: 'base',
      score: 0.95,
    });
    const safeCandidate = createRoutingCandidate({
      response: 'weekend plans later',
      responseTokens: ['weekend', 'plans', 'later'],
      routingPath: 'personalized',
      score: 0.6,
    });

    mockedGenerateNeatChatCandidates.mockReturnValue([
      promptEchoCandidate,
      safeCandidate,
    ]);

    // Act
    const result = runNeatChatExchange(session, 'how are you today');

    // Assert
    expect({
      response: result.response,
      selectedPath: result.updatedSession.routingLog.at(-1)?.selectedPath,
    }).toEqual({
      response: safeCandidate.response,
      selectedPath: safeCandidate.routingPath,
    });
  });

  it('uses the bounded local fallback when the only candidate is a short prompt-echo fragment', () => {
    // Arrange
    const session = createPretrainedLiveSession();
    const promptEchoCandidate = createRoutingCandidate({
      response: 'how was',
      responseTokens: ['how', 'was'],
      routingPath: 'base',
      score: 0.95,
    });

    mockedGenerateNeatChatCandidates.mockReturnValue([promptEchoCandidate]);

    // Act
    const result = runNeatChatExchange(session, 'what was that');

    // Assert
    expect({
      response: result.response,
      responseTokens: result.responseTokens,
    }).toEqual({
      response: NO_SAFE_CANDIDATE_FALLBACK_RESPONSE,
      responseTokens: NO_SAFE_CANDIDATE_FALLBACK_TOKENS,
    });
  });

  it('uses the bounded local fallback when every ranked candidate fails safety', () => {
    // Arrange
    const session = createPretrainedLiveSession();
    const topUnsafeCandidate = createRoutingCandidate({
      response: 'i will',
      responseTokens: ['i', 'will'],
      routingPath: 'base',
      score: 0.95,
    });
    const nextUnsafeCandidate = createRoutingCandidate({
      response: 'how was your',
      responseTokens: ['how', 'was', 'your'],
      routingPath: 'personalized',
      score: 0.7,
    });

    mockedGenerateNeatChatCandidates.mockReturnValue([
      topUnsafeCandidate,
      nextUnsafeCandidate,
    ]);
    mockedCheckSafety
      .mockReturnValueOnce({
        ok: false,
        violation: 'incomplete-fragment',
        detail:
          'Response ends on a dangling fragment tail and should not reach the live surface.',
      })
      .mockReturnValueOnce({
        ok: false,
        violation: 'incomplete-fragment',
        detail:
          'Response ends on a dangling fragment tail and should not reach the live surface.',
      });

    // Act
    const result = runNeatChatExchange(session, 'do you have plans');

    // Assert
    expect({
      response: result.response,
      responseTokens: result.responseTokens,
    }).toEqual({
      response: NO_SAFE_CANDIDATE_FALLBACK_RESPONSE,
      responseTokens: NO_SAFE_CANDIDATE_FALLBACK_TOKENS,
    });
  });

  it('uses the bounded local fallback when the only candidate fails safety', () => {
    // Arrange
    const session = createPretrainedLiveSession();
    const onlyUnsafeCandidate = createRoutingCandidate({
      response: 'i will',
      responseTokens: ['i', 'will'],
      routingPath: 'base',
      score: 0.95,
    });

    mockedGenerateNeatChatCandidates.mockReturnValue([onlyUnsafeCandidate]);
    mockedCheckSafety.mockReturnValueOnce({
      ok: false,
      violation: 'incomplete-fragment',
      detail:
        'Response ends on a dangling fragment tail and should not reach the live surface.',
    });

    // Act
    const result = runNeatChatExchange(session, 'do you have plans');

    // Assert
    expect({
      response: result.response,
      responseTokens: result.responseTokens,
    }).toEqual({
      response: NO_SAFE_CANDIDATE_FALLBACK_RESPONSE,
      responseTokens: NO_SAFE_CANDIDATE_FALLBACK_TOKENS,
    });
  });

  it('avoids selecting a fourth-turn duplicate live reply when a fresh safe alternative exists', () => {
    // Arrange
    const session = withExchangeHistory(createPretrainedLiveSession(), [
      'how was how',
      'how was how',
      'how was how',
    ]);
    const loopingCandidate = createRoutingCandidate({
      response: 'how was how',
      responseTokens: ['how', 'was', 'how'],
      routingPath: 'base',
      score: 0.95,
    });
    const safeCandidate = createRoutingCandidate({
      response: 'weekend plans later',
      responseTokens: ['weekend', 'plans', 'later'],
      routingPath: 'personalized',
      score: 0.6,
    });

    mockedGenerateNeatChatCandidates.mockReturnValue([
      loopingCandidate,
      safeCandidate,
    ]);
    mockedCheckSafety.mockReturnValue({
      ok: true,
      violation: null,
      detail: '',
    });

    // Act
    const result = runNeatChatExchange(session, 'hello');

    // Assert
    expect({
      response: result.response,
      selectedPath: result.updatedSession.routingLog.at(-1)?.selectedPath,
    }).toEqual({
      response: safeCandidate.response,
      selectedPath: safeCandidate.routingPath,
    });
  });

  it('avoids repeating the bounded fallback palette on the fourth unsafe turn', () => {
    // Arrange
    const session = createPretrainedLiveSession();
    const onlyUnsafeCandidate = createRoutingCandidate({
      response: 'i will',
      responseTokens: ['i', 'will'],
      routingPath: 'base',
      score: 0.95,
    });

    mockedGenerateNeatChatCandidates.mockReturnValue([onlyUnsafeCandidate]);
    mockedCheckSafety.mockReturnValue({
      ok: false,
      violation: 'incomplete-fragment',
      detail:
        'Response ends on a dangling fragment tail and should not reach the live surface.',
    });

    // Act
    const firstUnsafeTurn = runNeatChatExchange(session, 'how was your day');
    const secondUnsafeTurn = runNeatChatExchange(
      firstUnsafeTurn.updatedSession,
      'what about the weekend',
    );
    const thirdUnsafeTurn = runNeatChatExchange(
      secondUnsafeTurn.updatedSession,
      'hello',
    );
    const fourthUnsafeTurn = runNeatChatExchange(
      thirdUnsafeTurn.updatedSession,
      'tell me about the weekend',
    );

    // Assert
    expect([
      firstUnsafeTurn.response,
      secondUnsafeTurn.response,
      thirdUnsafeTurn.response,
    ]).not.toContain(fourthUnsafeTurn.response);
  });

  it('does not loop the shipped-snapshot fallback floor when rephrase tell and plans are unavailable', () => {
    // Arrange
    const session = createShippedSnapshotLikeFallbackSession();
    const onlyUnsafeCandidate = createRoutingCandidate({
      response: 'i will',
      responseTokens: ['i', 'will'],
      routingPath: 'base',
      score: 0.95,
    });

    mockedGenerateNeatChatCandidates.mockReturnValue([onlyUnsafeCandidate]);
    mockedCheckSafety.mockReturnValue({
      ok: false,
      violation: 'incomplete-fragment',
      detail:
        'Response ends on a dangling fragment tail and should not reach the live surface.',
    });

    // Act
    const firstUnsafeTurn = runNeatChatExchange(session, 'how was your day');
    const secondUnsafeTurn = runNeatChatExchange(
      firstUnsafeTurn.updatedSession,
      'what about the weekend',
    );
    const thirdUnsafeTurn = runNeatChatExchange(
      secondUnsafeTurn.updatedSession,
      'hello',
    );
    const fourthUnsafeTurn = runNeatChatExchange(
      thirdUnsafeTurn.updatedSession,
      'tell me about the weekend',
    );

    // Assert
    expect([
      firstUnsafeTurn.response,
      secondUnsafeTurn.response,
      thirdUnsafeTurn.response,
    ]).not.toContain(fourthUnsafeTurn.response);
  });
});

function createPretrainedLiveSession(
  corpusRetainedTerms: readonly string[] = [
    'hello',
    'steady',
    'answer',
    'i',
    'will',
    'stay',
    'home',
    'have',
    'plans',
    'PUNC_SENTENCE_END',
    'do',
    'you',
    'can',
    'rephrase',
    'that',
    'am',
    'feeling',
    'good',
    'how',
    'was',
    'day',
    'weekend',
    'later',
    'tell',
    'me',
    'about',
  ],
): NeatChatSession {
  const session = createNeatChatSession({
    corpusRetainedTerms,
    recurrentBlockSize: 4,
  });

  return {
    ...session,
    seededTokenPairCount: 7_776,
  };
}

function createShippedSnapshotLikeFallbackSession(): NeatChatSession {
  return createPretrainedLiveSession([
    'hello',
    'steady',
    'answer',
    'i',
    'will',
    'stay',
    'home',
    'have',
    'PUNC_SENTENCE_END',
    'do',
    'you',
    'can',
    'that',
    'am',
    'feeling',
    'good',
    'how',
    'was',
    'day',
    'weekend',
    'later',
    'me',
    'about',
  ]);
}

function createRoutingCandidate({
  response,
  responseTokens,
  routingPath,
  score,
}: {
  readonly response: string;
  readonly responseTokens: readonly string[];
  readonly routingPath: NeatChatRoutingPath;
  readonly score: number;
}): NeatChatRoutingCandidate {
  return {
    response,
    responseTokens,
    routingPath,
    score,
    retrievedMemoryCount: 0,
    retrievedMemoryKeys: [],
  };
}

function withExchangeHistory(
  session: NeatChatSession,
  responses: readonly string[],
): NeatChatSession {
  const exchanges = responses.map((response, responseIndex) =>
    createExchangeRecord(`earlier prompt ${responseIndex + 1}`, response),
  );

  return {
    ...session,
    exchanges,
    learnedExchangeCount: exchanges.length,
    replayBufferExchangeCount: exchanges.length,
  };
}

function createExchangeRecord(
  userMessage: string,
  response: string,
): NeatChatExchangeRecord {
  return {
    userMessage,
    response,
    trainedTokenPairCount: 0,
    userTokens: ['hello'],
    responseTokens: response.split(' '),
  };
}
