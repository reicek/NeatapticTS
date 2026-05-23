import {
  createNeatChatAdaptationManager,
  promoteNeatChatAdaptationCandidate,
  rejectNeatChatAdaptationCandidate,
  scheduleNeatChatAdaptation,
} from './neatChat.adaptation.services';
// @ts-ignore -- W4-03 red contract: adaptation helper types land in W4-04.
import type {
  NeatChatAdaptationCandidate,
  NeatChatAdaptationManager,
  NeatChatCandidateLogEntry,
  ScheduleNeatChatAdaptationOptions,
} from './neatChat.adaptation.types';
import {
  Network,
  toParameterVector,
  type ParameterVector,
} from '../../../src/neataptic.ts';
import * as neataptic from '../../../src/neataptic.ts';
import {
  exportNeatChatSessionV2,
  importNeatChatSessionV2,
} from './neatChat.snapshot.v2.services';
import {
  buildNeatChatVocabulary,
  createNeatChatSession,
  runNeatChatExchange,
} from './neatChat.session.services';

const ADAPTATION_TEST_RETAINED_TERMS = [
  'hello',
  'there',
  'general',
  'kenobi',
  'friend',
  'response',
] as const;

describe('neatChat adaptation services', () => {
  describe('createNeatChatAdaptationManager', () => {
    it('returns empty pendingCandidates and candidateLog arrays', () => {
      // Arrange
      const session = createSessionWithExchange();

      // Act
      const manager = createNeatChatAdaptationManager(session);

      // Assert
      expect({
        candidateLog: readCandidateLogEntries(manager),
        pendingCandidates: readPendingCandidates(manager),
      }).toEqual({
        candidateLog: [],
        pendingCandidates: [],
      });
    });
  });

  describe('scheduleNeatChatAdaptation', () => {
    it('returns a Promise when the session has at least one exchange', async () => {
      // Arrange
      const session = createSessionWithExchange();
      const manager = createNeatChatAdaptationManager(session);
      const options = createAdaptationOptions();

      // Act
      const adaptationPromise = scheduleNeatChatAdaptation(
        manager,
        session,
        options,
      );
      await adaptationPromise;

      // Assert
      expect(adaptationPromise).toBeInstanceOf(Promise);
    });

    it('resolves without throwing when the session has at least one exchange', async () => {
      // Arrange
      const session = createSessionWithExchange();
      const manager = createNeatChatAdaptationManager(session);

      // Act
      const scheduleOutcome = await captureAsyncOutcome(() =>
        scheduleNeatChatAdaptation(manager, session, createAdaptationOptions()),
      );

      // Assert
      expect(scheduleOutcome).toEqual({ didThrow: false });
    });

    it('stores one pending candidate after the scheduled adaptation resolves', async () => {
      // Arrange
      const { manager } = await createScheduledCandidateFixture();

      // Act
      const pendingCandidates = readPendingCandidates(manager);

      // Assert
      expect(pendingCandidates).toHaveLength(1);
    });

    it('stores a pending candidate with a trainedVector ParameterVector payload', async () => {
      // Arrange
      const { manager } = await createScheduledCandidateFixture();

      // Act
      const candidate = readPendingCandidates(manager).at(0);

      // Assert
      expect({
        hasTrainedVectorValues:
          isParameterVector(candidate?.trainedVector) &&
          candidate.trainedVector.values.length > 0,
        isParameterVector: isParameterVector(candidate?.trainedVector),
      }).toEqual({
        hasTrainedVectorValues: true,
        isParameterVector: true,
      });
    });

    it('stores a pending candidate with an evaluationScores object', async () => {
      // Arrange
      const { manager } = await createScheduledCandidateFixture();

      // Act
      const candidate = readPendingCandidates(manager).at(0);

      // Assert
      expect(isRecord(candidate?.evaluationScores)).toBe(true);
    });

    it('appends a candidate when adaptation runs without an options argument', async () => {
      // Arrange
      const session = createSessionWithExchange();
      const manager = createNeatChatAdaptationManager(session);

      // Act
      await scheduleNeatChatAdaptation(manager, session);

      // Assert
      expect(readPendingCandidates(manager)).toHaveLength(1);
    });

    it('appends a candidate when adaptation uses the epochs alias without steps', async () => {
      // Arrange
      const session = createSessionWithExchange();
      const manager = createNeatChatAdaptationManager(session);

      // Act
      await scheduleNeatChatAdaptation(manager, session, { epochs: 2 });

      // Assert
      expect(readPendingCandidates(manager)).toHaveLength(1);
    });

    it('appends a candidate when replayed exchanges contain out-of-vocabulary tokens', async () => {
      // Arrange
      const session = runNeatChatExchange(
        createEmptySession(),
        'mystery token',
      ).updatedSession;
      const manager = createNeatChatAdaptationManager(session);

      // Act
      await scheduleNeatChatAdaptation(
        manager,
        session,
        createAdaptationOptions(),
      );

      // Assert
      expect(readPendingCandidates(manager)).toHaveLength(1);
    });

    it('stores empty evaluationScores when fineTuneVector omits metrics', async () => {
      // Arrange
      const session = createSessionWithExchange();
      const manager = createNeatChatAdaptationManager(session);
      const trainedVector = toParameterVector(session.network);
      const fineTuneVectorSpy = jest
        .spyOn(neataptic, 'fineTuneVector')
        .mockReturnValue({ trainedVector });

      try {
        // Act
        await scheduleNeatChatAdaptation(
          manager,
          session,
          createAdaptationOptions(),
        );

        // Assert
        expect(readPendingCandidates(manager).at(0)?.evaluationScores).toEqual(
          {},
        );
      } finally {
        fineTuneVectorSpy.mockRestore();
      }
    });

    it('stores a pending candidate whose trainedOnExchangeCount matches the exchange count', async () => {
      // Arrange
      const { manager, session } = await createScheduledCandidateFixture();

      // Act
      const candidate = readPendingCandidates(manager).at(0);

      // Assert
      expect(candidate?.trainedOnExchangeCount).toBe(session.exchanges.length);
    });

    it('stores a pending candidate whose trainedOnExchangeCount honors a floored maxExchanges window', async () => {
      // Arrange
      const session = createSessionWithExchangeHistory([
        'hello friend',
        'general kenobi',
      ]);
      const manager = createNeatChatAdaptationManager(session);

      // Act
      await scheduleNeatChatAdaptation(manager, session, {
        ...createAdaptationOptions(),
        maxExchanges: 1.9,
      });
      const candidate = readPendingCandidates(manager).at(0);

      // Assert
      expect({
        exchangeCount: session.exchanges.length,
        trainedOnExchangeCount: candidate?.trainedOnExchangeCount,
      }).toEqual({
        exchangeCount: 2,
        trainedOnExchangeCount: 1,
      });
    });

    it('does not mutate session.network while adaptation is running', async () => {
      // Arrange
      const session = createSessionWithExchange();
      const manager = createNeatChatAdaptationManager(session);
      const originalNetworkJson = JSON.stringify(
        cloneSessionNetwork(session).toJSON(),
      );

      // Act
      await scheduleNeatChatAdaptation(
        manager,
        session,
        createAdaptationOptions(),
      );

      // Assert
      expect(JSON.stringify(session.network.toJSON())).toBe(
        originalNetworkJson,
      );
    });

    it('resolves even when the session has no exchanges', async () => {
      // Arrange
      const session = createEmptySession();
      const manager = createNeatChatAdaptationManager(session);

      // Act
      const scheduleOutcome = await captureAsyncOutcome(() =>
        scheduleNeatChatAdaptation(manager, session, createAdaptationOptions()),
      );

      // Assert
      expect(scheduleOutcome).toEqual({ didThrow: false });
    });

    it('rethrows fine-tune failures when replay training cases exist', async () => {
      // Arrange
      const session = createMalformedSessionWithVocabularyMismatch();
      const manager = createNeatChatAdaptationManager(session);

      // Act
      const scheduleOutcome = await captureAsyncOutcome(() =>
        scheduleNeatChatAdaptation(manager, session, createAdaptationOptions()),
      );

      // Assert
      expect(scheduleOutcome).toEqual({ didThrow: true });
    });
  });

  describe('promoteNeatChatAdaptationCandidate', () => {
    it('returns a new session with a different network object than the source session', async () => {
      // Arrange
      const { manager, session } = await createScheduledCandidateFixture();

      // Act
      const promotedSession = promoteNeatChatAdaptationCandidate(
        manager,
        session,
        0,
      );

      // Assert
      expect(promotedSession.network).not.toBe(session.network);
    });

    it('changes the promoted session parameter vector relative to the source session', async () => {
      // Arrange
      const { manager, session } = await createScheduledCandidateFixture();
      const sourceVectorValues = Array.from(
        toParameterVector(session.network).values,
      );

      // Act
      const promotedSession = promoteNeatChatAdaptationCandidate(
        manager,
        session,
        0,
      );

      // Assert
      expect(
        Array.from(toParameterVector(promotedSession.network).values),
      ).not.toEqual(sourceVectorValues);
    });

    it('appends a promoted candidate log entry to the manager', async () => {
      // Arrange
      const { manager, session } = await createScheduledCandidateFixture();

      // Act
      promoteNeatChatAdaptationCandidate(manager, session, 0);

      // Assert
      expect(readCandidateLogSummary(manager).firstStatus).toBe('promoted');
    });

    it('throws or rejects when candidateIndex is out of range', async () => {
      // Arrange
      const { manager, session } = await createScheduledCandidateFixture();

      // Act
      const promotionError = await captureAsyncError(() =>
        promoteNeatChatAdaptationCandidate(manager, session, 99),
      );

      // Assert
      expect(promotionError instanceof Error).toBe(true);
    });
  });

  describe('rejectNeatChatAdaptationCandidate', () => {
    it('removes the candidate from pendingCandidates', async () => {
      // Arrange
      const { manager } = await createScheduledCandidateFixture();

      // Act
      rejectNeatChatAdaptationCandidate(manager, 0);

      // Assert
      expect(readPendingCandidates(manager)).toHaveLength(0);
    });

    it('appends a rejected candidate log entry to the manager', async () => {
      // Arrange
      const { manager } = await createScheduledCandidateFixture();

      // Act
      rejectNeatChatAdaptationCandidate(manager, 0);

      // Assert
      expect(readCandidateLogSummary(manager).firstStatus).toBe('rejected');
    });
  });

  describe('snapshot integration', () => {
    it('round-trips candidateLog length and the first status after promotion', async () => {
      // Arrange
      const { manager, session } = await createScheduledCandidateFixture();
      const promotedSession = promoteNeatChatAdaptationCandidate(
        manager,
        session,
        0,
      );

      // Act
      const importedSession = importNeatChatSessionV2(
        exportNeatChatSessionV2(promotedSession),
      );

      // Assert
      expect(readCandidateLogSummary(importedSession)).toEqual({
        firstStatus: 'promoted',
        length: 1,
      });
    });

    it('imports v2 snapshots without candidateLog using an empty default summary', () => {
      // Arrange
      const legacySnapshot = createLegacySnapshotWithoutCandidateLog(
        createSessionWithExchange(),
      );

      // Act
      const importedSession = importNeatChatSessionV2(legacySnapshot);

      // Assert
      expect(readCandidateLogSummary(importedSession)).toEqual({
        firstStatus: undefined,
        length: 0,
      });
    });
  });
});

async function createScheduledCandidateFixture(
  session = createSessionWithExchange(),
): Promise<{
  readonly manager: NeatChatAdaptationManager;
  readonly options: ScheduleNeatChatAdaptationOptions;
  readonly session: ReturnType<typeof createNeatChatSession>;
}> {
  const manager = createNeatChatAdaptationManager(session);
  const options = createAdaptationOptions();

  await scheduleNeatChatAdaptation(manager, session, options);

  return { manager, options, session };
}

function createAdaptationOptions(): ScheduleNeatChatAdaptationOptions {
  return {
    learningRate: 0.05,
    seed: 7,
    steps: 2,
  };
}

function createEmptySession() {
  return createNeatChatSession({
    architectureFamily: 'lstm',
    contextWindowTokenCount: 6,
    corpusRetainedTerms: ADAPTATION_TEST_RETAINED_TERMS,
    recurrentBlockSize: 8,
    seedConversationLines: ['hello there', 'general kenobi'],
  });
}

function createSessionWithExchange() {
  return createSessionWithExchangeHistory(['hello friend']);
}

function createSessionWithExchangeHistory(userMessages: readonly string[]) {
  return userMessages.reduce(
    (currentSession, userMessage) =>
      runNeatChatExchange(currentSession, userMessage).updatedSession,
    createEmptySession(),
  );
}

function createMalformedSessionWithVocabularyMismatch() {
  const session = createSessionWithExchange();

  return {
    ...session,
    vocabulary: buildNeatChatVocabulary([
      ...ADAPTATION_TEST_RETAINED_TERMS,
      'mismatch',
      'token',
    ]),
  };
}

function readPendingCandidates(
  manager: NeatChatAdaptationManager,
): readonly NeatChatAdaptationCandidate[] {
  const pendingCandidates = (
    manager as { readonly pendingCandidates?: unknown }
  ).pendingCandidates;

  return Array.isArray(pendingCandidates)
    ? (pendingCandidates as readonly NeatChatAdaptationCandidate[])
    : [];
}

function readCandidateLogEntries(
  value: unknown,
): readonly NeatChatCandidateLogEntry[] {
  const candidateLog =
    value && typeof value === 'object' && 'candidateLog' in value
      ? (value as { readonly candidateLog?: unknown }).candidateLog
      : undefined;

  return Array.isArray(candidateLog)
    ? (candidateLog as readonly NeatChatCandidateLogEntry[])
    : [];
}

function readCandidateLogSummary(value: unknown) {
  const candidateLog = readCandidateLogEntries(value);

  return {
    firstStatus: candidateLog.at(0)?.status,
    length: candidateLog.length,
  };
}

function cloneSessionNetwork(
  session: ReturnType<typeof createNeatChatSession>,
) {
  return Network.fromJSON(session.network.toJSON());
}

function createLegacySnapshotWithoutCandidateLog(
  session: ReturnType<typeof createNeatChatSession>,
) {
  const snapshot = exportNeatChatSessionV2(session) as unknown as {
    readonly extensions: {
      readonly neatchat: Record<string, unknown>;
    };
  } & Record<string, unknown>;
  const { candidateLog: _candidateLog, ...legacyNeatChatExtension } =
    snapshot.extensions.neatchat;

  return {
    ...snapshot,
    extensions: {
      ...snapshot.extensions,
      neatchat: legacyNeatChatExtension,
    },
  };
}

async function captureAsyncOutcome(
  action: () => Promise<unknown> | unknown,
): Promise<{ readonly didThrow: boolean }> {
  try {
    await action();
    return { didThrow: false };
  } catch {
    return { didThrow: true };
  }
}

async function captureAsyncError(
  action: () => Promise<unknown> | unknown,
): Promise<unknown> {
  try {
    await action();
    return null;
  } catch (error) {
    return error;
  }
}

function isParameterVector(value: unknown): value is ParameterVector {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const parameterVector = value as {
    readonly layout?: unknown;
    readonly values?: unknown;
  };

  return Boolean(
    parameterVector.layout &&
    typeof parameterVector.layout === 'object' &&
    parameterVector.values instanceof Float64Array,
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}
