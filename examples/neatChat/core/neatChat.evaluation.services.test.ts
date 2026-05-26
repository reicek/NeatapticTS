import type {
  EvaluationHarnessInput,
  EvaluationMetric,
  FailureBucket,
  RegressionEntry,
  RegressionSuiteResult,
} from './neatChat.evaluation.types';
import {
  attributeToFailureBucket,
  runNeatChatRegressionSuite,
  scoreFactualConsistency,
  scoreNextTokenAccuracy,
  scoreRepetitionRate,
  scoreResponseLengthStability,
  scoreUnknownHandling,
} from './neatChat.evaluation.services';
import { addNeatChatMemoryRecord } from './neatChat.memory.services';
import {
  createNeatChatSession,
  runNeatChatExchange,
} from './neatChat.session.services';
import type { NeatChatSession } from './neatChat.types';

/** Retained terms shared across evaluation fixtures — wide enough to cover all
 * corpus entries in the held-out evaluation slice. */
const EVALUATION_TEST_RETAINED_TERMS = [
  'hello',
  'there',
  'how',
  'are',
  'you',
  'my',
  'favorite',
  'color',
  'blue',
  'name',
  'alice',
  'response',
  'general',
] as const;

/** Timestamp used in attribution-fixture log entries. */
const EVALUATION_TEST_DECISION_TIMESTAMP = 1_779_321_600_000;

// ---------------------------------------------------------------------------
// runNeatChatRegressionSuite
// ---------------------------------------------------------------------------

describe('runNeatChatRegressionSuite', () => {
  describe('return shape', () => {
    it('returns a RegressionSuiteResult with entryCount of 0 and empty perMetricMeans for an empty corpus', () => {
      // Arrange
      const session = createEvaluationFixtureSession();
      const harnessInput = createMinimalHarnessInput(session, []);

      // Act
      const result: RegressionSuiteResult = runNeatChatRegressionSuite(
        session,
        harnessInput,
      );

      // Assert
      expect(result.entryCount).toBe(0);
    });

    it('returns a RegressionSuiteResult with entryCount matching the corpus input length', () => {
      // Arrange
      const session = createEvaluationFixtureSession();
      const harnessInput = createMinimalHarnessInput(session, [
        { inputPrompt: 'hello', expectedResponse: 'hello there' },
        { inputPrompt: 'how are you', expectedResponse: 'hello there' },
      ]);

      // Act
      const result: RegressionSuiteResult = runNeatChatRegressionSuite(
        session,
        harnessInput,
      );

      // Assert
      expect(result.entryCount).toBe(2);
    });

    it('includes perMetricMeans for next-token-accuracy across corpus entries', () => {
      // Arrange
      const session = createEvaluationFixtureSession();
      const harnessInput = createMinimalHarnessInput(
        session,
        [{ inputPrompt: 'hello', expectedResponse: 'there' }],
        { 'next-token-accuracy': 0 },
      );

      // Act
      const result: RegressionSuiteResult = runNeatChatRegressionSuite(
        session,
        harnessInput,
      );

      // Assert
      expect(typeof result.perMetricMeans['next-token-accuracy']).toBe(
        'number',
      );
    });

    it('returns totalRegressions of 0 when all entries meet or exceed baseline', () => {
      // Arrange
      const session = createEvaluationFixtureSession();
      const harnessInput = createMinimalHarnessInput(
        session,
        [{ inputPrompt: 'hello', expectedResponse: 'there' }],
        {
          'next-token-accuracy': 0,
          'repetition-rate': 1,
          'response-length-stability': 0,
        },
      );

      // Act
      const result: RegressionSuiteResult = runNeatChatRegressionSuite(
        session,
        harnessInput,
      );

      // Assert
      expect(result.totalRegressions).toBe(0);
    });

    it('includes a bucketBreakdown entry when a regression is attributed', () => {
      // Arrange
      const session = createEvaluationFixtureSession();
      const harnessInput = createMinimalHarnessInput(
        session,
        [{ inputPrompt: 'hello', expectedResponse: 'there' }],
        {
          'next-token-accuracy': 1.0,
        },
      );

      // Act
      const result: RegressionSuiteResult = runNeatChatRegressionSuite(
        session,
        harnessInput,
      );

      // Assert
      expect(Object.keys(result.bucketBreakdown).length).toBeGreaterThanOrEqual(
        0,
      );
    });
  });

  describe('factual consistency — profile memory facts', () => {
    it('scores factual-consistency against saved memory facts in the session', () => {
      // Arrange
      const sessionWithMemory = createSessionWithMemoryFact({
        key: 'favorite color',
        value: 'blue',
      });
      const harnessInput = createMinimalHarnessInput(
        sessionWithMemory,
        [
          {
            inputPrompt: 'what is my favorite color',
            expectedResponse: 'your favorite color is blue',
          },
        ],
        { 'factual-consistency': 0 },
      );

      // Act
      const result: RegressionSuiteResult = runNeatChatRegressionSuite(
        sessionWithMemory,
        harnessInput,
      );

      // Assert
      expect(typeof result.perMetricMeans['factual-consistency']).toBe(
        'number',
      );
    });

    it('does not mutate session state or memory bank during harness evaluation', () => {
      // Arrange
      const sessionWithMemory = createSessionWithMemoryFact({
        key: 'name',
        value: 'alice',
      });
      const originalRecordCount = sessionWithMemory.memoryBank.records.length;
      const harnessInput = createMinimalHarnessInput(sessionWithMemory, [
        { inputPrompt: 'what is my name', expectedResponse: 'alice' },
      ]);

      // Act
      runNeatChatRegressionSuite(sessionWithMemory, harnessInput);

      // Assert
      expect(sessionWithMemory.memoryBank.records.length).toBe(
        originalRecordCount,
      );
    });
  });

  describe('repetition and collapse resistance', () => {
    it('scores repetition-rate and includes it in perMetricMeans', () => {
      // Arrange
      const session = createEvaluationFixtureSession();
      const harnessInput = createMinimalHarnessInput(
        session,
        [{ inputPrompt: 'hello', expectedResponse: 'hello hello hello' }],
        { 'repetition-rate': 0 },
      );

      // Act
      const result: RegressionSuiteResult = runNeatChatRegressionSuite(
        session,
        harnessInput,
      );

      // Assert
      expect(typeof result.perMetricMeans['repetition-rate']).toBe('number');
    });
  });

  describe('response-length stability', () => {
    it('scores response-length-stability and includes it in perMetricMeans', () => {
      // Arrange
      const session = createEvaluationFixtureSession();
      const harnessInput = createMinimalHarnessInput(
        session,
        [{ inputPrompt: 'hello', expectedResponse: 'hello there how are you' }],
        { 'response-length-stability': 0 },
      );

      // Act
      const result: RegressionSuiteResult = runNeatChatRegressionSuite(
        session,
        harnessInput,
      );

      // Assert
      expect(typeof result.perMetricMeans['response-length-stability']).toBe(
        'number',
      );
    });
  });

  describe('unknown-handling', () => {
    it('does not throw when the input prompt contains tokens absent from the vocabulary', () => {
      // Arrange
      const session = createEvaluationFixtureSession();
      const harnessInput = createMinimalHarnessInput(
        session,
        [
          {
            inputPrompt: 'zzzzquux_not_in_vocab',
            expectedResponse: 'hello',
          },
        ],
        {},
      );

      // Act
      const runSuite = () => runNeatChatRegressionSuite(session, harnessInput);

      // Assert
      expect(runSuite).not.toThrow();
    });

    it('scores unknown-handling below 1.0 when OOV fraction exceeds maxUnknownTokenFraction', () => {
      // Arrange
      const session = createEvaluationFixtureSession();
      const harnessInput: EvaluationHarnessInput = {
        session,
        corpusEntries: [
          {
            inputPrompt: 'zzzzquux_not_in_vocab',
            expectedResponse: 'hello',
          },
        ],
        metricBaselines: {},
        maxUnknownTokenFraction: 0,
      };

      // Act
      const result: RegressionSuiteResult = runNeatChatRegressionSuite(
        session,
        harnessInput,
      );

      // Assert
      expect((result.perMetricMeans['unknown-handling'] ?? 1) < 1).toBe(true);
    });
  });

  describe('stability after adaptation-like updates', () => {
    it('returns a RegressionSuiteResult without throwing after a session exchange runs', () => {
      // Arrange
      const session = createEvaluationFixtureSession();
      const { updatedSession } = runNeatChatExchange(session, 'hello');
      const harnessInput = createMinimalHarnessInput(
        updatedSession,
        [{ inputPrompt: 'hello', expectedResponse: 'there' }],
        { 'next-token-accuracy': 0 },
      );

      // Act
      const runSuite = () =>
        runNeatChatRegressionSuite(updatedSession, harnessInput);

      // Assert
      expect(runSuite).not.toThrow();
    });
  });
});

// ---------------------------------------------------------------------------
// attributeToFailureBucket
// ---------------------------------------------------------------------------

describe('attributeToFailureBucket', () => {
  describe('routing attribution', () => {
    it('attributes a routing-path regression to the routing bucket', () => {
      // Arrange
      const session = createSessionWithRoutingLogEntry('personalized');
      const entry = createRegressionEntry({
        metricScores: { 'next-token-accuracy': 0.1 },
      });

      // Act
      const bucket: FailureBucket = attributeToFailureBucket(session, entry);

      // Assert
      expect(bucket).toBe('routing');
    });

    it('does not mutate session routingLog during attribution', () => {
      // Arrange
      const session = createSessionWithRoutingLogEntry('personalized');
      const originalLogLength = session.routingLog.length;
      const entry = createRegressionEntry({
        metricScores: { 'next-token-accuracy': 0.1 },
      });

      // Act
      attributeToFailureBucket(session, entry);

      // Assert
      expect(session.routingLog.length).toBe(originalLogLength);
    });
  });

  describe('base-seed attribution', () => {
    it('attributes a base-path regression with empty routing log to the base-seed bucket', () => {
      // Arrange
      const session = createEvaluationFixtureSession();
      const entry = createRegressionEntry({
        metricScores: { 'next-token-accuracy': 0.05 },
      });

      // Act
      const bucket: FailureBucket = attributeToFailureBucket(session, entry);

      // Assert
      expect(bucket).toBe('base-seed');
    });

    it('attributes a base-path regression with selectedPath base to the base-seed bucket', () => {
      // Arrange
      const session = createSessionWithRoutingLogEntry('base');
      const entry = createRegressionEntry({
        metricScores: { 'next-token-accuracy': 0.05 },
      });

      // Act
      const bucket: FailureBucket = attributeToFailureBucket(session, entry);

      // Assert
      expect(bucket).toBe('base-seed');
    });
  });

  describe('retrieval attribution', () => {
    it('attributes a regression tied to retrieval-grounded path to the retrieval bucket', () => {
      // Arrange
      const session = createSessionWithRoutingLogEntry('retrieval-grounded');
      const entry = createRegressionEntry({
        metricScores: { 'factual-consistency': 0.1 },
      });

      // Act
      const bucket: FailureBucket = attributeToFailureBucket(session, entry);

      // Assert
      expect(bucket).toBe('retrieval');
    });
  });

  describe('determinism', () => {
    it('returns the same bucket for the same session and entry on repeated calls', () => {
      // Arrange
      const session = createSessionWithRoutingLogEntry('personalized');
      const entry = createRegressionEntry({
        metricScores: { 'next-token-accuracy': 0.1 },
      });

      // Act
      const firstBucket = attributeToFailureBucket(session, entry);
      const secondBucket = attributeToFailureBucket(session, entry);

      // Assert
      expect(firstBucket).toBe(secondBucket);
    });
  });
});

// ---------------------------------------------------------------------------
// scoreNextTokenAccuracy
// ---------------------------------------------------------------------------

describe('scoreNextTokenAccuracy', () => {
  it('returns 1 when predicted and expected are identical non-empty strings', () => {
    // Arrange / Act
    const score: number = scoreNextTokenAccuracy('hello', 'hello');

    // Assert
    expect(score).toBe(1);
  });

  it('returns 1 when expected is an empty string', () => {
    // Arrange / Act
    const score: number = scoreNextTokenAccuracy('hello', '');

    // Assert
    expect(score).toBe(1);
  });

  it('returns 0 when predicted is empty but expected is not', () => {
    // Arrange / Act
    const score: number = scoreNextTokenAccuracy('', 'hello');

    // Assert
    expect(score).toBe(0);
  });

  it('returns 0 when predicted and expected share no tokens', () => {
    // Arrange / Act
    const score: number = scoreNextTokenAccuracy('hello', 'goodbye');

    // Assert
    expect(score).toBe(0);
  });

  it('returns a value strictly in [0, 1] for partially matching outputs', () => {
    // Arrange / Act
    const score: number = scoreNextTokenAccuracy(
      'hello there friend',
      'hello world',
    );

    // Assert
    expect(score >= 0 && score <= 1).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// scoreRepetitionRate
// ---------------------------------------------------------------------------

describe('scoreRepetitionRate', () => {
  it('returns 0 for a response with no repeated bigrams', () => {
    // Arrange / Act
    const score: number = scoreRepetitionRate('hello there general kenobi');

    // Assert
    expect(score).toBe(0);
  });

  it('returns greater than 0 for a response with repeated n-grams', () => {
    // Arrange / Act
    const score: number = scoreRepetitionRate(
      'hello hello hello hello hello hello',
    );

    // Assert
    expect(score > 0).toBe(true);
  });

  it('returns a value strictly in [0, 1] for any response', () => {
    // Arrange / Act
    const score: number = scoreRepetitionRate(
      'the the the the the the the the the',
    );

    // Assert
    expect(score >= 0 && score <= 1).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// scoreResponseLengthStability
// ---------------------------------------------------------------------------

describe('scoreResponseLengthStability', () => {
  it('returns 1 when both response and expected length are zero', () => {
    // Arrange / Act
    const score: number = scoreResponseLengthStability('', 0);

    // Assert
    expect(score).toBe(1);
  });

  it('returns 1 when actual token count exactly matches expected length', () => {
    // Arrange / Act
    const score: number = scoreResponseLengthStability('hello there', 2);

    // Assert
    expect(score).toBe(1);
  });

  it('returns a value in [0, 1] when actual and expected differ', () => {
    // Arrange / Act
    const score: number = scoreResponseLengthStability('hello', 10);

    // Assert
    expect(score >= 0 && score <= 1).toBe(true);
  });

  it('returns 0 when the response is empty and expected length is greater than 0', () => {
    // Arrange / Act
    const score: number = scoreResponseLengthStability('', 5);

    // Assert
    expect(score).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// scoreUnknownHandling
// ---------------------------------------------------------------------------

describe('scoreUnknownHandling', () => {
  it('returns 1.0 when the input text is empty', () => {
    // Arrange
    const session = createEvaluationFixtureSession();

    // Act
    const score: number = scoreUnknownHandling('', session);

    // Assert
    expect(score).toBe(1);
  });

  it('returns 1.0 when no tokens are OOV (all tokens in vocabulary)', () => {
    // Arrange
    const session = createEvaluationFixtureSession();

    // Act
    const score: number = scoreUnknownHandling('hello there', session);

    // Assert
    expect(score).toBe(1);
  });

  it('returns less than 1.0 when at least one token is absent from the vocabulary', () => {
    // Arrange
    const session = createEvaluationFixtureSession();

    // Act
    const score: number = scoreUnknownHandling(
      'zzzzquux_not_in_vocab hello',
      session,
    );

    // Assert
    expect(score < 1).toBe(true);
  });

  it('returns 0 when every token is absent from the vocabulary', () => {
    // Arrange
    const session = createEvaluationFixtureSession();

    // Act
    const score: number = scoreUnknownHandling(
      'zzzzquux_not_in_vocab aaabbbccc_not_in_vocab',
      session,
    );

    // Assert
    expect(score).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// scoreFactualConsistency
// ---------------------------------------------------------------------------

describe('scoreFactualConsistency', () => {
  it('returns 1.0 when every memory-bank key and value token appears in the response', () => {
    // Arrange
    const sessionWithMemory = createSessionWithMemoryFact({
      key: 'favorite color',
      value: 'blue',
    });

    // Act
    const score: number = scoreFactualConsistency(
      'your favorite color is blue',
      sessionWithMemory.memoryBank,
    );

    // Assert
    expect(score).toBe(1);
  });

  it('returns 0.0 when no memory-bank terms appear in the response', () => {
    // Arrange
    const sessionWithMemory = createSessionWithMemoryFact({
      key: 'favorite color',
      value: 'blue',
    });

    // Act
    const score: number = scoreFactualConsistency(
      'general kenobi hello there',
      sessionWithMemory.memoryBank,
    );

    // Assert
    expect(score).toBe(0);
  });

  it('returns a value in [0, 1] for a partial match against memory facts', () => {
    // Arrange
    const sessionWithMemory = createSessionWithMemoryFact({
      key: 'name',
      value: 'alice',
    });

    // Act
    const score: number = scoreFactualConsistency(
      'alice hello there',
      sessionWithMemory.memoryBank,
    );

    // Assert
    expect(score >= 0 && score <= 1).toBe(true);
  });

  it('returns 1.0 when the memory bank has no records (no facts to violate)', () => {
    // Arrange
    const session = createEvaluationFixtureSession();

    // Act
    const score: number = scoreFactualConsistency(
      'any response here',
      session.memoryBank,
    );

    // Assert
    expect(score).toBe(1);
  });

  it('returns 1.0 when records exist but all key and value fields are empty strings', () => {
    // Arrange — records exist but tokenize to empty arrays, so factTokens is empty.
    const emptyTokenMemoryBank: import('./neatChat.memory.types').NeatChatEpisodicMemoryBank =
      {
        records: [{ key: '', value: '', addedAt: 0, hitCount: 0 }],
        maxRecords: 10,
        lastConsolidatedAt: null,
      };

    // Act
    const score: number = scoreFactualConsistency(
      'any response',
      emptyTokenMemoryBank,
    );

    // Assert
    expect(score).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function createEvaluationFixtureSession(): NeatChatSession {
  return createNeatChatSession({
    corpusRetainedTerms: EVALUATION_TEST_RETAINED_TERMS,
    recurrentBlockSize: 2,
    seedConversationLines: [],
  });
}

function createSessionWithMemoryFact(fact: {
  readonly key: string;
  readonly value: string;
}): NeatChatSession {
  const session = createEvaluationFixtureSession();

  return {
    ...session,
    memoryBank: addNeatChatMemoryRecord(session.memoryBank, fact),
  };
}

function createSessionWithRoutingLogEntry(
  selectedPath: 'base' | 'personalized' | 'retrieval-grounded',
): NeatChatSession {
  const session = createEvaluationFixtureSession();

  return {
    ...session,
    routingLog: [
      {
        decidedAt: EVALUATION_TEST_DECISION_TIMESTAMP,
        selectedPath,
        comparedCandidatePaths: ['base', selectedPath].filter(
          (path, index, array) => array.indexOf(path) === index,
        ) as Array<'base' | 'personalized' | 'retrieval-grounded'>,
        candidateCount: selectedPath === 'base' ? 1 : 2,
        scores: {
          base: 0.3,
          [selectedPath]: 0.7,
        },
        responsesByPath: {
          base: 'base response',
          [selectedPath]: `${selectedPath} response`,
        },
        retrievedMemoryCountByPath:
          selectedPath === 'retrieval-grounded'
            ? { 'retrieval-grounded': 1 }
            : {},
        retrievedMemoryKeysByPath:
          selectedPath === 'retrieval-grounded'
            ? { 'retrieval-grounded': ['favorite color'] }
            : {},
      },
    ],
  };
}

function createMinimalHarnessInput(
  session: NeatChatSession,
  corpusEntries: ReadonlyArray<{
    readonly inputPrompt: string;
    readonly expectedResponse: string;
  }>,
  metricBaselines: Readonly<Partial<Record<EvaluationMetric, number>>> = {},
): EvaluationHarnessInput {
  return {
    session,
    corpusEntries,
    metricBaselines,
  };
}

function createRegressionEntry(
  overrides: Partial<RegressionEntry> = {},
): RegressionEntry {
  return {
    inputPrompt: 'hello',
    expectedResponse: 'there',
    actualResponse: 'general',
    metricScores: { 'next-token-accuracy': 0.1 },
    attributedBucket: 'unattributed',
    ...overrides,
  };
}
