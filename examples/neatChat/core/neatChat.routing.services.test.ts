import {
  appendNeatChatRoutingDecision,
  generateNeatChatCandidates,
  selectNeatChatCandidate,
} from './neatChat.routing.services';
import type {
  NeatChatRoutingCandidate,
  NeatChatRoutingDecisionLogEntry,
  NeatChatRoutingPath,
} from './neatChat.routing.types';
import { toParameterVector } from '../../../src/neataptic.ts';
import { createNeatChatAdaptationManager } from './neatChat.adaptation.services';
import type {
  NeatChatAdaptationCandidate,
  NeatChatAdaptationManager,
} from './neatChat.adaptation.types';
import {
  addNeatChatMemoryRecord,
  retrieveNeatChatMemories,
} from './neatChat.memory.services';
import type { NeatChatMemoryRetrievalResult } from './neatChat.memory.types';
import { createNeatChatSession } from './neatChat.session.services';
import {
  exportNeatChatSessionV2,
  importNeatChatSessionV2,
} from './neatChat.snapshot.v2.services';
import type {
  NeatChatSession,
  NeatChatSessionSnapshotV2,
} from './neatChat.types';

const ROUTING_TEST_RETAINED_TERMS = [
  'favorite',
  'color',
  'blue',
  'ocean',
  'assistant',
  'response',
] as const;
const ROUTING_TEST_DECISION_TIMESTAMP = 1_779_321_600_000;

describe('neatChat routing services', () => {
  describe('generateNeatChatCandidates', () => {
    it('always includes the base path when no personalization or memory path is active', () => {
      // Arrange
      const session = createRoutingFixtureSession();
      const manager = createNeatChatAdaptationManager(session);

      // Act
      const candidates = generateNeatChatCandidates(session, manager, []);

      // Assert
      expect(readRoutingPaths(candidates)).toEqual(['base']);
    });

    it('falls back to the latest user message when promptText is omitted', () => {
      // Arrange
      const session = createRoutingFallbackSession({
        exchanges: [
          {
            userMessage: 'favorite',
            response: 'base response',
            responseTokens: ['base', 'response'],
            trainedTokenPairCount: 0,
            userTokens: ['favorite'],
          },
        ],
      });
      const manager = createNeatChatAdaptationManager(session);

      // Act
      const candidates = generateNeatChatCandidates(session, manager, []);

      // Assert
      expect(candidates[0]?.response).toBe('favorite');
    });

    it('skips the personalized path when the live network cannot be cloned', () => {
      // Arrange
      const session = createRoutingFixtureSession();
      removeNetworkClone(session);
      const manager = createManagerWithPendingPersonalization(session);

      // Act
      const candidates = generateNeatChatCandidates(session, manager, []);

      // Assert
      expect(readRoutingPaths(candidates)).toEqual(['base']);
    });

    it('compares base, personalized, and retrieval-grounded paths when all inputs are active', () => {
      // Arrange
      const session = createRoutingFixtureSession();
      const manager = createManagerWithPendingPersonalization(session);
      const retrievedMemories = createRetrievedMemoryFixture(session);

      // Act
      const candidates = generateNeatChatCandidates(
        session,
        manager,
        retrievedMemories,
      );

      // Assert
      expect(readRoutingPaths(candidates)).toEqual([
        'base',
        'personalized',
        'retrieval-grounded',
      ]);
    });

    it('keeps the personalized path when the first fallback evaluation score is used', () => {
      // Arrange
      const session = createRoutingFixtureSession();
      const manager = createManagerWithPendingPersonalization(session, {
        repetitionRate: 0.25,
      });

      // Act
      const candidates = generateNeatChatCandidates(session, manager, []);

      // Assert
      expect(readRoutingPaths(candidates)).toEqual(['base', 'personalized']);
    });

    it('keeps the personalized path when no evaluation scores are available', () => {
      // Arrange
      const session = createRoutingFixtureSession();
      const manager = createManagerWithPendingPersonalization(session, {});

      // Act
      const candidates = generateNeatChatCandidates(session, manager, []);

      // Assert
      expect(readRoutingPaths(candidates)).toEqual(['base', 'personalized']);
    });

    it('clamps non-finite adaptation scores instead of dropping the personalized path', () => {
      // Arrange
      const session = createRoutingFixtureSession();
      const manager = createManagerWithPendingPersonalization(session, {
        heldOutAccuracy: Number.POSITIVE_INFINITY,
      });

      // Act
      const candidates = generateNeatChatCandidates(session, manager, []);

      // Assert
      expect(readRoutingPaths(candidates)).toEqual(['base', 'personalized']);
    });

    it('surfaces retrieved-memory usage and provenance on the retrieval-grounded candidate', () => {
      // Arrange
      const session = createRoutingFixtureSession();
      const manager = createNeatChatAdaptationManager(session);
      const retrievedMemories = createRetrievedMemoryFixture(session);

      // Act
      const candidates = generateNeatChatCandidates(
        session,
        manager,
        retrievedMemories,
      );
      const retrievalCandidate = candidates.find(
        (candidate: NeatChatRoutingCandidate) =>
          candidate.routingPath === 'retrieval-grounded',
      );

      // Assert
      expect({
        retrievedMemoryCount: retrievalCandidate?.retrievedMemoryCount,
        retrievedMemoryKeys: retrievalCandidate?.retrievedMemoryKeys,
      }).toEqual({
        retrievedMemoryCount: 1,
        retrievedMemoryKeys: ['favorite color'],
      });
    });

    it('does not auto-promote pending adaptation candidates or mutate live weights', () => {
      // Arrange
      const session = createRoutingFixtureSession();
      const manager = createManagerWithPendingPersonalization(session);
      const originalNetworkJson = JSON.stringify(session.network.toJSON());

      // Act
      generateNeatChatCandidates(session, manager, []);

      // Assert
      expect({
        candidateLogLength: manager.candidateLog.length,
        networkJson: JSON.stringify(session.network.toJSON()),
        pendingCandidateCount: manager.pendingCandidates.length,
      }).toEqual({
        candidateLogLength: 0,
        networkJson: originalNetworkJson,
        pendingCandidateCount: 1,
      });
    });

    it('falls back to retrieval-memory coverage when grounded response decoding yields no tokens', () => {
      // Arrange
      const session = createRoutingFallbackSession();
      const manager = createNeatChatAdaptationManager(session);
      const retrievedMemories = [createRetrievedMemoryResult('   ', '   ')];

      // Act
      const candidates = generateNeatChatCandidates(
        session,
        manager,
        retrievedMemories,
        '   ',
      );
      const retrievalCandidate = candidates.find(
        (candidate: NeatChatRoutingCandidate) =>
          candidate.routingPath === 'retrieval-grounded',
      );

      // Assert
      expect(retrievalCandidate?.score).toBeCloseTo(1 / 6);
    });
  });

  describe('selectNeatChatCandidate', () => {
    it('throws when no routing candidates are available', () => {
      // Arrange
      const selectAction = () => selectNeatChatCandidate([]);

      // Assert
      expect(selectAction).toThrow(RangeError);
    });

    it('selects the highest-scoring candidate deterministically', () => {
      // Arrange
      const candidates = createScoredRoutingCandidates({
        baseScore: 0.2,
        personalizedScore: 0.8,
        retrievalGroundedScore: 0.5,
      });

      // Act
      const firstSelection = selectNeatChatCandidate(candidates);
      const secondSelection = selectNeatChatCandidate(candidates);

      // Assert
      expect([firstSelection.routingPath, secondSelection.routingPath]).toEqual(
        ['personalized', 'personalized'],
      );
    });

    it('breaks equal-score ties in favor of the base path', () => {
      // Arrange
      const candidates = createScoredRoutingCandidates({
        baseScore: 0.5,
        personalizedScore: 0.5,
        retrievalGroundedScore: 0.5,
      }).toReversed();

      // Act
      const selectedCandidate = selectNeatChatCandidate(candidates);

      // Assert
      expect(selectedCandidate.routingPath).toBe('base');
    });
  });

  describe('appendNeatChatRoutingDecision', () => {
    it('appends a decision entry with the selected path and compared alternatives', () => {
      // Arrange
      const candidates = createScoredRoutingCandidates({
        baseScore: 0.3,
        personalizedScore: 0.9,
        retrievalGroundedScore: 0.6,
      });
      const selectedCandidate = candidates[1]!;

      // Act
      const routingLog = appendNeatChatRoutingDecision(
        [],
        selectedCandidate,
        candidates,
      );
      const appendedEntry = routingLog.at(-1);

      // Assert
      expect({
        candidateCount: appendedEntry?.candidateCount,
        comparedCandidatePaths: appendedEntry?.comparedCandidatePaths,
        logLength: routingLog.length,
        scores: appendedEntry?.scores,
        selectedPath: appendedEntry?.selectedPath,
      }).toEqual({
        candidateCount: 3,
        comparedCandidatePaths: ['base', 'personalized', 'retrieval-grounded'],
        logLength: 1,
        scores: {
          base: 0.3,
          personalized: 0.9,
          'retrieval-grounded': 0.6,
        },
        selectedPath: 'personalized',
      });
    });

    it('preserves retrieved-memory provenance in the routing decision entry', () => {
      // Arrange
      const candidates = createScoredRoutingCandidates({
        baseScore: 0.3,
        personalizedScore: 0.4,
        retrievalGroundedScore: 0.9,
      });
      const selectedCandidate = candidates[2]!;

      // Act
      const routingLog = appendNeatChatRoutingDecision(
        [],
        selectedCandidate,
        candidates,
      );
      const appendedEntry = routingLog.at(-1);

      // Assert
      expect(appendedEntry?.retrievedMemoryKeysByPath).toEqual({
        'retrieval-grounded': ['favorite color'],
      });
    });
  });

  describe('snapshot routingLog persistence', () => {
    it('round-trips a non-empty routingLog through snapshot v2 import and export', () => {
      // Arrange
      const routingLogEntry = createRoutingDecisionLogEntry();
      const session = {
        ...createRoutingFixtureSession(),
        routingLog: [routingLogEntry],
      };

      // Act
      const bundle = exportNeatChatSessionV2(session);
      const importedSession = importNeatChatSessionV2(bundle);

      // Assert
      expect({
        exportedRoutingLog: bundle.extensions.neatchat.routingLog,
        importedRoutingLog: importedSession.routingLog,
      }).toEqual({
        exportedRoutingLog: [routingLogEntry],
        importedRoutingLog: [routingLogEntry],
      });
    });

    it('defaults routingLog to an empty array when importing a pre-W5 snapshot', () => {
      // Arrange
      const bundle = exportNeatChatSessionV2(createRoutingFixtureSession());
      const {
        routingLog: omittedRoutingLog,
        ...neatchatExtensionWithoutRoutingLog
      } = bundle.extensions.neatchat;
      const preW5Bundle = {
        ...bundle,
        extensions: {
          ...bundle.extensions,
          neatchat: neatchatExtensionWithoutRoutingLog,
        },
      } satisfies NeatChatSessionSnapshotV2;

      void omittedRoutingLog;

      // Act
      const importedSession = importNeatChatSessionV2(preW5Bundle);

      // Assert
      expect(importedSession.routingLog).toEqual([]);
    });
  });
});

function createRoutingFixtureSession(): NeatChatSession {
  return createNeatChatSession({
    corpusRetainedTerms: ROUTING_TEST_RETAINED_TERMS,
    recurrentBlockSize: 2,
    seedConversationLines: [],
  });
}

function createManagerWithPendingPersonalization(
  session: NeatChatSession,
  evaluationScores: Record<string, number> = { heldOutAccuracy: 0.8 },
): NeatChatAdaptationManager {
  const candidate: NeatChatAdaptationCandidate = {
    evaluationScores,
    proposedAt: ROUTING_TEST_DECISION_TIMESTAMP,
    trainedOnExchangeCount: 1,
    trainedVector: toParameterVector(session.network),
  };

  return {
    ...createNeatChatAdaptationManager(session),
    pendingCandidates: [candidate],
  };
}

function createRetrievedMemoryFixture(
  session: NeatChatSession,
): NeatChatMemoryRetrievalResult[] {
  const sessionWithMemory = {
    ...session,
    memoryBank: addNeatChatMemoryRecord(session.memoryBank, {
      key: 'favorite color',
      value: 'blue ocean',
    }),
  };

  return retrieveNeatChatMemories(sessionWithMemory, 'favorite color', {
    maxResults: 1,
  });
}

function createScoredRoutingCandidates(options: {
  readonly baseScore: number;
  readonly personalizedScore: number;
  readonly retrievalGroundedScore: number;
}): NeatChatRoutingCandidate[] {
  return [
    createRoutingCandidate('base', options.baseScore),
    createRoutingCandidate('personalized', options.personalizedScore),
    createRoutingCandidate(
      'retrieval-grounded',
      options.retrievalGroundedScore,
      ['favorite color'],
    ),
  ];
}

function createRoutingCandidate(
  routingPath: NeatChatRoutingPath,
  score: number,
  retrievedMemoryKeys: readonly string[] = [],
): NeatChatRoutingCandidate {
  return {
    response: `${routingPath} response`,
    responseTokens: [routingPath, 'response'],
    retrievedMemoryCount: retrievedMemoryKeys.length,
    retrievedMemoryKeys,
    routingPath,
    score,
  };
}

function createRoutingDecisionLogEntry(): NeatChatRoutingDecisionLogEntry {
  return {
    candidateCount: 3,
    comparedCandidatePaths: ['base', 'personalized', 'retrieval-grounded'],
    decidedAt: ROUTING_TEST_DECISION_TIMESTAMP,
    responsesByPath: {
      base: 'base response',
      personalized: 'personalized response',
      'retrieval-grounded': 'retrieval-grounded response',
    },
    retrievedMemoryCountByPath: {
      'retrieval-grounded': 1,
    },
    retrievedMemoryKeysByPath: {
      'retrieval-grounded': ['favorite color'],
    },
    scores: {
      base: 0.3,
      personalized: 0.6,
      'retrieval-grounded': 0.9,
    },
    selectedPath: 'retrieval-grounded',
  };
}

function readRoutingPaths(
  candidates: readonly NeatChatRoutingCandidate[],
): NeatChatRoutingPath[] {
  return candidates.map((candidate) => candidate.routingPath);
}

function createRoutingFallbackSession(
  options: {
    readonly exchanges?: NeatChatSession['exchanges'];
  } = {},
): NeatChatSession {
  const session = createRoutingFixtureSession();

  return {
    ...session,
    exchanges: options.exchanges ?? [],
    network: createEmptyOutputRoutingNetwork(session.vocabulary.size) as never,
  };
}

function createEmptyOutputRoutingNetwork(vocabularySize: number) {
  return {
    activate: () =>
      new Array<number>(vocabularySize).fill(Number.NEGATIVE_INFINITY),
    clear: () => undefined,
  };
}

function removeNetworkClone(session: NeatChatSession): void {
  Object.defineProperty(session.network, 'clone', {
    configurable: true,
    value: undefined,
    writable: true,
  });
}

function createRetrievedMemoryResult(
  key: string,
  value: string,
): NeatChatMemoryRetrievalResult {
  const record = {
    addedAt: ROUTING_TEST_DECISION_TIMESTAMP,
    hitCount: 0,
    key,
    value,
  };

  return {
    ...record,
    key,
    overlapCount: 1,
    record,
    value,
  };
}
