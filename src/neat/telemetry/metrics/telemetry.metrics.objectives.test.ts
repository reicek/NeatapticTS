import {
  applyHypervolumeTelemetry,
  applyObjectiveAges,
  applyObjectiveEvents,
  applyObjectiveImportance,
  applyObjectivesSnapshot,
  applySpeciesAllocation,
  computeHyperVolumeProxy,
  computeParetoFrontSizes,
} from './telemetry.metrics.objectives';
import type { TelemetryEntryRecord } from '../types/telemetry.types';

function createTelemetryEntryRecord(): TelemetryEntryRecord {
  return {
    gen: 4,
    best: 9,
    species: 2,
    hyper: 0,
    ops: [],
    objImportance: {},
  };
}

jest.retryTimes(2, { logErrorsBeforeRetry: true });

describe('neat telemetry metrics objectives chapter', () => {
  describe('computeHyperVolumeProxy() — multiObjective absent', () => {
    describe('given no multiObjective option', () => {
      it('defaults to connections as complexityMetric (line 26 fallback || arm)', () => {
        // Arrange: multiObjective?.complexityMetric is undefined → || 'connections' fires
        const options = {};
        const population = [
          { score: 5, _moRank: 0, nodes: [], connections: [1] },
        ] as unknown as Parameters<typeof computeHyperVolumeProxy>[1];

        // Act
        const result = computeHyperVolumeProxy(
          options as Parameters<typeof computeHyperVolumeProxy>[0],
          population,
        );

        // Assert: runs without error and returns a number
        expect(typeof result).toBe('number');
      });
    });
  });

  describe('computeHyperVolumeProxy()', () => {
    describe('given a population with mixed ranks and varying scores', () => {
      it('sums normalized contribution only for rank-0 genomes (lines 30-38 coverage)', () => {
        // Arrange: rank 0 genome contributes; rank 1 is skipped
        const options = {
          multiObjective: { complexityMetric: 'nodes' as const },
        };
        const population = [
          { score: 10, _moRank: 0, nodes: [1, 2], connections: [] },
          { score: 5, _moRank: 1, nodes: [1], connections: [] },
          { score: 0, _moRank: 0, nodes: [1], connections: [] },
        ] as unknown as Parameters<typeof computeHyperVolumeProxy>[1];

        // Act
        const result = computeHyperVolumeProxy(
          options as Parameters<typeof computeHyperVolumeProxy>[0],
          population,
        );

        // Assert: positive proxy because rank-0 genome with score 10 contributes
        expect(result).toBeGreaterThan(0);
      });
    });

    describe('given a population where all genomes have the same score', () => {
      it('uses normalizedScore=0 fallback (line 42 false arm — equal min/max)', () => {
        // Arrange: minPrimaryScore === maxPrimaryScore → normalizedScore = 0
        const options = {
          multiObjective: { complexityMetric: 'connections' as const },
        };
        const population = [
          { score: 5, _moRank: 0, nodes: [], connections: [1] },
          { score: 5, _moRank: 0, nodes: [], connections: [1] },
        ] as unknown as Parameters<typeof computeHyperVolumeProxy>[1];

        // Act
        const result = computeHyperVolumeProxy(
          options as Parameters<typeof computeHyperVolumeProxy>[0],
          population,
        );

        // Assert: proxy is 0 because normalized score is always 0
        expect(result).toBe(0);
      });
    });

    describe('given a genome with undefined _moRank', () => {
      it('treats _moRank as 0 via ?? fallback (line 38 ?? arm)', () => {
        // Arrange: _moRank = undefined → ?? 0 → treated as rank 0
        const options = {
          multiObjective: { complexityMetric: 'connections' as const },
        };
        const population = [
          { score: 10, _moRank: undefined, nodes: [], connections: [1, 2] },
          { score: 0, _moRank: undefined, nodes: [], connections: [1] },
        ] as unknown as Parameters<typeof computeHyperVolumeProxy>[1];

        // Act
        const result = computeHyperVolumeProxy(
          options as Parameters<typeof computeHyperVolumeProxy>[0],
          population,
        );

        // Assert: genome with rank undefined treated as 0 → contributes to proxy
        expect(result).toBeGreaterThan(0);
      });
    });
  });

  describe('computeParetoFrontSizes()', () => {
    describe('given a population spanning two Pareto fronts', () => {
      it('returns sizes for each populated front (line 69 ?? arm for defined _moRank)', () => {
        // Arrange: rank 0 has 2 genomes, rank 1 has 1 genome
        const population = [
          { score: 5, _moRank: 0, nodes: [], connections: [] },
          { score: 3, _moRank: 0, nodes: [], connections: [] },
          { score: 1, _moRank: 1, nodes: [], connections: [] },
        ] as unknown as Parameters<typeof computeParetoFrontSizes>[0];

        // Act
        const sizes = computeParetoFrontSizes(population);

        // Assert
        expect(sizes).toEqual([2, 1]);
      });
    });

    describe('given a population with undefined _moRank', () => {
      it('treats undefined _moRank as 0 via ?? fallback (line 69 ?? arm)', () => {
        // Arrange: _moRank = undefined → ?? 0 → counted as rank 0
        const population = [
          { score: 5, _moRank: undefined, nodes: [], connections: [] },
        ] as unknown as Parameters<typeof computeParetoFrontSizes>[0];

        // Act
        const sizes = computeParetoFrontSizes(population);

        // Assert: 1 genome at rank 0
        expect(sizes).toEqual([1]);
      });
    });
  });

  describe('applySpeciesAllocation() — no allocation', () => {
    describe('given no _lastOffspringAlloc on the context', () => {
      it('skips the entry update (line 172 false arm)', () => {
        // Arrange: _lastOffspringAlloc = undefined → if-guard is false
        const ctx = {};
        const entry = createTelemetryEntryRecord();

        // Act
        applySpeciesAllocation(ctx, entry);

        // Assert: speciesAlloc not set
        expect(entry.speciesAlloc).toBeUndefined();
      });
    });
  });

  describe('applySpeciesAllocation()', () => {
    describe('given a valid array allocation snapshot', () => {
      it('copies the allocation onto the entry (line 174 true arm)', () => {
        // Arrange
        const ctx = { _lastOffspringAlloc: [{ id: 1, alloc: 5 }] };
        const entry = createTelemetryEntryRecord();

        // Act
        applySpeciesAllocation(
          ctx as Parameters<typeof applySpeciesAllocation>[0],
          entry,
        );

        // Assert
        expect(entry.speciesAlloc).toEqual([{ id: 1, alloc: 5 }]);
      });
    });

    describe('given a non-array allocation snapshot', () => {
      it('skips attaching speciesAlloc (line 174 false arm — Array.isArray guard)', () => {
        // Arrange: _lastOffspringAlloc is a non-array value → Array.isArray returns false
        const ctx = {
          _lastOffspringAlloc: 'bad' as unknown as Parameters<
            typeof applySpeciesAllocation
          >[0]['_lastOffspringAlloc'],
        };
        const entry = createTelemetryEntryRecord();

        // Act
        applySpeciesAllocation(
          ctx as Parameters<typeof applySpeciesAllocation>[0],
          entry,
        );

        // Assert: no speciesAlloc attached
        expect(entry.speciesAlloc).toBeUndefined();
      });
    });
  });

  describe('applyHypervolumeTelemetry()', () => {
    describe('given telemetry.hypervolume and multiObjective.enabled are both true', () => {
      it('attaches rounded hv to the entry', () => {
        // Arrange
        const options = {
          telemetry: { hypervolume: true },
          multiObjective: { enabled: true },
        };
        const entry = createTelemetryEntryRecord();

        // Act
        applyHypervolumeTelemetry(
          options as Parameters<typeof applyHypervolumeTelemetry>[0],
          3.14159,
          entry,
        );

        // Assert
        expect(entry.hv).toBe(3.1416);
      });
    });

    describe('given multiObjective.enabled is false', () => {
      it('does not attach hv to the entry (false arm of combined guard)', () => {
        // Arrange
        const options = {
          telemetry: { hypervolume: true },
          multiObjective: { enabled: false },
        };
        const entry = createTelemetryEntryRecord();

        // Act
        applyHypervolumeTelemetry(
          options as Parameters<typeof applyHypervolumeTelemetry>[0],
          1.0,
          entry,
        );

        // Assert: hv not attached
        expect(entry.hv).toBeUndefined();
      });
    });
  });

  describe('applyObjectiveImportance', () => {
    describe('given entry.objImportance is null', () => {
      it('initializes it to an empty object (line 90 true arm — !entry.objImportance)', () => {
        // Arrange: objImportance = null → !null is true → initialization fires
        const ctx = {};
        const entry = {
          ...createTelemetryEntryRecord(),
          objImportance: null as unknown as Parameters<
            typeof applyObjectiveImportance
          >[1]['objImportance'],
        };

        // Act
        applyObjectiveImportance(ctx, entry);

        // Assert: objImportance initialized to {}
        expect(entry.objImportance).toEqual({});
      });
    });

    describe('given _lastObjImportance is a non-object value', () => {
      it('skips applying it (line 95 false arm — non-object guard)', () => {
        // Arrange: _lastObjImportance is not an object → guard on line 95 is false
        const ctx = {
          _lastObjImportance: 'invalid' as unknown as Parameters<
            typeof applyObjectiveImportance
          >[0]['_lastObjImportance'],
        };
        const entry = createTelemetryEntryRecord();

        // Act
        applyObjectiveImportance(
          ctx as Parameters<typeof applyObjectiveImportance>[0],
          entry,
        );

        // Assert: objImportance remains {} (not overwritten with invalid value)
        expect(entry.objImportance).toEqual({});
      });
    });

    describe('given an evolve-side importance snapshot already exists', () => {
      it('copies the latest objective importance map onto the telemetry entry', () => {
        // Arrange
        const telemetryContext = {
          _lastObjImportance: {
            fitness: { range: 2, var: 1 },
          },
        };
        const telemetryEntry = createTelemetryEntryRecord();

        // Act
        applyObjectiveImportance(telemetryContext, telemetryEntry);

        // Assert
        expect(telemetryEntry.objImportance).toEqual({
          fitness: { range: 2, var: 1 },
        });
      });
    });
  });

  describe('applyObjectiveAges() — no ages', () => {
    describe('given no _objectiveAges on the context', () => {
      it('skips the entry update (line 111 ?. false arm)', () => {
        // Arrange: _objectiveAges = undefined → ?. returns undefined → skip
        const ctx = {};
        const entry = createTelemetryEntryRecord();

        // Act
        applyObjectiveAges(ctx, entry);

        // Assert: objAges not set
        expect(entry.objAges).toBeUndefined();
      });
    });
  });

  describe('applyObjectiveAges', () => {
    describe('given objective ages are tracked in controller state', () => {
      it('serializes the age map into the telemetry entry payload', () => {
        // Arrange
        const telemetryContext = {
          _objectiveAges: new Map<string, number>([
            ['fitness', 7],
            ['complexity', 3],
            ['entropy', 1],
          ]),
        };
        const telemetryEntry = createTelemetryEntryRecord();

        // Act
        applyObjectiveAges(telemetryContext, telemetryEntry);

        // Assert
        expect(telemetryEntry.objAges).toEqual({
          fitness: 7,
          complexity: 3,
          entropy: 1,
        });
      });
    });
  });

  describe('applyObjectiveEvents', () => {
    describe('given _pendingObjectiveAdds and _pendingObjectiveRemoves are both undefined', () => {
      it('uses empty arrays via ?? fallback (lines 143,146 ?? arms) and exits early', () => {
        // Arrange: both arrays undefined → both ?? [] fire; but both are empty → early return
        const ctx = {
          _pendingObjectiveAdds: undefined,
          _pendingObjectiveRemoves: undefined,
        };
        const entry = createTelemetryEntryRecord();

        // Act
        applyObjectiveEvents(
          ctx as Parameters<typeof applyObjectiveEvents>[0],
          entry,
          0,
        );

        // Assert: early return — no events
        expect(entry.objEvents).toBeUndefined();
      });
    });

    describe('given _pendingObjectiveAdds is undefined but removes has elements', () => {
      it('uses ?? [] for adds-loop (line 143 ?? arm) and runs removes only', () => {
        // Arrange: _pendingObjectiveAdds = undefined → ?? [] fires; removes runs
        const ctx = {
          _pendingObjectiveAdds: undefined,
          _pendingObjectiveRemoves: ['entropy'],
          _objectiveEvents: [],
        };
        const entry = createTelemetryEntryRecord();

        // Act
        applyObjectiveEvents(
          ctx as Parameters<typeof applyObjectiveEvents>[0],
          entry,
          2,
        );

        // Assert: only remove event in the entry (adds ?? [] produces no iterations)
        expect(entry.objEvents).toEqual([
          { gen: 2, type: 'remove', key: 'entropy' },
        ]);
      });
    });

    describe('given _pendingObjectiveRemoves is undefined but adds has elements', () => {
      it('uses ?? [] for removes-loop (line 146 ?? arm) and runs adds only', () => {
        // Arrange: _pendingObjectiveRemoves = undefined → ?? [] fires; adds runs
        const ctx = {
          _pendingObjectiveAdds: ['novelty'],
          _pendingObjectiveRemoves: undefined,
          _objectiveEvents: [],
        };
        const entry = createTelemetryEntryRecord();

        // Act
        applyObjectiveEvents(
          ctx as Parameters<typeof applyObjectiveEvents>[0],
          entry,
          3,
        );

        // Assert: only add event in the entry
        expect(entry.objEvents).toEqual([
          { gen: 3, type: 'add', key: 'novelty' },
        ]);
      });
    });

    describe('given _pendingObjectiveAdds has events but _objectiveEvents is undefined', () => {
      it('initializes _objectiveEvents via ?? [] fallback (line 155 ?? arm)', () => {
        // Arrange: _objectiveEvents = undefined → ?? [] fires on line 155
        const ctx = {
          _pendingObjectiveAdds: ['entropy'],
          _pendingObjectiveRemoves: [],
          _objectiveEvents: undefined,
        };
        const entry = createTelemetryEntryRecord();

        // Act
        applyObjectiveEvents(
          ctx as Parameters<typeof applyObjectiveEvents>[0],
          entry,
          1,
        );

        // Assert: _objectiveEvents initialized and populated
        expect(ctx._objectiveEvents).toEqual([
          { gen: 1, type: 'add', key: 'entropy' },
        ]);
      });
    });

    describe('given pending removes are present', () => {
      it('persists remove events and clears pending removes (line 147 — removes loop body)', () => {
        // Arrange: only removes pending → loop body at line 147 fires
        const ctx = {
          _pendingObjectiveAdds: [],
          _pendingObjectiveRemoves: ['complexity'],
          _objectiveEvents: [],
        };
        const entry = createTelemetryEntryRecord();

        // Act
        applyObjectiveEvents(ctx, entry, 7);

        // Assert: remove event recorded
        expect(entry.objEvents).toEqual([
          { gen: 7, type: 'remove', key: 'complexity' },
        ]);
      });
    });

    describe('given delayed objective additions are still pending at telemetry record time', () => {
      it('persists the add events and clears the pending queues', () => {
        // Arrange
        const telemetryContext = {
          _pendingObjectiveAdds: ['complexity', 'entropy'],
          _pendingObjectiveRemoves: [],
          _objectiveEvents: [],
        };
        const telemetryEntry = createTelemetryEntryRecord();

        // Act
        applyObjectiveEvents(telemetryContext, telemetryEntry, 4);

        // Assert
        expect({
          entryEvents: telemetryEntry.objEvents,
          persistedEvents: telemetryContext._objectiveEvents,
          pendingAdds: telemetryContext._pendingObjectiveAdds,
          pendingRemoves: telemetryContext._pendingObjectiveRemoves,
        }).toEqual({
          entryEvents: [
            { gen: 4, type: 'add', key: 'complexity' },
            { gen: 4, type: 'add', key: 'entropy' },
          ],
          persistedEvents: [
            { gen: 4, type: 'add', key: 'complexity' },
            { gen: 4, type: 'add', key: 'entropy' },
          ],
          pendingAdds: [],
          pendingRemoves: [],
        });
      });
    });
  });

  describe('applyObjectivesSnapshot() — no provider', () => {
    describe('given no _getObjectives on the context', () => {
      it('sets objectives to empty array via || [] fallback (line 191 ?. false arm)', () => {
        // Arrange: _getObjectives = undefined → ?. returns undefined → || [] fires
        const ctx = {};
        const entry = createTelemetryEntryRecord();

        // Act
        applyObjectivesSnapshot(ctx, entry);

        // Assert: objectives set to []
        expect(entry.objectives).toEqual([]);
      });
    });
  });

  describe('applyObjectivesSnapshot', () => {
    describe('given an objective provider exposes the active objective list', () => {
      it('records the active objective keys onto the telemetry entry', () => {
        // Arrange
        const telemetryContext = {
          _getObjectives: () => [{ key: 'fitness' }, { key: 'entropy' }],
        };
        const telemetryEntry = createTelemetryEntryRecord();

        // Act
        applyObjectivesSnapshot(telemetryContext, telemetryEntry);

        // Assert
        expect(telemetryEntry.objectives).toEqual(['fitness', 'entropy']);
      });
    });
  });
});
