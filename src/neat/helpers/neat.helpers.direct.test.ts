import Network from '../../architecture/network';
import { createInnovationTracker } from '../innovation-tracker/innovation-tracker';
import { addGenome, createPool, spawnFromParent } from './neat.helpers';

type DirectHelperHost = {
  input: number;
  output: number;
  generation: number;
  population: Array<Record<string, unknown>>;
  options: {
    minHidden?: number;
    mutation?: unknown;
    popsize?: number;
    reenableProb?: number;
  };
  _getRNG: () => () => number;
  _innovationTracker: ReturnType<typeof createInnovationTracker>;
  _invalidateGenomeCaches?: jest.Mock<void, [Record<string, unknown>]>;
  _lineageEnabled?: boolean;
  _nextGenomeId: number;
  ensureMinHiddenNodes?: (genome: Record<string, unknown>) => void;
  ensureNoDeadEnds?: (genome: Record<string, unknown>) => void;
  selectMutationMethod?: () => unknown;
};

function createHelperHost(
  overrides?: Partial<DirectHelperHost>,
): DirectHelperHost {
  return {
    input: 2,
    output: 1,
    generation: 0,
    population: [],
    options: {
      popsize: 2,
      reenableProb: 0.4,
    },
    _getRNG: () => () => 0.5,
    _innovationTracker: createInnovationTracker(),
    _invalidateGenomeCaches: jest.fn(),
    _nextGenomeId: 1,
    ensureMinHiddenNodes: () => {},
    ensureNoDeadEnds: () => {},
    ...overrides,
  };
}

function createSeedClone(): Record<string, unknown> {
  return {
    connections: [
      {
        from: { index: 0 },
        to: { index: 1 },
      },
    ],
    clone: () => createSeedClone(),
  };
}

function createSelfConnectionOnlySeed(): Record<string, unknown> {
  return {
    clone: () => createSelfConnectionOnlySeed(),
    selfconns: [
      {
        from: { index: 0 },
        to: { index: 0 },
      },
    ],
  };
}

describe('neat helpers direct utility wrappers', () => {
  describe('spawnFromParent', () => {
    describe('given the parent genome exposes neither clone nor serialization support', () => {
      it('rejects while attempting the JSON clone fallback', async () => {
        // Arrange
        const helperHost = createHelperHost();

        // Act
        const spawnPromise = spawnFromParent.call(
          helperHost as never,
          {} as never,
          0,
        );

        // Assert
        await expect(spawnPromise).rejects.toBeInstanceOf(Error);
      });
    });

    describe('given the parent genome only exposes JSON serialization', () => {
      it('rebuilds the child through the JSON clone fallback and assigns fresh lineage metadata', async () => {
        // Arrange
        const helperHost = createHelperHost();
        const serializableParent = new Network(2, 1, { seed: 1_901 });
        const parentGenome = {
          _depth: 2,
          _id: 7,
          toJSON: () => serializableParent.toJSON(),
        };

        // Act
        const childGenome = await spawnFromParent.call(
          helperHost as never,
          parentGenome as never,
          0,
        );

        // Assert
        expect({
          depth: childGenome._depth,
          id: childGenome._id,
          parents: childGenome._parents,
        }).toEqual({
          depth: 3,
          id: 1,
          parents: [7],
        });
      });
    });

    describe('given the mutation selector returns one candidate list and mutateCount is omitted', () => {
      it('uses the RNG-selected candidate and skips mutation when that candidate has no name', async () => {
        // Arrange
        const clonedGenome = {
          connections: [],
          mutate: jest.fn(),
          selfconns: [],
        };
        const helperHost = createHelperHost({
          _getRNG: () => () => 0,
          selectMutationMethod: () => [{}, { name: 'unused' }],
        });
        const parentGenome = {
          clone: () => clonedGenome,
        };

        // Act
        const childGenome = await spawnFromParent.call(
          helperHost as never,
          parentGenome as never,
        );

        // Assert
        expect({
          mutateCalls: clonedGenome.mutate.mock.calls,
          parents: childGenome._parents,
        }).toEqual({
          mutateCalls: [],
          parents: [0],
        });
      });
    });
  });

  describe('addGenome', () => {
    describe('given invariant enforcement throws while an exogenous genome is being admitted', () => {
      it('still appends the genome through the fallback path', () => {
        // Arrange
        const admittedGenome: Record<string, unknown> = {};
        const helperHost = createHelperHost({
          ensureMinHiddenNodes: () => {
            throw new Error('expected fallback path');
          },
        });

        // Act
        addGenome.call(helperHost as never, admittedGenome as never);

        // Assert
        expect({
          depth: admittedGenome._depth,
          parents: admittedGenome._parents,
          populationLength: helperHost.population.length,
        }).toEqual({
          depth: 0,
          parents: [],
          populationLength: 1,
        });
      });
    });

    describe('given all provided parent ids are absent from the live population', () => {
      it('falls back to depth one for the admitted genome', () => {
        // Arrange
        const admittedGenome: Record<string, unknown> = {};
        const helperHost = createHelperHost();

        // Act
        addGenome.call(helperHost as never, admittedGenome as never, [999]);

        // Assert
        expect({
          depth: admittedGenome._depth,
          parents: admittedGenome._parents,
        }).toEqual({
          depth: 1,
          parents: [999],
        });
      });
    });
  });

  describe('createPool', () => {
    describe('given no explicit population size is configured and lineage tracking is enabled', () => {
      it('creates the default generation-zero pool and assigns empty lineage metadata', () => {
        // Arrange
        const helperHost = createHelperHost({
          _lineageEnabled: true,
          options: {
            reenableProb: 0.4,
          },
        });

        // Act
        createPool.call(helperHost as never, null);

        // Assert
        expect({
          depth: helperHost.population[0]?._depth,
          parents: helperHost.population[0]?._parents,
          populationLength: helperHost.population.length,
        }).toEqual({
          depth: 0,
          parents: [],
          populationLength: 50,
        });
      });
    });

    describe('given one seeded template omits self connections and connection innovations', () => {
      it('reseeds the innovation tracker from the missing-innovation fallback', () => {
        // Arrange
        const helperHost = createHelperHost();
        const seedNetwork = createSeedClone();

        // Act
        createPool.call(helperHost as never, seedNetwork as never);

        // Assert
        expect(helperHost._innovationTracker.nextInnovationId).toBe(0);
      });
    });

    describe('given one seeded template only exposes self connections', () => {
      it('uses the missing connections fallback while reseeding the innovation tracker', () => {
        // Arrange
        const helperHost = createHelperHost();
        const seedNetwork = createSelfConnectionOnlySeed();

        // Act
        createPool.call(helperHost as never, seedNetwork as never);

        // Assert
        expect(helperHost._innovationTracker.nextInnovationId).toBe(0);
      });
    });

    describe('given one seeded template only exposes JSON serialization', () => {
      it('rebuilds each generation-zero genome through the JSON clone fallback', () => {
        // Arrange
        const helperHost = createHelperHost();
        const seedNetwork = new Network(2, 1, { seed: 1_902 });
        const jsonOnlySeed = {
          toJSON: () => seedNetwork.toJSON(),
        };

        // Act
        createPool.call(helperHost as never, jsonOnlySeed as never);

        // Assert
        expect(
          helperHost.population.map((genome) => {
            return (genome as Network).connections.length;
          }),
        ).toEqual([
          seedNetwork.connections.length,
          seedNetwork.connections.length,
        ]);
      });
    });

    describe('given one seeded template lacks both cloning and serialization support', () => {
      it('keeps the pool empty through the best-effort fallback path', () => {
        // Arrange
        const helperHost = createHelperHost();

        // Act
        createPool.call(helperHost as never, {} as never);

        // Assert
        expect(helperHost.population).toEqual([]);
      });
    });
  });
});