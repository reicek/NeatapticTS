import Network from '../../../architecture/network';
import Neat from '../../../neat';
import {
  applyDynamicObjectiveSchedule,
  captureObjectiveImportanceSnapshot,
  updateObjectiveScheduleAndAges,
} from './evolve.objectives.utils';
import type {
  GenomeWithMetadata,
  NeatControllerForEvolution,
} from '../evolve.types';

type EvolutionObjectivesControllerInput = {
  generation?: number;
  registeredObjectiveKeys?: string[];
  objectiveAges?: Array<[string, number]>;
  population?: GenomeWithMetadata[];
};

function createGenome(input: {
  genomeId: number;
  score?: number;
  connectionCount?: number;
}): GenomeWithMetadata {
  return {
    _id: input.genomeId,
    score: input.score,
    nodes: [],
    connections: Array.from(
      { length: input.connectionCount ?? 0 },
      () => ({}),
    ) as GenomeWithMetadata['connections'],
  };
}

function createEvolutionObjectivesController(
  input?: EvolutionObjectivesControllerInput,
): NeatControllerForEvolution {
  const population = input?.population ?? [createGenome({ genomeId: 1 })];
  const registeredObjectives = (
    input?.registeredObjectiveKeys ?? ['fitness']
  ).map((objectiveKey) => createObjectiveDescriptor(objectiveKey));

  const evolutionController = {
    input: 1,
    output: 1,
    population,
    generation: input?.generation ?? 0,
    options: {
      multiObjective: {
        enabled: true,
        dynamic: {
          enabled: true,
          addComplexityAt: 2,
          addEntropyAt: 4,
        },
        objectives: registeredObjectives,
      },
    },
    _bestGlobalScore: 1,
    _species: [],
    _speciesHistory: [],
    _nextGenomeId: population.length + 1,
    _paretoArchive: [],
    _paretoObjectivesArchive: [],
    _lastEpsilonAdjustGen: 0,
    _objectiveStale: new Map<string, number>(),
    _pendingObjectiveAdds: [],
    _pendingObjectiveRemoves: [],
    _objectiveAges: new Map(input?.objectiveAges ?? []),
    _lastOffspringAlloc: [],
    _prevInbreedingCount: 0,
    _lastInbreedingCount: 0,
    _getRNG: () => () => 0.5,
    _getObjectives: () => registeredObjectives,
    _sortSpeciesMembers: () => {},
    _updateSpeciesStagnation: () => {},
    _lastEvolveDuration: 0,
    _structuralEntropy: () => 0.5,
    evaluate: async () => {},
    sort: () => {},
    mutate: async () => {},
    getOffspring: async () => population[0],
    selectParent: () => population[0],
    registerObjective: (key, direction, accessor) => {
      const nextObjective = { key, direction, accessor };
      const updatedObjectives = registeredObjectives.filter(
        (objective) => objective.key !== key,
      );

      updatedObjectives.push(nextObjective);
      registeredObjectives.splice(
        0,
        registeredObjectives.length,
        ...updatedObjectives,
      );
      evolutionController.options.multiObjective!.objectives =
        registeredObjectives;
    },
    ensureMinHiddenNodes: async () => {},
    ensureNoDeadEnds: () => {},
  } as NeatControllerForEvolution;

  return evolutionController;
}

function createObjectiveDescriptor(objectiveKey: string) {
  if (objectiveKey === 'complexity') {
    return {
      key: objectiveKey,
      direction: 'min' as const,
      accessor: (genome: GenomeWithMetadata) => genome.connections.length,
    };
  }

  if (objectiveKey === 'entropy') {
    return {
      key: objectiveKey,
      direction: 'max' as const,
      accessor: () => 0.5,
    };
  }

  return {
    key: objectiveKey,
    direction: 'max' as const,
    accessor: (genome: GenomeWithMetadata) => genome.score ?? 0,
  };
}

async function captureObjectiveKeysByGeneration(
  neat: Neat,
  generationCount: number,
): Promise<string[][]> {
  const objectiveKeysByGeneration: string[][] = [];

  for (
    let generationIndex = 0;
    generationIndex < generationCount;
    generationIndex++
  ) {
    await neat.evaluate();
    await neat.evolve();
    objectiveKeysByGeneration.push(neat.getObjectiveKeys().toSorted());
  }

  return objectiveKeysByGeneration;
}

describe('neat evolve objectives chapter', () => {
  describe('captureObjectiveImportanceSnapshot', () => {
    describe('given two objectives that separate the current population', () => {
      it('records per-objective range and variance for later telemetry reads', () => {
        // Arrange
        const evolutionController = createEvolutionObjectivesController({
          registeredObjectiveKeys: ['fitness', 'complexity'],
          population: [
            createGenome({ genomeId: 1, score: 1, connectionCount: 1 }),
            createGenome({ genomeId: 2, score: 3, connectionCount: 3 }),
          ],
        });

        // Act
        captureObjectiveImportanceSnapshot(evolutionController);

        // Assert
        expect(evolutionController._lastObjImportance).toEqual({
          fitness: { range: 2, var: 1 },
          complexity: { range: 2, var: 1 },
        });
      });
    });
  });

  describe('applyDynamicObjectiveSchedule', () => {
    describe('given the run just reached the configured complexity generation gate', () => {
      it('queues the complexity objective for addition', () => {
        // Arrange
        const evolutionController = createEvolutionObjectivesController({
          generation: 1,
          registeredObjectiveKeys: ['fitness'],
        });

        // Act
        applyDynamicObjectiveSchedule(evolutionController, ['fitness'], {
          autoEntropyAddAt: Number.POSITIVE_INFINITY,
        });

        // Assert
        expect(evolutionController._pendingObjectiveAdds).toEqual([
          'complexity',
        ]);
      });
    });

    describe('given the run later reaches the configured entropy generation gate', () => {
      it('queues the entropy objective for addition', () => {
        // Arrange
        const evolutionController = createEvolutionObjectivesController({
          generation: 3,
          registeredObjectiveKeys: ['fitness', 'complexity'],
        });

        // Act
        applyDynamicObjectiveSchedule(
          evolutionController,
          ['fitness', 'complexity'],
          {
            autoEntropyAddAt: Number.POSITIVE_INFINITY,
          },
        );

        // Assert
        expect(evolutionController._pendingObjectiveAdds).toEqual(['entropy']);
      });
    });
  });

  describe('updateObjectiveScheduleAndAges', () => {
    describe('given a scheduled entropy addition lands after older objectives already accumulated age', () => {
      it('increments the existing ages and resets the newly added objective to age zero', async () => {
        // Arrange
        const evolutionController = createEvolutionObjectivesController({
          generation: 3,
          registeredObjectiveKeys: ['fitness', 'complexity'],
          objectiveAges: [
            ['fitness', 4],
            ['complexity', 2],
          ],
        });

        // Act
        await updateObjectiveScheduleAndAges(evolutionController, {
          applyDynamicObjectiveSchedule: (currentObjectiveKeys: string[]) =>
            applyDynamicObjectiveSchedule(
              evolutionController,
              currentObjectiveKeys,
              {
                autoEntropyAddAt: Number.POSITIVE_INFINITY,
              },
            ),
        });

        // Assert
        expect(Object.fromEntries(evolutionController._objectiveAges)).toEqual({
          fitness: 5,
          complexity: 3,
          entropy: 0,
        });
      });
    });
  });

  describe('dynamic objective schedule integration', () => {
    describe('given a scheduled multi-objective run observed through public objective keys over several generations', () => {
      const scoreByEnabledConnections = (network: Network) =>
        network.connections.filter((connection) => connection.enabled !== false)
          .length;

      let objectiveKeysByGeneration: string[][] = [];

      beforeAll(async () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByEnabledConnections, {
          popsize: 25,
          seed: 42,
          multiObjective: {
            enabled: true,
            autoEntropy: true,
            complexityMetric: 'nodes',
            dynamic: {
              enabled: true,
              addComplexityAt: 4,
              addEntropyAt: 3,
              dropEntropyOnStagnation: 6,
              readdEntropyAfter: 2,
            },
          },
          telemetry: { enabled: true },
          lineageTracking: false,
        });

        // Act
        objectiveKeysByGeneration = await captureObjectiveKeysByGeneration(
          neat,
          10,
        );
      });

      it('keeps complexity absent before the configured generation gate', () => {
        // Assert
        expect(
          objectiveKeysByGeneration
            .slice(0, 3)
            .every((objectiveKeys) => !objectiveKeys.includes('complexity')),
        ).toBe(true);
      });

      it('adds complexity once the configured generation gate is reached', () => {
        // Assert
        expect(objectiveKeysByGeneration[3].includes('complexity')).toBe(true);
      });

      it('keeps entropy absent before the configured entropy gate', () => {
        // Assert
        expect(
          objectiveKeysByGeneration
            .slice(0, 2)
            .every((objectiveKeys) => !objectiveKeys.includes('entropy')),
        ).toBe(true);
      });

      it('adds entropy once the configured entropy gate is reached', () => {
        // Assert
        expect(objectiveKeysByGeneration[2].includes('entropy')).toBe(true);
      });

      it('keeps the default fitness objective available throughout the schedule', () => {
        // Assert
        expect(
          objectiveKeysByGeneration.every((objectiveKeys) =>
            objectiveKeys.includes('fitness'),
          ),
        ).toBe(true);
      });
    });
  });
});
