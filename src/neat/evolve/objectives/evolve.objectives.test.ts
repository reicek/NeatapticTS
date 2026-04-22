import Network from '../../../architecture/network';
import Neat from '../../../neat';
import {
  applyFitnessSuppressionForTests,
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

    describe('given no resolved objectives are available on the controller', () => {
      it('keeps the last importance snapshot undefined', () => {
        // Arrange
        const evolutionController = createEvolutionObjectivesController();

        Reflect.deleteProperty(evolutionController as object, '_getObjectives');

        // Act
        captureObjectiveImportanceSnapshot(evolutionController);

        // Assert
        expect(evolutionController._lastObjImportance).toBeUndefined();
      });
    });

    describe('given one objective is resolved but the population is empty', () => {
      it('uses the empty-population variance divisor fallback', () => {
        // Arrange
        const evolutionController = createEvolutionObjectivesController({
          population: [],
          registeredObjectiveKeys: ['fitness'],
        });

        // Act
        captureObjectiveImportanceSnapshot(evolutionController);

        // Assert
        expect(evolutionController._lastObjImportance).toEqual({
          fitness: { range: Number.NEGATIVE_INFINITY, var: 0 },
        });
      });
    });
  });

  describe('applyFitnessSuppressionForTests', () => {
    describe('given pruneInactive is disabled and fitness is one of multiple resolved objectives', () => {
      it('suppresses fitness once and clears the cached objectives list', () => {
        // Arrange
        const evolutionController = createEvolutionObjectivesController({
          registeredObjectiveKeys: ['fitness', 'complexity'],
        });

        evolutionController.options.multiObjective!.pruneInactive = {
          enabled: false,
        };
        evolutionController._objectivesList = [
          createObjectiveDescriptor('fitness'),
          createObjectiveDescriptor('complexity'),
        ];

        // Act
        applyFitnessSuppressionForTests(evolutionController);

        // Assert
        expect({
          fitnessSuppressedOnce: evolutionController._fitnessSuppressedOnce,
          objectivesList: evolutionController._objectivesList,
          suppressFitnessObjective: evolutionController._suppressFitnessObjective,
        }).toEqual({
          fitnessSuppressedOnce: true,
          objectivesList: undefined,
          suppressFitnessObjective: true,
        });
      });
    });

    describe('given pruneInactive is disabled but the controller does not expose a resolved objective getter', () => {
      it('leaves the fitness suppression flags untouched', () => {
        // Arrange
        const evolutionController = createEvolutionObjectivesController({
          registeredObjectiveKeys: ['fitness', 'complexity'],
        });

        evolutionController.options.multiObjective!.pruneInactive = {
          enabled: false,
        };
        Reflect.deleteProperty(evolutionController as object, '_getObjectives');

        // Act
        applyFitnessSuppressionForTests(evolutionController);

        // Assert
        expect({
          fitnessSuppressedOnce: evolutionController._fitnessSuppressedOnce,
          suppressFitnessObjective: evolutionController._suppressFitnessObjective,
        }).toEqual({
          fitnessSuppressedOnce: undefined,
          suppressFitnessObjective: undefined,
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

    describe('given the dynamic generation gates are omitted', () => {
      it('uses the Infinity defaults and queues no scheduled objectives', () => {
        // Arrange
        const evolutionController = createEvolutionObjectivesController({
          generation: 100,
          registeredObjectiveKeys: ['fitness'],
        });

        evolutionController.options.multiObjective!.dynamic = {
          enabled: true,
        };

        // Act
        applyDynamicObjectiveSchedule(evolutionController, ['fitness'], {
          autoEntropyAddAt: Number.POSITIVE_INFINITY,
        });

        // Assert
        expect(evolutionController._pendingObjectiveAdds).toEqual([]);
      });
    });

    describe('given multi-objective scheduling is disabled entirely', () => {
      it('returns without queueing any scheduled objectives', () => {
        // Arrange
        const evolutionController = createEvolutionObjectivesController({
          generation: 10,
          registeredObjectiveKeys: ['fitness'],
        });

        evolutionController.options.multiObjective!.enabled = false;

        // Act
        applyDynamicObjectiveSchedule(evolutionController, ['fitness'], {
          autoEntropyAddAt: 0,
        });

        // Assert
        expect(evolutionController._pendingObjectiveAdds).toEqual([]);
      });
    });

    describe('given dynamic scheduling is disabled and auto entropy is also disabled', () => {
      it('leaves the pending objective queue unchanged', () => {
        // Arrange
        const evolutionController = createEvolutionObjectivesController({
          generation: 10,
          registeredObjectiveKeys: ['fitness'],
        });

        evolutionController.options.multiObjective!.autoEntropy = false;
        evolutionController.options.multiObjective!.dynamic = {
          enabled: false,
        };

        // Act
        applyDynamicObjectiveSchedule(evolutionController, ['fitness'], {
          autoEntropyAddAt: 0,
        });

        // Assert
        expect(evolutionController._pendingObjectiveAdds).toEqual([]);
      });
    });

    describe('given dynamic scheduling is disabled but auto entropy reached its fallback generation gate', () => {
      it('queues entropy through the fallback path', () => {
        // Arrange
        const evolutionController = createEvolutionObjectivesController({
          generation: 4,
          registeredObjectiveKeys: ['fitness'],
        });

        evolutionController.options.multiObjective!.autoEntropy = true;
        evolutionController.options.multiObjective!.dynamic = {
          enabled: false,
        };

        // Act
        applyDynamicObjectiveSchedule(evolutionController, ['fitness'], {
          autoEntropyAddAt: 4,
        });

        // Assert
        expect(evolutionController._pendingObjectiveAdds).toEqual(['entropy']);
      });
    });

    describe('given dynamic scheduling is disabled but auto entropy has not reached its fallback generation gate', () => {
      it('does not queue entropy yet', () => {
        // Arrange
        const evolutionController = createEvolutionObjectivesController({
          generation: 2,
          registeredObjectiveKeys: ['fitness'],
        });

        evolutionController.options.multiObjective!.autoEntropy = true;
        evolutionController.options.multiObjective!.dynamic = {
          enabled: false,
        };

        // Act
        applyDynamicObjectiveSchedule(evolutionController, ['fitness'], {
          autoEntropyAddAt: 4,
        });

        // Assert
        expect(evolutionController._pendingObjectiveAdds).toEqual([]);
      });
    });

    describe('given entropy is active when stagnation reaches the configured drop generation', () => {
      it('removes entropy from the configured objectives and records the drop generation', () => {
        // Arrange
        const evolutionController = createEvolutionObjectivesController({
          generation: 5,
          registeredObjectiveKeys: ['fitness', 'entropy'],
        });

        evolutionController.options.multiObjective!.dynamic = {
          enabled: true,
          addComplexityAt: Number.POSITIVE_INFINITY,
          addEntropyAt: Number.POSITIVE_INFINITY,
          dropEntropyOnStagnation: 5,
          readdEntropyAfter: 2,
        };

        // Act
        applyDynamicObjectiveSchedule(
          evolutionController,
          ['fitness', 'entropy'],
          {
            autoEntropyAddAt: Number.POSITIVE_INFINITY,
          },
        );

        // Assert
        expect({
          entropyDropped: evolutionController._entropyDropped,
          objectiveKeys:
            evolutionController.options.multiObjective!.objectives?.map(
              (objective) => objective.key,
            ),
          pendingObjectiveRemoves: evolutionController._pendingObjectiveRemoves,
        }).toEqual({
          entropyDropped: 5,
          objectiveKeys: ['fitness'],
          pendingObjectiveRemoves: ['entropy'],
        });
      });
    });

    describe('given entropy is active but the configured objective array is missing at drop time', () => {
      it('skips the drop bookkeeping without mutating pending removals', () => {
        // Arrange
        const evolutionController = createEvolutionObjectivesController({
          generation: 5,
          registeredObjectiveKeys: ['fitness', 'entropy'],
        });

        evolutionController.options.multiObjective!.dynamic = {
          enabled: true,
          addComplexityAt: Number.POSITIVE_INFINITY,
          addEntropyAt: Number.POSITIVE_INFINITY,
          dropEntropyOnStagnation: 5,
          readdEntropyAfter: 2,
        };
        evolutionController.options.multiObjective!.objectives = undefined;

        // Act
        applyDynamicObjectiveSchedule(
          evolutionController,
          ['fitness', 'entropy'],
          {
            autoEntropyAddAt: Number.POSITIVE_INFINITY,
          },
        );

        // Assert
        expect({
          entropyDropped: evolutionController._entropyDropped,
          pendingObjectiveRemoves: evolutionController._pendingObjectiveRemoves,
        }).toEqual({
          entropyDropped: undefined,
          pendingObjectiveRemoves: [],
        });
      });
    });

    describe('given entropy was dropped earlier and the configured cooldown elapsed', () => {
      it('re-adds entropy and clears the dropped marker', () => {
        // Arrange
        const evolutionController = createEvolutionObjectivesController({
          generation: 5,
          registeredObjectiveKeys: ['fitness'],
        });

        evolutionController._entropyDropped = 3;
        evolutionController.options.multiObjective!.dynamic = {
          enabled: true,
          addComplexityAt: Number.POSITIVE_INFINITY,
          addEntropyAt: Number.POSITIVE_INFINITY,
          dropEntropyOnStagnation: 9,
          readdEntropyAfter: 2,
        };

        // Act
        applyDynamicObjectiveSchedule(evolutionController, ['fitness'], {
          autoEntropyAddAt: Number.POSITIVE_INFINITY,
        });

        // Assert
        expect({
          entropyDropped: evolutionController._entropyDropped,
          objectiveKeys:
            evolutionController.options.multiObjective!.objectives?.map(
              (objective) => objective.key,
            ),
          pendingObjectiveAdds: evolutionController._pendingObjectiveAdds,
        }).toEqual({
          entropyDropped: undefined,
          objectiveKeys: ['fitness', 'entropy'],
          pendingObjectiveAdds: ['entropy'],
        });
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

    describe('given the controller does not expose the resolved objective getter', () => {
      it('still initializes the ages of newly queued objectives', async () => {
        // Arrange
        const evolutionController = createEvolutionObjectivesController();

        Reflect.deleteProperty(evolutionController as object, '_getObjectives');

        // Act
        await updateObjectiveScheduleAndAges(evolutionController, {
          applyDynamicObjectiveSchedule: () => {
            evolutionController._pendingObjectiveAdds.push('entropy');
          },
        });

        // Assert
        expect(Object.fromEntries(evolutionController._objectiveAges)).toEqual({
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
