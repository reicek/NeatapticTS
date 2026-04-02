import { runAutoEntropyObjectiveInjection } from './evaluate.objectives';
import type {
  GenomeForEvaluation,
  NeatControllerForEval,
  ObjectiveDef,
} from '../shared/evaluate.types';

type EvaluateObjectivesController = NeatControllerForEval & {
  _objectiveDescriptors: ObjectiveDef[];
  _pendingObjectiveAdds: string[];
  _objectivesList?: unknown;
};

function createEvaluateObjectivesController(): EvaluateObjectivesController {
  const objectiveDescriptors: ObjectiveDef[] = [
    { key: 'fitness', direction: 'max', fn: () => 0 },
  ];

  return {
    options: {
      multiObjective: {
        enabled: true,
        autoEntropy: true,
      },
    },
    population: [],
    fitness: async (
      genomeOrPopulation: GenomeForEvaluation | GenomeForEvaluation[],
    ) => {
      void genomeOrPopulation;
      return 0;
    },
    _objectiveDescriptors: objectiveDescriptors,
    _pendingObjectiveAdds: [],
    _objectivesList: ['stale-cache'],
    _getObjectives: () => objectiveDescriptors,
    _structuralEntropy: () => 0.5,
    registerObjective: (key, direction, fn) => {
      objectiveDescriptors.push({ key, direction, fn });
    },
  };
}

describe('neat evaluate objectives chapter', () => {
  describe('runAutoEntropyObjectiveInjection', () => {
    describe('given multi-objective evaluation requests automatic entropy registration', () => {
      it('registers entropy, records the pending add, and invalidates the cached objective list', () => {
        // Arrange
        const evaluationController = createEvaluateObjectivesController();

        // Act
        runAutoEntropyObjectiveInjection(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect({
          objectiveKeys: evaluationController
            ._getObjectives?.()
            ?.map((objective) => objective.key),
          pendingObjectiveAdds: evaluationController._pendingObjectiveAdds,
          objectiveListCache: evaluationController._objectivesList,
        }).toEqual({
          objectiveKeys: ['fitness', 'entropy'],
          pendingObjectiveAdds: ['entropy'],
          objectiveListCache: undefined,
        });
      });
    });
  });
});
