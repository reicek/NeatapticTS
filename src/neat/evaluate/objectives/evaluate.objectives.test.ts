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

type EvaluateObjectivesControllerOptions = {
  autoEntropy?: boolean;
  dynamicEnabled?: boolean;
  objectiveKeys?: string[];
  getObjectives?: () => ObjectiveDef[];
  multiObjectiveEnabled?: boolean;
  registerObjective?: EvaluateObjectivesController['registerObjective'];
  structuralEntropy?: EvaluateObjectivesController['_structuralEntropy'];
};

function createEvaluateObjectivesController(
  controllerOptions: EvaluateObjectivesControllerOptions = {},
): EvaluateObjectivesController {
  const objectiveDescriptors: ObjectiveDef[] = (
    controllerOptions.objectiveKeys ?? ['fitness']
  ).map((objectiveKey) => ({
    key: objectiveKey,
    direction: 'max',
    fn: () => 0,
  }));

  const getObjectives =
    controllerOptions.getObjectives ?? (() => objectiveDescriptors);
  const registerObjective =
    controllerOptions.registerObjective ??
    ((key, direction, fn) => {
      objectiveDescriptors.push({ key, direction, fn });
    });

  return {
    options: {
      multiObjective: {
        enabled: controllerOptions.multiObjectiveEnabled ?? true,
        autoEntropy: controllerOptions.autoEntropy ?? true,
        dynamic: {
          enabled: controllerOptions.dynamicEnabled,
        },
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
    _getObjectives: getObjectives,
    _structuralEntropy: controllerOptions.structuralEntropy ?? (() => 0.5),
    registerObjective,
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

    describe('given multi-objective evaluation is disabled', () => {
      it('leaves the objective list unchanged', () => {
        // Arrange
        const evaluationController = createEvaluateObjectivesController({
          multiObjectiveEnabled: false,
        });

        // Act
        runAutoEntropyObjectiveInjection(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect(
          evaluationController
            ._getObjectives?.()
            ?.map((objective) => objective.key),
        ).toEqual(['fitness']);
      });
    });

    describe('given automatic entropy registration is disabled', () => {
      it('keeps the pending registration list empty', () => {
        // Arrange
        const evaluationController = createEvaluateObjectivesController({
          autoEntropy: false,
        });

        // Act
        runAutoEntropyObjectiveInjection(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect(evaluationController._pendingObjectiveAdds).toEqual([]);
      });
    });

    describe('given dynamic objective mode already owns the policy', () => {
      it('does not invalidate the cached objective list', () => {
        // Arrange
        const evaluationController = createEvaluateObjectivesController({
          dynamicEnabled: true,
        });

        // Act
        runAutoEntropyObjectiveInjection(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect(evaluationController._objectivesList).toEqual(['stale-cache']);
      });
    });

    describe('given entropy is already registered', () => {
      it('does not append a duplicate entropy objective', () => {
        // Arrange
        const evaluationController = createEvaluateObjectivesController({
          objectiveKeys: ['fitness', 'entropy'],
        });

        // Act
        runAutoEntropyObjectiveInjection(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect(
          evaluationController
            ._getObjectives?.()
            ?.map((objective) => objective.key),
        ).toEqual(['fitness', 'entropy']);
      });
    });

    describe('given the controller has no objective accessor', () => {
      it('still registers entropy by falling back to an empty objective-key list', () => {
        // Arrange
        const evaluationController = createEvaluateObjectivesController();
        evaluationController._getObjectives = undefined;

        // Act
        runAutoEntropyObjectiveInjection(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect(evaluationController._pendingObjectiveAdds).toEqual(['entropy']);
      });
    });

    describe('given the injected entropy objective is evaluated later', () => {
      it('delegates the score to the controller structural-entropy helper', () => {
        // Arrange
        const evaluationController = createEvaluateObjectivesController();

        // Act
        runAutoEntropyObjectiveInjection(
          evaluationController,
          evaluationController.options,
        );
        const entropyObjectiveDescriptor = evaluationController
          ._getObjectives?.()
          ?.find((objective) => objective.key === 'entropy');
        const entropyScore = entropyObjectiveDescriptor?.fn?.(
          {} as GenomeForEvaluation,
        );

        // Assert
        expect(entropyScore).toBe(0.5);
      });
    });

    describe('given the controller has no structural-entropy helper', () => {
      it('falls back to zero when the injected entropy objective is evaluated', () => {
        // Arrange
        const evaluationController = createEvaluateObjectivesController();
        evaluationController._structuralEntropy = undefined;

        // Act
        runAutoEntropyObjectiveInjection(
          evaluationController,
          evaluationController.options,
        );
        const entropyObjectiveDescriptor = evaluationController
          ._getObjectives?.()
          ?.find((objective) => objective.key === 'entropy');
        const entropyScore = entropyObjectiveDescriptor?.fn?.(
          {} as GenomeForEvaluation,
        );

        // Assert
        expect(entropyScore).toBe(0);
      });
    });

    describe('given the controller cannot resolve objectives safely', () => {
      it('swallows the lookup failure instead of throwing from evaluation', () => {
        // Arrange
        const evaluationController = createEvaluateObjectivesController({
          getObjectives: () => {
            throw new Error('lookup failed');
          },
        });

        // Act
        const runInjection = () =>
          runAutoEntropyObjectiveInjection(
            evaluationController,
            evaluationController.options,
          );

        // Assert
        expect(runInjection).not.toThrow();
      });
    });
  });
});
