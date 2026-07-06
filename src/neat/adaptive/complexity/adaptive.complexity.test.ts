import Network from '../../../architecture/network';
import Neat from '../../../neat';
import { applyComplexityBudget, applyPhasedComplexity } from '../adaptive';
import type {
  ComplexityBudgetConfig,
  NeatLikeWithAdaptive,
} from '../core/adaptive.core.types';

function readPhaseState(target: object): {
  phase: string | undefined;
  phaseStartGeneration: number | undefined;
} {
  return {
    phase: Reflect.get(target, '_phase') as string | undefined,
    phaseStartGeneration: Reflect.get(target, '_phaseStartGeneration') as
      number | undefined,
  };
}

function createComplexityController(
  complexityBudget: ComplexityBudgetConfig,
  initialScore = 1,
): NeatLikeWithAdaptive {
  return {
    options: { complexityBudget },
    population: [{ score: initialScore }],
    input: 2,
    output: 1,
    generation: 0,
  };
}

function collectBudgetSnapshots(
  adaptiveController: NeatLikeWithAdaptive,
  scoreTimeline: number[],
): Array<{ maxNodes: number | undefined; maxConns: number | undefined }> {
  return scoreTimeline.map((score) => {
    adaptiveController.population[0].score = score;
    applyComplexityBudget.call(adaptiveController);

    return {
      maxNodes: adaptiveController.options.maxNodes,
      maxConns: adaptiveController.options.maxConns,
    };
  });
}

function collectNodeBudgetByGeneration(
  adaptiveController: NeatLikeWithAdaptive,
  generationTimeline: number[],
): Array<number | undefined> {
  return generationTimeline.map((generation) => {
    adaptiveController.generation = generation;
    applyComplexityBudget.call(adaptiveController);
    return adaptiveController.options.maxNodes;
  });
}

describe('neat adaptive complexity chapter', () => {
  describe('applyComplexityBudget', () => {
    describe('given adaptive mode with an explicit starting node budget', () => {
      it('initializes maxNodes from maxNodesStart on the first evolve call', async () => {
        // Arrange
        const fitness = (network: Network) => network.connections.length;
        const neatController = new Neat(2, 1, fitness, {
          popsize: 10,
          complexityBudget: {
            enabled: true,
            mode: 'adaptive',
            maxNodesStart: 6,
            maxNodesEnd: 100,
            improvementWindow: 5,
          },
          mutationRate: 1,
          mutationAmount: 2,
        });

        // Act
        await neatController.evolve();

        // Assert
        expect(neatController.options.maxNodes).toBe(6);
      });
    });

    describe('given improving scores and a sufficiently large novelty archive', () => {
      it('grows the node and connection budgets within the configured ceilings', () => {
        // Arrange
        const adaptiveController = createComplexityController({
          enabled: true,
          mode: 'adaptive',
          improvementWindow: 3,
          maxNodesStart: 8,
          maxNodesEnd: 20,
          maxConnsStart: 10,
          maxConnsEnd: 30,
          increaseFactor: 1.1,
          stagnationFactor: 0.95,
        });
        adaptiveController._noveltyArchive = [1, 2, 3, 4, 5, 6];

        // Act
        const budgetSnapshots = collectBudgetSnapshots(
          adaptiveController,
          [1, 2, 3, 4],
        );

        // Assert
        expect(budgetSnapshots.at(-1)).toEqual({ maxNodes: 10, maxConns: 13 });
      });
    });

    describe('given repeated stagnation without an explicit minimum node budget', () => {
      it('falls back to the minimal topology floor when shrink pressure would go lower', () => {
        // Arrange
        const adaptiveController = createComplexityController({
          enabled: true,
          mode: 'adaptive',
          improvementWindow: 2,
          maxNodesStart: 6,
          stagnationFactor: 0.1,
        });

        // Act
        const budgetSnapshots = collectBudgetSnapshots(
          adaptiveController,
          [1, 1, 1, 1],
        );

        // Assert
        expect(budgetSnapshots.at(-1)).toEqual({
          maxNodes: 5,
          maxConns: undefined,
        });
      });
    });

    describe('given repeated stagnation with an explicit minimum node budget', () => {
      it('clamps the node budget at the configured minimum after shrink attempts', () => {
        // Arrange
        const adaptiveController = createComplexityController({
          enabled: true,
          mode: 'adaptive',
          improvementWindow: 2,
          maxNodesStart: 9,
          minNodes: 8,
          stagnationFactor: 0.1,
        });

        // Act
        const budgetSnapshots = collectBudgetSnapshots(
          adaptiveController,
          [1, 1, 1, 1, 1],
        );

        // Assert
        expect(budgetSnapshots.at(-1)).toEqual({
          maxNodes: 8,
          maxConns: undefined,
        });
      });
    });

    describe('given repeated stagnation with an explicit connection budget start', () => {
      it('keeps maxConns pinned to the configured start floor when shrink pressure would go lower', () => {
        // Arrange
        const adaptiveController = createComplexityController({
          enabled: true,
          mode: 'adaptive',
          improvementWindow: 3,
          maxNodesStart: 6,
          maxConnsStart: 18,
          maxConnsEnd: 30,
          stagnationFactor: 0.5,
        });

        // Act
        const budgetSnapshots = collectBudgetSnapshots(
          adaptiveController,
          [1, 1, 1, 1],
        );

        // Assert
        expect(budgetSnapshots.map((snapshot) => snapshot.maxConns)).toEqual([
          18, 18, 18, 18,
        ]);
      });
    });

    describe('given a linear budget schedule across a fixed horizon', () => {
      it('interpolates maxNodes from the configured start toward the configured end', () => {
        // Arrange
        const adaptiveController = createComplexityController({
          enabled: true,
          mode: 'linear',
          maxNodesStart: 6,
          maxNodesEnd: 12,
          horizon: 5,
        });

        // Act
        const nodeBudgets = collectNodeBudgetByGeneration(
          adaptiveController,
          [0, 1, 2, 3, 4, 5],
        );

        // Assert
        expect(nodeBudgets).toEqual([6, 7, 8, 9, 10, 12]);
      });
    });
  });

  describe('applyPhasedComplexity', () => {
    describe('given a phase window that expires exactly at the current generation', () => {
      it('toggles the phase and resets the phase start generation at the exact boundary', () => {
        // Arrange
        const adaptiveController: NeatLikeWithAdaptive = {
          options: {
            phasedComplexity: {
              enabled: true,
              phaseLength: 2,
              initialPhase: 'complexify',
            },
          },
          population: [],
          input: 2,
          output: 1,
          generation: 0,
        };

        // Act
        const phaseTimeline = Array.from(
          { length: 5 },
          (_, generationIndex) => {
            adaptiveController.generation = generationIndex;
            applyPhasedComplexity.call(adaptiveController);
            return readPhaseState(adaptiveController);
          },
        );

        // Assert
        expect(phaseTimeline).toEqual([
          { phase: 'complexify', phaseStartGeneration: 0 },
          { phase: 'complexify', phaseStartGeneration: 0 },
          { phase: 'simplify', phaseStartGeneration: 2 },
          { phase: 'simplify', phaseStartGeneration: 2 },
          { phase: 'complexify', phaseStartGeneration: 4 },
        ]);
      });
    });

    describe('given evolve runs with a one-generation phase window', () => {
      it('records the expected phase history across consecutive evolve calls', async () => {
        // Arrange
        const fitness = (network: Network) => network.connections.length;
        const neatController = new Neat(2, 1, fitness, {
          popsize: 8,
          phasedComplexity: { enabled: true, phaseLength: 1 },
          mutationRate: 1,
          mutationAmount: 1,
        });

        // Act
        const phaseHistory: Array<string | undefined> = [];
        await neatController.evolve();
        phaseHistory.push(readPhaseState(neatController).phase);
        await neatController.evolve();
        phaseHistory.push(readPhaseState(neatController).phase);

        // Assert
        expect(phaseHistory).toEqual(['complexify', 'simplify']);
      });
    });
  });
});
