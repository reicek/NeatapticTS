/**
 * Red-phase test contract for the NGE collective multi-agent evaluation shelf.
 *
 * Target production module: `./neat.nge-collective.evaluation`
 * Phase G Step 03 — intended failures before any implementation exists.
 *
 * Covered behaviors:
 * 1. Evaluation context is constructed with the declared agent count and an initial
 *    generation tick of zero.
 * 2. `runCollectiveEvaluationTick` invokes evaluators in declared order [0, 1, ..., N-1]
 *    and collects one fitness value per agent.
 * 3. Sequential ordering guarantee: the evaluator for agent index 1 observes writes
 *    committed by agent index 0 within the same tick.
 * 4. `resetCollectiveEvaluationState` increments the generation tick and returns a field
 *    where all cells are zero.
 */

import {
  createSharedField,
  readCell,
  writeCell,
} from './neat.nge-collective.shared-field';
import type { SharedField } from './neat.nge-collective.shared-field';
import {
  createCollectiveEvaluationContext,
  resetCollectiveEvaluationState,
  runCollectiveEvaluationTick,
} from './neat.nge-collective.evaluation';
import {
  NGE_COLLECTIVE_DEFAULT_DECAY_FACTOR,
  NGE_COLLECTIVE_DEFAULT_DIFFUSION_RATE,
  NGE_COLLECTIVE_DEFAULT_SNAPSHOT_POOL_CAPACITY,
  NGE_COLLECTIVE_INITIAL_GENERATION_TICK,
} from './neat.nge-collective.constants';
import {
  NgeCollective_EvaluationError,
  NgeCollective_FieldDimensionError,
} from './neat.nge-collective.errors';

/** Minimal 4×4 shared field used as a deterministic fixture. */
function buildFixtureField() {
  return createSharedField(4, 4);
}

describe('neat.nge-collective multi-agent evaluation', () => {
  describe('createCollectiveEvaluationContext', () => {
    describe('given agentCount=3 and a 4×4 field', () => {
      it('returns a context with agentCount equal to 3', () => {
        // Arrange
        const field = buildFixtureField();

        // Act
        const context = createCollectiveEvaluationContext(3, field);

        // Assert
        expect(context.agentCount).toBe(3);
      });

      it('initializes the generation tick to zero', () => {
        // Arrange
        const field = buildFixtureField();

        // Act
        const context = createCollectiveEvaluationContext(3, field);

        // Assert
        expect(context.generationTick).toBe(0);
      });

      it('holds a reference to the provided shared field', () => {
        // Arrange
        const field = buildFixtureField();

        // Act
        const context = createCollectiveEvaluationContext(3, field);

        // Assert
        expect(context.field.width).toBe(field.width);
      });
    });
  });

  describe('runCollectiveEvaluationTick', () => {
    describe('given fewer evaluators than the declared agent count', () => {
      it('skips the agent index when no evaluator is provided for that slot', () => {
        // Arrange — 3 agents declared but only 1 evaluator provided
        const context = createCollectiveEvaluationContext(
          3,
          buildFixtureField(),
        );
        const evaluators = [() => 10];

        // Act
        const result = runCollectiveEvaluationTick(context, evaluators);

        // Assert — only 1 fitness value collected (agent indices 1 and 2 skipped)
        expect(result.agentFitness.length).toBe(1);
      });
    });

    describe('given 3 agents each returning a distinct constant fitness', () => {
      it('returns agentFitness with length equal to the agent count', () => {
        // Arrange
        const context = createCollectiveEvaluationContext(
          3,
          buildFixtureField(),
        );
        const evaluators = [() => 10, () => 20, () => 30];

        // Act
        const result = runCollectiveEvaluationTick(context, evaluators);

        // Assert
        expect(result.agentFitness.length).toBe(3);
      });

      it('maps evaluator return values to their corresponding agent indices', () => {
        // Arrange
        const context = createCollectiveEvaluationContext(
          3,
          buildFixtureField(),
        );
        const evaluators = [() => 10, () => 20, () => 30];

        // Act
        const result = runCollectiveEvaluationTick(context, evaluators);

        // Assert — each agent gets its own evaluator's return value
        expect(result.agentFitness).toEqual([10, 20, 30]);
      });

      it('records evaluation order as [0, 1, 2] (declared sequence)', () => {
        // Arrange
        const context = createCollectiveEvaluationContext(
          3,
          buildFixtureField(),
        );
        const evaluators = [() => 1, () => 2, () => 3];

        // Act
        const result = runCollectiveEvaluationTick(context, evaluators);

        // Assert
        expect(result.evaluationOrder).toEqual([0, 1, 2]);
      });
    });

    describe('given sequential write-then-read dependency between agents', () => {
      it('allows agent index 1 to observe the cell written by agent index 0 in the same tick', () => {
        // Arrange
        const initialField = buildFixtureField();
        const context = createCollectiveEvaluationContext(2, initialField);
        let fieldSeenByAgent1CellZero = -1;

        // Agent 0 writes 0.5 to cell (0,0); Agent 1 reads it back via the live shared field.
        const evaluators = [
          (_agentIndex: number, field: SharedField) => {
            writeCell(field, 0, 0, 0.5);
            return 1;
          },
          (_agentIndex: number, field: SharedField) => {
            // Sequential ordering guarantee: agent 0's write is already committed
            fieldSeenByAgent1CellZero = readCell(field, 0, 0);
            return 2;
          },
        ];

        // Act
        runCollectiveEvaluationTick(context, evaluators);

        // Assert — agent 1 observed the value written by agent 0
        expect(fieldSeenByAgent1CellZero).toBeCloseTo(0.5, 6);
      });
    });
  });

  describe('resetCollectiveEvaluationState', () => {
    describe('given a context that has completed one tick', () => {
      it('increments the generation tick by one', () => {
        // Arrange
        const context = createCollectiveEvaluationContext(
          2,
          buildFixtureField(),
        );

        // Act
        const resetContext = resetCollectiveEvaluationState(context);

        // Assert
        expect(resetContext.generationTick).toBe(1);
      });

      it('returns a context whose field has all cells zeroed', () => {
        // Arrange
        const seedField = writeCell(buildFixtureField(), 1, 1, 0.9);
        // Construct a minimal context literal matching the expected CollectiveEvaluationContext shape
        const context = { agentCount: 1, field: seedField, generationTick: 0 };

        // Act
        const resetContext = resetCollectiveEvaluationState(context);

        // Assert
        expect(
          (Array.from(resetContext.field.cells) as number[]).every(
            (value) => value === 0,
          ),
        ).toBe(true);
      });
    });
  });
});

describe('neat.nge-collective constants', () => {
  describe('NGE_COLLECTIVE_DEFAULT_DECAY_FACTOR', () => {
    it('is 0.95', () => {
      expect(NGE_COLLECTIVE_DEFAULT_DECAY_FACTOR).toBe(0.95);
    });
  });

  describe('NGE_COLLECTIVE_DEFAULT_DIFFUSION_RATE', () => {
    it('is 0.1', () => {
      expect(NGE_COLLECTIVE_DEFAULT_DIFFUSION_RATE).toBe(0.1);
    });
  });

  describe('NGE_COLLECTIVE_DEFAULT_SNAPSHOT_POOL_CAPACITY', () => {
    it('is 5', () => {
      expect(NGE_COLLECTIVE_DEFAULT_SNAPSHOT_POOL_CAPACITY).toBe(5);
    });
  });

  describe('NGE_COLLECTIVE_INITIAL_GENERATION_TICK', () => {
    it('is 0', () => {
      expect(NGE_COLLECTIVE_INITIAL_GENERATION_TICK).toBe(0);
    });
  });
});

describe('neat.nge-collective error classes', () => {
  describe('NgeCollective_FieldDimensionError', () => {
    it('sets name to NgeCollective_FieldDimensionError', () => {
      // Arrange / Act
      const error = new NgeCollective_FieldDimensionError(
        'width must be positive',
      );

      // Assert
      expect(error.name).toBe('NgeCollective_FieldDimensionError');
    });

    it('inherits from Error', () => {
      // Arrange / Act
      const error = new NgeCollective_FieldDimensionError('test message');

      // Assert
      expect(error).toBeInstanceOf(Error);
    });

    it('carries the provided message', () => {
      // Arrange / Act
      const error = new NgeCollective_FieldDimensionError('bad dimension');

      // Assert
      expect(error.message).toBe('bad dimension');
    });
  });

  describe('NgeCollective_EvaluationError', () => {
    it('sets name to NgeCollective_EvaluationError', () => {
      // Arrange / Act
      const error = new NgeCollective_EvaluationError(
        'no evaluator for agent 2',
      );

      // Assert
      expect(error.name).toBe('NgeCollective_EvaluationError');
    });

    it('inherits from Error', () => {
      // Arrange / Act
      const error = new NgeCollective_EvaluationError('test message');

      // Assert
      expect(error).toBeInstanceOf(Error);
    });

    it('carries the provided message', () => {
      // Arrange / Act
      const error = new NgeCollective_EvaluationError('evaluation failure');

      // Assert
      expect(error.message).toBe('evaluation failure');
    });
  });
});
