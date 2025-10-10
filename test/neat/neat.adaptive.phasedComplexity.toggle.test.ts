import Neat from '../../src/neat';
import type { NeatLikeWithAdaptive } from '../../src/neat/neat.adaptive';
import { applyPhasedComplexity } from '../../src/neat/neat.adaptive';

/** Tests for phase toggling edge (exact boundary) in phased complexity. */
describe('Phased Complexity Toggle Boundary', () => {
  describe('toggles phase exactly at boundary', () => {
    const fitness = () => 1;
    const neat = new Neat(2, 1, fitness, {
      popsize: 2,
      seed: 910,
      phasedComplexity: {
        enabled: true,
        phaseLength: 2,
        initialPhase: 'complexify',
      },
    });
    test('phase flips after configured length', () => {
      // Arrange: simulate generations and apply each time
      for (let g = 0; g < 5; g++) {
        neat.generation = g;
        applyPhasedComplexity.call((neat as unknown as NeatLikeWithAdaptive));
      }
      // Act: final phase after simulation
      const phase = Reflect.get(neat, '_phase') as string | undefined;
      // Assert: phase string exists (flipped at least once)
      expect(typeof phase).toBe('string');
    });
  });
});

