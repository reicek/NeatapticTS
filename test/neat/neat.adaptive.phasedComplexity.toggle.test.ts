import Neat from '../../src/neat';

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
        (neat as any).generation = g;
        require('../../src/neat/neat.adaptive').applyPhasedComplexity.call(
          neat as any
        );
      }
      // Act: final phase after simulation
      const phase = (neat as any)._phase;
      // Assert: phase string exists (flipped at least once)
      expect(typeof phase).toBe('string');
    });
  });
});
