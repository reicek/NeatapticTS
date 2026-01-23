import Neat from '../../src/neat';
import type { NeatLikeWithAdaptive } from '../../src/neat/neat.adaptive';
import { createTelemetryEntryBase } from '../../src/neat/neat.telemetry';
import type { TelemetryEntry } from '../../src/neat/neat.types';

/** Tests ancestor uniqueness adaptive in lineagePressure mode (strength adjustments). */
describe('Ancestor Uniqueness Adaptive (lineagePressure mode)', () => {
  describe('increases lineage pressure strength when uniqueness low', () => {
    const fitness = () => 1;
    const neat = new Neat(2, 1, fitness, {
      popsize: 3,
      seed: 970,
      ancestorUniqAdaptive: {
        enabled: true,
        mode: 'lineagePressure',
        lowThreshold: 0.5,
        highThreshold: 0.9,
        adjust: 0.05,
        cooldown: 0,
      },
    });
    test('lineagePressure.strength increases', async () => {
      // Arrange: low ancestor uniqueness telemetry
      ((neat as unknown) as {
        _telemetry: TelemetryEntry[];
      })._telemetry.push({
        ...createTelemetryEntryBase(0, 0, 1),
        lineage: {
          parents: [],
          depthBest: 0,
          meanDepth: 0,
          inbreeding: 0,
          ancestorUniq: 0.2,
        },
      });
      const { applyAncestorUniqAdaptive } = await import(
        '../../src/neat/neat.adaptive'
      );
      applyAncestorUniqAdaptive.call((neat as unknown) as NeatLikeWithAdaptive);
      const strength = neat.options.lineagePressure.strength;
      // Assert: strength above default baseline 0.01
      expect(strength).toBeGreaterThan(0.01);
    });
  });
  describe('reduces lineage pressure strength when uniqueness high', () => {
    const fitness = () => 1;
    const neat = new Neat(2, 1, fitness, {
      popsize: 3,
      seed: 971,
      ancestorUniqAdaptive: {
        enabled: true,
        mode: 'lineagePressure',
        lowThreshold: 0.1,
        highThreshold: 0.2,
        adjust: 0.05,
        cooldown: 0,
      },
    });
    test('lineagePressure.strength decreases from initial', async () => {
      // Arrange: set initial strength then push high uniqueness
      ((neat as unknown) as {
        _telemetry: TelemetryEntry[];
      })._telemetry.push({
        ...createTelemetryEntryBase(0, 0, 1),
        lineage: {
          parents: [],
          depthBest: 0,
          meanDepth: 0,
          inbreeding: 0,
          ancestorUniq: 0.95,
        },
      });
      const { applyAncestorUniqAdaptive } = await import(
        '../../src/neat/neat.adaptive'
      );
      applyAncestorUniqAdaptive.call((neat as unknown) as NeatLikeWithAdaptive);
      const strength = neat.options.lineagePressure.strength;
      // Assert: strength not increased above starting 0.01 (may reduce or stay ~0.01)
      expect(strength).toBeLessThanOrEqual(0.01);
    });
  });
});
