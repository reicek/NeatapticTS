/**
 * Red tests for the NGE juvenile lifecycle acceleration policy.
 *
 * `src/neat/nge-juvenile/neat.nge-juvenile.lifecycle-policy.ts` maps NGE
 * lifecycle stages (`embryo`, `baby`, `juvenile`, `adult`, `equilibrium`) to
 * stage-specific `AccelerationConfig` values. This suite defines the expected
 * contract before the implementation slice lands.
 *
 * Tests import from `../neat.nge-juvenile.lifecycle-policy`, which does not
 * exist yet, so the suite fails with TS2307 until P8S3-05.
 */

import { buildJuvenileLifecyclePolicy } from '../neat.nge-juvenile.lifecycle-policy';

describe('nge-juvenile.lifecycle-policy', () => {
  describe('buildJuvenileLifecyclePolicy', () => {
    it('exports a policy factory', () => {
      expect(typeof buildJuvenileLifecyclePolicy).toBe('function');
    });

    it('returns a policy with a stages map', () => {
      const policy = buildJuvenileLifecyclePolicy();

      expect(policy.stages).toBeDefined();
    });

    it('includes a baby stage configuration', () => {
      const policy = buildJuvenileLifecyclePolicy();

      expect(policy.stages.baby).toBeDefined();
    });

    it('includes an adult stage configuration', () => {
      const policy = buildJuvenileLifecyclePolicy();

      expect(policy.stages.adult).toBeDefined();
    });

    it('includes an equilibrium stage configuration', () => {
      const policy = buildJuvenileLifecyclePolicy();

      expect(policy.stages.equilibrium).toBeDefined();
    });

    it('exposes per-stage backend preferences as strings', () => {
      const policy = buildJuvenileLifecyclePolicy();
      const baby = policy.stages.baby ?? {};

      expect(typeof baby.backend).toBe('string');
    });
  });
});
