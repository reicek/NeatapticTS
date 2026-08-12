/**
 * Red-phase test contracts for Phase 4 Step 05 — combat-pressure to
 * reproduction-mode mapping.
 *
 * Covers AC-401-S05-004: raw combat metrics are mapped to a deterministic
 * ReproductionModePressureSignal where exactly one of isDominating,
 * isStruggling, or isStalemate is true.
 *
 * All tests fail because the imported source module does not exist yet. The
 * expected failure reason is TS2307 "Cannot find module".
 *
 * Single-expect rule enforced. AAA structure in every test.
 */

import {
  evaluateCombatPressure,
  type CombatMetrics,
} from './neat.nge-evolution.combat-pressure';
import type { ReproductionModePressureSignal } from './neat.nge-evolution.reproduction-mode';

describe('evaluateCombatPressure', () => {
  it('marks the signal as dominating when damage dealt and kills are high', () => {
    // Arrange — high damage dealt, high kills, low damage taken, low deaths
    const metrics: CombatMetrics = {
      damageDealt: 900,
      damageTaken: 50,
      kills: 8,
      deaths: 0,
      generation: 7,
    };

    // Act
    const signal: ReproductionModePressureSignal =
      evaluateCombatPressure(metrics);

    // Assert
    expect(signal.isDominating).toBe(true);
  });

  it('marks the signal as struggling when damage taken and deaths are high', () => {
    // Arrange — high damage taken, high deaths, low damage dealt, low kills
    const metrics: CombatMetrics = {
      damageDealt: 40,
      damageTaken: 850,
      kills: 0,
      deaths: 9,
      generation: 7,
    };

    // Act
    const signal: ReproductionModePressureSignal =
      evaluateCombatPressure(metrics);

    // Assert
    expect(signal.isStruggling).toBe(true);
  });

  it('marks the signal as stalemate when damage and kills/deaths are balanced', () => {
    // Arrange — balanced damage, equal kills and deaths
    const metrics: CombatMetrics = {
      damageDealt: 450,
      damageTaken: 450,
      kills: 4,
      deaths: 4,
      generation: 7,
    };

    // Act
    const signal: ReproductionModePressureSignal =
      evaluateCombatPressure(metrics);

    // Assert
    expect(signal.isStalemate).toBe(true);
  });

  it('preserves the input generation number on the returned signal', () => {
    // Arrange — arbitrary deterministic generation
    const metrics: CombatMetrics = {
      damageDealt: 900,
      damageTaken: 50,
      kills: 8,
      deaths: 0,
      generation: 12,
    };

    // Act
    const signal: ReproductionModePressureSignal =
      evaluateCombatPressure(metrics);

    // Assert
    expect(signal.generation).toBe(12);
  });

  it('returns a signal with exactly one pressure flag set', () => {
    // Arrange — arbitrary balanced combat metrics
    const metrics: CombatMetrics = {
      damageDealt: 300,
      damageTaken: 300,
      kills: 2,
      deaths: 2,
      generation: 3,
    };

    // Act
    const signal: ReproductionModePressureSignal =
      evaluateCombatPressure(metrics);

    // Assert
    const trueCount = [
      signal.isDominating,
      signal.isStruggling,
      signal.isStalemate,
    ].filter(Boolean).length;
    expect(trueCount).toBe(1);
  });
});
