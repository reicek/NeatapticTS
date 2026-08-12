import { describe, expect, it } from '@jest/globals';
import {
  NEATENSTEIN_MAIN_NEAT_INPUTS,
  NEATENSTEIN_MAIN_NEAT_OUTPUTS,
  NEATENSTEIN_FALLBACK_TURN_RATE,
  NEATENSTEIN_MOVE_ACCEL_PER_TICK,
  NEATENSTEIN_MOVE_DECEL_PER_TICK,
  MAX_TURN_RATE,
  ENEMY_VISIBLE_SENSOR_INDEX,
  FIRE_GATE_HYSTERESIS_LOW,
  FIRE_GATE_HYSTERESIS_HIGH,
  NEATENSTEIN_LOW_AMMO_RATIO,
  NEATENSTEIN_AMMO_PICKUP_SENSOR_COUNT,
  NEATENSTEIN_AMMO_PICKUP_START_INDEX,
  createFireGateState,
  applyFireGate,
  networkOutputToTickInput,
  smoothCommand,
} from './neat-io-config';

describe('neat-io-config: constants and configuration', () => {
  it('NEATENSTEIN_MAIN_NEAT_INPUTS is 22', () => {
    expect(NEATENSTEIN_MAIN_NEAT_INPUTS).toBe(22);
  });

  it('NEATENSTEIN_MAIN_NEAT_OUTPUTS is 5', () => {
    expect(NEATENSTEIN_MAIN_NEAT_OUTPUTS).toBe(5);
  });

  it('AC-P2S1-001: NEATENSTEIN_LOW_AMMO_RATIO is 0.25', () => {
    expect(NEATENSTEIN_LOW_AMMO_RATIO).toBe(0.25);
  });

  it('AC-P2S1-001: NEATENSTEIN_AMMO_PICKUP_SENSOR_COUNT is 7', () => {
    expect(NEATENSTEIN_AMMO_PICKUP_SENSOR_COUNT).toBe(7);
  });

  it('AC-P2S1-001: NEATENSTEIN_AMMO_PICKUP_START_INDEX is 15', () => {
    expect(NEATENSTEIN_AMMO_PICKUP_START_INDEX).toBe(15);
  });

  it('AC-P2S1-001: ammo pickup block ends at the main input count', () => {
    expect(
      NEATENSTEIN_AMMO_PICKUP_START_INDEX +
        NEATENSTEIN_AMMO_PICKUP_SENSOR_COUNT,
    ).toBe(NEATENSTEIN_MAIN_NEAT_INPUTS);
  });

  it('MAX_TURN_RATE is π/4', () => {
    expect(MAX_TURN_RATE).toBeCloseTo(Math.PI / 4);
  });

  it('NEATENSTEIN_FALLBACK_TURN_RATE is π/12', () => {
    expect(NEATENSTEIN_FALLBACK_TURN_RATE).toBeCloseTo(Math.PI / 12);
  });
});

describe('P5S1-fire-gate: hysteresis constants', () => {
  it('AC-P5S1a-003: FIRE_GATE_HYSTERESIS_LOW is 0.15', () => {
    expect(FIRE_GATE_HYSTERESIS_LOW).toBe(0.15);
  });

  it('AC-P5S1a-003: FIRE_GATE_HYSTERESIS_HIGH is 0.18', () => {
    expect(FIRE_GATE_HYSTERESIS_HIGH).toBe(0.18);
  });

  it('AC-P5S1a-003: LOW < HIGH (hysteresis band exists)', () => {
    expect(FIRE_GATE_HYSTERESIS_LOW).toBeLessThan(FIRE_GATE_HYSTERESIS_HIGH);
  });

  it('ENEMY_VISIBLE_SENSOR_INDEX is 12', () => {
    expect(ENEMY_VISIBLE_SENSOR_INDEX).toBe(12);
  });
});

describe('P5S1-fire-gate: createFireGateState', () => {
  it('creates state with fireActive = false (gate closed)', () => {
    const state = createFireGateState();
    expect(state.fireActive).toBe(false);
  });

  it('returns a new object each call (no shared reference)', () => {
    const a = createFireGateState();
    const b = createFireGateState();
    expect(a).not.toBe(b);
    a.fireActive = true;
    expect(b.fireActive).toBe(false);
  });
});

describe('P5S1-fire-gate: applyFireGate', () => {
  it('AC-P5S1a-001: suppresses fire when enemyVisible below hysteresis floor', () => {
    const state = createFireGateState();
    // enemyVisible = 0 (no enemy) → below 0.15 floor → fire suppressed
    expect(applyFireGate(state, 0, 0.9)).toBe(false);
  });

  it('AC-P5S1a-001: suppresses fire even when rawFireOutput is strongly positive', () => {
    const state = createFireGateState();
    expect(applyFireGate(state, 0, 5.0)).toBe(false);
  });

  it('AC-P5S1a-002: allows fire when enemyVisible above hysteresis ceiling', () => {
    const state = createFireGateState();
    // enemyVisible = 1 (enemy visible) → above 0.18 ceiling → fire allowed
    expect(applyFireGate(state, 1, 0.5)).toBe(true);
  });

  it('AC-P5S1a-002: does NOT suppress fire when enemy is visible but rawFireOutput ≤ 0', () => {
    const state = createFireGateState();
    // Gate opens, but raw fire output is not positive → no fire
    expect(applyFireGate(state, 1, -0.1)).toBe(false);
    expect(applyFireGate(state, 1, 0)).toBe(false);
  });

  it('AC-P5S1a-003: hysteresis — gate stays open when enemyVisible drops into hysteresis band', () => {
    const state = createFireGateState();
    // Open the gate with enemyVisible = 1
    applyFireGate(state, 1, 0.5);
    expect(state.fireActive).toBe(true);

    // enemyVisible drops to 0.16 (between 0.15 and 0.18) — gate stays open
    expect(applyFireGate(state, 0.16, 0.5)).toBe(true);
    expect(state.fireActive).toBe(true);
  });

  it('AC-P5S1a-003: hysteresis — gate stays closed when enemyVisible rises into hysteresis band', () => {
    const state = createFireGateState();
    // Gate starts closed
    expect(state.fireActive).toBe(false);

    // enemyVisible rises to 0.16 (between 0.15 and 0.18) — gate stays closed
    expect(applyFireGate(state, 0.16, 0.5)).toBe(false);
    expect(state.fireActive).toBe(false);
  });

  it('AC-P5S1a-003: hysteresis — gate closes when enemyVisible drops below floor', () => {
    const state = createFireGateState();
    // Open the gate
    applyFireGate(state, 1, 0.5);
    expect(state.fireActive).toBe(true);

    // enemyVisible drops to 0.10 (below 0.15 floor) — gate closes
    expect(applyFireGate(state, 0.1, 0.5)).toBe(false);
    expect(state.fireActive).toBe(false);
  });

  it('AC-P5S1a-003: hysteresis — gate opens when enemyVisible rises above ceiling', () => {
    const state = createFireGateState();
    // Gate starts closed
    expect(state.fireActive).toBe(false);

    // enemyVisible rises to 0.20 (above 0.18 ceiling) — gate opens
    expect(applyFireGate(state, 0.2, 0.5)).toBe(true);
    expect(state.fireActive).toBe(true);
  });

  it('AC-P5S1a-003: hysteresis — no rapid oscillation at boundary', () => {
    const state = createFireGateState();
    // Simulate enemyVisible oscillating around the boundary
    // Tick 1: enemy visible → gate opens
    expect(applyFireGate(state, 1, 0.5)).toBe(true);
    // Tick 2: brief visibility loss (sensor = 0.16, in hysteresis band) → gate stays open
    expect(applyFireGate(state, 0.16, 0.5)).toBe(true);
    // Tick 3: enemy still barely visible (sensor = 0.17, in hysteresis band) → gate stays open
    expect(applyFireGate(state, 0.17, 0.5)).toBe(true);
    // Tick 4: enemy visible again → gate stays open
    expect(applyFireGate(state, 1, 0.5)).toBe(true);
    expect(state.fireActive).toBe(true);
  });

  it('AC-P5S1a-003: hysteresis — gate closes and stays closed through band', () => {
    const state = createFireGateState();
    // Open the gate
    applyFireGate(state, 1, 0.5);
    expect(state.fireActive).toBe(true);

    // enemyVisible drops below floor → gate closes
    expect(applyFireGate(state, 0.1, 0.5)).toBe(false);
    expect(state.fireActive).toBe(false);

    // enemyVisible rises into hysteresis band → gate stays closed
    expect(applyFireGate(state, 0.16, 0.5)).toBe(false);
    expect(state.fireActive).toBe(false);
  });

  it('mutates state in-place (persists between ticks)', () => {
    const state = createFireGateState();
    applyFireGate(state, 1, 0.5);
    expect(state.fireActive).toBe(true);
    // Same state object should retain the change
    applyFireGate(state, 0.16, 0.5);
    expect(state.fireActive).toBe(true);
  });
});

describe('P5S1-fire-gate: networkOutputToTickInput with fire gate', () => {
  it('AC-P5S1a-001: suppresses fire when no enemy visible (fireGate provided)', () => {
    const state = createFireGateState();
    const result = networkOutputToTickInput([0, 0, 0, 0.9, 0], {
      state,
      enemyVisible: 0,
    });
    expect(result.fire).toBe(false);
  });

  it('AC-P5S1a-002: allows fire when enemy visible (fireGate provided)', () => {
    const state = createFireGateState();
    const result = networkOutputToTickInput([0, 0, 0, 0.5, 0], {
      state,
      enemyVisible: 1,
    });
    expect(result.fire).toBe(true);
  });

  it('without fireGate, fire follows normal threshold (backward compat)', () => {
    const result = networkOutputToTickInput([0, 0, 0, 0.5, 0]);
    expect(result.fire).toBe(true);

    const result2 = networkOutputToTickInput([0, 0, 0, -0.1, 0]);
    expect(result2.fire).toBe(false);
  });

  it('hysteresis maintained across multiple calls with same state object', () => {
    const state = createFireGateState();
    // Tick 1: enemy visible → fire allowed
    let result = networkOutputToTickInput([0, 0, 0, 0.5, 0], {
      state,
      enemyVisible: 1,
    });
    expect(result.fire).toBe(true);

    // Tick 2: enemy briefly not visible (in hysteresis band) → fire still allowed
    result = networkOutputToTickInput([0, 0, 0, 0.5, 0], {
      state,
      enemyVisible: 0.16,
    });
    expect(result.fire).toBe(true);

    // Tick 3: enemy truly gone (below floor) → fire suppressed
    result = networkOutputToTickInput([0, 0, 0, 0.5, 0], {
      state,
      enemyVisible: 0,
    });
    expect(result.fire).toBe(false);
  });

  it('non-fire outputs are unaffected by fire gate', () => {
    const state = createFireGateState();
    const result = networkOutputToTickInput([0.8, -0.6, 0.4, 0.9, 0.7], {
      state,
      enemyVisible: 0,
    });
    // Fire suppressed but move/look/dash are normal
    expect(result.fire).toBe(false);
    expect(result.move.x).toBeCloseTo(Math.tanh(0.8));
    expect(result.move.y).toBeCloseTo(Math.tanh(-0.6));
    expect(result.lookDelta).toBeCloseTo(Math.tanh(0.4) * MAX_TURN_RATE);
    expect(result.dash).toBe(true);
  });

  it('pads short output arrays even with fire gate', () => {
    const state = createFireGateState();
    const result = networkOutputToTickInput([0.5, 0.5], {
      state,
      enemyVisible: 1,
    });
    // Padded: out[2]=0, out[3]=0, out[4]=0
    expect(result.fire).toBe(false); // out[3] = 0, not > 0
    expect(result.dash).toBe(false); // out[4] = 0, not > 0.5
    expect(result.move.x).toBeCloseTo(Math.tanh(0.5));
    expect(result.move.y).toBeCloseTo(Math.tanh(0.5));
  });
});

describe('P1S1-smoothing: movement smoothing constants and helper', () => {
  it('AC-002: NEATENSTEIN_MOVE_ACCEL_PER_TICK is exported and positive', () => {
    expect(NEATENSTEIN_MOVE_ACCEL_PER_TICK).toBe(0.2);
  });

  it('AC-002: NEATENSTEIN_MOVE_DECEL_PER_TICK is exported and positive', () => {
    expect(NEATENSTEIN_MOVE_DECEL_PER_TICK).toBe(0.25);
  });

  it('AC-003: smoothCommand ramps from 0 toward the target by the accel limit', () => {
    expect(smoothCommand(0, 1)).toBe(NEATENSTEIN_MOVE_ACCEL_PER_TICK);
  });

  it('AC-003: smoothCommand reaches the target once the delta is within the limit', () => {
    const current = 0.85;
    const target = 1;
    expect(smoothCommand(current, target)).toBe(target);
  });

  it('AC-003: smoothCommand ramps down from 0 toward a negative target by the decel limit', () => {
    expect(smoothCommand(0, -1)).toBe(-NEATENSTEIN_MOVE_DECEL_PER_TICK);
  });

  it('AC-004: smoothCommand can override defaults with custom accel/decel limits', () => {
    expect(smoothCommand(0, 1, 0.5, 0.5)).toBe(0.5);
    expect(smoothCommand(0, -1, 0.5, 0.5)).toBe(-0.5);
  });

  it('AC-004: smoothCommand leaves the value unchanged when current equals target', () => {
    expect(smoothCommand(0.5, 0.5)).toBe(0.5);
  });
});
