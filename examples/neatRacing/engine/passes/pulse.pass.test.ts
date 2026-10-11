/**
 * Red-first contract tests for the NeatRacing ambient pulse pass (slice
 * P1-S2-C; AC-006 / spec §6.4.3).
 *
 * Locks the pulse contract: the `#B7FF00` (185, 255, 0) glow color, the
 * `PULSE_GLOW_ALPHA` mirror at 0.42, a lifetime-based attack/decay alpha
 * envelope that peaks at the glow alpha and decays deterministically to
 * zero, a ring radius that expands monotonically over the pulse lifetime,
 * seed-deterministic world-space placement (TP-20 visual seed 1337), a
 * deterministic spawn cadence capped at the concurrency limit, GL_LESS
 * depth-test semantics, and the `'pulse'` warmup key parity with the
 * context-service program registry (AC-040).
 *
 * The Jest project runs in a plain Node environment, so every contract is
 * exercised through pure functions and a recording mock WebGL2 context —
 * no GPU or DOM is required.
 */
import { describe, expect, it } from '@jest/globals';
import { NEATRACING_WARMED_PROGRAM_KEYS } from '../context.service';
import {
  AmbientPulse,
  NEATRACING_PULSE_AMBIENT_INTERVAL_MS,
  NEATRACING_PULSE_COLOR_RGB,
  NEATRACING_PULSE_ENVELOPE_ATTACK_FRACTION,
  NEATRACING_PULSE_GLOW_ALPHA,
  NEATRACING_PULSE_LIFETIME_MS,
  NEATRACING_PULSE_MAX_CONCURRENT,
  NEATRACING_PULSE_PROGRAM_KEY,
  NEATRACING_PULSE_RING_END_RADIUS_M,
  NEATRACING_PULSE_RING_START_RADIUS_M,
  NEATRACING_PULSE_SPAWN_HALF_EXTENT_M,
  NEATRACING_PULSE_VISUAL_SEED_DEFAULT,
  collectActivePulses,
  computePulseEnvelope,
  computePulseRingRadius,
  createPulsePass,
  derivePulseOrigin,
  isPulseDepthVisible,
  resolvePulseVisuals,
  toPulseWarmupProgram,
} from './pulse.pass';

/** Steady-state timestamp where exactly the concurrency cap would be alive. */
const STEADY_STATE_SIM_TIME_MS = NEATRACING_PULSE_LIFETIME_MS - 1;

/** Progress probe points spanning the attack ramp (strictly below attack end). */
const ATTACK_PROBES = [0.02, 0.07, NEATRACING_PULSE_ENVELOPE_ATTACK_FRACTION];

/** Progress probe points spanning the decay tail (strictly above attack end). */
const DECAY_PROBES = [0.2, 0.4, 0.6, 0.8];

/** Spawn indices probed for deterministic placement coverage. */
const PLACEMENT_PROBE_SPAWN_INDICES = [0, 1, 2, 7, 19];

/** Column-major identity view-projection used by the mock draw frames. */
const IDENTITY_VIEW_PROJECTION_FIXTURE = [
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
];

/** `gl.BLEND` enum value (0x0BE2) asserted on the mock glow draw state. */
const MOCK_GL_BLEND = 0x0be2;

/** `gl.CULL_FACE` enum value (0x0B44) asserted on the mock glow draw state. */
const MOCK_GL_CULL_FACE = 0x0b44;

/** `gl.DEPTH_TEST` enum value (0x0B71) asserted on the mock glow draw state. */
const MOCK_GL_DEPTH_TEST = 0x0b71;

/** `gl.ONE` blend factor (1) asserted on the mock additive blend call. */
const MOCK_GL_ONE = 1;

describe('neatRacing pulse pass constants (AC-006)', () => {
  it('exports the #B7FF00 pulse color normalized to the [0, 1] uniform range', () => {
    expect(NEATRACING_PULSE_COLOR_RGB).toEqual([185 / 255, 255 / 255, 0]);
  });

  it('mirrors the PULSE_GLOW_ALPHA constant at 0.42', () => {
    expect(NEATRACING_PULSE_GLOW_ALPHA).toBe(0.42);
  });

  it('registers under the pulse warmup key from the program registry', () => {
    expect(NEATRACING_PULSE_PROGRAM_KEY).toBe('pulse');
    expect(NEATRACING_WARMED_PROGRAM_KEYS).toContain(
      NEATRACING_PULSE_PROGRAM_KEY,
    );
  });

  it('defaults the visual seed to the TP-20 base 1337', () => {
    expect(NEATRACING_PULSE_VISUAL_SEED_DEFAULT).toBe(1337);
  });
});

describe('lifetime alpha envelope (AC-006)', () => {
  it('holds the envelope at zero before the pulse is born', () => {
    expect(computePulseEnvelope(-0.25)).toBe(0);
    expect(computePulseEnvelope(0)).toBe(0);
  });

  it('ramps up linearly across the attack window to the glow alpha peak', () => {
    const attackAlphas = ATTACK_PROBES.map(computePulseEnvelope);
    expect(attackAlphas[0]).toBeLessThan(attackAlphas[1]);
    expect(attackAlphas[1]).toBeLessThan(attackAlphas[2]);
    expect(attackAlphas[2]).toBeCloseTo(NEATRACING_PULSE_GLOW_ALPHA, 12);
  });

  it('decays monotonically from the peak back to zero at end of lifetime', () => {
    const decayAlphas = DECAY_PROBES.map(computePulseEnvelope);
    expect(decayAlphas[0]).toBeLessThan(NEATRACING_PULSE_GLOW_ALPHA);
    expect(decayAlphas[0]).toBeGreaterThan(decayAlphas[1]);
    expect(decayAlphas[1]).toBeGreaterThan(decayAlphas[2]);
    expect(decayAlphas[2]).toBeGreaterThan(decayAlphas[3]);
    expect(computePulseEnvelope(1)).toBe(0);
  });

  it('clamps progress beyond the lifetime back to zero', () => {
    expect(computePulseEnvelope(1.5)).toBe(0);
  });

  it('is deterministic for identical progress inputs', () => {
    expect(computePulseEnvelope(0.5)).toBe(computePulseEnvelope(0.5));
  });
});

describe('ring expansion curve', () => {
  it('starts at the ring start radius at birth', () => {
    expect(computePulseRingRadius(0)).toBe(
      NEATRACING_PULSE_RING_START_RADIUS_M,
    );
  });

  it('expands to the ring end radius at end of lifetime', () => {
    expect(computePulseRingRadius(1)).toBe(NEATRACING_PULSE_RING_END_RADIUS_M);
  });

  it('expands monotonically across the lifetime', () => {
    const radii = [0, 0.25, 0.5, 0.75, 1].map(computePulseRingRadius);
    expect(radii[0]).toBeLessThan(radii[1]);
    expect(radii[1]).toBeLessThan(radii[2]);
    expect(radii[2]).toBeLessThan(radii[3]);
    expect(radii[3]).toBeLessThan(radii[4]);
  });

  it('clamps progress outside the lifetime onto the radius endpoints', () => {
    expect(computePulseRingRadius(-1)).toBe(
      NEATRACING_PULSE_RING_START_RADIUS_M,
    );
    expect(computePulseRingRadius(2)).toBe(NEATRACING_PULSE_RING_END_RADIUS_M);
  });
});

describe('deterministic placement from the seed (TP-20)', () => {
  it('places every derived origin inside the spawn bounds', () => {
    for (const spawnIndex of PLACEMENT_PROBE_SPAWN_INDICES) {
      const origin = derivePulseOrigin(
        NEATRACING_PULSE_VISUAL_SEED_DEFAULT,
        spawnIndex,
      );
      expect(Math.abs(origin.x)).toBeLessThanOrEqual(
        NEATRACING_PULSE_SPAWN_HALF_EXTENT_M,
      );
      expect(Math.abs(origin.z)).toBeLessThanOrEqual(
        NEATRACING_PULSE_SPAWN_HALF_EXTENT_M,
      );
    }
  });

  it('reproduces identical origins for the same seed and spawn index', () => {
    const first = derivePulseOrigin(NEATRACING_PULSE_VISUAL_SEED_DEFAULT, 3);
    const second = derivePulseOrigin(NEATRACING_PULSE_VISUAL_SEED_DEFAULT, 3);
    expect(first).toEqual(second);
  });

  it('differs across seeds so the visual seed is meaningful', () => {
    const seeded = derivePulseOrigin(1337, 0);
    const other = derivePulseOrigin(4242, 0);
    expect(seeded).not.toEqual(other);
  });
});

describe('ambient pulse cadence', () => {
  it('spawns the first pulse at simulation time zero with zero age', () => {
    const pulses = collectActivePulses(0, NEATRACING_PULSE_VISUAL_SEED_DEFAULT);
    expect(pulses).toHaveLength(1);
    expect(pulses[0].spawnIndex).toBe(0);
    expect(pulses[0].ageMs).toBe(0);
    expect(pulses[0].active).toBe(true);
  });

  it('keeps exactly the concurrency cap alive at steady state', () => {
    const pulses = collectActivePulses(
      STEADY_STATE_SIM_TIME_MS,
      NEATRACING_PULSE_VISUAL_SEED_DEFAULT,
    );
    expect(pulses).toHaveLength(NEATRACING_PULSE_MAX_CONCURRENT);
  });

  it('caps the alive set and orders pulses newest-first when the window overflows', () => {
    const overflowTimeMs = 2_500;
    const pulses = collectActivePulses(
      overflowTimeMs,
      NEATRACING_PULSE_VISUAL_SEED_DEFAULT,
    );
    expect(pulses).toHaveLength(NEATRACING_PULSE_MAX_CONCURRENT);
    const spawnIndices = pulses.map((pulse) => pulse.spawnIndex);
    const isDescending = spawnIndices.every(
      (spawnIndex, position) =>
        position === 0 || spawnIndex < spawnIndices[position - 1],
    );
    expect(isDescending).toBe(true);
  });

  it('matches each alive pulse origin to its deterministic seed derivation', () => {
    const pulses = collectActivePulses(
      2_500,
      NEATRACING_PULSE_VISUAL_SEED_DEFAULT,
    );
    for (const pulse of pulses) {
      const origin = derivePulseOrigin(
        NEATRACING_PULSE_VISUAL_SEED_DEFAULT,
        pulse.spawnIndex,
      );
      expect(pulse.originX).toBe(origin.x);
      expect(pulse.originZ).toBe(origin.z);
    }
  });

  it('expires pulses once the lifetime elapses', () => {
    const expiredTimeMs =
      NEATRACING_PULSE_LIFETIME_MS + NEATRACING_PULSE_AMBIENT_INTERVAL_MS;
    const pulses = collectActivePulses(
      expiredTimeMs,
      NEATRACING_PULSE_VISUAL_SEED_DEFAULT,
    );
    const spawnIndexOfOldest = pulses[pulses.length - 1].spawnIndex;
    const oldestAgeMs =
      expiredTimeMs - spawnIndexOfOldest * NEATRACING_PULSE_AMBIENT_INTERVAL_MS;
    expect(oldestAgeMs).toBeLessThan(NEATRACING_PULSE_LIFETIME_MS);
  });

  it('yields no pulses for negative simulation time', () => {
    expect(
      collectActivePulses(-1, NEATRACING_PULSE_VISUAL_SEED_DEFAULT),
    ).toHaveLength(0);
  });

  it('is deterministic for identical time and seed inputs', () => {
    const first = collectActivePulses(
      1_234,
      NEATRACING_PULSE_VISUAL_SEED_DEFAULT,
    );
    const second = collectActivePulses(
      1_234,
      NEATRACING_PULSE_VISUAL_SEED_DEFAULT,
    );
    expect(first).toEqual(second);
  });
});

describe('pulse visuals composition', () => {
  it('derives envelope alpha and expanded radius from the pulse age', () => {
    const pulse: AmbientPulse = {
      spawnIndex: 3,
      originX: 1,
      originZ: -1,
      ageMs: NEATRACING_PULSE_LIFETIME_MS / 2,
      active: true,
    };
    const visuals = resolvePulseVisuals(pulse);
    expect(visuals.radiusM).toBeCloseTo(
      NEATRACING_PULSE_RING_START_RADIUS_M +
        (NEATRACING_PULSE_RING_END_RADIUS_M -
          NEATRACING_PULSE_RING_START_RADIUS_M) *
          0.5,
      12,
    );
    expect(visuals.alpha).toBeCloseTo(
      NEATRACING_PULSE_GLOW_ALPHA *
        (1 -
          (0.5 - NEATRACING_PULSE_ENVELOPE_ATTACK_FRACTION) /
            (1 - NEATRACING_PULSE_ENVELOPE_ATTACK_FRACTION)),
      12,
    );
  });
});

describe('pulse depth testing (AC-006)', () => {
  it('renders the pulse when it is strictly nearer than the depth buffer', () => {
    expect(isPulseDepthVisible(0.5, 1)).toBe(true);
  });

  it('occludes the pulse when the buffer is at least as near', () => {
    expect(isPulseDepthVisible(0.5, 0.5)).toBe(false);
    expect(isPulseDepthVisible(0.9, 0.5)).toBe(false);
  });
});

describe('pulse pass WebGL2 orchestration', () => {
  it('draws one call per active pulse with the glow depth state', () => {
    const harness = createPulseMockWebGlHarness();
    const pass = createPulsePass(harness.gl);
    const pulses = collectActivePulses(
      1_000,
      NEATRACING_PULSE_VISUAL_SEED_DEFAULT,
    );

    pass.draw({ pulses, viewProjection: IDENTITY_VIEW_PROJECTION_FIXTURE });

    expect(harness.calls.drawArrays).toBe(pulses.length);
    expect(harness.calls.depthMask).toBe(1);
  });

  it('feeds the #B7FF00 pulse color as [0, 1]-normalized uniform channels', () => {
    const harness = createPulseMockWebGlHarness();
    const pass = createPulsePass(harness.gl);

    pass.draw({ pulses: [], viewProjection: IDENTITY_VIEW_PROJECTION_FIXTURE });

    expect(harness.uniform3fFeeds).toEqual([
      { name: 'uColor', channels: [185 / 255, 255 / 255, 0] },
    ]);
  });

  it('owns the per-frame glow state machine with blend enabled before the additive factors', () => {
    const harness = createPulseMockWebGlHarness();
    const pass = createPulsePass(harness.gl);

    pass.draw({ pulses: [], viewProjection: IDENTITY_VIEW_PROJECTION_FIXTURE });

    expect(harness.stateOps).toEqual([
      { op: 'enable', args: [MOCK_GL_DEPTH_TEST] },
      { op: 'depthMask', args: [false] },
      { op: 'disable', args: [MOCK_GL_CULL_FACE] },
      { op: 'enable', args: [MOCK_GL_BLEND] },
      { op: 'blendFunc', args: [MOCK_GL_ONE, MOCK_GL_ONE] },
    ]);
  });

  it('exposes the AC-040 warmup seam under the pulse registry key', () => {
    const harness = createPulseMockWebGlHarness();

    const warmup = toPulseWarmupProgram(harness.gl);

    expect(warmup.key).toBe(NEATRACING_WARMED_PROGRAM_KEYS[2]);
  });

  it('draws the warmed pipeline offscreen through the shared pass draw path', () => {
    const harness = createPulseMockWebGlHarness();
    const warmup = toPulseWarmupProgram(harness.gl);
    warmup.compileAndLink();
    const drawsBeforeWarmup = harness.calls.drawArrays;

    warmup.drawOffscreen(harness.gl);

    expect(harness.calls.drawArrays - drawsBeforeWarmup).toBe(1);
  });

  it('disposes the GPU program and vertex buffer', () => {
    const harness = createPulseMockWebGlHarness();
    const pass = createPulsePass(harness.gl);

    pass.dispose();

    expect(harness.calls.deleteProgram).toBe(1);
    expect(harness.calls.deleteBuffer).toBe(1);
  });
});

/** One recorded frame-state GL op from the pulse draw state machine. */
interface PulseMockGlStateOp {
  readonly op: 'enable' | 'disable' | 'depthMask' | 'blendFunc';
  readonly args: readonly (number | boolean)[];
}

/** One recorded uniform3f feed keyed by uniform name. */
interface PulseMockUniform3fFeed {
  readonly name: string;
  readonly channels: readonly [number, number, number];
}

/** Mutable call recorder mirroring the mock-GL harness shape. */
interface PulseMockGlHarness {
  readonly gl: WebGL2RenderingContext;
  readonly calls: Record<
    'drawArrays' | 'depthMask' | 'disable' | 'deleteProgram' | 'deleteBuffer',
    number
  >;
  /** Uniform3f feeds keyed by uniform name (mock-uniform color pin). */
  readonly uniform3fFeeds: ReadonlyArray<PulseMockUniform3fFeed>;
  /** Frame-state ops in call order (glow draw state machine pin). */
  readonly stateOps: ReadonlyArray<PulseMockGlStateOp>;
}

/**
 * Creates a recording mock WebGL2 context covering every GL entry point the
 * pulse pass touches, mirroring the context-service test harness pattern.
 */
function createPulseMockWebGlHarness(): PulseMockGlHarness {
  const calls: PulseMockGlHarness['calls'] = {
    drawArrays: 0,
    depthMask: 0,
    disable: 0,
    deleteProgram: 0,
    deleteBuffer: 0,
  };
  const uniform3fFeeds: PulseMockUniform3fFeed[] = [];
  const stateOps: PulseMockGlStateOp[] = [];
  const gl = {
    createShader: (): object => ({}),
    shaderSource: (): void => undefined,
    compileShader: (): void => undefined,
    getShaderParameter: (): boolean => true,
    createProgram: (): object => ({}),
    attachShader: (): void => undefined,
    linkProgram: (): void => undefined,
    getProgramParameter: (): boolean => true,
    getAttribLocation: (): number => 0,
    getUniformLocation: (_program: unknown, name: string): object => ({ name }),
    useProgram: (): void => undefined,
    enable: (target: number): void => {
      stateOps.push({ op: 'enable', args: [target] });
    },
    disable: (target: number): void => {
      calls.disable += 1;
      stateOps.push({ op: 'disable', args: [target] });
    },
    depthMask: (flag: boolean): void => {
      calls.depthMask += 1;
      stateOps.push({ op: 'depthMask', args: [flag] });
    },
    blendFunc: (sfactor: number, dfactor: number): void => {
      stateOps.push({ op: 'blendFunc', args: [sfactor, dfactor] });
    },
    createBuffer: (): object => ({}),
    bindBuffer: (): void => undefined,
    bufferData: (): void => undefined,
    vertexAttribPointer: (): void => undefined,
    enableVertexAttribArray: (): void => undefined,
    uniformMatrix4fv: (): void => undefined,
    uniform1f: (): void => undefined,
    uniform2f: (): void => undefined,
    uniform3f: (
      location: { name: string },
      red: number,
      green: number,
      blue: number,
    ): void => {
      uniform3fFeeds.push({
        name: location.name,
        channels: [red, green, blue],
      });
    },
    viewport: (): void => undefined,
    clearColor: (): void => undefined,
    clear: (): void => undefined,
    drawArrays: (): void => {
      calls.drawArrays += 1;
    },
    deleteProgram: (): void => {
      calls.deleteProgram += 1;
    },
    deleteShader: (): void => undefined,
    deleteBuffer: (): void => {
      calls.deleteBuffer += 1;
    },
  };
  return {
    gl: gl as unknown as WebGL2RenderingContext,
    calls,
    uniform3fFeeds,
    stateOps,
  };
}
