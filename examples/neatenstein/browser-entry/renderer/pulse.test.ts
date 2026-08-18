import { describe, expect, it } from '@jest/globals';
import { NEATENSTEIN_PULSE_MAX_CONCURRENT } from '../constants';
import { PARK_MILLER_MODULUS } from './renderer.rng.constants';
import type { NeatensteinPulse } from './pulse';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const loadModule = (path: string): Promise<any> => import(path);

function createPulseFixture(seed: number): NeatensteinPulse {
  return {
    worldX: 5,
    worldY: 7,
    seed,
    active: true,
    lifetimeTicks: 100,
    axis: 'x',
    travelDirection: 1,
    travelSpeed: 0.05,
    screenColumn: 80,
    distance: 5,
    layer: 'floor',
  };
}

function createCeilingPulseFixture(
  seed: number,
): NeatensteinPulse & { layer: 'ceiling' } {
  return {
    worldX: 5,
    worldY: 7,
    seed,
    active: true,
    lifetimeTicks: 100,
    axis: 'x',
    travelDirection: 1,
    travelSpeed: 0.05,
    screenColumn: 80,
    distance: 5,
    layer: 'ceiling',
  };
}

describe('Neatenstein floor pulse system', () => {
  it('emits ambient pulses only on qualifying sim ticks', async () => {
    const { emitNeatensteinAmbientPulse } = await loadModule('./pulse.ts');
    const emitted = emitNeatensteinAmbientPulse(0, 12345);
    const notEmitted = emitNeatensteinAmbientPulse(1, 12345);
    expect({ emitted: !!emitted, notEmitted: !!notEmitted }).toEqual({
      emitted: true,
      notEmitted: false,
    });
  });

  it('produces deterministic pulses for the same seed and sim tick', async () => {
    const { emitNeatensteinAmbientPulse } = await loadModule('./pulse.ts');
    const first = emitNeatensteinAmbientPulse(0, 12345);
    const second = emitNeatensteinAmbientPulse(0, 12345);
    expect(second).toEqual(first);
  });

  it('spawns an ambient pulse at tick 0', async () => {
    const { emitNeatensteinAmbientPulse } = await loadModule('./pulse.ts');
    expect(emitNeatensteinAmbientPulse(0, 12345)).not.toBeNull();
  });

  it('spawns ambient pulses on integer grid lines', async () => {
    const { emitNeatensteinAmbientPulse } = await loadModule('./pulse.ts');
    const pulse = emitNeatensteinAmbientPulse(0, 12345);
    const fixedWorldValue =
      pulse === null ? null : pulse.axis === 'x' ? pulse.worldX : pulse.worldY;
    expect(
      fixedWorldValue === null
        ? NaN
        : Math.abs(fixedWorldValue - Math.round(fixedWorldValue)),
    ).toBe(0);
  });

  it('advances an x-axis ambient pulse along Y', async () => {
    const { updateNeatensteinPulses } = await loadModule('./pulse.ts');
    const pulse = createPulseFixture(12345);
    const startX = pulse.worldX;
    const startY = pulse.worldY;
    const next = updateNeatensteinPulses([pulse], 0);
    expect({
      length: next.length,
      worldY: next[0]?.worldY,
      worldX: next[0]?.worldX,
    }).toEqual({
      length: 1,
      worldY: startY + next[0]!.travelDirection * next[0]!.travelSpeed,
      worldX: startX,
    });
  });

  it('advances a y-axis ambient pulse along X', async () => {
    const { updateNeatensteinPulses } = await loadModule('./pulse.ts');
    const pulse = createPulseFixture(12345);
    pulse.axis = 'y';
    const startX = pulse.worldX;
    const startY = pulse.worldY;
    const next = updateNeatensteinPulses([pulse], 0);
    expect({
      length: next.length,
      worldX: next[0]?.worldX,
      worldY: next[0]?.worldY,
    }).toEqual({
      length: 1,
      worldX: startX + next[0]!.travelDirection * next[0]!.travelSpeed,
      worldY: startY,
    });
  });

  it('enforces the concurrent pulse ceiling', async () => {
    const { updateNeatensteinPulses } = await loadModule('./pulse.ts');
    const pulses: ReturnType<typeof createPulseFixture>[] = [];
    for (let i = 0; i < NEATENSTEIN_PULSE_MAX_CONCURRENT + 2; i++) {
      pulses.push(createPulseFixture(i));
    }
    const next = updateNeatensteinPulses(pulses, 0);
    expect(next.length).toBeLessThanOrEqual(NEATENSTEIN_PULSE_MAX_CONCURRENT);
  });

  it('fades a pulse only by lifetime, not by bearing tolerance', async () => {
    const { updateNeatensteinPulses } = await loadModule('./pulse.ts');
    const pulse = createPulseFixture(12345);
    const before = pulse.lifetimeTicks;
    const next = updateNeatensteinPulses([pulse], 0);
    expect(next[0].lifetimeTicks).toBe(before - 1);
  });

  it('hides a pulse behind a closer wall in the z-buffer', async () => {
    const { depthTestPulse } = await loadModule('./pulse.ts');
    const pulse = { screenColumn: 12, distance: 5 };
    const zBuffer = new Float32Array(320);
    zBuffer[12] = 2;
    expect(depthTestPulse(pulse, zBuffer)).toBe(false);
  });

  it('occludes a pulse at exactly the wall distance so the wall wins ties', async () => {
    // Arrange: pulse distance equals the stored wall distance exactly.
    const { depthTestPulse } = await loadModule('./pulse.ts');
    const pulse = { screenColumn: 0, distance: 5 };
    const zBuffer = new Float32Array([5]);

    // Act
    const visible = depthTestPulse(pulse, zBuffer);

    // Assert — fails today: depthTestPulse uses <= (inclusive), so a pulse
    // at exactly the wall distance is visible. The fix standardizes on
    // strict < so the wall wins ties, matching clipNeatensteinSpriteSpan.
    expect(visible).toBe(false);
  });

  it('schedules generation-up audio and pulse on the same sim tick', async () => {
    const {
      emitNeatensteinGenerationUpPulse,
      scheduleNeatensteinGenerationUpSound,
    } = await loadModule('./pulse.ts');
    const simTick = 42;
    const pulseScheduled = emitNeatensteinGenerationUpPulse(simTick);
    const soundScheduled = scheduleNeatensteinGenerationUpSound(simTick);
    expect(pulseScheduled && soundScheduled).toBe(true);
  });
});

describe('Neatenstein ceiling pulse layer', () => {
  it('exports floor and ceiling pulse layer constants', async () => {
    const { NEATENSTEIN_PULSE_LAYER_FLOOR, NEATENSTEIN_PULSE_LAYER_CEILING } =
      await loadModule('./pulse.ts');
    expect({
      floor: NEATENSTEIN_PULSE_LAYER_FLOOR,
      ceiling: NEATENSTEIN_PULSE_LAYER_CEILING,
    }).toEqual({
      floor: 'floor',
      ceiling: 'ceiling',
    });
  });

  it('tags ambient pulses as floor by default', async () => {
    const { emitNeatensteinAmbientPulse } = await loadModule('./pulse.ts');
    const pulse = emitNeatensteinAmbientPulse(0, 42) as unknown as {
      layer: string;
    };
    expect(pulse.layer).toBe('floor');
  });

  it('emits a ceiling ambient pulse when asked for the ceiling layer', async () => {
    const { emitNeatensteinAmbientPulse, NEATENSTEIN_PULSE_LAYER_CEILING } =
      await loadModule('./pulse.ts');
    const pulse = emitNeatensteinAmbientPulse(
      0,
      42,
      NEATENSTEIN_PULSE_LAYER_CEILING,
    ) as unknown as { layer: string };
    expect(pulse.layer).toBe('ceiling');
  });

  it('moves a ceiling pulse upward along its x-axis grid line', async () => {
    const { updateNeatensteinPulses } = await loadModule('./pulse.ts');
    const pulse = createCeilingPulseFixture(42);
    const beforeY = pulse.worldY;
    const next = updateNeatensteinPulses([pulse], 0) as unknown as Array<{
      worldY: number;
      layer: string;
    }>;
    expect(next[0]?.worldY).toBeLessThan(beforeY);
  });
});

describe('Neatenstein ambient pulse branch coverage', () => {
  it('returns null for a negative sim tick', async () => {
    const { emitNeatensteinAmbientPulse } = await loadModule('./pulse.ts');
    expect(emitNeatensteinAmbientPulse(-1, 42)).toBeNull();
  });

  it('selects the x axis when the LCG float is below the threshold', async () => {
    const { emitNeatensteinAmbientPulse } = await loadModule('./pulse.ts');
    const pulse = emitNeatensteinAmbientPulse(0, 2);
    expect(pulse).not.toBeNull();
    expect(pulse!.axis).toBe('x');
  });

  it('selects the y axis when the LCG float is at or above the threshold', async () => {
    const { emitNeatensteinAmbientPulse } = await loadModule('./pulse.ts');
    const pulse = emitNeatensteinAmbientPulse(0, 4);
    expect(pulse).not.toBeNull();
    expect(pulse!.axis).toBe('y');
  });

  it('sets travel direction to +1 when the LCG float is below the threshold', async () => {
    const { emitNeatensteinAmbientPulse } = await loadModule('./pulse.ts');
    const pulse = emitNeatensteinAmbientPulse(0, 2);
    expect(pulse).not.toBeNull();
    expect(pulse!.travelDirection).toBe(1);
  });

  it('sets travel direction to -1 when the LCG float is at or above the threshold', async () => {
    const { emitNeatensteinAmbientPulse } = await loadModule('./pulse.ts');
    const pulse = emitNeatensteinAmbientPulse(0, 1);
    expect(pulse).not.toBeNull();
    expect(pulse!.travelDirection).toBe(-1);
  });
});

describe('LCG seed normalization', () => {
  it('produces identical computed values for a negative seed and its positive modular equivalent', async () => {
    const { emitNeatensteinAmbientPulse } = await loadModule('./pulse.ts');
    const negativeSeedPulse = emitNeatensteinAmbientPulse(0, -500);
    const positiveSeedPulse = emitNeatensteinAmbientPulse(
      0,
      PARK_MILLER_MODULUS - 500,
    );
    expect(negativeSeedPulse).not.toBeNull();
    expect(positiveSeedPulse).not.toBeNull();
    if (negativeSeedPulse === null || positiveSeedPulse === null) return;
    expect({
      worldX: negativeSeedPulse.worldX,
      worldY: negativeSeedPulse.worldY,
      axis: negativeSeedPulse.axis,
      travelDirection: negativeSeedPulse.travelDirection,
      travelSpeed: negativeSeedPulse.travelSpeed,
    }).toEqual({
      worldX: positiveSeedPulse.worldX,
      worldY: positiveSeedPulse.worldY,
      axis: positiveSeedPulse.axis,
      travelDirection: positiveSeedPulse.travelDirection,
      travelSpeed: positiveSeedPulse.travelSpeed,
    });
  });

  it('produces identical computed values for a large negative seed and its positive modular equivalent', async () => {
    const { emitNeatensteinAmbientPulse } = await loadModule('./pulse.ts');
    const negativeSeedPulse = emitNeatensteinAmbientPulse(0, -2_147_483_000);
    const positiveSeedPulse = emitNeatensteinAmbientPulse(0, 647);
    expect(negativeSeedPulse).not.toBeNull();
    expect(positiveSeedPulse).not.toBeNull();
    if (negativeSeedPulse === null || positiveSeedPulse === null) return;
    expect({
      worldX: negativeSeedPulse.worldX,
      worldY: negativeSeedPulse.worldY,
      axis: negativeSeedPulse.axis,
      travelDirection: negativeSeedPulse.travelDirection,
      travelSpeed: negativeSeedPulse.travelSpeed,
    }).toEqual({
      worldX: positiveSeedPulse.worldX,
      worldY: positiveSeedPulse.worldY,
      axis: positiveSeedPulse.axis,
      travelDirection: positiveSeedPulse.travelDirection,
      travelSpeed: positiveSeedPulse.travelSpeed,
    });
  });
});

describe('pooled pulse updates', () => {
  it('returns the same object reference for a surviving pulse', async () => {
    const { updateNeatensteinPulses } = await loadModule('./pulse.ts');
    const pulse = createPulseFixture(12345);
    const next = updateNeatensteinPulses([pulse], 0);
    // Fails today: .map() creates a new object, so next[0] !== pulse
    expect(next[0]).toBe(pulse);
  });

  it('decrements lifetimeTicks on the original pulse object in place', async () => {
    const { updateNeatensteinPulses } = await loadModule('./pulse.ts');
    const pulse = createPulseFixture(12345);
    const originalLifetime = pulse.lifetimeTicks;
    updateNeatensteinPulses([pulse], 0);
    // Fails today: .map() creates a copy, original lifetimeTicks is unchanged
    expect(pulse.lifetimeTicks).toBe(originalLifetime - 1);
  });

  it('marks an expired pulse as inactive on the original object', async () => {
    const { updateNeatensteinPulses } = await loadModule('./pulse.ts');
    const pulse = createPulseFixture(12345);
    pulse.lifetimeTicks = 1;
    updateNeatensteinPulses([pulse], 0);
    // Fails today: .map() creates a copy with active:false, original stays active:true
    expect(pulse.active).toBe(false);
  });
});
