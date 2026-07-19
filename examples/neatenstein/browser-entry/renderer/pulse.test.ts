import { describe, expect, it } from '@jest/globals';

const loadModule = (path: string): Promise<any> => import(path);

function createPulseFixture(seed: number) {
  return {
    worldBearingRad: 0,
    seed,
    active: true,
    lifetimeTicks: 100,
    distance: 5,
    screenX: 80,
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

  it('enforces the eight-concurrent pulse ceiling', async () => {
    const { updateNeatensteinPulses } = await loadModule('./pulse.ts');
    const pulses: ReturnType<typeof createPulseFixture>[] = [];
    for (let i = 0; i < 10; i++) {
      pulses.push(createPulseFixture(i));
    }
    const next = updateNeatensteinPulses(pulses, 0);
    expect(next.length).toBeLessThanOrEqual(8);
  });

  it('fades a vertical pulse when ray bearing drifts beyond tolerance', async () => {
    const { updateNeatensteinPulses } = await loadModule('./pulse.ts');
    const pulse = createPulseFixture(12345);
    pulse.worldBearingRad = 0;
    const before = pulse.lifetimeTicks;
    const next = updateNeatensteinPulses([pulse], 0, Math.PI / 2);
    expect(next[0].lifetimeTicks).toBeLessThan(before);
  });

  it('hides a pulse behind a closer wall in the z-buffer', async () => {
    const { depthTestPulse } = await loadModule('./pulse.ts');
    const pulse = { screenColumnStart: 10, screenColumnEnd: 15, distance: 5 };
    const zBuffer = new Float32Array(160);
    for (let x = 10; x <= 15; x++) zBuffer[x] = 2;
    expect(depthTestPulse(pulse, zBuffer)).toBe(false);
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
