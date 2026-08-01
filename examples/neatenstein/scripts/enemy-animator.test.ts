import { describe, expect, it } from '@jest/globals';
import { getEnemyAnimationFrame } from './enemy-animator';

describe('enemy animator (AC-603 red contracts)', () => {
  it('exports getEnemyAnimationFrame', () => {
    expect(typeof getEnemyAnimationFrame).toBe('function');
  });

  it('returns the approved idle frame count and a valid index', () => {
    const frame = getEnemyAnimationFrame('idle', 0, 42);
    expect(frame.frameCount).toBe(6);
    expect(frame.frameIndex).toBeGreaterThanOrEqual(0);
    expect(frame.frameIndex).toBeLessThan(6);
  });

  it('returns the approved move frame count and a valid index', () => {
    const frame = getEnemyAnimationFrame('move', 100, 42);
    expect(frame.frameCount).toBe(12);
    expect(frame.frameIndex).toBeGreaterThanOrEqual(0);
    expect(frame.frameIndex).toBeLessThan(12);
  });

  it('returns the approved fire frame count and a valid index', () => {
    const frame = getEnemyAnimationFrame('fire', 50, 42);
    expect(frame.frameCount).toBe(3);
    expect(frame.frameIndex).toBeGreaterThanOrEqual(0);
    expect(frame.frameIndex).toBeLessThan(3);
  });

  it('returns the approved death frame count and a valid index', () => {
    const frame = getEnemyAnimationFrame('death', 250, 42);
    expect(frame.frameCount).toBe(12);
    expect(frame.frameIndex).toBeGreaterThanOrEqual(0);
    expect(frame.frameIndex).toBeLessThan(12);
  });

  it('is deterministic for the same state, elapsed time, and seed', () => {
    const a = getEnemyAnimationFrame('move', 333, 7);
    const b = getEnemyAnimationFrame('move', 333, 7);
    expect(a.frameIndex).toBe(b.frameIndex);
  });

  it('throws for an unknown animation state', () => {
    expect(() =>
      getEnemyAnimationFrame('unknown' as unknown as 'idle', 0, 0),
    ).toThrow('Unknown enemy animation state: unknown');
  });

  it('works when the optional seed is omitted', () => {
    const frame = getEnemyAnimationFrame('idle', 0);
    expect(frame.frameCount).toBe(6);
    expect(frame.frameIndex).toBe(0);
  });

  it('clamps negative elapsed time to zero', () => {
    const negative = getEnemyAnimationFrame('move', -1000, 42);
    const zero = getEnemyAnimationFrame('move', 0, 42);
    expect(negative.frameIndex).toBe(zero.frameIndex);
  });
});
