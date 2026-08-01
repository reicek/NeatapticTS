import { describe, expect, it } from '@jest/globals';
import { buildVoxelEnemy, resolveColor } from './voxel-enemy';

describe('voxel enemy descriptor (AC-601 red contracts)', () => {
  it('exports buildVoxelEnemy', () => {
    expect(typeof buildVoxelEnemy).toBe('function');
  });

  it('models a 192-voxel height grid', () => {
    const grid = buildVoxelEnemy();
    expect(grid.height).toBe(192);
  });

  it('chooses width and depth from the approved silhouettes with both ≤192', () => {
    const grid = buildVoxelEnemy();
    expect(grid.width).toBeGreaterThan(0);
    expect(grid.width).toBeLessThanOrEqual(192);
    expect(grid.depth).toBeLessThanOrEqual(192);
  });

  it('includes the required body parts', () => {
    const grid = buildVoxelEnemy();
    expect(grid.parts).toEqual(
      expect.arrayContaining([
        'head',
        'torso',
        'arms',
        'legs',
        'cannon',
        'back disk',
      ]),
    );
  });

  it('defaults the accent color to Ares Red', () => {
    const grid = buildVoxelEnemy();
    expect(grid.palette.accent).toBe('#DD2200');
  });

  it('exposes a swappable accent palette', () => {
    const grid = buildVoxelEnemy('#00FFFF');
    expect(grid.palette).toMatchObject({
      accent: '#00FFFF',
      neon: expect.any(String),
      suit: expect.any(String),
      damage: expect.any(String),
    });
  });

  it('uses 2–4 voxel thickness for edges, neon strips, and the back disk', () => {
    const grid = buildVoxelEnemy();
    expect(grid.thickness).toBeGreaterThanOrEqual(2);
    expect(grid.thickness).toBeLessThanOrEqual(4);
  });

  it('resolves every material slot to the expected palette color', () => {
    const palette = buildVoxelEnemy('#00FF00').palette;
    expect(resolveColor('accent', palette)).toEqual({ r: 0, g: 255, b: 0 });
    expect(resolveColor('neon', palette)).toEqual({
      r: 0xfb,
      g: 0xff,
      b: 0xff,
    });
    expect(resolveColor('suit', palette)).toEqual({
      r: 0x12,
      g: 0x14,
      b: 0x18,
    });
    expect(resolveColor('dark', palette)).toEqual({
      r: 0x0a,
      g: 0x0b,
      b: 0x0e,
    });
    expect(resolveColor('damage', palette)).toEqual({
      r: 0xff,
      g: 0x33,
      b: 0x00,
    });
  });

  it('resolves colors with or without a leading hash', () => {
    const palette = buildVoxelEnemy().palette;
    expect(resolveColor('accent', { ...palette, accent: 'DD2200' })).toEqual({
      r: 0xdd,
      g: 0x22,
      b: 0x00,
    });
  });
});
