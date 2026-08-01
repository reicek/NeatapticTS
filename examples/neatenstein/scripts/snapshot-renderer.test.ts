import { describe, expect, it } from '@jest/globals';
import { renderVoxelSnapshot } from './snapshot-renderer';
import type { VoxelGrid } from './voxel-enemy';

function makeMinimalVoxelGrid(): VoxelGrid {
  return {
    width: 1,
    height: 192,
    depth: 1,
    voxels: [
      {
        x: 0,
        y: 96,
        z: 0,
        part: 'torso',
        material: 'accent',
        r: 0xdd,
        g: 0x22,
        b: 0x00,
        emissive: true,
        alpha: 1.0,
      },
    ],
    parts: ['torso'],
    palette: {
      accent: '#DD2200',
      neon: '#FBFFFF',
      suit: '#121418',
      dark: '#0A0B0E',
      damage: '#880808',
    },
    thickness: 3,
  };
}

describe('snapshot renderer (AC-602 red contracts)', () => {
  it('exports renderVoxelSnapshot', () => {
    expect(typeof renderVoxelSnapshot).toBe('function');
  });

  it('renders a non-empty pixel buffer', () => {
    const frame = renderVoxelSnapshot(makeMinimalVoxelGrid(), 0);
    expect(frame.width).toBeGreaterThan(0);
    expect(frame.height).toBeGreaterThan(0);
    expect(frame.data.length).toBeGreaterThan(0);
  });

  it('supports all 8 yaw angles', () => {
    const grid = makeMinimalVoxelGrid();
    for (let yaw = 0; yaw < 8; yaw++) {
      const frame = renderVoxelSnapshot(grid, yaw);
      expect(frame.data.length).toBeGreaterThan(0);
    }
  });

  it('renders deterministically for the same inputs', () => {
    const grid = makeMinimalVoxelGrid();
    const first = renderVoxelSnapshot(grid, 3);
    const second = renderVoxelSnapshot(grid, 3);
    expect(second.data).toEqual(first.data);
  });

  it('rejects an out-of-range yaw index', () => {
    expect(() => renderVoxelSnapshot(makeMinimalVoxelGrid(), 8)).toThrow(/yaw/);
  });

  it('ignores projected pixels that fall outside the buffer bounds', () => {
    const grid: VoxelGrid = {
      width: 1,
      height: 192,
      depth: 1,
      voxels: [
        {
          x: 0,
          y: 380,
          z: 0,
          part: 'torso',
          material: 'accent',
          r: 0xdd,
          g: 0x22,
          b: 0x00,
          emissive: true,
          alpha: 1.0,
        },
      ],
      parts: ['torso'],
      palette: {
        accent: '#DD2200',
        neon: '#FBFFFF',
        suit: '#121418',
        dark: '#0A0B0E',
        damage: '#880808',
      },
      thickness: 3,
    };
    const snapshot = renderVoxelSnapshot(grid, 0, { width: 16, height: 16 });
    expect(snapshot.width).toBe(16);
    expect(snapshot.height).toBe(16);
    // The voxel projects far above the top edge, so nothing is drawn.
    expect(snapshot.data.every((v) => v === 0)).toBe(true);
  });
});
