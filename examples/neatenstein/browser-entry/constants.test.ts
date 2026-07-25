import * as fs from 'node:fs';
import { describe, expect, it } from '@jest/globals';

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
const loadModule = (path: string): Promise<any> => import(path);

describe('Neatenstein constants module', () => {
  it('has a README at examples/neatenstein/README.md', () => {
    expect(fs.existsSync('examples/neatenstein/README.md')).toBe(true);
  });

  it('exports the versioned frame format string', async () => {
    const constants = await loadModule('./constants.ts');
    expect(constants.NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION).toBe(
      'neatenstein-frame-v1',
    );
  });

  it('exports tier-aware column counts', async () => {
    const constants = await loadModule('./constants.ts');
    expect({
      gpu: constants.NEATENSTEIN_GPU_COLUMN_COUNT,
      worker: constants.NEATENSTEIN_WORKER_COLUMN_COUNT,
      cpu: constants.NEATENSTEIN_CPU_COLUMN_COUNT,
    }).toEqual({
      gpu: 640,
      worker: 480,
      cpu: 320,
    });
  });

  it('exports pulse timing and concurrency constants', async () => {
    const constants = await loadModule('./constants.ts');
    expect({
      ambientIntervalMs: constants.NEATENSTEIN_PULSE_AMBIENT_INTERVAL_MS,
      ambientLifetimeMs: constants.NEATENSTEIN_PULSE_AMBIENT_LIFETIME_MS,
      maxConcurrent: constants.NEATENSTEIN_PULSE_MAX_CONCURRENT,
    }).toEqual({
      ambientIntervalMs: 2000,
      ambientLifetimeMs: 2700,
      maxConcurrent: 11,
    });
  });

  it('exports the complete ordered sound-name list', async () => {
    const constants = await loadModule('./constants.ts');
    expect(constants.NEATENSTEIN_AUDIO_SOUND_NAMES).toEqual([
      'fire',
      'enemy-hit',
      'player-damage',
      'dash',
      'kill',
      'generation-up',
    ]);
  });

  it('exports the fixed 60x60 map size', async () => {
    const constants = await loadModule('./constants.ts');
    expect(constants.NEATENSTEIN_MAP_SIZE).toBe(120);
  });
});
