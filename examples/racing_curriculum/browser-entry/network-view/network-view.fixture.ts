import { Network } from '../../../../src/browser-entry.ts';

/**
 * Deterministic dense feed-forward network for level-of-detail red tests.
 *
 * The fixture intentionally starts with 4,000 hidden nodes so the LOD renderer
 * has a strong reason to abstract the hidden graph while keeping every input
 * and output node visible.
 */
export function buildDenseRacingNetwork(): Network {
  return new Network(70, 2, {
    minHidden: 4000,
    topologyIntent: 'feed-forward',
    seed: 42,
  });
}
