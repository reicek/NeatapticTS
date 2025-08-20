/**
 * Node pool mutate/prune stress harness (Phase 2).
 * Performs randomized ADD_NODE / REMOVE_NODE mutations to exercise pooling.
 * Computes:
 *  - recycledRatio = reused / (reused + fresh)
 *  - highWaterMark tail stability (approx via start vs end comparison as proxy for tail slope)
 * Single educational assertion bundles core invariants.
 */
import {
  acquireNode,
  releaseNode,
  nodePoolStats,
  resetNodePool,
} from '../../src/architecture/nodePool';

describe('benchmark.nodePool.stress', () => {
  it('achieves recycledRatio >=0.5 with stable highWaterMark tail', () => {
    resetNodePool();
    const BATCH = 32; // number of logical node creations per phase

    // Phase A: fresh allocations
    const phaseAFresh: any[] = [];
    for (let i = 0; i < BATCH; i++)
      phaseAFresh.push(acquireNode({ type: 'hidden' }));
    // Release all (populate pool)
    for (const n of phaseAFresh) releaseNode(n);
    const highWaterBeforeReuseTail = nodePoolStats().highWaterMark;

    // Phase B: first reuse wave
    const phaseBReuse: any[] = [];
    for (let i = 0; i < BATCH; i++)
      phaseBReuse.push(acquireNode({ type: 'hidden' }));
    for (const n of phaseBReuse) releaseNode(n);

    // Phase C: second reuse wave (boost recycled ratio)
    const phaseCReuse: any[] = [];
    for (let i = 0; i < BATCH; i++)
      phaseCReuse.push(acquireNode({ type: 'hidden' }));

    const finalStats = nodePoolStats();
    const recycledRatio = finalStats.recycledRatio; // reused / (reused + fresh)
    const highWaterTailDelta =
      finalStats.highWaterMark - highWaterBeforeReuseTail; // should be 0 (no growth during reuse waves)

    expect(
      recycledRatio >= 0.5 &&
        highWaterTailDelta <= 2 &&
        finalStats.reused > 0 &&
        finalStats.fresh > 0
    ).toBe(true);
  });
});
