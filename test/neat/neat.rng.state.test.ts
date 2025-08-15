import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('RNG state snapshot/restore', () => {
  test('restoring RNG state reproduces future random subsequence', () => {
    const neat = new Neat(2, 1, (n: Network) => 0, { popsize: 4, seed: 42 });
    (neat as any).sampleRandom(10); // warm up consumes state
    const snap = neat.snapshotRNGState();
    const seqA = (neat as any).sampleRandom(5);
    neat.restoreRNGState(snap as any);
    const seqARepeat = (neat as any).sampleRandom(5);
    expect(seqARepeat).toEqual(seqA);
    // Export/import snapshot and replicate on a fresh instance (unseeded start)
    const exported = JSON.stringify(snap);
    const neat2 = new Neat(2, 1, (n: Network) => 0, { popsize: 4 });
    neat2.importRNGState(exported);
    const seqB = (neat2 as any).sampleRandom(5);
    neat.restoreRNGState(snap as any);
    const seqAThird = (neat as any).sampleRandom(5);
    expect(seqB).toEqual(seqAThird);
  });
});
