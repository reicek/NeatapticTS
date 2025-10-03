import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('RNG state snapshot/restore', () => {
  test('restoring RNG state reproduces future random subsequence', () => {
    const neat = new Neat(2, 1, (network: Network) => {
      void network; // Fitness stub keeps interface satisfied without side effects
      return 0;
    }, { popsize: 4, seed: 42 });
    // Step 1: consume a handful of samples to move RNG forward.
    neat.sampleRandom(10);
    // Step 2: capture the RNG state for later restoration.
    const snapshotState = neat.snapshotRNGState();
    const initialSequence = neat.sampleRandom(5);
    neat.restoreRNGState(snapshotState);
    const repeatedSequence = neat.sampleRandom(5);
    expect(repeatedSequence).toEqual(initialSequence);
    // Step 3: import the state on a fresh instance and verify parity.
    const exported = JSON.stringify(snapshotState);
    const neatReplica = new Neat(2, 1, (network: Network) => {
      void network;
      return 0;
    }, { popsize: 4 });
    neatReplica.importRNGState(exported);
    const replicaSequence = neatReplica.sampleRandom(5);
    neat.restoreRNGState(snapshotState);
    const restoredSequence = neat.sampleRandom(5);
    expect(replicaSequence).toEqual(restoredSequence);
  });
});
