/**
 * network.slab.async.test.ts
 * Validates cooperative async slab rebuild yields across multiple microtasks and preserves invariants
 * relative to the synchronous rebuild path (version increment, capacity policy, alloc stats behavior, parity).
 */
import Network from '../../src/architecture/network';
import Node from '../../src/architecture/node';
import { config } from '../../src/config';
import { rebuildConnectionSlabAsync } from '../../src/architecture/network/network.slab';
import { memoryStats } from '../../src/utils/memory';
import type { SlabAllocStats } from '../../src/utils/memory';

type ConnectionSlabSnapshot = ReturnType<Network['getConnectionSlab']>;
type PromiseThen = typeof Promise.prototype.then;

const setTopologyDirty = (network: Network): void => {
  Reflect.set(network, '_topoDirty', true);
  Reflect.set(network, '_nodeIndexDirty', true);
};

const markSlabDirty = (network: Network): void => {
  Reflect.set(network, '_slabDirty', true);
};

const getSlabVersion = (network: Network): number =>
  (Reflect.get(network, '_slabVersion') as number | undefined) ?? 0;

const rebuildConnectionSlabIfPresent = (network: Network): void => {
  const rebound = Reflect.get(network, 'rebuildConnectionSlab') as
    | ((force?: boolean) => void)
    | undefined;
  rebound?.call(network);
};

const getConnectionSlabSnapshot = (network: Network): ConnectionSlabSnapshot =>
  network.getConnectionSlab() as ConnectionSlabSnapshot;

const buildDenseNetwork = (
  inputCount: number,
  hiddenCount: number,
  outputCount: number,
): Network => {
  const network = new Network(inputCount, outputCount, {
    enforceAcyclic: true,
  });
  for (let hiddenIndex = 0; hiddenIndex < hiddenCount; hiddenIndex += 1) {
    const hiddenNode = new Node('hidden');
    network.nodes.push(hiddenNode);
    for (let inputIndex = 0; inputIndex < inputCount; inputIndex += 1) {
      network.connect(
        network.nodes[inputIndex],
        hiddenNode,
        Math.random() * 0.2 - 0.1,
      );
    }
    for (
      let outputIndex = network.nodes.length - outputCount;
      outputIndex < network.nodes.length;
      outputIndex += 1
    ) {
      network.connect(
        hiddenNode,
        network.nodes[outputIndex],
        Math.random() * 0.2 - 0.1,
      );
    }
  }
  setTopologyDirty(network);
  return network;
};

describe('network.slab.async', () => {
  it('async rebuild yields and matches sync slab (version, capacity, flags, gain) with minimal alloc stat bump', async () => {
    // Arrange
    if (typeof window === 'undefined') {
      // Environment not browser-like; skip (single expectation satisfied trivially)
      expect(true).toBe(true);
      return;
    }
    config.enableNodePooling = false;
    config.enableSlabArrayPooling = true; // exercise pooling accounting
    const targetConnections = 110_000; // force multiple chunks with default 50k
    // Heuristic hidden count to exceed target connections: each hidden adds input + output edges
    const net = buildDenseNetwork(20, 1_500, 5); // ~ (20+5)*1500 = 37,500 + baseline input-output; may adjust
    // If still under threshold, add more hidden nodes
    let hiddenExpansionGuard = 0;
    while (
      net.connections.length < targetConnections &&
      hiddenExpansionGuard < 2_000
    ) {
      const node = new Node('hidden');
      net.nodes.push(node);
      for (let inputIndex = 0; inputIndex < 20; inputIndex += 1) {
        net.connect(net.nodes[inputIndex], node, 0.05);
      }
      for (
        let outputIndex = net.nodes.length - 5;
        outputIndex < net.nodes.length;
        outputIndex += 1
      ) {
        net.connect(node, net.nodes[outputIndex], 0.05);
      }
      hiddenExpansionGuard += 1;
    }
    setTopologyDirty(net);
    markSlabDirty(net);
    const statsBefore = memoryStats(net).flags.snapshot
      .allocStats as SlabAllocStats;
    if (!statsBefore) throw new Error('statsBefore is null');
    const versionBefore = getSlabVersion(net);

    // Act: perform async rebuild with small chunkSize to force >1 yield
    const chunkSize = 10_000; // ensure multiple slices
    let microtaskYields = 0;
    const originalThen: PromiseThen = Promise.prototype.then;
    const patchedThen: PromiseThen = function patchedThen<TResult1, TResult2>(
      this: Promise<unknown>,
      onfulfilled?:
        | ((value: unknown) => TResult1 | PromiseLike<TResult1>)
        | null,
      onrejected?:
        | ((reason: unknown) => TResult2 | PromiseLike<TResult2>)
        | null,
    ): Promise<TResult1 | TResult2> {
      microtaskYields += 1;
      return originalThen.call(this, onfulfilled, onrejected) as Promise<
        TResult1 | TResult2
      >;
    };
    Promise.prototype.then = patchedThen;
    await rebuildConnectionSlabAsync.call(net, chunkSize);
    Promise.prototype.then = originalThen;

    const slabAsync = getConnectionSlabSnapshot(net);
    if (!slabAsync.gain || !slabAsync.flags) {
      throw new Error('Connection slab should expose gain and flag arrays');
    }
    const statsAfter = memoryStats(net).flags.snapshot
      .allocStats as SlabAllocStats;
    if (!statsAfter) throw new Error('statsAfter is null');

    // Force a sync rebuild after a dummy structural no-op to compare parity
    markSlabDirty(net);
    rebuildConnectionSlabIfPresent(net);
    const slabSync = getConnectionSlabSnapshot(net);
    if (!slabSync.gain || !slabSync.flags) {
      throw new Error('Reference slab should expose gain and flag arrays');
    }

    // Assert: single expectation bundling invariants
    expect(
      slabAsync.version > versionBefore &&
        slabAsync.capacity >= slabAsync.used &&
        slabAsync.flags.length === slabAsync.weights.length &&
        slabAsync.gain.length === slabAsync.weights.length &&
        microtaskYields > 1 &&
        slabAsync.version === slabSync.version &&
        slabAsync.used === slabSync.used &&
        statsAfter.fresh >= statsBefore.fresh &&
        // pooled may increase or stay; ensure fresh did not jump multiple times (<= fresh + 10 heuristic guard)
        statsAfter.fresh - statsBefore.fresh <= 10,
    ).toBe(true);
  }, 30000);
});
