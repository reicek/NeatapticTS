import Network from '../network';
import Node from '../../node';
import { config } from '../../../config';
import { rebuildConnectionSlabAsync } from '../network.utils';
import { memoryStats } from '../../../utils/memory';
import type { SlabAllocStats } from '../../../utils/memory';

type ConnectionSlabSnapshot = ReturnType<Network['getConnectionSlab']>;
type PromiseThen = typeof Promise.prototype.then;

function buildDeterministicWeight(
  fromIndex: number,
  toIndex: number,
  offset = 0,
): number {
  return (((fromIndex + toIndex + offset) % 9) - 4) / 20;
}

function buildDenseNetwork(
  inputCount: number,
  hiddenCount: number,
  outputCount: number,
): Network {
  const network = new Network(inputCount, outputCount, {
    enforceAcyclic: true,
  });

  for (let hiddenIndex = 0; hiddenIndex < hiddenCount; hiddenIndex++) {
    const hiddenNode = new Node('hidden');
    network.nodes.push(hiddenNode);

    for (let inputIndex = 0; inputIndex < inputCount; inputIndex++) {
      network.connect(
        network.nodes[inputIndex],
        hiddenNode,
        buildDeterministicWeight(inputIndex, hiddenIndex),
      );
    }

    for (
      let outputIndex = network.nodes.length - outputCount;
      outputIndex < network.nodes.length;
      outputIndex++
    ) {
      network.connect(
        hiddenNode,
        network.nodes[outputIndex],
        buildDeterministicWeight(hiddenIndex, outputIndex, 3),
      );
    }
  }

  setTopologyDirty(network);
  return network;
}

function getConnectionSlabSnapshot(network: Network): ConnectionSlabSnapshot {
  return network.getConnectionSlab() as ConnectionSlabSnapshot;
}

function getSlabVersion(network: Network): number {
  return (Reflect.get(network, '_slabVersion') as number | undefined) ?? 0;
}

function markSlabDirty(network: Network): void {
  Reflect.set(network, '_slabDirty', true);
}

function rebuildConnectionSlabIfPresent(network: Network): void {
  const rebuildConnectionSlab = Reflect.get(
    network,
    'rebuildConnectionSlab',
  ) as ((force?: boolean) => void) | undefined;

  rebuildConnectionSlab?.call(network);
}

function setTopologyDirty(network: Network): void {
  Reflect.set(network, '_topoDirty', true);
  Reflect.set(network, '_nodeIndexDirty', true);
}

describe('network slab chapter', () => {
  describe('async rebuild', () => {
    describe('given a large browser-like network rebuild is chunked across microtasks', () => {
      describe('when async and sync slab snapshots are compared', () => {
        it('yields while preserving slab invariants and sync parity', async () => {
          // Arrange
          if (typeof window === 'undefined') {
            expect(true).toBe(true);
            return;
          }

          config.enableNodePooling = false;
          config.enableSlabArrayPooling = true;
          const targetConnections = 110_000;
          const network = buildDenseNetwork(20, 1_500, 5);

          let hiddenExpansionGuard = 0;
          while (
            network.connections.length < targetConnections &&
            hiddenExpansionGuard < 2_000
          ) {
            const hiddenNode = new Node('hidden');
            network.nodes.push(hiddenNode);

            for (let inputIndex = 0; inputIndex < 20; inputIndex++) {
              network.connect(
                network.nodes[inputIndex],
                hiddenNode,
                buildDeterministicWeight(inputIndex, hiddenExpansionGuard, 5),
              );
            }

            for (
              let outputIndex = network.nodes.length - 5;
              outputIndex < network.nodes.length;
              outputIndex++
            ) {
              network.connect(
                hiddenNode,
                network.nodes[outputIndex],
                buildDeterministicWeight(hiddenExpansionGuard, outputIndex, 7),
              );
            }

            hiddenExpansionGuard++;
          }

          setTopologyDirty(network);
          markSlabDirty(network);
          const initialStats = memoryStats(network).flags.snapshot
            .allocStats as SlabAllocStats;

          if (!initialStats) {
            throw new Error('Expected initial slab allocation stats');
          }

          const initialVersion = getSlabVersion(network);
          const chunkSize = 10_000;
          let microtaskYields = 0;
          const originalThen: PromiseThen = Promise.prototype.then;

          const patchedThen: PromiseThen = function patchedThen<
            TResult1,
            TResult2,
          >(
            this: Promise<unknown>,
            onfulfilled?:
              ((value: unknown) => TResult1 | PromiseLike<TResult1>) | null,
            onrejected?:
              ((reason: unknown) => TResult2 | PromiseLike<TResult2>) | null,
          ): Promise<TResult1 | TResult2> {
            microtaskYields++;
            return originalThen.call(this, onfulfilled, onrejected) as Promise<
              TResult1 | TResult2
            >;
          };

          try {
            Promise.prototype.then = patchedThen;
            await rebuildConnectionSlabAsync.call(network, chunkSize);
          } finally {
            Promise.prototype.then = originalThen;
          }

          const asyncSlab = getConnectionSlabSnapshot(network);

          if (!asyncSlab.gain || !asyncSlab.flags) {
            throw new Error(
              'Expected async slab to expose gain and flags arrays',
            );
          }

          const finalStats = memoryStats(network).flags.snapshot
            .allocStats as SlabAllocStats;

          if (!finalStats) {
            throw new Error('Expected final slab allocation stats');
          }

          markSlabDirty(network);
          rebuildConnectionSlabIfPresent(network);
          const syncSlab = getConnectionSlabSnapshot(network);

          if (!syncSlab.gain || !syncSlab.flags) {
            throw new Error(
              'Expected sync slab to expose gain and flags arrays',
            );
          }

          // Assert
          expect(
            asyncSlab.version > initialVersion &&
              asyncSlab.capacity >= asyncSlab.used &&
              asyncSlab.flags.length === asyncSlab.weights.length &&
              asyncSlab.gain.length === asyncSlab.weights.length &&
              microtaskYields > 1 &&
              asyncSlab.version === syncSlab.version &&
              asyncSlab.used === syncSlab.used &&
              finalStats.fresh >= initialStats.fresh &&
              finalStats.fresh - initialStats.fresh <= 10,
          ).toBe(true);
        }, 30000);
      });
    });
  });
});
