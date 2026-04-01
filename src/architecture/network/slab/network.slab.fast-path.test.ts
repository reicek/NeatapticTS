import Network from '../network';
import { config } from '../../../config';
import Node from '../../node';

function clearConnectionWeights(network: Network): void {
  Reflect.set(network, '_connWeights', null);
}

function createDeterministicInputVector(inputCount: number): number[] {
  return Array.from({ length: inputCount }, (_, inputIndex) => {
    return ((inputIndex % 5) - 2) / 5;
  });
}

function getGateCollection(network: Network): Array<unknown> {
  return Reflect.get(network, 'gates') as Array<unknown>;
}

function markSlabDirty(network: Network): void {
  Reflect.set(network, '_slabDirty', true);
}

function markTopologyDirty(network: Network): void {
  Reflect.set(network, '_topoDirty', true);
  Reflect.set(network, '_nodeIndexDirty', true);
}

function runInternalFastSlabActivate(
  network: Network,
  inputVector: number[],
): number[] {
  const fastSlabActivate = Reflect.get(network, '_fastSlabActivate') as (
    input: number[],
  ) => number[];
  return fastSlabActivate.call(network, inputVector);
}

function runPublicFastSlabActivate(
  network: Network,
  inputVector: number[],
): number[] {
  const fastSlabActivate = Reflect.get(network, 'fastSlabActivate') as (
    input: number[],
  ) => number[];
  return fastSlabActivate.call(network, inputVector);
}

describe('network slab chapter', () => {
  describe('fast path execution', () => {
    describe('given slab prerequisites are manually corrupted after a rebuild', () => {
      describe('when internal fast slab activation runs', () => {
        it('falls back to the standard output width', () => {
          // Arrange
          const network = new Network(2, 2, {
            seed: 22,
            enforceAcyclic: true,
          });
          const expectedLength = 2;

          network.rebuildConnectionSlab(true);
          clearConnectionWeights(network);

          // Act
          const outputVector = runInternalFastSlabActivate(network, [0.9, 0.1]);

          // Assert
          expect(outputVector.length).toBe(expectedLength);
        });
      });
    });

    describe('given an eligible acyclic network without gating or noise', () => {
      describe('when fast slab activation is compared with legacy activation', () => {
        it('produces identical outputs', () => {
          // Arrange
          config.enableNodePooling = false;
          const network = new Network(5, 3, { enforceAcyclic: true });

          for (let mutationIndex = 0; mutationIndex < 8; mutationIndex++) {
            if (network.connections.length) {
              network.addNodeBetween();
            }
          }

          markSlabDirty(network);
          const inputVector = createDeterministicInputVector(5);
          const legacyOutput = network.activate([...inputVector], false);

          // Act
          const fastOutput = runPublicFastSlabActivate(network, [
            ...inputVector,
          ]);

          // Assert
          expect(fastOutput).toStrictEqual(legacyOutput);
        });
      });
    });

    describe('given non-neutral gains force lazy gain slab allocation', () => {
      describe('when fast slab activation is compared with legacy activation', () => {
        it('still produces identical outputs', () => {
          // Arrange
          config.enableNodePooling = false;
          const network = new Network(4, 2, { enforceAcyclic: true });

          for (let mutationIndex = 0; mutationIndex < 6; mutationIndex++) {
            if (network.connections.length) {
              network.addNodeBetween();
            }
          }

          for (
            let connectionIndex = 0;
            connectionIndex < network.connections.length;
            connectionIndex += 2
          ) {
            network.connections[connectionIndex].gain = 1.2;
          }

          markSlabDirty(network);
          const inputVector = [0.1, -0.2, 0.05, 0.9];
          const legacyOutput = network.activate([...inputVector], false);

          // Act
          const fastOutput = runPublicFastSlabActivate(network, [
            ...inputVector,
          ]);

          // Assert
          expect(fastOutput).toStrictEqual(legacyOutput);
        });
      });
    });

    describe('given gating is present on the network', () => {
      describe('when fast slab activation is requested', () => {
        it('falls back to the legacy activation path', () => {
          // Arrange
          config.enableNodePooling = false;
          const network = new Network(2, 1, { enforceAcyclic: true });
          const hiddenNode = new Node('hidden');

          network.nodes.push(hiddenNode);
          getGateCollection(network).push({ dummy: true });
          markTopologyDirty(network);

          const inputVector = [0.3, -0.1];
          const legacyOutput = network.activate([...inputVector], false);

          // Act
          const fastOutput = runPublicFastSlabActivate(network, [
            ...inputVector,
          ]);

          // Assert
          expect(fastOutput).toStrictEqual(legacyOutput);
        });
      });
    });
  });
});
