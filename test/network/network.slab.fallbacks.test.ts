import Network from '../../src/architecture/network';

const clearConnectionWeights = (network: Network): void => {
  Reflect.set(network, '_connWeights', null);
};

const runInternalFastSlabActivate = (
  network: Network,
  values: number[]
): number[] => {
  const fastSlab = Reflect.get(network, '_fastSlabActivate') as (
    input: number[]
  ) => number[];
  return fastSlab.call(network, values);
};

/**
 * fastSlabActivate fallback branch when adjacency prerequisites missing (after manual corruption).
 * We simulate by nulling internal slab arrays post-build so guard triggers and generic path is used.
 */
describe('Network.fastSlabActivate prerequisites', () => {
  describe('Scenario: missing adjacency arrays triggers generic activate fallback', () => {
    it('returns output of expected length via standard activate', () => {
      // Arrange
      const networkUnderTest = new Network(2, 2, {
        seed: 22,
        enforceAcyclic: true,
      });
      // Build slab once so fields exist, then corrupt them.
      networkUnderTest.rebuildConnectionSlab(true);
      clearConnectionWeights(networkUnderTest); // force missing prerequisite guard
      const inputVector = [0.9, 0.1];
      // Act
      const outputVector = runInternalFastSlabActivate(
        networkUnderTest,
        inputVector
      );
      // Assert
      expect(outputVector.length).toBe(2);
    });
  });
});
