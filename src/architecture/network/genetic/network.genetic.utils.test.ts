import Network from '../network';
import {
  crossOver,
  crossOverWithRandomGenerator,
} from './network.genetic.utils';

describe('network genetic utils', () => {
  describe('crossOver()', () => {
    describe('given the equal flag is omitted', () => {
      describe('when crossover runs directly', () => {
        it('uses the default fitter-parent bias path', () => {
          // Arrange
          const parentNetwork1 = Network.createMLP(2, [2], 1);
          const parentNetwork2 = Network.createMLP(2, [2], 1);

          // Act
          const offspringNetwork = crossOver(parentNetwork1, parentNetwork2);

          // Assert
          expect(offspringNetwork.nodes.length).toBeGreaterThan(0);
        });
      });
    });
  });

  describe('crossOverWithRandomGenerator()', () => {
    describe('given the equal flag is omitted', () => {
      describe('when crossover runs with an explicit RNG', () => {
        it('uses the default equal=false path', () => {
          // Arrange
          const parentNetwork1 = Network.createMLP(2, [2], 1);
          const parentNetwork2 = Network.createMLP(2, [2], 1);

          // Act
          const offspringNetwork = crossOverWithRandomGenerator(
            parentNetwork1,
            parentNetwork2,
            undefined as unknown as boolean,
            () => 0.5,
          );

          // Assert
          expect(offspringNetwork.nodes.length).toBeGreaterThan(0);
        });
      });
    });
  });
});