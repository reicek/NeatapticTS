import Network from '../network';

describe('network runtime chapter', () => {
  describe('DropConnect', () => {
    describe('given an invalid negative probability', () => {
      describe('when DropConnect is enabled', () => {
        it('throws the range error', () => {
          // Arrange
          const network = new Network(1, 1);

          // Act
          const enableWithInvalidProbability = () =>
            network.enableDropConnect(-0.1);

          // Assert
          expect(enableWithInvalidProbability).toThrow(
            'DropConnect probability must be in [0,1)',
          );
        });
      });
    });

    describe('given an invalid probability greater than or equal to one', () => {
      describe('when DropConnect is enabled', () => {
        it('throws the range error', () => {
          // Arrange
          const network = new Network(1, 1);

          // Act
          const enableWithInvalidProbability = () =>
            network.enableDropConnect(1);

          // Assert
          expect(enableWithInvalidProbability).toThrow(
            'DropConnect probability must be in [0,1)',
          );
        });
      });
    });

    describe('given DropConnect is enabled for a training activation', () => {
      describe('when connection masks are inspected after activation', () => {
        it('assigns only binary DropConnect masks', () => {
          // Arrange
          const network = new Network(3, 2, { minHidden: 2 });

          for (const connection of network.connections) {
            connection.weight = 0.5;
          }

          network.enableDropConnect(0.9);

          // Act
          network.activate([0.1, 0.2, 0.3], true);
          const hasBinaryMasks = network.connections.every(
            (connection) => connection.dcMask === 0 || connection.dcMask === 1,
          );

          // Assert
          expect(hasBinaryMasks).toBe(true);
        });
      });
    });

    describe('given DropConnect was enabled earlier', () => {
      describe('when DropConnect is disabled', () => {
        it('resets the internal probability to zero', () => {
          // Arrange
          const network = new Network(2, 1);
          network.enableDropConnect(0.4);

          // Act
          network.disableDropConnect();
          const probability = Reflect.get(network, '_dropConnectProb');

          // Assert
          expect(probability).toBe(0);
        });
      });
    });
  });
});
