import Network from '../network';
import { Architect } from '../../../neataptic';

describe('network runtime chapter', () => {
  describe('weight noise', () => {
    describe('given one global weight-noise standard deviation is enabled', () => {
      describe('when runtime weight-noise state is inspected', () => {
        it('stores the global standard deviation and clears per-layer state', () => {
          // Arrange
          const network = new Network(2, 1, { minHidden: 2 });

          // Act
          network.enableWeightNoise(0.5);
          const standardDeviation = Reflect.get(network, '_weightNoiseStd');
          const perLayerState = Reflect.get(network, '_weightNoisePerHidden');

          // Assert
          expect(
            standardDeviation === 0.5 &&
              Array.isArray(perLayerState) &&
              perLayerState.length === 0,
          ).toBe(true);
        });
      });
    });

    describe('given weight noise was enabled earlier', () => {
      describe('when weight noise is disabled', () => {
        it('clears both global and per-layer configuration', () => {
          // Arrange
          const network = new Network(2, 1, { minHidden: 1 });
          network.enableWeightNoise(0.3);

          // Act
          network.disableWeightNoise();
          const standardDeviation = Reflect.get(network, '_weightNoiseStd');
          const perLayerState = Reflect.get(network, '_weightNoisePerHidden');

          // Assert
          expect(
            standardDeviation === 0 &&
              Array.isArray(perLayerState) &&
              perLayerState.length === 0,
          ).toBe(true);
        });
      });
    });

    describe('given per-hidden-layer weight noise is configured on a seeded layered network', () => {
      describe('when several training activations run', () => {
        it('keeps later hidden layers noisier than earlier zero-noise layers', () => {
          // Arrange
          const network = Architect.perceptron(3, 5, 7, 9, 2);
          const noiseByLayer: number[][] = [[], [], []];

          network.enableWeightNoise({ perHiddenLayer: [0.0, 0.05, 0.1] });
          network.setSeed(7);

          for (let iteration = 0; iteration < 30; iteration++) {
            network.activate([0.01, 0.02, 0.03], true);

            for (const connection of network.connections) {
              let fromLayer = -1;

              if (network.layers) {
                for (
                  let layerIndex = 0;
                  layerIndex < network.layers.length;
                  layerIndex++
                ) {
                  if (
                    network.layers[layerIndex].nodes.includes(connection.from)
                  ) {
                    fromLayer = layerIndex;
                    break;
                  }
                }
              }

              if (
                fromLayer > 0 &&
                fromLayer < (network.layers?.length ?? 0) - 1
              ) {
                const hiddenLayerIndex = fromLayer - 1;
                noiseByLayer[hiddenLayerIndex].push(
                  (connection as typeof connection & { _wnLast?: number })
                    ._wnLast || 0,
                );
              }
            }
          }

          const averageAbsoluteNoise = noiseByLayer.map((layerNoise) => {
            return (
              layerNoise.reduce((sum, value) => sum + Math.abs(value), 0) /
              (layerNoise.length || 1)
            );
          });

          // Act
          const respectsPerLayerOrdering =
            averageAbsoluteNoise[0] < 1e-6 &&
            averageAbsoluteNoise[1] > 0.0001 &&
            averageAbsoluteNoise[2] > averageAbsoluteNoise[1];

          // Assert
          expect(respectsPerLayerOrdering).toBe(true);
        });
      });
    });
  });
});
