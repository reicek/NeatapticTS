import { Architect, methods } from '../../../neataptic';
import Network from '../network';

function createSerializableNetwork(seed: number): Network {
  return new Network(2, 1, { seed });
}

function createSingleValueSerializableNetwork(seed: number): Network {
  return new Network(1, 1, { seed });
}

type JsonRoundTripScenario = {
  architectureName: string;
  createNetwork: () => Network;
};

const JSON_ROUND_TRIP_SCENARIOS: JsonRoundTripScenario[] = [
  {
    architectureName: 'Perceptron',
    createNetwork: () => Architect.perceptron(2, 3, 1),
  },
  {
    architectureName: 'Basic Network',
    createNetwork: () => new Network(3, 2),
  },
  {
    architectureName: 'LSTM',
    createNetwork: () => Architect.lstm(2, 3, 1),
  },
  {
    architectureName: 'GRU',
    createNetwork: () => Architect.gru(2, 3, 2, 1),
  },
  {
    architectureName: 'Random',
    createNetwork: () => Architect.random(2, 5, 1),
  },
  {
    architectureName: 'NARX',
    createNetwork: () => Architect.narx(2, 2, 2, 1, 1),
  },
  {
    architectureName: 'Hopfield',
    createNetwork: () => Architect.hopfield(3),
  },
];

function createDeterministicInputValues(inputCount: number): number[] {
  return Array.from({ length: inputCount }, (_, inputIndex) =>
    Number(((inputIndex + 1) / 10).toFixed(2)),
  );
}

describe('network serialize chapter', () => {
  describe('Network.deserialize()', () => {
    describe('given a compact payload is rebuilt without explicit size overrides', () => {
      describe('when the source network shape is compared to the rebuilt network', () => {
        it('preserves the node count', () => {
          // Arrange
          const network = createSerializableNetwork(360);
          network.activate([0.5, 0.5]);
          const serializedNetwork = network.serialize();

          // Act
          const deserialized = Network.deserialize(serializedNetwork);
          const deserializedNodeCount = deserialized.nodes.length;

          // Assert
          expect(deserializedNodeCount).toBe(network.nodes.length);
        });
      });

      describe('when the rebuilt node collection is inspected', () => {
        it('returns a nodes array', () => {
          // Arrange
          const network = createSerializableNetwork(361);
          network.activate([0.5, 0.5]);
          const serializedNetwork = network.serialize();

          // Act
          const deserialized = Network.deserialize(serializedNetwork);
          const hasNodesArray = Array.isArray(deserialized.nodes);

          // Assert
          expect(hasNodesArray).toBe(true);
        });
      });

      describe('when the rebuilt network is activated with the original input', () => {
        it('preserves the output width', () => {
          // Arrange
          const network = createSerializableNetwork(362);
          const input = [0.2, 0.8];
          const originalOutput = network.activate(input);
          const serializedNetwork = network.serialize();

          // Act
          const deserialized = Network.deserialize(
            serializedNetwork,
            network.input,
            network.output,
          );
          const deserializedOutput = deserialized.activate(input);

          // Assert
          expect(deserializedOutput.length).toBe(originalOutput.length);
        });
      });

      describe('when the rebuilt network output is compared to the original output', () => {
        it('stays numerically close', () => {
          // Arrange
          const network = createSerializableNetwork(363);
          const input = [0.2, 0.8];
          const originalOutput = network.activate(input);
          const serializedNetwork = network.serialize();

          // Act
          const deserialized = Network.deserialize(
            serializedNetwork,
            network.input,
            network.output,
          );
          const deserializedOutput = deserialized.activate(input);
          const epsilon = 0.05;
          const allOutputsStayClose = deserializedOutput.every(
            (outputValue, outputIndex) =>
              Math.abs(outputValue - originalOutput[outputIndex]) < epsilon,
          );

          // Assert
          expect(allOutputsStayClose).toBe(true);
        });
      });
    });

    describe('given the compact payload has no connections', () => {
      describe('when deserialize() is called', () => {
        it('rebuilds a network with no connections', () => {
          // Arrange
          const network = createSerializableNetwork(364);
          network.connections = [];
          const serializedNetwork = network.serialize();

          // Act
          const deserialized = Network.deserialize(
            serializedNetwork,
            network.input,
            network.output,
          );
          const connectionCount = deserialized.connections.length;

          // Assert
          expect(connectionCount).toBe(0);
        });
      });
    });

    describe('given the compact payload contains invalid connection indices', () => {
      describe('when deserialize() rebuilds the payload', () => {
        it('skips the invalid connection entries', () => {
          // Arrange
          const network = createSerializableNetwork(365);
          const serializedNetwork = network.serialize();
          serializedNetwork[3][0].from = 999;
          serializedNetwork[3][0].to = 999;

          // Act
          const deserialized = Network.deserialize(
            serializedNetwork,
            network.input,
            network.output,
          );

          // Assert
          expect(deserialized.connections.length).toBeLessThanOrEqual(
            network.connections.length,
          );
        });
      });

      describe('when warnings are observed during rebuild', () => {
        it('warns about the invalid connection indices', () => {
          // Arrange
          const network = createSerializableNetwork(366);
          const serializedNetwork = network.serialize();
          serializedNetwork[3][0].from = 999;
          serializedNetwork[3][0].to = 999;
          const warnSpy = jest
            .spyOn(console, 'warn')
            .mockImplementation(() => {});

          try {
            // Act
            Network.deserialize(
              serializedNetwork,
              network.input,
              network.output,
            );
            const warnedAboutInvalidConnectionIndices = warnSpy.mock.calls.some(
              (call) =>
                typeof call[0] === 'string' &&
                call[0].includes('Invalid connection indices'),
            );

            // Assert
            expect(warnedAboutInvalidConnectionIndices).toBe(true);
          } finally {
            warnSpy.mockRestore();
          }
        });
      });
    });

    describe('given the compact payload contains an invalid gater index', () => {
      describe('when deserialize() rebuilds the payload', () => {
        it('skips the invalid gater assignment', () => {
          // Arrange
          const network = createSerializableNetwork(367);
          const serializedNetwork = network.serialize();
          serializedNetwork[3][0].gater = 999;

          // Act
          const deserialized = Network.deserialize(
            serializedNetwork,
            network.input,
            network.output,
          );

          // Assert
          expect(deserialized.gates.length).toBeLessThanOrEqual(
            network.gates.length,
          );
        });
      });

      describe('when warnings are observed during rebuild', () => {
        it('warns about the invalid gater index', () => {
          // Arrange
          const network = createSerializableNetwork(368);
          const serializedNetwork = network.serialize();
          serializedNetwork[3][0].gater = 999;
          const warnSpy = jest
            .spyOn(console, 'warn')
            .mockImplementation(() => {});

          try {
            // Act
            Network.deserialize(
              serializedNetwork,
              network.input,
              network.output,
            );
            const warnedAboutInvalidGaterIndex = warnSpy.mock.calls.some(
              (call) =>
                typeof call[0] === 'string' &&
                call[0].includes('Invalid gater index'),
            );

            // Assert
            expect(warnedAboutInvalidGaterIndex).toBe(true);
          } finally {
            warnSpy.mockRestore();
          }
        });
      });
    });

    describe('given the compact payload contains an unknown squash key', () => {
      describe('when deserialize() rebuilds the payload', () => {
        it('falls back to identity activation', () => {
          // Arrange
          const network = createSerializableNetwork(369);
          const serializedNetwork = network.serialize();
          serializedNetwork[2][0] = 'notARealSquashFn' as never;

          // Act
          const deserialized = Network.deserialize(
            serializedNetwork,
            network.input,
            network.output,
          );

          // Assert
          expect(deserialized.nodes[0].squash).toBe(
            methods.Activation.identity,
          );
        });
      });

      describe('when warnings are observed during rebuild', () => {
        it('warns about the unknown squash key', () => {
          // Arrange
          const network = createSerializableNetwork(370);
          const serializedNetwork = network.serialize();
          serializedNetwork[2][0] = 'notARealSquashFn' as never;
          const warnSpy = jest
            .spyOn(console, 'warn')
            .mockImplementation(() => {});

          try {
            // Act
            Network.deserialize(
              serializedNetwork,
              network.input,
              network.output,
            );
            const warnedAboutUnknownSquash = warnSpy.mock.calls.some(
              (call) =>
                typeof call[0] === 'string' &&
                call[0].includes('Unknown squash function'),
            );

            // Assert
            expect(warnedAboutUnknownSquash).toBe(true);
          } finally {
            warnSpy.mockRestore();
          }
        });
      });
    });
  });

  describe('Network.fromJSON()', () => {
    describe('given a supported architecture is serialized to JSON', () => {
      for (const jsonRoundTripScenario of JSON_ROUND_TRIP_SCENARIOS) {
        describe(`${jsonRoundTripScenario.architectureName}`, () => {
          let originalNetwork: Network;
          let deserializedNetwork: Network;
          let inputValues: number[];
          let originalOutputValues: number[];
          let deserializedOutputValues: number[];

          beforeAll(() => {
            // Arrange
            originalNetwork = jsonRoundTripScenario.createNetwork();
            const serializedJson = originalNetwork.toJSON();
            deserializedNetwork = Network.fromJSON(serializedJson);
            inputValues = createDeterministicInputValues(originalNetwork.input);
            originalOutputValues = originalNetwork.activate(inputValues);
            deserializedOutputValues =
              deserializedNetwork.activate(inputValues);
          });

          describe('when the rebuilt output width is compared to the original', () => {
            it('preserves the output length', () => {
              // Assert
              expect(deserializedOutputValues.length).toBe(
                originalOutputValues.length,
              );
            });
          });

          describe('when the rebuilt output values are compared to the original', () => {
            it('stays numerically close', () => {
              // Arrange
              const outputsStayClose = deserializedOutputValues.every(
                (outputValue, outputIndex) =>
                  Math.abs(outputValue - originalOutputValues[outputIndex]) <
                  1e-9,
              );

              // Assert
              expect(outputsStayClose).toBe(true);
            });
          });

          describe('when the nodes field is removed from the JSON payload', () => {
            it('throws', () => {
              // Arrange
              const serializedJson = originalNetwork.toJSON() as {
                nodes?: unknown;
              } & Record<string, unknown>;
              delete serializedJson.nodes;

              // Act
              const deserializeWithoutNodes = () =>
                Network.fromJSON(serializedJson);

              // Assert
              expect(deserializeWithoutNodes).toThrow();
            });
          });

          describe('when the connections field is removed from the JSON payload', () => {
            it('throws', () => {
              // Arrange
              const serializedJson = originalNetwork.toJSON() as {
                connections?: unknown;
              } & Record<string, unknown>;
              delete serializedJson.connections;

              // Act
              const deserializeWithoutConnections = () =>
                Network.fromJSON(serializedJson);

              // Assert
              expect(deserializeWithoutConnections).toThrow();
            });
          });
        });
      }
    });

    describe('given the JSON payload contains an unknown squash key', () => {
      describe('when fromJSON() rebuilds the payload', () => {
        it('falls back to identity activation', () => {
          // Arrange
          const network = new Network(1, 1);
          const serializedJson = network.toJSON() as {
            nodes: Array<Record<string, unknown>>;
          };
          serializedJson.nodes[0].squash = 'UNKNOWN_FUNCTION';

          // Act
          const deserialized = Network.fromJSON(
            serializedJson as unknown as Record<string, unknown>,
          );

          // Assert
          expect(deserialized.nodes[0].squash).toBe(
            methods.Activation.identity,
          );
        });
      });
    });

    describe('given the JSON payload contains invalid connection indices', () => {
      describe('when fromJSON() rebuilds the payload', () => {
        it('skips the invalid connection row', () => {
          // Arrange
          const network = createSerializableNetwork(371);
          const inputNodeIndex = network.nodes.findIndex(
            (node) => node.type === 'input',
          );
          const outputNodeIndex = network.nodes.findIndex(
            (node) => node.type === 'output',
          );
          const serializedJson: Record<string, unknown> = {
            formatVersion: 1,
            nodes: network.nodes.map((node) => ({
              bias: node.bias,
              type: node.type,
              squash: 'LOGISTIC',
            })),
            connections: [
              { from: inputNodeIndex, to: outputNodeIndex, weight: 0.5 },
              { from: 999, to: outputNodeIndex, weight: 0.5 },
            ],
            input: network.input,
            output: network.output,
          };

          // Act
          const deserialized = Network.fromJSON(
            serializedJson as unknown as Record<string, unknown>,
          );

          // Assert
          expect(deserialized.connections.length).toBe(1);
        });
      });
    });

    describe('given the JSON payload contains an invalid gater index', () => {
      describe('when fromJSON() rebuilds the payload', () => {
        it('skips the invalid gater assignment', () => {
          // Arrange
          const network = createSerializableNetwork(372);
          const inputNodeIndex = network.nodes.findIndex(
            (node) => node.type === 'input',
          );
          const outputNodeIndex = network.nodes.findIndex(
            (node) => node.type === 'output',
          );
          const serializedJson: Record<string, unknown> = {
            formatVersion: 1,
            nodes: network.nodes.map((node) => ({
              bias: node.bias,
              type: node.type,
              squash: 'LOGISTIC',
            })),
            connections: [
              {
                from: inputNodeIndex,
                to: outputNodeIndex,
                weight: 0.5,
                gater: 999,
              },
            ],
            input: network.input,
            output: network.output,
          };

          // Act
          const deserialized = Network.fromJSON(
            serializedJson as unknown as Record<string, unknown>,
          );

          // Assert
          expect(deserialized.gates.length).toBe(0);
        });
      });
    });

    describe('given the JSON payload contains extra unexpected fields', () => {
      describe('when fromJSON() rebuilds the payload', () => {
        it('ignores the extra fields', () => {
          // Arrange
          const network = createSerializableNetwork(373);
          const serializedJson = network.toJSON() as Record<string, unknown>;
          serializedJson.extraField = 'shouldBeIgnored';

          // Act
          const deserialized = Network.fromJSON(serializedJson);

          // Assert
          expect(deserialized).toBeInstanceOf(Network);
        });
      });
    });

    describe('given the JSON payload omits optional squash and gater fields', () => {
      describe('when fromJSON() rebuilds the payload', () => {
        it('uses the fallback squash function', () => {
          // Arrange
          const network = createSerializableNetwork(374);
          const serializedJson = network.toJSON() as {
            nodes: Array<Record<string, unknown>>;
            connections: Array<Record<string, unknown>>;
          };
          delete serializedJson.nodes[0].squash;
          delete serializedJson.connections[0].gater;

          // Act
          const deserialized = Network.fromJSON(
            serializedJson as unknown as Record<string, unknown>,
          );

          // Assert
          expect(deserialized.nodes[0].squash).toBeDefined();
        });
      });

      describe('when the rebuilt connection metadata is inspected', () => {
        it('defaults the gater to null', () => {
          // Arrange
          const network = createSerializableNetwork(375);
          const serializedJson = network.toJSON() as {
            nodes: Array<Record<string, unknown>>;
            connections: Array<Record<string, unknown>>;
          };
          delete serializedJson.nodes[0].squash;
          delete serializedJson.connections[0].gater;

          // Act
          const deserialized = Network.fromJSON(
            serializedJson as unknown as Record<string, unknown>,
          );

          // Assert
          expect(deserialized.connections[0].gater).toBeNull();
        });
      });
    });

    describe('given the JSON payload contains empty node and connection arrays', () => {
      describe('when fromJSON() rebuilds the payload', () => {
        it('returns a network with zero nodes', () => {
          // Arrange
          const serializedJson = {
            nodes: [],
            connections: [],
            input: 1,
            output: 1,
          } as Record<string, unknown>;

          // Act
          const deserialized = Network.fromJSON(
            serializedJson as unknown as Record<string, unknown>,
          );

          // Assert
          expect(deserialized.nodes.length).toBe(0);
        });
      });

      describe('when the rebuilt connection collection is inspected', () => {
        it('returns zero connections', () => {
          // Arrange
          const serializedJson = {
            nodes: [],
            connections: [],
            input: 1,
            output: 1,
          } as Record<string, unknown>;

          // Act
          const deserialized = Network.fromJSON(
            serializedJson as unknown as Record<string, unknown>,
          );

          // Assert
          expect(deserialized.connections.length).toBe(0);
        });
      });
    });

    describe('given a serialized network used a custom activation function', () => {
      describe('when the rebuilt squash reference is compared to the original', () => {
        it('does not preserve the original custom function', () => {
          // Arrange
          const network = createSingleValueSerializableNetwork(376);
          const customSquash = (value: number, derivative = false) =>
            derivative ? 0 : value * 100;
          network.nodes[0].squash = customSquash;
          Object.defineProperty(network.nodes[0].squash, 'name', {
            value: 'MY_CUSTOM_SQUASH',
          });
          const serializedJson = network.toJSON();

          // Act
          const deserialized = Network.fromJSON(serializedJson);

          // Assert
          expect(deserialized.nodes[0].squash).not.toBe(customSquash);
        });
      });

      describe('when the rebuilt squash output is compared to the original custom output', () => {
        it('changes the functional behavior', () => {
          // Arrange
          const network = createSingleValueSerializableNetwork(377);
          const customSquash = (value: number, derivative = false) =>
            derivative ? 0 : value * 100;
          network.nodes[0].squash = customSquash;
          Object.defineProperty(network.nodes[0].squash, 'name', {
            value: 'MY_CUSTOM_SQUASH',
          });
          const serializedJson = network.toJSON();

          // Act
          const deserialized = Network.fromJSON(serializedJson);
          const rebuiltOutputChanged = deserialized.nodes[0].squash(0.5) !== 50;

          // Assert
          expect(rebuiltOutputChanged).toBe(true);
        });
      });

      describe('when the rebuilt network is activated', () => {
        it('remains usable after deserialization', () => {
          // Arrange
          const network = createSingleValueSerializableNetwork(378);
          const customSquash = (value: number, derivative = false) =>
            derivative ? 0 : value * 100;
          network.nodes[0].squash = customSquash;
          Object.defineProperty(network.nodes[0].squash, 'name', {
            value: 'MY_CUSTOM_SQUASH',
          });
          const serializedJson = network.toJSON();

          // Act
          const deserialized = Network.fromJSON(serializedJson);
          const activateDeserializedNetwork = () =>
            deserialized.activate([0.5]);

          // Assert
          expect(activateDeserializedNetwork).not.toThrow();
        });
      });
    });

    describe('given a second custom activation function was serialized', () => {
      describe('when the rebuilt squash is inspected', () => {
        it('still exposes a callable squash function', () => {
          // Arrange
          const network = createSingleValueSerializableNetwork(379);
          const customSquash = (value: number, derivative = false) =>
            derivative ? 0 : value * value;
          network.nodes[0].squash = customSquash;
          Object.defineProperty(network.nodes[0].squash, 'name', {
            value: 'MY_CUSTOM_SQUASH',
          });
          const serializedJson = network.toJSON();

          // Act
          const deserialized = Network.fromJSON(serializedJson);
          const rebuiltSquashIsCallable =
            typeof deserialized.nodes[0].squash === 'function';

          // Assert
          expect(rebuiltSquashIsCallable).toBe(true);
        });
      });

      describe('when the rebuilt output is compared to the original custom output', () => {
        it('does not preserve the custom activation semantics', () => {
          // Arrange
          const network = createSingleValueSerializableNetwork(380);
          const customSquash = (value: number, derivative = false) =>
            derivative ? 0 : value * value;
          network.nodes[0].squash = customSquash;
          Object.defineProperty(network.nodes[0].squash, 'name', {
            value: 'MY_CUSTOM_SQUASH',
          });
          const serializedJson = network.toJSON();
          const testValue = 0.5;

          // Act
          const deserialized = Network.fromJSON(serializedJson);
          const rebuiltOutputChanged =
            deserialized.nodes[0].squash(testValue) !== customSquash(testValue);

          // Assert
          expect(rebuiltOutputChanged).toBe(true);
        });
      });

      describe('when identity is selected as the fallback squash function', () => {
        it('keeps the identity behavior intact', () => {
          // Arrange
          const network = createSingleValueSerializableNetwork(381);
          const customSquash = (value: number, derivative = false) =>
            derivative ? 0 : value * value;
          network.nodes[0].squash = customSquash;
          Object.defineProperty(network.nodes[0].squash, 'name', {
            value: 'MY_CUSTOM_SQUASH',
          });
          const serializedJson = network.toJSON();
          const testValue = 0.5;

          // Act
          const deserialized = Network.fromJSON(serializedJson);
          const identityFallbackStayedCorrect =
            deserialized.nodes[0].squash !== methods.Activation.identity ||
            deserialized.nodes[0].squash(testValue) === testValue;

          // Assert
          expect(identityFallbackStayedCorrect).toBe(true);
        });
      });
    });

    describe('given a mutated network is serialized to JSON', () => {
      describe('when fromJSON() rebuilds the payload', () => {
        it('preserves the expanded node count', () => {
          // Arrange
          const network = createSerializableNetwork(382);
          network.mutate(methods.mutation.ADD_NODE);
          const serializedJson = network.toJSON();

          // Act
          const deserialized = Network.fromJSON(serializedJson);

          // Assert
          expect(deserialized.nodes.length).toBeGreaterThan(2);
        });
      });
    });
  });
});
