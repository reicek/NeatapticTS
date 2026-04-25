import Node from '../node';
import type Network from './network';
import {
  appendTemporalDescriptorSet,
  buildLstmTemporalDescriptorSet,
  buildNarxMemoryTemporalDescriptorSet,
  describeTemporalStructure,
  resolveTemporalRecurrentModuleNodeGeneIds,
  splitGruLayerNodes,
  splitLstmLayerNodes,
  synchronizeTemporalDescriptorExtensions,
} from './network.temporal.extensions.utils';

type SerializedExtensionBag = {
  version?: number;
  values?: Record<string, unknown>;
};

type TemporalRuntimeSurface = {
  nodes: Node[];
  _serializedExtensions?: SerializedExtensionBag;
};

describe('network.temporal.extensions.utils', () => {
  describe('split helpers', () => {
    describe('given the requested recurrent block shape is invalid', () => {
      it('returns undefined for both LSTM and GRU splits', () => {
        // Arrange
        const splitResult = {
          gru: splitGruLayerNodes([], 0),
          lstm: splitLstmLayerNodes([], 0),
        };

        // Assert
        expect(splitResult).toEqual({
          gru: undefined,
          lstm: undefined,
        });
      });
    });
  });

  describe('buildNarxMemoryTemporalDescriptorSet', () => {
    describe('given the delay-line memory block has no registered boundary connections', () => {
      it('returns undefined instead of emitting an empty descriptor bag', () => {
        // Arrange
        const delayNode = createNode(701);
        const runtimeNetwork = createRuntimeSurface([delayNode]);

        // Act
        const descriptorSet = buildNarxMemoryTemporalDescriptorSet(
          runtimeNetwork as unknown as Network,
          'history',
          [[delayNode]],
        );

        // Assert
        expect(descriptorSet).toBeUndefined();
      });
    });
  });

  describe('buildLstmTemporalDescriptorSet', () => {
    describe('given a recurrent block has internal structure but no active gate ownership', () => {
      it('emits only the recurrent-module descriptor and de-duplicates the registered self-connection', () => {
        // Arrange
        const inputGateNode = createNode(101);
        const forgetGateNode = createNode(102);
        const memoryCellNode = createNode(103);
        const outputGateNode = createNode(104);
        const outputBlockNode = createNode(105);
        const sharedSelfConnection = connectNodeToSelf(inputGateNode, 9_001);
        inputGateNode.connections.out.push(sharedSelfConnection);
        const runtimeNetwork = createRuntimeSurface([
          inputGateNode,
          forgetGateNode,
          memoryCellNode,
          outputGateNode,
          outputBlockNode,
        ]);

        // Act
        const descriptorSet = buildLstmTemporalDescriptorSet(
          runtimeNetwork as unknown as Network,
          {
            forgetGate: [forgetGateNode],
            inputGate: [inputGateNode],
            memoryCell: [memoryCellNode],
            outputBlock: [outputBlockNode],
            outputGate: [outputGateNode],
          },
        );

        // Assert
        expect(descriptorSet).toEqual({
          recurrentModules: [
            {
              connectionInnovations: [9_001],
              kind: 'lstm',
              moduleId: 'module:lstm:101',
              nodeGeneIdsByRole: {
                forgetGate: [102],
                inputGate: [101],
                memoryCell: [103],
                outputBlock: [105],
                outputGate: [104],
              },
            },
          ],
        });
      });
    });
  });

  describe('describeTemporalStructure', () => {
    describe('given the serialized extension bag contains malformed temporal descriptors', () => {
      it('drops the invalid temporal entries while preserving non-temporal extension values', () => {
        // Arrange
        const runtimeNetwork = createRuntimeSurface([], {
          values: {
            gatedBlocks: [
              {
                blockId: '',
                connectionInnovations: [9],
                gaterGeneIds: [4],
              },
            ],
            marker: 'keep-me',
            recurrentModules: [
              {
                connectionInnovations: [7],
                kind: 'gru',
                moduleId: '',
                nodeGeneIdsByRole: { output: [3] },
              },
              {
                connectionInnovations: [8],
                kind: 'gru',
                moduleId: 'module:gru:valid',
                nodeGeneIdsByRole: [],
              },
            ],
          },
          version: 1,
        });

        // Act
        const temporalState = {
          descriptor: describeTemporalStructure(runtimeNetwork as unknown as Network),
          extensions: runtimeNetwork._serializedExtensions,
        };

        // Assert
        expect(temporalState).toEqual({
          descriptor: {
            gatedBlocks: [],
            recurrentModules: [],
          },
          extensions: {
            values: {
              marker: 'keep-me',
            },
            version: 1,
          },
        });
      });
    });

    describe('given the temporal descriptors reference missing live genes or connections', () => {
      it('removes the stale descriptors from the hydrated extension bag', () => {
        // Arrange
        const liveNode = createNode(401);
        const runtimeNetwork = createRuntimeSurface([liveNode], {
          values: {
            gatedBlocks: [
              {
                blockId: 'gated:block:gru:401',
                connectionInnovations: [55],
                gaterGeneIds: [999],
              },
              {
                blockId: 'gated:block:gru:402',
                connectionInnovations: [56],
                gaterGeneIds: [401],
              },
            ],
            marker: 'keep-me-too',
            recurrentModules: [
              {
                connectionInnovations: [57],
                kind: 'gru',
                moduleId: 'module:gru:401',
                nodeGeneIdsByRole: {
                  output: [999],
                },
              },
            ],
          },
          version: 1,
        });

        // Act
        const temporalState = {
          descriptor: describeTemporalStructure(runtimeNetwork as unknown as Network),
          extensions: runtimeNetwork._serializedExtensions,
        };

        // Assert
        expect(temporalState).toEqual({
          descriptor: {
            gatedBlocks: [],
            recurrentModules: [],
          },
          extensions: {
            values: {
              marker: 'keep-me-too',
            },
            version: 1,
          },
        });
      });
    });

    describe('given a live narx-memory descriptor carries a nonstandard module id and nonpositive version', () => {
      it('keeps the descriptor without a derived module label and normalizes the extension version', () => {
        // Arrange
        const memoryNode = createNode(701);
        const memoryConnection = connectNodeToSelf(memoryNode, 123);
        const runtimeNetwork = createRuntimeSurface([memoryNode], {
          values: {
            marker: 'still-here',
            recurrentModules: [
              {
                connectionInnovations: [memoryConnection.innovation],
                kind: 'narx-memory',
                moduleId: 'broken',
                nodeGeneIdsByRole: {
                  delayStep0: [701],
                },
              },
            ],
          },
          version: 0,
        });

        // Act
        const temporalState = {
          descriptor: describeTemporalStructure(runtimeNetwork as unknown as Network),
          extensions: runtimeNetwork._serializedExtensions,
        };

        // Assert
        expect(temporalState).toEqual({
          descriptor: {
            gatedBlocks: [],
            recurrentModules: [
              {
                connectionInnovations: [123],
                kind: 'narx-memory',
                moduleId: 'broken',
                nodeGeneIdsByRole: {
                  delayStep0: [701],
                },
              },
            ],
          },
          extensions: {
            values: {
              marker: 'still-here',
              recurrentModules: [
                {
                  connectionInnovations: [123],
                  kind: 'narx-memory',
                  moduleId: 'broken',
                  nodeGeneIdsByRole: {
                    delayStep0: [701],
                  },
                },
              ],
            },
            version: 1,
          },
        });
      });
    });
  });

  describe('resolveTemporalRecurrentModuleNodeGeneIds', () => {
    describe('given the runtime carries one valid recurrent module descriptor', () => {
      it('returns the live gene ids owned by that module', () => {
        // Arrange
        const firstModuleNode = createNode(501);
        const secondModuleNode = createNode(502);
        const liveConnection = connectNodes(firstModuleNode, secondModuleNode, 88);
        const runtimeNetwork = createRuntimeSurface(
          [firstModuleNode, secondModuleNode],
          {
            values: {
              recurrentModules: [
                {
                  connectionInnovations: [liveConnection.innovation],
                  kind: 'gru',
                  moduleId: 'module:gru:501',
                  nodeGeneIdsByRole: {
                    output: [501, 502],
                  },
                },
              ],
            },
            version: 1,
          },
        );

        // Act
        const ownedGeneIds = [...resolveTemporalRecurrentModuleNodeGeneIds(
          runtimeNetwork as unknown as Network,
        )];

        // Assert
        expect(ownedGeneIds).toEqual([501, 502]);
      });
    });
  });

  describe('appendTemporalDescriptorSet', () => {
    describe('given multiple valid gated blocks are appended out of order', () => {
      it('stores the gated block descriptors in sorted block-id order', () => {
        // Arrange
        const leftSourceNode = createNode(601);
        const leftTargetNode = createNode(602);
        const leftGaterNode = createNode(603);
        const rightSourceNode = createNode(611);
        const rightTargetNode = createNode(612);
        const rightGaterNode = createNode(613);
        const leftConnection = connectNodes(leftSourceNode, leftTargetNode, 91);
        const rightConnection = connectNodes(rightSourceNode, rightTargetNode, 92);
        Reflect.set(leftConnection, 'gater', leftGaterNode);
        Reflect.set(rightConnection, 'gater', rightGaterNode);
        const runtimeNetwork = createRuntimeSurface([
          leftSourceNode,
          leftTargetNode,
          leftGaterNode,
          rightSourceNode,
          rightTargetNode,
          rightGaterNode,
        ]);

        // Act
        appendTemporalDescriptorSet(runtimeNetwork as unknown as Network, {
          gatedBlocks: [
            {
              blockId: 'gated:block:gru:zeta',
              connectionInnovations: [92],
              gaterGeneIds: [613],
            },
            {
              blockId: 'gated:block:gru:alpha',
              connectionInnovations: [91],
              gaterGeneIds: [603],
            },
          ],
        });
        const temporalDescriptor = describeTemporalStructure(
          runtimeNetwork as unknown as Network,
        );

        // Assert
        expect(temporalDescriptor.gatedBlocks.map((gatedBlock) => gatedBlock.blockId)).toEqual([
          'gated:block:gru:alpha',
          'gated:block:gru:zeta',
        ]);
      });
    });

    describe('given gated metadata is available and the extension bag has no version', () => {
      it('stores the reachable LSTM descriptors and seeds the default extension version', () => {
        // Arrange
        const externalSourceNode = createNode(801);
        const externalTargetNode = createNode(802);
        const inputGateNode = createNode(803);
        const forgetGateNode = createNode(804);
        const memoryCellNode = createNode(805);
        const outputGateNode = createNode(806);
        const outputBlockNode = createNode(807);
        const gatedConnection = connectNodes(externalSourceNode, externalTargetNode, 222);
        Reflect.set(gatedConnection, 'gater', inputGateNode);
        const runtimeNetwork = createRuntimeSurface(
          [
            externalSourceNode,
            externalTargetNode,
            inputGateNode,
            forgetGateNode,
            memoryCellNode,
            outputGateNode,
            outputBlockNode,
          ],
          {
            values: {
              marker: 'seed-version',
            },
          },
        );
        const descriptorSet = buildLstmTemporalDescriptorSet(
          runtimeNetwork as unknown as Network,
          {
            forgetGate: [forgetGateNode],
            inputGate: [inputGateNode],
            memoryCell: [memoryCellNode],
            outputBlock: [outputBlockNode],
            outputGate: [outputGateNode],
          },
        );

        // Act
        appendTemporalDescriptorSet(runtimeNetwork as unknown as Network, descriptorSet);
        const temporalState = {
          descriptor: describeTemporalStructure(runtimeNetwork as unknown as Network),
          extensions: runtimeNetwork._serializedExtensions,
        };

        // Assert
        expect(temporalState).toEqual({
          descriptor: {
            gatedBlocks: [
              {
                blockId: 'gated:block:lstm:803',
                connectionInnovations: [222],
                gaterGeneIds: [803],
              },
            ],
            recurrentModules: [
              {
                connectionInnovations: [222],
                kind: 'lstm',
                moduleId: 'module:lstm:803',
                nodeGeneIdsByRole: {
                  forgetGate: [804],
                  inputGate: [803],
                  memoryCell: [805],
                  outputBlock: [807],
                  outputGate: [806],
                },
              },
            ],
          },
          extensions: {
            values: {
              gatedBlocks: [
                {
                  blockId: 'gated:block:lstm:803',
                  connectionInnovations: [222],
                  gaterGeneIds: [803],
                },
              ],
              marker: 'seed-version',
              recurrentModules: [
                {
                  connectionInnovations: [222],
                  kind: 'lstm',
                  moduleId: 'module:lstm:803',
                  nodeGeneIdsByRole: {
                    forgetGate: [804],
                    inputGate: [803],
                    memoryCell: [805],
                    outputBlock: [807],
                    outputGate: [806],
                  },
                },
              ],
            },
            version: 1,
          },
        });
      });
    });
  });

  describe('appendTemporalDescriptorSet', () => {
    describe('given a network that already has a valid versioned extension bag', () => {
      it('preserves the existing version number instead of resetting to default', () => {
        // Arrange — build LSTM descriptor set once and append it to seed versioned extensions
        const inputGateNode = createNode(901);
        const forgetGateNode = createNode(902);
        const memoryCellNode = createNode(903);
        const outputGateNode = createNode(904);
        const outputBlockNode = createNode(905);
        const externalSourceNode = createNode(906);
        const externalTargetNode = createNode(907);
        const gatedConnection = connectNodes(externalSourceNode, externalTargetNode, 333);
        Reflect.set(gatedConnection, 'gater', inputGateNode);
        const runtimeNetwork = createRuntimeSurface([
          externalSourceNode,
          externalTargetNode,
          inputGateNode,
          forgetGateNode,
          memoryCellNode,
          outputGateNode,
          outputBlockNode,
        ]);
        const lstmLayerNodes = {
          forgetGate: [forgetGateNode],
          inputGate: [inputGateNode],
          memoryCell: [memoryCellNode],
          outputBlock: [outputBlockNode],
          outputGate: [outputGateNode],
        };
        const firstDescriptorSet = buildLstmTemporalDescriptorSet(
          runtimeNetwork as unknown as Network,
          lstmLayerNodes,
        );
        // First append seeds _serializedExtensions with version=1
        appendTemporalDescriptorSet(runtimeNetwork as unknown as Network, firstDescriptorSet);
        // Now _serializedExtensions.version === 1; build a second LSTM descriptor set to trigger
        // resolveExtensionVersion with a valid pre-existing version (the true arm at line 716)
        const secondDescriptorSet = buildLstmTemporalDescriptorSet(
          runtimeNetwork as unknown as Network,
          lstmLayerNodes,
        );

        // Act — second append hits resolveExtensionVersion with extensions.version === 1
        appendTemporalDescriptorSet(runtimeNetwork as unknown as Network, secondDescriptorSet);

        // Assert
        expect(runtimeNetwork._serializedExtensions?.version).toBe(1);
      });
    });

    describe('given a network with pre-seeded version-1 extensions and valid live descriptors', () => {
      it('retains version 1 when synchronize re-writes the extension bag', () => {
        // Arrange — build live LSTM descriptor nodes with a live gated connection
        const inputGateNode = createNode(911);
        const forgetGateNode = createNode(912);
        const memoryCellNode = createNode(913);
        const outputGateNode = createNode(914);
        const outputBlockNode = createNode(915);
        const externalSourceNode = createNode(916);
        const externalTargetNode = createNode(917);
        const gatedConnection = connectNodes(externalSourceNode, externalTargetNode, 444);
        Reflect.set(gatedConnection, 'gater', inputGateNode);
        const runtimeNetwork = createRuntimeSurface([
          externalSourceNode,
          externalTargetNode,
          inputGateNode,
          forgetGateNode,
          memoryCellNode,
          outputGateNode,
          outputBlockNode,
        ]);
        const lstmLayerNodes = {
          forgetGate: [forgetGateNode],
          inputGate: [inputGateNode],
          memoryCell: [memoryCellNode],
          outputBlock: [outputBlockNode],
          outputGate: [outputGateNode],
        };
        const descriptorSet = buildLstmTemporalDescriptorSet(
          runtimeNetwork as unknown as Network,
          lstmLayerNodes,
        );
        // Manually seed the extension bag with version=1 and the built descriptor values
        // so that synchronizeTemporalDescriptorExtensions receives a versioned bag directly
        Reflect.set(runtimeNetwork, '_serializedExtensions', {
          version: 1,
          values: descriptorSet,
        });

        // Act — synchronize with an already-versioned bag; resolveExtensionVersion takes true arm
        synchronizeTemporalDescriptorExtensions(runtimeNetwork as unknown as Network);

        // Assert
        expect(runtimeNetwork._serializedExtensions?.version).toBe(1);
      });
    });
  });
});

function createNode(geneId: number): Node {
  const node = new Node('hidden', undefined, () => 0.5);
  Reflect.set(node, 'geneId', geneId);
  return node;
}

function connectNodeToSelf(node: Node, innovation: number) {
  const connection = node.connect(node)[0];

  if (!connection) {
    throw new Error('Expected one self-connection for the temporal descriptor test.');
  }

  Reflect.set(connection, 'innovation', innovation);
  return connection;
}

function connectNodes(fromNode: Node, toNode: Node, innovation: number) {
  const connection = fromNode.connect(toNode)[0];

  if (!connection) {
    throw new Error('Expected one connection for the temporal descriptor test.');
  }

  Reflect.set(connection, 'innovation', innovation);
  return connection;
}

function createRuntimeSurface(
  nodes: Node[],
  serializedExtensions?: SerializedExtensionBag,
): TemporalRuntimeSurface {
  return {
    ...(serializedExtensions ? { _serializedExtensions: serializedExtensions } : {}),
    nodes,
  };
}