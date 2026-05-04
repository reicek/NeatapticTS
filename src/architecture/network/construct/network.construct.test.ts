import Network from '../network';
import Group from '../../group';
import Layer from '../../layer';
import Node from '../../node';
import { formatConstructSummary } from './network.construct.summary.utils';

describe('network construct chapter', () => {
  describe('Network.construct()', () => {
    describe('given mixed node, group, and layer parts', () => {
      describe('when explicit input and output ids define the public vector order', () => {
        it('builds one deterministic feed-forward runtime and compiled schedule', () => {
          // Arrange
          const sensorLeft = new Node('input');
          const sensorRight = new Node('input');
          const hiddenStage = new Group(2);
          const readoutLayer = Layer.dense(1, 'output');
          const readoutNode = readoutLayer.nodes[0];

          sensorLeft.describe({ label: 'sensorLeft' });
          sensorRight.describe({ label: 'sensorRight' });
          readoutNode.describe({ label: 'readout' });

          sensorLeft.connect(hiddenStage);
          sensorRight.connect(hiddenStage);
          hiddenStage.connect(readoutLayer);

          const expectedSummary = {
            nodeRoles: ['input', 'input', 'hidden', 'hidden', 'output'],
            inputNodeIds: [sensorLeft.geneId, sensorRight.geneId],
            outputNodeIds: [readoutNode.geneId],
            diagnostics: {
              nodeCount: 5,
              edgeCount: 6,
              detectedCycles: false,
              activationOrder: [0, 1, 2, 3, 4],
            },
            graph: {
              requestedMode: 'acyclic',
              topologyIntent: 'feed-forward',
              inputNodeIds: [sensorLeft.geneId, sensorRight.geneId],
              outputNodeIds: [readoutNode.geneId],
              activationOrder: [0, 1, 2, 3, 4],
              nodes: [
                {
                  index: 0,
                  geneId: sensorLeft.geneId,
                  label: 'sensorLeft',
                  role: 'input',
                  inputOrder: 0,
                  outputOrder: null,
                },
                {
                  index: 1,
                  geneId: sensorRight.geneId,
                  label: 'sensorRight',
                  role: 'input',
                  inputOrder: 1,
                  outputOrder: null,
                },
                {
                  index: 2,
                  geneId: hiddenStage.nodes[0].geneId,
                  label: null,
                  role: 'hidden',
                  inputOrder: null,
                  outputOrder: null,
                },
                {
                  index: 3,
                  geneId: hiddenStage.nodes[1].geneId,
                  label: null,
                  role: 'hidden',
                  inputOrder: null,
                  outputOrder: null,
                },
                {
                  index: 4,
                  geneId: readoutNode.geneId,
                  label: 'readout',
                  role: 'output',
                  inputOrder: null,
                  outputOrder: 0,
                },
              ],
              connections: [
                {
                  innovation: sensorLeft.connections.out[0].innovation,
                  fromIndex: 0,
                  toIndex: 2,
                  fromGeneId: sensorLeft.geneId,
                  toGeneId: hiddenStage.nodes[0].geneId,
                  gaterGeneId: null,
                  isSelfConnection: false,
                  enabled: true,
                  weight: sensorLeft.connections.out[0].weight,
                },
                {
                  innovation: sensorLeft.connections.out[1].innovation,
                  fromIndex: 0,
                  toIndex: 3,
                  fromGeneId: sensorLeft.geneId,
                  toGeneId: hiddenStage.nodes[1].geneId,
                  gaterGeneId: null,
                  isSelfConnection: false,
                  enabled: true,
                  weight: sensorLeft.connections.out[1].weight,
                },
                {
                  innovation: sensorRight.connections.out[0].innovation,
                  fromIndex: 1,
                  toIndex: 2,
                  fromGeneId: sensorRight.geneId,
                  toGeneId: hiddenStage.nodes[0].geneId,
                  gaterGeneId: null,
                  isSelfConnection: false,
                  enabled: true,
                  weight: sensorRight.connections.out[0].weight,
                },
                {
                  innovation: sensorRight.connections.out[1].innovation,
                  fromIndex: 1,
                  toIndex: 3,
                  fromGeneId: sensorRight.geneId,
                  toGeneId: hiddenStage.nodes[1].geneId,
                  gaterGeneId: null,
                  isSelfConnection: false,
                  enabled: true,
                  weight: sensorRight.connections.out[1].weight,
                },
                {
                  innovation:
                    hiddenStage.nodes[0].connections.out[0].innovation,
                  fromIndex: 2,
                  toIndex: 4,
                  fromGeneId: hiddenStage.nodes[0].geneId,
                  toGeneId: readoutNode.geneId,
                  gaterGeneId: null,
                  isSelfConnection: false,
                  enabled: true,
                  weight: hiddenStage.nodes[0].connections.out[0].weight,
                },
                {
                  innovation:
                    hiddenStage.nodes[1].connections.out[0].innovation,
                  fromIndex: 3,
                  toIndex: 4,
                  fromGeneId: hiddenStage.nodes[1].geneId,
                  toGeneId: readoutNode.geneId,
                  gaterGeneId: null,
                  isSelfConnection: false,
                  enabled: true,
                  weight: hiddenStage.nodes[1].connections.out[0].weight,
                },
              ],
            },
            scheduling: {
              topologyIntent: 'feed-forward',
              requestedMode: 'acyclic',
              topologyDirty: false,
              executionPath: 'compiled-schedule',
              issue: null,
              message:
                'Activation is using the compiled acyclic schedule with stable wave ordering.',
              inputNodeIds: [sensorLeft.geneId, sensorRight.geneId],
              outputNodeIds: [readoutNode.geneId],
              stepCount: 3,
              recurrentComponentCount: 0,
              stateSemantics: null,
              cycleNodeIds: [],
              suggestions: [],
            },
          };

          // Act
          const construction = Network.construct(
            [hiddenStage, sensorRight, readoutLayer, sensorLeft],
            {
              inputNodes: ['sensorLeft', 'sensorRight'],
              outputNodes: ['readout'],
            },
          );
          const actualSummary = {
            nodeRoles: construction.network.nodes.map((node) => node.type),
            inputNodeIds: construction.network.inputNodeIds,
            outputNodeIds: construction.network.outputNodeIds,
            diagnostics: construction.diagnostics,
            graph: construction.graph,
            scheduling:
              construction.network.getActivationSchedulingDiagnostics(),
          };

          // Assert
          expect(actualSummary).toEqual(expectedSummary);
        });
      });
    });

    describe('given explicit public IO ids', () => {
      describe('when one requested label matches multiple input nodes', () => {
        it('rejects the ambiguous construct id with a direct label error', () => {
          // Arrange
          const sharedInputLeft = new Node('input');
          const sharedInputRight = new Node('input');
          const readout = new Node('output');

          sharedInputLeft.describe({ label: 'sharedInput' });
          sharedInputRight.describe({ label: 'sharedInput' });

          // Act
          const constructWithAmbiguousInputLabel = () => {
            Network.construct([sharedInputLeft, sharedInputRight, readout], {
              inputNodes: ['sharedInput'],
            });
          };

          // Assert
          expect(constructWithAmbiguousInputLabel).toThrow(
            'Node label "sharedInput" matched multiple nodes. Labels used as explicit construct ids must be unique within the provided parts.',
          );
        });
      });

      describe('when explicit input ids repeat the same node', () => {
        it('rejects the duplicate public input selection', () => {
          // Arrange
          const sensorInput = new Node('input');
          const readout = new Node('output');

          sensorInput.describe({ label: 'sensorInput' });

          // Act
          const constructWithDuplicateInputIds = () => {
            Network.construct([sensorInput, readout], {
              inputNodes: ['sensorInput', 'sensorInput'],
            });
          };

          // Assert
          expect(constructWithDuplicateInputIds).toThrow(
            'Explicit input node ids must resolve to unique nodes.',
          );
        });
      });

      describe('when one explicit output id omits another output node from the provided parts', () => {
        it('rejects the partial public output selection with the missing label', () => {
          // Arrange
          const sensorInput = new Node('input');
          const primaryOutput = new Node('output');
          const secondaryOutput = new Node('output');

          primaryOutput.describe({ label: 'primaryOutput' });
          secondaryOutput.describe({ label: 'secondaryOutput' });

          // Act
          const constructWithIncompleteOutputCoverage = () => {
            Network.construct([sensorInput, primaryOutput, secondaryOutput], {
              outputNodes: ['primaryOutput'],
            });
          };

          // Assert
          expect(constructWithIncompleteOutputCoverage).toThrow(
            'Explicit output node ids must cover every output node in the provided parts. Missing: "secondaryOutput".',
          );
        });
      });

      describe('when an input node is requested as a public output id', () => {
        it('rejects the wrong-role selection before materialization', () => {
          // Arrange
          const sensorInput = new Node('input');
          const readout = new Node('output');

          sensorInput.describe({ label: 'sensorInput' });

          // Act
          const constructWithWrongRoleOutputId = () => {
            Network.construct([sensorInput, readout], {
              outputNodes: ['sensorInput'],
            });
          };

          // Assert
          expect(constructWithWrongRoleOutputId).toThrow(
            'Node "sensorInput" cannot be used as a output node because its runtime role is input.',
          );
        });
      });
    });

    describe('given referenced edges escape the provided part set', () => {
      describe('when the network is constructed', () => {
        it('fails with a missing referenced node error', () => {
          // Arrange
          const externalInput = new Node('input');
          const hiddenStage = new Group(2);
          const readoutLayer = Layer.dense(1, 'output');

          externalInput.describe({ label: 'externalInput' });
          externalInput.connect(hiddenStage);
          hiddenStage.connect(readoutLayer);

          // Act
          const constructWithMissingSource = () => {
            Network.construct([hiddenStage, readoutLayer]);
          };

          // Assert
          expect(constructWithMissingSource).toThrow(
            'were not included in parts',
          );
        });
      });
    });

    describe('given a public input node receives an incoming edge', () => {
      describe('when the graph is materialized', () => {
        it('rejects the graph because input nodes must stay pure sources', () => {
          // Arrange
          const sensorInput = new Node('input');
          const hiddenDriver = new Node('hidden');
          const readout = new Node('output');

          sensorInput.describe({ label: 'sensorInput' });
          hiddenDriver.describe({ label: 'hiddenDriver' });
          hiddenDriver.connect(sensorInput);
          sensorInput.connect(readout);

          // Act
          const constructWithIncomingInputEdge = () => {
            Network.construct([hiddenDriver, sensorInput, readout]);
          };

          // Assert
          expect(constructWithIncomingInputEdge).toThrow(
            'Input nodes must be pure sources in construct-from-parts graphs. Incoming edges: "hiddenDriver" -> "sensorInput".',
          );
        });
      });
    });

    describe('given a public output node projects onward', () => {
      describe('when sink-only output validation stays enabled', () => {
        it('rejects the graph because output nodes must stay pure sinks by default', () => {
          // Arrange
          const sensorInput = new Node('input');
          const primaryOutput = new Node('output');
          const secondaryOutput = new Node('output');

          primaryOutput.describe({ label: 'primaryOutput' });
          secondaryOutput.describe({ label: 'secondaryOutput' });
          sensorInput.connect(primaryOutput);
          primaryOutput.connect(secondaryOutput);

          // Act
          const constructWithOutputFeedback = () => {
            Network.construct([sensorInput, primaryOutput, secondaryOutput]);
          };

          // Assert
          expect(constructWithOutputFeedback).toThrow(
            'Output nodes must be pure sinks unless validate.allowOutputNodeOutgoingEdges is true. Outgoing edges: "primaryOutput" -> "secondaryOutput".',
          );
        });
      });
    });

    describe('given a public output node gates another connection', () => {
      describe('when sink-only output validation stays enabled', () => {
        it('rejects the graph because output nodes must not modulate downstream edges by default', () => {
          // Arrange
          const sensorInput = new Node('input');
          const hiddenCarrier = new Node('hidden');
          const primaryOutput = new Node('output');

          sensorInput.describe({ label: 'sensorInput' });
          hiddenCarrier.describe({ label: 'hiddenCarrier' });
          primaryOutput.describe({ label: 'primaryOutput' });

          sensorInput.connect(hiddenCarrier);
          hiddenCarrier.connect(primaryOutput);
          primaryOutput.gate(sensorInput.connections.out[0]);

          // Act
          const constructWithOutputGater = () => {
            Network.construct([sensorInput, hiddenCarrier, primaryOutput]);
          };

          // Assert
          expect(constructWithOutputGater).toThrow(
            'Output nodes must be pure sinks unless validate.allowOutputNodeOutgoingEdges is true. Gated connections: "primaryOutput" gates "sensorInput" -> "hiddenCarrier".',
          );
        });
      });
    });

    describe('given a graph intentionally uses output feedback', () => {
      describe('when sink-only output validation is relaxed explicitly', () => {
        it('still constructs one deterministic runtime', () => {
          // Arrange
          const sensorInput = new Node('input');
          const primaryOutput = new Node('output');
          const secondaryOutput = new Node('output');

          sensorInput.connect(primaryOutput);
          primaryOutput.connect(secondaryOutput);

          const expectedSummary = {
            outputNodeIds: [primaryOutput.geneId, secondaryOutput.geneId],
            diagnostics: {
              nodeCount: 3,
              edgeCount: 2,
              detectedCycles: false,
              activationOrder: [0, 1, 2],
            },
          };

          // Act
          const construction = Network.construct(
            [sensorInput, primaryOutput, secondaryOutput],
            {
              validate: { allowOutputNodeOutgoingEdges: true },
            },
          );
          const actualSummary = {
            outputNodeIds: construction.network.outputNodeIds,
            diagnostics: construction.diagnostics,
          };

          // Assert
          expect(actualSummary).toEqual(expectedSummary);
        });
      });
    });

    describe('given a graph intentionally uses output gating', () => {
      describe('when sink-only output validation is relaxed explicitly', () => {
        it('preserves the gated connection in the detached construct graph', () => {
          // Arrange
          const sensorInput = new Node('input');
          const hiddenCarrier = new Node('hidden');
          const primaryOutput = new Node('output');

          sensorInput.connect(hiddenCarrier);
          hiddenCarrier.connect(primaryOutput);
          primaryOutput.gate(sensorInput.connections.out[0]);

          const expectedSummary = {
            outputNodeIds: [primaryOutput.geneId],
            diagnostics: {
              nodeCount: 3,
              edgeCount: 2,
              detectedCycles: false,
              activationOrder: [0, 1, 2],
            },
            gatedConnection: {
              fromGeneId: sensorInput.geneId,
              toGeneId: hiddenCarrier.geneId,
              gaterGeneId: primaryOutput.geneId,
            },
          };

          // Act
          const construction = Network.construct(
            [sensorInput, hiddenCarrier, primaryOutput],
            {
              validate: { allowOutputNodeOutgoingEdges: true },
            },
          );
          const actualSummary = {
            outputNodeIds: construction.network.outputNodeIds,
            diagnostics: construction.diagnostics,
            gatedConnection: {
              fromGeneId: construction.graph.connections[0].fromGeneId,
              toGeneId: construction.graph.connections[0].toGeneId,
              gaterGeneId: construction.graph.connections[0].gaterGeneId,
            },
          };

          // Assert
          expect(actualSummary).toEqual(expectedSummary);
        });
      });
    });

    describe('given a cyclic hidden component is compiled in acyclic mode', () => {
      describe('when the graph is materialized', () => {
        it('rejects the back-connection with a concrete cycle path', () => {
          // Arrange
          const sensor = new Node('input');
          const hiddenStage = new Group(2);
          const readout = new Node('output');

          hiddenStage.nodes[0].describe({ label: 'hiddenA' });
          hiddenStage.nodes[1].describe({ label: 'hiddenB' });
          sensor.connect(hiddenStage);
          hiddenStage.nodes[0].connect(hiddenStage.nodes[1]);
          hiddenStage.nodes[1].connect(hiddenStage.nodes[0]);
          hiddenStage.connect(readout);

          // Act
          const constructAcyclicGraph = () => {
            Network.construct([hiddenStage, readout, sensor]);
          };

          // Assert
          expect(constructAcyclicGraph).toThrow(
            'Cycle path: "hiddenA" -> "hiddenB" -> "hiddenA".',
          );
        });
      });
    });

    describe('given a cyclic hidden component is compiled in recurrent mode', () => {
      describe('when the graph is materialized', () => {
        it('returns a recurrent runtime with carried-state scheduling diagnostics', () => {
          // Arrange
          const sensor = new Node('input');
          const hiddenStage = new Group(2);
          const readout = new Node('output');

          sensor.connect(hiddenStage);
          hiddenStage.nodes[0].connect(hiddenStage.nodes[1]);
          hiddenStage.nodes[1].connect(hiddenStage.nodes[0]);
          hiddenStage.connect(readout);

          const expectedSummary = {
            nodeRoles: ['input', 'hidden', 'hidden', 'output'],
            diagnostics: {
              nodeCount: 4,
              edgeCount: 6,
              detectedCycles: true,
              activationOrder: [0, 1, 2, 3],
            },
            graph: {
              requestedMode: 'recurrent',
              topologyIntent: 'unconstrained',
              inputNodeIds: [sensor.geneId],
              outputNodeIds: [readout.geneId],
              activationOrder: [0, 1, 2, 3],
              nodes: [
                {
                  index: 0,
                  geneId: sensor.geneId,
                  label: null,
                  role: 'input',
                  inputOrder: 0,
                  outputOrder: null,
                },
                {
                  index: 1,
                  geneId: hiddenStage.nodes[0].geneId,
                  label: null,
                  role: 'hidden',
                  inputOrder: null,
                  outputOrder: null,
                },
                {
                  index: 2,
                  geneId: hiddenStage.nodes[1].geneId,
                  label: null,
                  role: 'hidden',
                  inputOrder: null,
                  outputOrder: null,
                },
                {
                  index: 3,
                  geneId: readout.geneId,
                  label: null,
                  role: 'output',
                  inputOrder: null,
                  outputOrder: 0,
                },
              ],
              connections: [
                {
                  innovation: sensor.connections.out[0].innovation,
                  fromIndex: 0,
                  toIndex: 1,
                  fromGeneId: sensor.geneId,
                  toGeneId: hiddenStage.nodes[0].geneId,
                  gaterGeneId: null,
                  isSelfConnection: false,
                  enabled: true,
                  weight: sensor.connections.out[0].weight,
                },
                {
                  innovation: sensor.connections.out[1].innovation,
                  fromIndex: 0,
                  toIndex: 2,
                  fromGeneId: sensor.geneId,
                  toGeneId: hiddenStage.nodes[1].geneId,
                  gaterGeneId: null,
                  isSelfConnection: false,
                  enabled: true,
                  weight: sensor.connections.out[1].weight,
                },
                {
                  innovation:
                    hiddenStage.nodes[0].connections.out[0].innovation,
                  fromIndex: 1,
                  toIndex: 2,
                  fromGeneId: hiddenStage.nodes[0].geneId,
                  toGeneId: hiddenStage.nodes[1].geneId,
                  gaterGeneId: null,
                  isSelfConnection: false,
                  enabled: true,
                  weight: hiddenStage.nodes[0].connections.out[0].weight,
                },
                {
                  innovation:
                    hiddenStage.nodes[0].connections.out[1].innovation,
                  fromIndex: 1,
                  toIndex: 3,
                  fromGeneId: hiddenStage.nodes[0].geneId,
                  toGeneId: readout.geneId,
                  gaterGeneId: null,
                  isSelfConnection: false,
                  enabled: true,
                  weight: hiddenStage.nodes[0].connections.out[1].weight,
                },
                {
                  innovation:
                    hiddenStage.nodes[1].connections.out[0].innovation,
                  fromIndex: 2,
                  toIndex: 1,
                  fromGeneId: hiddenStage.nodes[1].geneId,
                  toGeneId: hiddenStage.nodes[0].geneId,
                  gaterGeneId: null,
                  isSelfConnection: false,
                  enabled: true,
                  weight: hiddenStage.nodes[1].connections.out[0].weight,
                },
                {
                  innovation:
                    hiddenStage.nodes[1].connections.out[1].innovation,
                  fromIndex: 2,
                  toIndex: 3,
                  fromGeneId: hiddenStage.nodes[1].geneId,
                  toGeneId: readout.geneId,
                  gaterGeneId: null,
                  isSelfConnection: false,
                  enabled: true,
                  weight: hiddenStage.nodes[1].connections.out[1].weight,
                },
              ],
            },
            scheduling: {
              topologyIntent: 'unconstrained',
              requestedMode: 'recurrent',
              topologyDirty: false,
              executionPath: 'compiled-schedule',
              issue: null,
              message:
                'Activation is using the compiled recurrent schedule with explicit recurrent-component steps and carried recurrent state.',
              inputNodeIds: [sensor.geneId],
              outputNodeIds: [readout.geneId],
              stepCount: 3,
              recurrentComponentCount: 1,
              stateSemantics: 'carry',
              cycleNodeIds: [],
              suggestions: [
                'Call clear() before a new independent sequence when carried recurrent state should reset.',
              ],
            },
          };

          // Act
          const construction = Network.construct(
            [hiddenStage, readout, sensor],
            { mode: 'recurrent' },
          );
          const actualSummary = {
            nodeRoles: construction.network.nodes.map((node) => node.type),
            diagnostics: construction.diagnostics,
            graph: construction.graph,
            scheduling:
              construction.network.getActivationSchedulingDiagnostics(),
          };

          // Assert
          expect(actualSummary).toEqual(expectedSummary);
        });
      });
    });
  });

  describe('formatConstructSummary()', () => {
    describe('given a labeled feed-forward construct result', () => {
      describe('when the summary is formatted', () => {
        it('returns one compact human-readable summary', () => {
          // Arrange
          const sensorLeft = new Node('input');
          const sensorRight = new Node('input');
          const hiddenStage = new Group(2);
          const readoutLayer = Layer.dense(1, 'output');
          const readoutNode = readoutLayer.nodes[0];

          sensorLeft.describe({ label: 'sensorLeft' });
          sensorRight.describe({ label: 'sensorRight' });
          readoutNode.describe({ label: 'readout' });

          sensorLeft.connect(hiddenStage);
          sensorRight.connect(hiddenStage);
          hiddenStage.connect(readoutLayer);

          const construction = Network.construct(
            [hiddenStage, sensorRight, readoutLayer, sensorLeft],
            {
              inputNodes: ['sensorLeft', 'sensorRight'],
              outputNodes: ['readout'],
            },
          );
          const expectedSummary = [
            'Construct summary',
            'Mode: acyclic (feed-forward)',
            'Graph: 5 nodes, 6 connections, cycles detected: no',
            'Roles: 2 input, 2 hidden, 1 output',
            `Inputs: [0] "sensorLeft" (geneId: ${sensorLeft.geneId}), [1] "sensorRight" (geneId: ${sensorRight.geneId})`,
            `Outputs: [0] "readout" (geneId: ${readoutNode.geneId})`,
            'Activation order: 0 -> 1 -> 2 -> 3 -> 4',
            `Connections: [0] geneId:${sensorLeft.geneId}[0] -> geneId:${hiddenStage.nodes[0].geneId}[2], [1] geneId:${sensorLeft.geneId}[0] -> geneId:${hiddenStage.nodes[1].geneId}[3], [2] geneId:${sensorRight.geneId}[1] -> geneId:${hiddenStage.nodes[0].geneId}[2], [3] geneId:${sensorRight.geneId}[1] -> geneId:${hiddenStage.nodes[1].geneId}[3], [4] geneId:${hiddenStage.nodes[0].geneId}[2] -> geneId:${readoutNode.geneId}[4], [5] geneId:${hiddenStage.nodes[1].geneId}[3] -> geneId:${readoutNode.geneId}[4]`,
          ].join('\n');

          // Act
          const actualSummary = formatConstructSummary(construction);

          // Assert
          expect(actualSummary).toBe(expectedSummary);
        });
      });
    });

    describe('given a recurrent construct result with unlabeled hidden nodes', () => {
      describe('when the summary is formatted', () => {
        it('reports the recurrent topology and cycle presence clearly', () => {
          // Arrange
          const sensor = new Node('input');
          const hiddenStage = new Group(2);
          const readout = new Node('output');

          sensor.connect(hiddenStage);
          hiddenStage.nodes[0].connect(hiddenStage.nodes[1]);
          hiddenStage.nodes[1].connect(hiddenStage.nodes[0]);
          hiddenStage.connect(readout);

          const construction = Network.construct(
            [hiddenStage, readout, sensor],
            { mode: 'recurrent' },
          );
          const expectedSummary = [
            'Construct summary',
            'Mode: recurrent (unconstrained)',
            'Graph: 4 nodes, 6 connections, cycles detected: yes',
            'Roles: 1 input, 2 hidden, 1 output',
            `Inputs: [0] geneId:${sensor.geneId} (geneId: ${sensor.geneId})`,
            `Outputs: [0] geneId:${readout.geneId} (geneId: ${readout.geneId})`,
            'Activation order: 0 -> 1 -> 2 -> 3',
            `Connections: [0] geneId:${sensor.geneId}[0] -> geneId:${hiddenStage.nodes[0].geneId}[1], [1] geneId:${sensor.geneId}[0] -> geneId:${hiddenStage.nodes[1].geneId}[2], [2] geneId:${hiddenStage.nodes[0].geneId}[1] -> geneId:${hiddenStage.nodes[1].geneId}[2], [3] geneId:${hiddenStage.nodes[0].geneId}[1] -> geneId:${readout.geneId}[3], [4] geneId:${hiddenStage.nodes[1].geneId}[2] -> geneId:${hiddenStage.nodes[0].geneId}[1], [5] geneId:${hiddenStage.nodes[1].geneId}[2] -> geneId:${readout.geneId}[3]`,
          ].join('\n');

          // Act
          const actualSummary = formatConstructSummary(construction);

          // Assert
          expect(actualSummary).toBe(expectedSummary);
        });
      });
    });
  });
});
