import Connection from '../../connection';
import Group from '../../group';
import Layer from '../../layer';
import Node from '../../node';
import Network from '../network';
import type {
  ActivationSchedule,
  ActivationSchedulingDiagnostics,
  NetworkConstructorOptions,
  NetworkTopologyIntent,
} from '../network.types';
import * as topologyUtils from '../topology/network.topology.utils';
import { constructNetwork } from './network.construct.utils';
import type {
  ConstructOptions,
  ConstructPart,
} from './network.construct.utils.types';

type FakeConstructScenario = {
  activationSchedule?: ActivationSchedule | null;
  diagnostics?: Partial<ActivationSchedulingDiagnostics>;
  onClear?: (network: FakeConstructNetwork) => void;
  topoOrder?: Node[] | null;
};

class FakeConstructNetwork {
  static scenario: FakeConstructScenario = {};

  inputNodeIds: number[] = [];
  outputNodeIds: number[] = [];
  nodes: Node[] = [];
  connections: Connection[] = [];
  selfconns: Connection[] = [];
  gates: Connection[] = [];
  layers?: unknown[];
  _activationSchedule: ActivationSchedule | null = null;
  _activationSchedulingDiagnostics: ActivationSchedulingDiagnostics = {
    topologyIntent: 'feed-forward',
    requestedMode: 'acyclic',
    topologyDirty: false,
    executionPath: 'compiled-schedule',
    issue: null,
    message: 'Synthetic construct scheduling diagnostics.',
    inputNodeIds: [],
    outputNodeIds: [],
    stepCount: 0,
    recurrentComponentCount: 0,
    stateSemantics: null,
    cycleNodeIds: [],
    suggestions: [],
  };
  _topoOrder: Node[] | null = null;
  _topoDirty = true;
  _slabDirty = true;
  _adjDirty = true;
  _nodeIndexDirty = true;

  private _topologyIntent: NetworkTopologyIntent = 'feed-forward';

  constructor(
    readonly input: number,
    readonly output: number,
    readonly options?: NetworkConstructorOptions,
  ) {
    void this.input;
    void this.output;
    void this.options;
  }

  refreshExplicitIORoles(): void {
    this.inputNodeIds = this.nodes
      .filter((node) => node.type === 'input')
      .map((node) => node.geneId);
    this.outputNodeIds = this.nodes
      .filter((node) => node.type === 'output')
      .map((node) => node.geneId);
  }

  setTopologyIntent(intent: NetworkTopologyIntent): void {
    this._topologyIntent = intent;
  }

  getTopologyIntent(): NetworkTopologyIntent {
    return this._topologyIntent;
  }

  clear(): void {
    FakeConstructNetwork.scenario.onClear?.(this);
  }

  getActivationSchedulingDiagnostics(): ActivationSchedulingDiagnostics {
    return this._activationSchedulingDiagnostics;
  }
}

function createNode(role: 'input' | 'hidden' | 'output', label?: string): Node {
  const node = new Node(role);

  if (label) {
    node.describe({ label });
  }

  return node;
}

function constructWithRealNetwork(
  parts: readonly ConstructPart[],
  options: ConstructOptions = {},
) {
  return constructNetwork.call(
    Network as unknown as new (
      input: number,
      output: number,
      options?: NetworkConstructorOptions,
    ) => Network,
    parts,
    options,
  );
}

function createSchedulingDiagnostics(
  network: FakeConstructNetwork,
  overrides: Partial<ActivationSchedulingDiagnostics> = {},
): ActivationSchedulingDiagnostics {
  const requestedMode =
    network.getTopologyIntent() === 'feed-forward' ? 'acyclic' : 'recurrent';

  return {
    topologyIntent: network.getTopologyIntent(),
    requestedMode,
    topologyDirty: false,
    executionPath: 'compiled-schedule',
    issue: null,
    message: 'Synthetic construct scheduling diagnostics.',
    inputNodeIds: [...network.inputNodeIds],
    outputNodeIds: [...network.outputNodeIds],
    stepCount: network._activationSchedule?.steps.length ?? 0,
    recurrentComponentCount:
      network._activationSchedule?.steps.filter(
        (step) => step.kind === 'recurrent-component',
      ).length ?? 0,
    stateSemantics: network._activationSchedule?.stateSemantics ?? null,
    cycleNodeIds: [],
    suggestions: [],
    ...overrides,
  };
}

function constructWithFakeNetwork(
  parts: readonly ConstructPart[],
  options: ConstructOptions,
  scenario: FakeConstructScenario,
) {
  FakeConstructNetwork.scenario = scenario;

  const computeTopoOrderSpy = jest
    .spyOn(topologyUtils, 'computeTopoOrder')
    .mockImplementation(function (this: FakeConstructNetwork) {
      this._activationSchedule = scenario.activationSchedule ?? null;
      this._topoOrder = scenario.topoOrder ?? null;
      this._activationSchedulingDiagnostics = createSchedulingDiagnostics(
        this,
        scenario.diagnostics,
      );
    });

  try {
    return constructNetwork.call(
      FakeConstructNetwork as unknown as new (
        input: number,
        output: number,
        options?: NetworkConstructorOptions,
      ) => Network,
      parts,
      options,
    );
  } finally {
    computeTopoOrderSpy.mockRestore();
    FakeConstructNetwork.scenario = {};
  }
}

jest.retryTimes(2, { logErrorsBeforeRetry: true });

describe('network construct utility chapter', () => {
  describe('constructNetwork()', () => {
    describe('given the parts list includes one unsupported runtime value', () => {
      it('ignores the unsupported entry while materializing the supported nodes', () => {
        // Arrange
        const sensorInput = createNode('input', 'sensorInput');
        const readout = createNode('output', 'readout');

        // Act
        const construction = constructWithRealNetwork(
          [
            sensorInput,
            readout,
            { unsupported: true } as unknown as ConstructPart,
          ],
          {},
        );

        // Assert
        expect({
          nodeCount: construction.diagnostics.nodeCount,
          nodeRoles: construction.network.nodes.map((node) => node.type),
        }).toEqual({
          nodeCount: 2,
          nodeRoles: ['input', 'output'],
        });
      });
    });

    describe('given an explicit numeric input id does not match any provided node', () => {
      it('throws the numeric node-id resolution error', () => {
        // Arrange
        const sensorInput = createNode('input', 'sensorInput');
        const readout = createNode('output', 'readout');
        const constructWithUnknownInputId = () => {
          constructWithRealNetwork([sensorInput, readout], {
            inputNodes: [999_999],
          });
        };

        // Act / Assert
        expect(constructWithUnknownInputId).toThrow(
          'Could not resolve input node id 999999 from the provided parts.',
        );
      });
    });

    describe('given an explicit output label does not match any provided node', () => {
      it('throws the missing-label node-id resolution error', () => {
        // Arrange
        const sensorInput = createNode('input', 'sensorInput');
        const readout = createNode('output', 'readout');
        const constructWithUnknownOutputLabel = () => {
          constructWithRealNetwork([sensorInput, readout], {
            outputNodes: ['missingOutput'],
          });
        };

        // Act / Assert
        expect(constructWithUnknownOutputLabel).toThrow(
          'Could not resolve output node id "missingOutput" from the provided parts.',
        );
      });
    });

    describe('given explicit input ids omit one provided input node', () => {
      it('rejects the partial public input selection with the missing label', () => {
        // Arrange
        const sensorLeft = createNode('input', 'sensorLeft');
        const sensorRight = createNode('input', 'sensorRight');
        const readout = createNode('output', 'readout');
        const constructWithIncompleteInputCoverage = () => {
          constructWithRealNetwork([sensorLeft, sensorRight, readout], {
            inputNodes: ['sensorLeft'],
          });
        };

        // Act / Assert
        expect(constructWithIncompleteInputCoverage).toThrow(
          'Explicit input node ids must cover every input node in the provided parts. Missing: "sensorRight".',
        );
      });
    });

    describe('given the provided parts contain no input-role nodes', () => {
      it('rejects the construct request before runtime materialization', () => {
        // Arrange
        const hiddenNode = createNode('hidden', 'hiddenNode');
        const readout = createNode('output', 'readout');
        const constructWithoutInputs = () => {
          constructWithRealNetwork([hiddenNode, readout]);
        };

        // Act / Assert
        expect(constructWithoutInputs).toThrow(
          'Constructed graph must include at least one input node.',
        );
      });
    });

    describe('given the provided parts contain no output-role nodes', () => {
      it('rejects the construct request before runtime materialization', () => {
        // Arrange
        const sensorInput = createNode('input', 'sensorInput');
        const hiddenNode = createNode('hidden', 'hiddenNode');
        const constructWithoutOutputs = () => {
          constructWithRealNetwork([sensorInput, hiddenNode]);
        };

        // Act / Assert
        expect(constructWithoutOutputs).toThrow(
          'Constructed graph must include at least one output node.',
        );
      });
    });

    describe('given one malformed node resolves as both input and output across explicit role resolution', () => {
      it('rejects the overlapping public role selection', () => {
        // Arrange
        const sharedRoleNode = createNode('input', 'sharedRoleNode');
        let typeReadCount = 0;
        Object.defineProperty(sharedRoleNode, 'type', {
          configurable: true,
          get: () => {
            typeReadCount += 1;
            return typeReadCount === 1 ? 'input' : 'output';
          },
        });

        const constructWithOverlappingRoles = () => {
          constructWithRealNetwork([sharedRoleNode], {
            inputNodes: [sharedRoleNode.geneId],
            outputNodes: [sharedRoleNode.geneId],
          });
        };

        // Act / Assert
        expect(constructWithOverlappingRoles).toThrow(
          'Input and output node selections must not overlap. Shared nodes: "sharedRoleNode".',
        );
      });
    });

    describe('given self-edge validation is enabled on a hidden node', () => {
      it('rejects the construct graph with the self-edge error', () => {
        // Arrange
        const sensorInput = createNode('input', 'sensorInput');
        const hiddenCarrier = createNode('hidden', 'hiddenCarrier');
        const readout = createNode('output', 'readout');

        sensorInput.connect(readout);
        hiddenCarrier.connect(hiddenCarrier);

        const constructWithSelfEdgeValidation = () => {
          constructWithRealNetwork([sensorInput, hiddenCarrier, readout], {
            validate: { forbidSelfEdges: true },
          });
        };

        // Act / Assert
        expect(constructWithSelfEdgeValidation).toThrow(
          'Constructed graph contains a self edge on node "hiddenCarrier" while self-edge validation is enabled.',
        );
      });
    });

    describe('given duplicate edges are permitted explicitly', () => {
      it('keeps both parallel edges and orders them by innovation id', () => {
        // Arrange
        const sensorInput = createNode('input', 'sensorInput');
        const readout = createNode('output', 'readout');

        sensorInput.connect(readout);
        sensorInput.connect(readout);

        // Act
        const construction = constructWithRealNetwork([sensorInput, readout], {
          validate: { forbidDuplicateEdges: false },
        });

        // Assert
        expect({
          edgeCount: construction.diagnostics.edgeCount,
          innovations: construction.graph.connections.map(
            (connection) => connection.innovation,
          ),
        }).toEqual({
          edgeCount: 2,
          innovations: sensorInput.connections.out
            .map((connection) => connection.innovation)
            .toSorted((leftInnovation, rightInnovation) => {
              return leftInnovation - rightInnovation;
            }),
        });
      });
    });

    describe('given duplicate edges remain forbidden', () => {
      it('rejects the duplicate source-to-target edge pair', () => {
        // Arrange
        const sensorInput = createNode('input', 'sensorInput');
        const readout = createNode('output', 'readout');

        sensorInput.connect(readout);
        sensorInput.connect(readout);

        const constructWithDuplicateEdges = () => {
          constructWithRealNetwork([sensorInput, readout]);
        };

        // Act / Assert
        expect(constructWithDuplicateEdges).toThrow(
          'Constructed graph contains duplicate edges from "sensorInput" to "readout".',
        );
      });
    });

    describe('given one referenced connection targets a node outside the provided parts', () => {
      it('reports the missing target node directly', () => {
        // Arrange
        const sensorInput = createNode('input', 'sensorInput');
        const hiddenCarrier = createNode('hidden', 'hiddenCarrier');
        const readout = createNode('output', 'readout');

        sensorInput.connect(hiddenCarrier);

        const constructWithMissingTargetNode = () => {
          constructWithRealNetwork([sensorInput, readout]);
        };

        // Act / Assert
        expect(constructWithMissingTargetNode).toThrow(
          'Connection references target node "hiddenCarrier" that were not included in parts.',
        );
      });
    });

    describe('given one referenced connection is still gated by a node outside the provided parts', () => {
      it('reports the missing gater node directly', () => {
        // Arrange
        const sensorInput = createNode('input', 'sensorInput');
        const hiddenCarrier = createNode('hidden', 'hiddenCarrier');
        const readout = createNode('output', 'readout');
        const externalGater = createNode('hidden', 'externalGater');

        sensorInput.connect(hiddenCarrier);
        hiddenCarrier.connect(readout);
        externalGater.gate(sensorInput.connections.out[0]);

        const constructWithMissingGaterNode = () => {
          constructWithRealNetwork([sensorInput, hiddenCarrier, readout]);
        };

        // Act / Assert
        expect(constructWithMissingGaterNode).toThrow(
          'Connection references gater node "externalGater" that were not included in parts.',
        );
      });
    });

    describe('given hidden isolation is allowed explicitly', () => {
      it('constructs the runtime even when one hidden node carries no edges', () => {
        // Arrange
        const sensorInput = createNode('input', 'sensorInput');
        const isolatedHidden = createNode('hidden', 'isolatedHidden');
        const readout = createNode('output', 'readout');

        // Act
        const construction = constructWithRealNetwork(
          [sensorInput, isolatedHidden, readout],
          { allowIsolatedHiddenNodes: true },
        );

        // Assert
        expect({
          edgeCount: construction.diagnostics.edgeCount,
          nodeRoles: construction.network.nodes.map((node) => node.type),
        }).toEqual({
          edgeCount: 0,
          nodeRoles: ['input', 'hidden', 'output'],
        });
      });
    });

    describe('given one hidden node only carries a self connection', () => {
      it('counts the self-connected hidden node as non-isolated without double-counting its endpoint', () => {
        // Arrange
        const sensorInput = createNode('input', 'sensorInput');
        const hiddenCarrier = createNode('hidden', 'hiddenCarrier');
        const readout = createNode('output', 'readout');

        sensorInput.connect(readout);
        hiddenCarrier.connect(hiddenCarrier);

        // Act
        const construction = constructWithRealNetwork(
          [sensorInput, hiddenCarrier, readout],
          {},
        );

        // Assert
        expect({
          edgeCount: construction.diagnostics.edgeCount,
          selfConnectionFlags: construction.graph.connections.map(
            (connection) => connection.isSelfConnection,
          ),
        }).toEqual({
          edgeCount: 2,
          selfConnectionFlags: [false, true],
        });
      });
    });

    describe('given hidden isolation is not allowed', () => {
      it('rejects the disconnected hidden node set', () => {
        // Arrange
        const sensorInput = createNode('input', 'sensorInput');
        const isolatedHidden = createNode('hidden', 'isolatedHidden');
        const readout = createNode('output', 'readout');
        const constructWithIsolatedHiddenNode = () => {
          constructWithRealNetwork([sensorInput, isolatedHidden, readout]);
        };

        // Act / Assert
        expect(constructWithIsolatedHiddenNode).toThrow(
          'Hidden nodes must participate in at least one edge unless allowIsolatedHiddenNodes is true. Isolated nodes: "isolatedHidden".',
        );
      });
    });

    describe('given a gated connection loses its gater identity during error formatting', () => {
      it('falls back to the plain connection identity in the validation message', () => {
        // Arrange
        const sensorInput = createNode('input', 'sensorInput');
        const hiddenCarrier = createNode('hidden', 'hiddenCarrier');
        const primaryOutput = createNode('output', 'primaryOutput');
        let gaterReadCount = 0;

        sensorInput.connect(hiddenCarrier);
        hiddenCarrier.connect(primaryOutput);
        primaryOutput.gate(sensorInput.connections.out[0]);

        Object.defineProperty(sensorInput.connections.out[0], 'gater', {
          configurable: true,
          get: () => {
            gaterReadCount += 1;
            return gaterReadCount <= 4 ? primaryOutput : null;
          },
        });

        const constructWithTransientGaterIdentity = () => {
          constructWithRealNetwork([sensorInput, hiddenCarrier, primaryOutput]);
        };

        // Act / Assert
        expect(constructWithTransientGaterIdentity).toThrow(
          'Output nodes must be pure sinks unless validate.allowOutputNodeOutgoingEdges is true. Gated connections: "sensorInput" -> "hiddenCarrier".',
        );
      });
    });

    describe('given one unresolved referenced node has no label', () => {
      it('formats the missing node identity through its gene id in the validation error', () => {
        // Arrange
        const sensorInput = createNode('input', 'sensorInput');
        const hiddenCarrier = createNode('hidden');
        const readout = createNode('output', 'readout');

        sensorInput.connect(hiddenCarrier);

        const constructWithUnlabeledMissingTarget = () => {
          constructWithRealNetwork([sensorInput, readout]);
        };

        // Act / Assert
        expect(constructWithUnlabeledMissingTarget).toThrow(
          `Connection references target node geneId:${hiddenCarrier.geneId} that were not included in parts.`,
        );
      });
    });
  });

  describe('constructNetwork() fallback diagnostics', () => {
    describe('given acyclic validation receives a cycle report without implicated node ids', () => {
      it('throws the base cycle-mode guidance without a cycle suffix', () => {
        // Arrange
        const sensorInput = createNode('input', 'sensorInput');
        const readout = createNode('output', 'readout');
        sensorInput.connect(readout);

        const constructWithEmptyCycleIds = () => {
          constructWithFakeNetwork(
            [sensorInput, readout],
            {},
            {
              diagnostics: {
                issue: 'cycle-detected',
                executionPath: 'cycle-fallback-order',
                cycleNodeIds: [],
                message: 'Synthetic cycle fallback.',
              },
            },
          );
        };

        // Act / Assert
        expect(constructWithEmptyCycleIds).toThrow(
          'Constructed graph contains a cycle while mode is "acyclic". Use mode: "recurrent" or remove the reported back-connections.',
        );
      });
    });

    describe('given acyclic validation receives cycle node ids without a traversable cycle path', () => {
      it('falls back to reporting the implicated cycle node ids directly', () => {
        // Arrange
        const sensorInput = createNode('input', 'sensorInput');
        const hiddenA = createNode('hidden', 'hiddenA');
        const hiddenB = createNode('hidden', 'hiddenB');
        const readout = createNode('output', 'readout');

        sensorInput.connect(hiddenA);
        hiddenA.connect(hiddenB);
        hiddenA.connect(readout);
        hiddenB.connect(readout);

        const constructWithFallbackCycleIds = () => {
          constructWithFakeNetwork(
            [sensorInput, hiddenA, hiddenB, readout],
            {},
            {
              diagnostics: {
                issue: 'cycle-detected',
                executionPath: 'cycle-fallback-order',
                cycleNodeIds: [hiddenA.geneId, hiddenB.geneId, readout.geneId],
                message: 'Synthetic cycle fallback.',
              },
            },
          );
        };

        // Act / Assert
        expect(constructWithFallbackCycleIds).toThrow(
          `Constructed graph contains a cycle while mode is "acyclic". Use mode: "recurrent" or remove the reported back-connections. Cycle nodes: ${hiddenA.geneId}, ${hiddenB.geneId}, ${readout.geneId}.`,
        );
      });
    });

    describe('given the compiled schedule references one missing node id but a topo order still exists', () => {
      it('falls back to topo-order activation indices while using array-index fallback for an isolated node snapshot', () => {
        // Arrange
        const sensorInput = createNode('input', 'sensorInput');
        const isolatedHidden = createNode('hidden', 'isolatedHidden');
        const readout = createNode('output', 'readout');

        sensorInput.connect(readout);

        // Act
        const construction = constructWithFakeNetwork(
          [sensorInput, isolatedHidden, readout],
          { allowIsolatedHiddenNodes: true },
          {
            onClear: (network) => {
              network.nodes[1].index = undefined as unknown as number;
            },
            activationSchedule: {
              mode: 'acyclic',
              steps: [{ kind: 'wave', nodeIds: [999_999] }],
              outputNodeIds: [readout.geneId],
            },
            topoOrder: [isolatedHidden, readout],
          },
        );

        // Assert
        expect({
          activationOrder: construction.diagnostics.activationOrder,
          hiddenNodeIndex: construction.graph.nodes[1].index,
        }).toEqual({
          activationOrder: [2],
          hiddenNodeIndex: 1,
        });
      });
    });

    describe('given no compiled schedule or topo-order cache is available', () => {
      it('falls back to the raw runtime node order for activation diagnostics', () => {
        // Arrange
        const sensorInput = createNode('input', 'sensorInput');
        const readout = createNode('output', 'readout');

        sensorInput.connect(readout);

        // Act
        const construction = constructWithFakeNetwork(
          [sensorInput, readout],
          {},
          {},
        );

        // Assert
        expect(construction.diagnostics.activationOrder).toEqual([0, 1]);
      });
    });

    describe('given a recurrent-component schedule omits its explicit iteration count', () => {
      it('defaults the recurrent-component step to one traversal pass', () => {
        // Arrange
        const sensorInput = createNode('input', 'sensorInput');
        const readout = createNode('output', 'readout');

        sensorInput.connect(readout);

        // Act
        const construction = constructWithFakeNetwork(
          [sensorInput, readout],
          { mode: 'recurrent' },
          {
            activationSchedule: {
              mode: 'recurrent',
              steps: [
                {
                  kind: 'recurrent-component',
                  nodeIds: [sensorInput.geneId, readout.geneId],
                },
              ],
              outputNodeIds: [readout.geneId],
              stateSemantics: 'carry',
            },
          },
        );

        // Assert
        expect(construction.diagnostics.activationOrder).toEqual([0, 1]);
      });
    });

    describe('given one connected node loses its runtime index before graph snapshotting', () => {
      it('throws the unresolved construct-node index error', () => {
        // Arrange
        const sensorInput = createNode('input', 'sensorInput');
        const readout = createNode('output', 'readout');

        sensorInput.connect(readout);

        const constructWithMissingRuntimeIndex = () => {
          constructWithFakeNetwork(
            [sensorInput, readout],
            {},
            {
              onClear: (network) => {
                network.nodes[0].index = undefined as unknown as number;
              },
            },
          );
        };

        // Act / Assert
        expect(constructWithMissingRuntimeIndex).toThrow(
          'Construct graph snapshot could not resolve a runtime index for node "sensorInput".',
        );
      });
    });
  });
});
