import { jest } from '@jest/globals';

import type Connection from '../connection/connection';
import Layer from '../layer/layer';
import Node from '../node/node';
import { config } from '../../config';
import * as methods from '../../methods/methods';
import { GroupOneToOneSizeMismatchError } from './group.errors';
import Group from './group';
import {
  collectUniqueSourceNodes,
  connectAllToAll,
  connectGroupToGroup,
  connectGroupToLayer,
  connectGroupToNode,
  connectOneToOne,
  disconnectGroupFromGroup,
  disconnectGroupFromNode,
  gateByInput,
  gateByOutput,
  gateBySelf,
  removeInboundConnection,
  removeOutboundConnection,
  resolveDefaultGroupConnectionMethod,
} from './group.utils';

describe('group structural helper chapter', () => {
  afterEach(() => {
    config.warnings = false;
  });

  describe('resolveDefaultGroupConnectionMethod', () => {
    describe('given warnings are enabled and disabled across distinct and self targets', () => {
      it('chooses the documented defaults and only warns when configured to do so', () => {
        const sourceGroup = new Group(1);
        const targetGroup = new Group(1);
        const warningSpy = jest
          .spyOn(console, 'warn')
          .mockImplementation(() => undefined);

        try {
          config.warnings = true;
          const warnedDistinctMethod = resolveDefaultGroupConnectionMethod(
            sourceGroup,
            targetGroup,
          );
          const warnedSelfMethod = resolveDefaultGroupConnectionMethod(
            sourceGroup,
            sourceGroup,
          );

          config.warnings = false;
          const quietDistinctMethod = resolveDefaultGroupConnectionMethod(
            sourceGroup,
            targetGroup,
          );
          const quietSelfMethod = resolveDefaultGroupConnectionMethod(
            sourceGroup,
            sourceGroup,
          );

          expect({
            quietDistinctMethod,
            quietSelfMethod,
            warnedDistinctMethod,
            warnedSelfMethod,
            warningMessages: warningSpy.mock.calls.map(([message]) => message),
          }).toStrictEqual({
            quietDistinctMethod: methods.groupConnection.ALL_TO_ALL,
            quietSelfMethod: methods.groupConnection.ONE_TO_ONE,
            warnedDistinctMethod: methods.groupConnection.ALL_TO_ALL,
            warnedSelfMethod: methods.groupConnection.ONE_TO_ONE,
            warningMessages: [
              'No group connection specified, using ALL_TO_ALL by default.',
              'Connecting group to itself, using ONE_TO_ONE by default.',
            ],
          });
        } finally {
          warningSpy.mockRestore();
        }
      });
    });
  });

  describe('connectAllToAll', () => {
    describe('given one group connects to itself with all-to-else semantics', () => {
      it('skips self-pairs while registering the remaining bookkeeping entries', () => {
        const group = new Group(2);
        const createdConnections = connectAllToAll(
          group,
          group,
          methods.groupConnection.ALL_TO_ELSE,
          0.25,
        );

        expect({
          created: createdConnections.length,
          inbound: group.connections.in.length,
          outbound: group.connections.out.length,
        }).toStrictEqual({
          created: 2,
          inbound: 2,
          outbound: 2,
        });
      });
    });
  });

  describe('connectOneToOne', () => {
    describe('given source and target groups have different sizes', () => {
      it('throws the dedicated size-mismatch error', () => {
        expect(() => connectOneToOne(new Group(1), new Group(2), 0.1)).toThrow(
          GroupOneToOneSizeMismatchError,
        );
      });
    });

    describe('given one group connects one-to-one to itself', () => {
      it('stores the created links on the self-connection shelf', () => {
        const group = new Group(2);
        const createdConnections = connectOneToOne(group, group, 0.5);

        expect({
          created: createdConnections.length,
          selfConnections: group.connections.self.length,
        }).toStrictEqual({
          created: 2,
          selfConnections: 2,
        });
      });
    });
  });

  describe('connectGroupToGroup', () => {
    describe('given explicit and implicit connection methods are both exercised', () => {
      it('dispatches to one-to-one and default all-to-all wiring correctly', () => {
        const explicitSourceGroup = new Group(2);
        const explicitTargetGroup = new Group(2);
        const implicitSourceGroup = new Group(2);
        const implicitTargetGroup = new Group(2);

        config.warnings = false;
        const oneToOneConnections = connectGroupToGroup(
          explicitSourceGroup,
          explicitTargetGroup,
          methods.groupConnection.ONE_TO_ONE,
          0.4,
        );
        const defaultConnections = connectGroupToGroup(
          implicitSourceGroup,
          implicitTargetGroup,
          undefined,
          0.3,
        );

        expect({
          defaultCreated: defaultConnections.length,
          defaultTargetInbound: implicitTargetGroup.connections.in.length,
          oneToOneCreated: oneToOneConnections.length,
          oneToOneTargetInbound: explicitTargetGroup.connections.in.length,
        }).toStrictEqual({
          defaultCreated: 4,
          defaultTargetInbound: 4,
          oneToOneCreated: 2,
          oneToOneTargetInbound: 2,
        });
      });
    });
  });

  describe('connectGroupToNode and connectGroupToLayer', () => {
    describe('given node and layer targets are available', () => {
      it('creates node connections and forwards layer connections through the layer input hook', () => {
        const sourceGroup = new Group(2);
        const targetNode = new Node('hidden');
        const delegatedConnections = [
          sourceGroup.nodes[0].connect(new Node('hidden'))[0],
        ];
        const targetLayer = {
          input: jest.fn(() => delegatedConnections),
        } as unknown as Layer;

        const nodeConnections = connectGroupToNode(
          sourceGroup,
          targetNode,
          0.2,
        );
        const layerConnections = connectGroupToLayer(
          sourceGroup,
          targetLayer,
          methods.groupConnection.ALL_TO_ALL,
          0.7,
        );

        expect({
          delegatedArguments: (targetLayer.input as jest.Mock).mock.calls,
          layerConnections,
          nodeConnections: nodeConnections.length,
          outbound: sourceGroup.connections.out.length,
        }).toStrictEqual({
          delegatedArguments: [
            [sourceGroup, methods.groupConnection.ALL_TO_ALL, 0.7],
          ],
          layerConnections: delegatedConnections,
          nodeConnections: 2,
          outbound: 2,
        });
      });
    });
  });

  describe('collectUniqueSourceNodes', () => {
    describe('given multiple connections share the same source node', () => {
      it('preserves first-seen source-node order while removing duplicates', () => {
        const sourceNodeOne = new Node('hidden');
        const sourceNodeTwo = new Node('hidden');
        const connections = [
          sourceNodeOne.connect(new Node('hidden'))[0],
          sourceNodeOne.connect(new Node('hidden'))[0],
          sourceNodeTwo.connect(new Node('hidden'))[0],
        ];

        expect(collectUniqueSourceNodes(connections)).toStrictEqual([
          sourceNodeOne,
          sourceNodeTwo,
        ]);
      });
    });
  });

  describe('gateByInput, gateByOutput, and gateBySelf', () => {
    describe('given input, output, and self gating fixtures', () => {
      it('assigns gaters only to the targeted connections across all gating modes', () => {
        const inputGatingGroup = new Group(2);
        const inputConnections = [
          new Node('hidden').connect(new Node('hidden'))[0],
          new Node('hidden').connect(new Node('hidden'))[0],
          new Node('hidden').connect(new Node('hidden'))[0],
        ];

        gateByInput(inputGatingGroup, inputConnections);

        const outputGatingGroup = new Group(1);
        const outputSourceNode = new Node('hidden');
        const selectedOutputConnectionOne = outputSourceNode.connect(
          new Node('hidden'),
        )[0];
        const selectedOutputConnectionTwo = outputSourceNode.connect(
          new Node('hidden'),
        )[0];
        const unselectedOutputConnection = outputSourceNode.connect(
          new Node('hidden'),
        )[0];

        gateByOutput(
          outputGatingGroup,
          [selectedOutputConnectionOne, selectedOutputConnectionTwo],
          [outputSourceNode],
        );

        const selfGatingGroup = new Group(2);
        const arraySelfNode = new Node('hidden');
        const arraySelfConnection = arraySelfNode.connect(arraySelfNode)[0];
        const legacySelfNode = new Node('hidden');
        const legacySelfConnection = legacySelfNode.connect(legacySelfNode)[0];
        (legacySelfNode.connections as { self: unknown }).self =
          legacySelfConnection;

        gateBySelf(
          selfGatingGroup,
          [arraySelfConnection],
          [arraySelfNode, legacySelfNode],
        );

        expect({
          inputGaters: inputConnections.map(
            (candidateConnection) => candidateConnection.gater,
          ),
          outputGaters: [
            selectedOutputConnectionOne.gater,
            selectedOutputConnectionTwo.gater,
            unselectedOutputConnection.gater,
          ],
          selfGaters: [arraySelfConnection.gater, legacySelfConnection.gater],
        }).toStrictEqual({
          inputGaters: [
            inputGatingGroup.nodes[0],
            inputGatingGroup.nodes[1],
            inputGatingGroup.nodes[0],
          ],
          outputGaters: [
            outputGatingGroup.nodes[0],
            outputGatingGroup.nodes[0],
            null,
          ],
          selfGaters: [selfGatingGroup.nodes[0], null],
        });
      });
    });
  });

  describe('removeOutboundConnection and removeInboundConnection', () => {
    describe('given a bookkeeping list contains matched and unmatched edges', () => {
      it('removes only the first matching entry from each list', () => {
        const sourceNode = new Node('hidden');
        const targetNode = new Node('hidden');
        const matchedOutboundConnection = sourceNode.connect(targetNode)[0];
        const unmatchedOutboundConnection = sourceNode.connect(
          new Node('hidden'),
        )[0];
        const matchedInboundConnection = new Node('hidden').connect(
          targetNode,
        )[0];
        const unmatchedInboundConnection = new Node('hidden').connect(
          new Node('hidden'),
        )[0];
        const outboundConnections = [
          matchedOutboundConnection,
          unmatchedOutboundConnection,
        ];
        const inboundConnections = [
          matchedInboundConnection,
          unmatchedInboundConnection,
        ];

        removeOutboundConnection(outboundConnections, sourceNode, targetNode);
        removeInboundConnection(
          inboundConnections,
          matchedInboundConnection.from,
          targetNode,
        );

        expect({
          inboundConnections,
          outboundConnections,
        }).toStrictEqual({
          inboundConnections: [unmatchedInboundConnection],
          outboundConnections: [unmatchedOutboundConnection],
        });
      });
    });
  });

  describe('disconnectGroupFromGroup', () => {
    describe('given one-sided and two-sided reciprocal group links exist', () => {
      it('removes only the bookkeeping entries implied by the disconnect mode', () => {
        const oneSidedSourceGroup = new Group(1);
        const oneSidedTargetGroup = new Group(1);
        oneSidedSourceGroup.connect(
          oneSidedTargetGroup,
          methods.groupConnection.ONE_TO_ONE,
        );
        disconnectGroupFromGroup(
          oneSidedSourceGroup,
          oneSidedTargetGroup,
          false,
        );

        const twoSidedSourceGroup = new Group(1);
        const twoSidedTargetGroup = new Group(1);
        twoSidedSourceGroup.connect(
          twoSidedTargetGroup,
          methods.groupConnection.ONE_TO_ONE,
        );
        twoSidedTargetGroup.connect(
          twoSidedSourceGroup,
          methods.groupConnection.ONE_TO_ONE,
        );
        disconnectGroupFromGroup(
          twoSidedSourceGroup,
          twoSidedTargetGroup,
          true,
        );

        expect({
          oneSided: {
            sourceInbound: oneSidedSourceGroup.connections.in.length,
            sourceOutbound: oneSidedSourceGroup.connections.out.length,
            targetInbound: oneSidedTargetGroup.connections.in.length,
            targetOutbound: oneSidedTargetGroup.connections.out.length,
          },
          twoSided: {
            sourceInbound: twoSidedSourceGroup.connections.in.length,
            sourceOutbound: twoSidedSourceGroup.connections.out.length,
            targetInbound: twoSidedTargetGroup.connections.in.length,
            targetOutbound: twoSidedTargetGroup.connections.out.length,
          },
        }).toStrictEqual({
          oneSided: {
            sourceInbound: 0,
            sourceOutbound: 0,
            targetInbound: 0,
            targetOutbound: 0,
          },
          twoSided: {
            sourceInbound: 0,
            sourceOutbound: 0,
            targetInbound: 0,
            targetOutbound: 0,
          },
        });
      });
    });
  });

  describe('disconnectGroupFromNode', () => {
    describe('given one-sided and two-sided node links exist', () => {
      it('removes outbound bookkeeping and optionally removes reciprocal inbound links', () => {
        const oneSidedGroup = new Group(2);
        const oneSidedTargetNode = new Node('hidden');
        connectGroupToNode(oneSidedGroup, oneSidedTargetNode, 0.2);
        disconnectGroupFromNode(oneSidedGroup, oneSidedTargetNode, false);

        const twoSidedGroup = new Group(2);
        const twoSidedTargetNode = new Node('hidden');
        connectGroupToNode(twoSidedGroup, twoSidedTargetNode, 0.2);
        const reciprocalInboundConnections = twoSidedTargetNode.connect(
          twoSidedGroup,
        ) as Connection[];
        twoSidedGroup.connections.in.push(...reciprocalInboundConnections);
        disconnectGroupFromNode(twoSidedGroup, twoSidedTargetNode, true);

        expect({
          oneSided: {
            inbound: oneSidedGroup.connections.in.length,
            outbound: oneSidedGroup.connections.out.length,
          },
          twoSided: {
            inbound: twoSidedGroup.connections.in.length,
            outbound: twoSidedGroup.connections.out.length,
          },
        }).toStrictEqual({
          oneSided: {
            inbound: 0,
            outbound: 0,
          },
          twoSided: {
            inbound: 0,
            outbound: 0,
          },
        });
      });
    });
  });
});
