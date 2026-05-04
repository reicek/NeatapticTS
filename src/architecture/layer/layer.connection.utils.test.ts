import Connection from '../connection/connection';
import Group from '../group/group';
import * as methods from '../../methods/methods';
import Node from '../node/node';
import {
  clearLayer,
  connectLayer,
  disconnectLayer,
  gateLayer,
  inputLayer,
} from './layer.connection.utils';
import type { LayerConnectionContext, LayerLike } from './layer.utils.types';

describe('layer connection utility chapter', () => {
  describe('connectLayer', () => {
    describe('given the layer has no output group', () => {
      it('throws the missing-output connection error', () => {
        // Arrange
        const context = createLayerConnectionContext({ output: null });

        // Act and Assert
        expect(() => connectLayer(context, new Group(1))).toThrow(
          'Layer output is not defined. Cannot connect from this layer.',
        );
      });
    });

    describe('given the target is a group', () => {
      it('creates the forward connections from the output group', () => {
        // Arrange
        const context = createLayerConnectionContext();

        // Act
        const createdConnections = connectLayer(
          context,
          new Group(1),
          methods.groupConnection.ALL_TO_ALL,
        );

        // Assert
        expect(createdConnections).toHaveLength(2);
      });
    });

    describe('given the target is another layer', () => {
      it('delegates to the target input surface', () => {
        // Arrange
        const context = createLayerConnectionContext();
        const expectedConnections: Connection[] = [];
        const targetLayer: LayerLike = {
          input: jest.fn(() => expectedConnections),
          nodes: [],
          output: null,
        };

        // Act
        const createdConnections = connectLayer(context, targetLayer);

        // Assert
        expect(createdConnections).toBe(expectedConnections);
      });
    });

    describe('given the target is not a supported layer, group, or node', () => {
      it('returns an empty connection list', () => {
        // Arrange
        const context = createLayerConnectionContext();

        // Act
        const createdConnections = connectLayer(context, {} as Group);

        // Assert
        expect(createdConnections).toEqual([]);
      });
    });
  });

  describe('gateLayer', () => {
    describe('given the layer has no output group', () => {
      it('throws the missing-output gating error', () => {
        // Arrange
        const context = createLayerConnectionContext({ output: null });
        const gatedConnections = [
          new Connection(new Node('hidden'), new Node('hidden')),
        ];

        // Act and Assert
        expect(() =>
          gateLayer(context, gatedConnections, methods.gating.INPUT),
        ).toThrow('Layer output is not defined. Cannot gate from this layer.');
      });
    });

    describe('given the layer has an output group', () => {
      it('delegates gating to the output group', () => {
        // Arrange
        const context = createLayerConnectionContext();
        const gatedConnections = [
          new Connection(new Node('hidden'), new Node('hidden')),
        ];
        const gateSpy = jest
          .spyOn(context.output as Group, 'gate')
          .mockImplementation(() => undefined);

        try {
          // Act
          gateLayer(context, gatedConnections, methods.gating.INPUT);

          // Assert
          expect(gateSpy).toHaveBeenCalledWith(
            gatedConnections,
            methods.gating.INPUT,
          );
        } finally {
          gateSpy.mockRestore();
        }
      });
    });
  });

  describe('inputLayer', () => {
    describe('given the source is a group and no method is provided', () => {
      it('defaults to all-to-all wiring into the layer output', () => {
        // Arrange
        const context = createLayerConnectionContext();

        // Act
        const createdConnections = inputLayer(context, new Group(2));

        // Assert
        expect(createdConnections).toHaveLength(4);
      });
    });

    describe('given the layer has no output group', () => {
      it('throws the missing-input-target error', () => {
        // Arrange
        const context = createLayerConnectionContext({ output: null });

        // Act and Assert
        expect(() => inputLayer(context, new Group(1))).toThrow(
          'Layer output (acting as input target) is not defined.',
        );
      });
    });

    describe('given the source layer has no output group', () => {
      it('throws the missing-input-source error', () => {
        // Arrange
        const context = createLayerConnectionContext();
        const sourceLayer: LayerLike = {
          input: jest.fn(() => []),
          nodes: [],
          output: null,
        };

        // Act and Assert
        expect(() => inputLayer(context, sourceLayer)).toThrow(
          'Layer output (acting as input source) is not defined.',
        );
      });
    });
  });

  describe('disconnectLayer', () => {
    describe('given the target is a group without a twoSided argument', () => {
      it('uses the default one-sided disconnect', () => {
        // Arrange – no third arg → covers default-param branch (line 189)
        const context = createLayerConnectionContext();
        const targetGroup = new Group(1);
        const connection = context.nodes[0].connect(targetGroup.nodes[0])[0];
        context.connections.out.push(connection);

        // Act
        disconnectLayer(context, targetGroup);

        // Assert
        expect(context.connections.out).toHaveLength(0);
      });
    });

    describe('given the target is a group and two-sided cleanup is disabled', () => {
      it('removes outgoing entries without touching incoming entries', () => {
        // Arrange – covers line 234 FALSE arm (removeTwoSided = false)
        const context = createLayerConnectionContext();
        const targetGroup = new Group(1);
        const forwardConnection = context.nodes[0].connect(
          targetGroup.nodes[0],
        )[0];
        const reverseConnection = targetGroup.nodes[0].connect(
          context.nodes[0],
        )[0];
        context.connections.out.push(forwardConnection);
        context.connections.in.push(reverseConnection);

        // Act
        disconnectLayer(context, targetGroup, false);

        // Assert
        expect(context.connections.in).toHaveLength(1);
      });
    });

    describe('given incoming connections contain a non-matching entry after the target', () => {
      it('skips the non-matching connection and removes only the correct one', () => {
        // Arrange – line 317 FALSE arm: reverse scan checks index 1 (non-match) before index 0 (match)
        const context = createLayerConnectionContext();
        const targetNode = new Node('hidden');
        const unrelatedNode = new Node('hidden');
        // Reverse connection that matches: targetNode → layerNode[0]
        const matchingReverse = targetNode.connect(context.nodes[0])[0];
        // Non-matching: unrelatedNode → layerNode[0]
        const nonMatchingReverse = unrelatedNode.connect(context.nodes[0])[0];
        // Push matching at index 0, non-matching at index 1 → reverse iter checks index 1 first (FALSE arm)
        context.connections.in.push(matchingReverse, nonMatchingReverse);
        const forwardConn = context.nodes[0].connect(targetNode)[0];
        context.connections.out.push(forwardConn);

        // Act
        disconnectLayer(context, targetNode, true);

        // Assert – matching incoming removed, non-matching remains
        expect(context.connections.in).toHaveLength(1);
        expect(context.connections.in[0]).toBe(nonMatchingReverse);
      });
    });

    describe('given the target is a group and two-sided cleanup is enabled', () => {
      it('removes both outgoing and incoming layer tracking entries', () => {
        // Arrange
        const context = createLayerConnectionContext();
        const targetGroup = new Group(1);
        const firstConnection = context.nodes[0].connect(
          targetGroup.nodes[0],
        )[0];
        const secondConnection = context.nodes[1].connect(
          targetGroup.nodes[0],
        )[0];
        const reverseConnection = targetGroup.nodes[0].connect(
          context.nodes[0],
        )[0];
        context.connections.out.push(firstConnection, secondConnection);
        context.connections.in.push(reverseConnection);

        // Act
        disconnectLayer(context, targetGroup, true);

        // Assert
        expect({
          outgoingCount: context.connections.out.length,
          incomingCount: context.connections.in.length,
        }).toEqual({ outgoingCount: 0, incomingCount: 0 });
      });
    });

    describe('given the target is a single node and two-sided cleanup is disabled', () => {
      it('removes only the outgoing tracking entry', () => {
        // Arrange
        const context = createLayerConnectionContext();
        const targetNode = new Node('hidden');
        const forwardConnection = context.nodes[0].connect(targetNode)[0];
        const reverseConnection = targetNode.connect(context.nodes[0])[0];
        context.connections.out.push(forwardConnection);
        context.connections.in.push(reverseConnection);

        // Act
        disconnectLayer(context, targetNode, false);

        // Assert
        expect({
          outgoingCount: context.connections.out.length,
          incomingCount: context.connections.in.length,
        }).toEqual({ outgoingCount: 0, incomingCount: 1 });
      });
    });

    describe('given the target is a single node and two-sided cleanup is enabled', () => {
      it('removes both outgoing and incoming tracking entries', () => {
        // Arrange
        const context = createLayerConnectionContext();
        const targetNode = new Node('hidden');
        const forwardConnection = context.nodes[0].connect(targetNode)[0];
        const reverseConnection = targetNode.connect(context.nodes[0])[0];
        context.connections.out.push(forwardConnection);
        context.connections.in.push(reverseConnection);

        // Act
        disconnectLayer(context, targetNode, true);

        // Assert
        expect({
          outgoingCount: context.connections.out.length,
          incomingCount: context.connections.in.length,
        }).toEqual({ outgoingCount: 0, incomingCount: 0 });
      });
    });
  });

  describe('clearLayer', () => {
    describe('given the layer exposes ordinary nodes', () => {
      it('clears each node exactly once', () => {
        // Arrange
        const context = createLayerConnectionContext();
        const clearSpies = context.nodes.map((node) =>
          jest.spyOn(node, 'clear').mockImplementation(() => undefined),
        );

        // Act
        clearLayer(context);
        const everyNodeWasCleared = clearSpies.every(
          (clearSpy) => clearSpy.mock.calls.length === 1,
        );
        clearSpies.forEach((clearSpy) => clearSpy.mockRestore());

        // Assert
        expect(everyNodeWasCleared).toBe(true);
      });
    });
  });
});

function createLayerConnectionContext(input?: {
  output?: Group | null;
}): LayerConnectionContext {
  const output = input?.output === undefined ? new Group(2) : input.output;
  const layer: LayerLike = {
    input: jest.fn(() => []),
    nodes: output?.nodes ?? [],
    output,
  };

  return {
    connections: { in: [], out: [], self: [] },
    isLayer: (value: unknown): value is LayerLike =>
      typeof value === 'object' &&
      value !== null &&
      'input' in value &&
      typeof (value as LayerLike).input === 'function',
    layer,
    nodes: output?.nodes ?? [new Node('hidden'), new Node('hidden')],
    output,
  };
}
