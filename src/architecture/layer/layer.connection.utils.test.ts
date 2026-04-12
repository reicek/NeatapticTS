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
    describe('given the target is a group and two-sided cleanup is enabled', () => {
      it('removes both outgoing and incoming layer tracking entries', () => {
        // Arrange
        const context = createLayerConnectionContext();
        const targetGroup = new Group(1);
        const firstConnection = context.nodes[0].connect(targetGroup.nodes[0])[0];
        const secondConnection = context.nodes[1].connect(targetGroup.nodes[0])[0];
        const reverseConnection = targetGroup.nodes[0].connect(context.nodes[0])[0];
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