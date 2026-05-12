import Connection from '../../connection';
import Node from '../../node/node';
import { activationArrayPool } from '../../activationArrayPool/activationArrayPool';
import type Network from '../network';
import {
  buildEvolutionaryPruneSelection,
  buildEvolutionaryTarget,
  disconnectEvolutionaryConnections,
  getOrCaptureEvolutionaryBaseline,
  markEvolutionaryTopologyDirty,
  normalizeEvolutionaryTargetSparsity,
} from './network.prune.evolutionary.utils';

describe('network prune evolutionary utility chapter', () => {
  describe('normalizeEvolutionaryTargetSparsity()', () => {
    describe('given a sparsity at or below zero', () => {
      it('returns zero for a negative input', () => {
        // Act
        const result = normalizeEvolutionaryTargetSparsity(-0.1);

        // Assert
        expect(result).toBe(0);
      });
    });

    describe('given a sparsity at or above one', () => {
      it('clamps to the maximum safe sparsity for an input of one', () => {
        // Act
        const result = normalizeEvolutionaryTargetSparsity(1);

        // Assert
        expect(result).toBeLessThan(1);
      });
    });

    describe('given a sparsity in range', () => {
      it('returns the value unchanged', () => {
        // Act
        const result = normalizeEvolutionaryTargetSparsity(0.5);

        // Assert
        expect(result).toBe(0.5);
      });
    });
  });

  describe('getOrCaptureEvolutionaryBaseline()', () => {
    describe('given a network with no prior baseline', () => {
      it('captures the current connection count as baseline on first call', () => {
        // Arrange
        const network = { connections: [1, 2, 3] } as unknown as Parameters<
          typeof getOrCaptureEvolutionaryBaseline
        >[0];

        // Act
        const baseline = getOrCaptureEvolutionaryBaseline(network);

        // Assert
        expect(baseline).toBe(3);
      });
    });

    describe('given a network with an existing baseline', () => {
      it('returns the existing baseline without overwriting it', () => {
        // Arrange
        const network = {
          connections: [1, 2, 3, 4, 5],
        } as unknown as Parameters<typeof getOrCaptureEvolutionaryBaseline>[0];
        Reflect.set(network, '_evoInitialConnCount', 10);

        // Act
        const baseline = getOrCaptureEvolutionaryBaseline(network);

        // Assert
        expect(baseline).toBe(10);
      });
    });
  });

  describe('buildEvolutionaryTarget()', () => {
    describe('given a target sparsity and baseline', () => {
      it('derives remaining and excess connection counts', () => {
        // Arrange
        const context = { baselineConnectionCount: 10, targetSparsity: 0.4 };

        // Act
        const result = buildEvolutionaryTarget(context, 10);

        // Assert
        expect(result.excessConnectionCount).toBe(4);
      });
    });
  });

  describe('buildEvolutionaryPruneSelection()', () => {
    describe('given SNIP method with non-zero delta connections', () => {
      it('scores by weight times gradient magnitude and prunes lowest saliency', () => {
        // Arrange — high-weight but low-delta vs low-weight but high-delta
        const lowSaliencyConnection = new Connection(
          new Node('hidden'),
          new Node('hidden'),
          0.5,
        );
        lowSaliencyConnection.totalDeltaWeight = 0.1; // saliency = 0.5 * 0.1 = 0.05

        const highSaliencyConnection = new Connection(
          new Node('hidden'),
          new Node('hidden'),
          0.5,
        );
        highSaliencyConnection.totalDeltaWeight = 2.0; // saliency = 0.5 * 2.0 = 1.0

        // Act
        const result = buildEvolutionaryPruneSelection({
          connections: [highSaliencyConnection, lowSaliencyConnection],
          method: 'snip',
          removalCount: 1,
        });

        // Assert
        expect(result.connectionsToPrune).toEqual([lowSaliencyConnection]);
      });
    });

    describe('given SNIP method with zero-delta connections', () => {
      it('falls back to magnitude ranking when gradient info is absent', () => {
        // Arrange — connections with default zero deltas so saliency degrades to |weight|
        const heavyConnection = new Connection(
          new Node('hidden'),
          new Node('hidden'),
          0.9,
        );
        const lightConnection = new Connection(
          new Node('hidden'),
          new Node('hidden'),
          0.1,
        );
        // Both totalDeltaWeight and previousDeltaWeight are 0 by default

        // Act — request removal of 1 connection; lightest magnitude pruned first
        const result = buildEvolutionaryPruneSelection({
          connections: [heavyConnection, lightConnection],
          method: 'snip',
          removalCount: 1,
        });

        // Assert
        expect(result.connectionsToPrune).toEqual([lightConnection]);
      });
    });

    describe('given magnitude method with two connections', () => {
      it('selects the lower-weight connection for removal', () => {
        // Arrange
        const heavyConnection = new Connection(
          new Node('hidden'),
          new Node('hidden'),
          0.8,
        );
        const lightConnection = new Connection(
          new Node('hidden'),
          new Node('hidden'),
          0.2,
        );

        // Act
        const result = buildEvolutionaryPruneSelection({
          connections: [heavyConnection, lightConnection],
          method: 'magnitude',
          removalCount: 1,
        });

        // Assert
        expect(result.connectionsToPrune).toEqual([lightConnection]);
      });
    });
  });

  describe('disconnectEvolutionaryConnections()', () => {
    describe('given a list of connections to remove', () => {
      it('calls disconnect on the network for each connection', () => {
        // Arrange
        const sourceNode = new Node('hidden');
        const targetNode = new Node('hidden');
        const connection = new Connection(sourceNode, targetNode, 0.5);
        const disconnectSpy = jest.fn();
        const fakeNetwork = { disconnect: disconnectSpy } as unknown as Network;

        // Act
        disconnectEvolutionaryConnections(fakeNetwork, [connection]);

        // Assert
        expect(disconnectSpy).toHaveBeenCalledWith(sourceNode, targetNode);
      });
    });

    describe('given many connections are removed at once', () => {
      it('forwards the large-prune size to the activation pool after disconnecting', () => {
        // Arrange
        const disconnectSpy = jest.fn();
        const fakeNetwork = { disconnect: disconnectSpy } as unknown as Network;
        const connectionsToDisconnect = Array.from({ length: 64 }, () => {
          return new Connection(new Node('hidden'), new Node('hidden'), 0.2);
        });
        const originalScheduler = Reflect.get(
          activationArrayPool,
          'scheduleCompactionAfterLargePrune',
        );
        const scheduleSpy = jest.fn();
        Reflect.set(
          activationArrayPool,
          'scheduleCompactionAfterLargePrune',
          scheduleSpy,
        );

        // Act
        disconnectEvolutionaryConnections(fakeNetwork, connectionsToDisconnect);

        if (originalScheduler === undefined) {
          Reflect.deleteProperty(
            activationArrayPool,
            'scheduleCompactionAfterLargePrune',
          );
        } else {
          Reflect.set(
            activationArrayPool,
            'scheduleCompactionAfterLargePrune',
            originalScheduler,
          );
        }

        // Assert
        expect({
          disconnectedEdgeCount: disconnectSpy.mock.calls.length,
          scheduledPruneCount: scheduleSpy.mock.calls[0]?.[0],
        }).toEqual({
          disconnectedEdgeCount: 64,
          scheduledPruneCount: 64,
        });
      });
    });
  });

  describe('markEvolutionaryTopologyDirty()', () => {
    describe('given a network object', () => {
      it('sets the topology dirty flag', () => {
        // Arrange
        const fakeNetwork = {} as unknown as Network;

        // Act
        markEvolutionaryTopologyDirty(fakeNetwork);

        // Assert
        expect(Reflect.get(fakeNetwork, '_topoDirty')).toBe(true);
      });
    });
  });
});
