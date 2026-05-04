import Group from '../group/group';
import Layer from './layer';
import Node from '../node';

describe('layer factory recurrent utility chapter', () => {
  describe('Layer.lstm()', () => {
    describe('given the memory cell self-connection is not created', () => {
      it('warns about the missing self-connection', () => {
        // Arrange
        const originalConnect = Group.prototype.connect;
        const connectSpy = jest
          .spyOn(Group.prototype, 'connect')
          .mockImplementation(function (
            this: Group,
            target: Layer | Group | Node,
            method?: unknown,
            weight?: number,
          ) {
            if (target === this) {
              return [];
            }

            return originalConnect.call(this, target, method, weight);
          });
        const warnSpy = jest
          .spyOn(console, 'warn')
          .mockImplementation(() => undefined);

        try {
          // Act
          Layer.lstm(1);

          // Assert
          expect(warnSpy).toHaveBeenCalledWith(
            'LSTM Warning: No self-connection found for memory cell node 0',
          );
        } finally {
          connectSpy.mockRestore();
          warnSpy.mockRestore();
        }
      });
    });

    describe('given the forget gate already tracks the self-connection', () => {
      it('keeps the gated self-connection list deduplicated', () => {
        // Arrange
        const originalConnect = Group.prototype.connect;
        const connectCountsByGroup = new Map<Group, number>();
        let forgetGateNode: Node | undefined;
        jest.spyOn(Group.prototype, 'connect').mockImplementation(function (
          this: Group,
          target: Layer | Group | Node,
          method?: unknown,
          weight?: number,
        ) {
          const connections = originalConnect.call(
            this,
            target,
            method,
            weight,
          );

          if (target instanceof Group && this !== target) {
            const nextConnectCount = (connectCountsByGroup.get(this) ?? 0) + 1;
            connectCountsByGroup.set(this, nextConnectCount);

            if (nextConnectCount === 2) {
              forgetGateNode = target.nodes[0];
            }
          }

          if (this === target && forgetGateNode && connections[0]) {
            forgetGateNode.connections.gated.push(connections[0]);
          }

          return connections;
        });

        // Act
        const lstmLayer = Layer.lstm(1);

        // Assert
        expect(lstmLayer.nodes[1].connections.gated.length).toBe(1);
      });
    });

    describe('given a group source is connected without an explicit method', () => {
      it('returns the flattened LSTM input connections', () => {
        // Arrange
        const lstmLayer = Layer.lstm(1);
        const sourceGroup = new Group(1);

        // Act
        const inputConnections = lstmLayer.input(sourceGroup);

        // Assert
        expect(inputConnections.length).toBe(4);
      });
    });
  });

  describe('Layer.memory()', () => {
    describe('given the terminal memory block is replaced with a node', () => {
      it('throws the memory input block type error', () => {
        // Arrange
        const memoryLayer = Layer.memory(2, 2);
        memoryLayer.nodes[memoryLayer.nodes.length - 1] = new Node('hidden');
        const connectMemoryLayer = () => memoryLayer.input(Layer.dense(2));

        // Assert
        expect(connectMemoryLayer).toThrow(
          'Memory layer input block is not a Group.',
        );
      });
    });

    describe('given a group source matches the memory block size', () => {
      it('returns the one-to-one memory input connection list', () => {
        // Arrange
        const memoryLayer = Layer.memory(2, 2);
        const sourceGroup = new Group(2);

        // Act
        const inputConnections = memoryLayer.input(sourceGroup);

        // Assert
        expect(inputConnections.length).toBe(2);
      });
    });

    describe('given the runtime toReversed helper yields no value', () => {
      it('falls back to the legacy reversed ordering path', () => {
        // Arrange
        const arrayPrototype = Array.prototype as typeof Array.prototype & {
          toReversed: () => unknown[];
        };
        const toReversedSpy = jest
          .spyOn(arrayPrototype, 'toReversed')
          .mockImplementation(() => undefined as never);

        try {
          // Act
          Layer.memory(2, 2);

          // Assert
          expect(toReversedSpy).toHaveBeenCalled();
        } finally {
          toReversedSpy.mockRestore();
        }
      });
    });

    describe('given the source layer size does not match the memory block size', () => {
      it('throws the memory size mismatch error', () => {
        // Arrange
        const memoryLayer = Layer.memory(2, 2);
        const sourceLayer = Layer.dense(3);
        const connectMemoryLayer = () => memoryLayer.input(sourceLayer);

        // Assert
        expect(connectMemoryLayer).toThrow(
          'Previous layer size (3) must be same as memory size (2)',
        );
      });
    });
  });

  describe('Layer.gru()', () => {
    describe('given a group source is connected without an explicit method', () => {
      it('builds the GRU layer and returns the flattened input connections', () => {
        // Arrange
        const gruLayer = Layer.gru(1);
        const sourceGroup = new Group(1);

        // Act
        const inputConnections = gruLayer.input(sourceGroup);

        // Assert
        expect(inputConnections.length).toBe(3);
      });
    });
  });
});
