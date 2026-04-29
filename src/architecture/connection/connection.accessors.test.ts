import Connection from './connection';
import Node from '../node/node';

describe('connection accessor chapter', () => {
  let fromNode: Node;
  let toNode: Node;

  beforeEach(() => {
    // Arrange
    Connection.resetInnovationCounter(1);
    fromNode = new Node('input');
    toNode = new Node('output');
  });

  describe('resetInnovationCounter()', () => {
    describe('when an explicit starting value is provided', () => {
      it('uses that value for the next constructed connection innovation', () => {
        // Act
        Connection.resetInnovationCounter(40);
        const connection = new Connection(fromNode, toNode, 0.2);

        // Assert
        expect(connection.innovation).toBe(40);
      });
    });

    describe('when no starting value is provided', () => {
      it('resets the next innovation back to one', () => {
        // Arrange
        Connection.resetInnovationCounter(12);

        // Act
        Connection.resetInnovationCounter();
        const connection = new Connection(fromNode, toNode, 0.2);

        // Assert
        expect(connection.innovation).toBe(1);
      });
    });
  });

  describe('syncInnovationCounter()', () => {
    describe('when the restored maximum innovation is larger than the current cursor', () => {
      it('advances the next constructed innovation above that restored maximum', () => {
        // Act
        Connection.syncInnovationCounter(9);
        const connection = new Connection(fromNode, toNode, 0.2);

        // Assert
        expect(connection.innovation).toBe(10);
      });
    });

    describe('when the restored maximum innovation is smaller than the current cursor', () => {
      it('keeps the next constructed innovation monotonic', () => {
        // Arrange
        Connection.resetInnovationCounter(8);

        // Act
        Connection.syncInnovationCounter(3);
        const connection = new Connection(fromNode, toNode, 0.2);

        // Assert
        expect(connection.innovation).toBe(8);
      });
    });

    describe('when the restored innovation is not finite', () => {
      it('leaves the next innovation cursor unchanged', () => {
        // Arrange
        Connection.resetInnovationCounter(6);

        // Act
        Connection.syncInnovationCounter(Number.NaN);
        const connection = new Connection(fromNode, toNode, 0.2);

        // Assert
        expect(connection.innovation).toBe(6);
      });
    });
  });

  describe('acquire()', () => {
    describe('when the internal pool is empty', () => {
      it('constructs a fresh connection with the requested endpoints and weight', () => {
        // Act
        const connection = Connection.acquire(fromNode, toNode, 0.45);

        // Assert
        expect({
          from: connection.from,
          innovation: connection.innovation,
          to: connection.to,
          weight: connection.weight,
        }).toEqual({
          from: fromNode,
          innovation: 1,
          to: toNode,
          weight: 0.45,
        });
      });
    });

    describe('when a released connection is available in the pool', () => {
      it('reuses the pooled instance after resetting its optional state', () => {
        // Arrange
        const originalConnection = new Connection(fromNode, toNode, 0.5);
        const originalGater = new Node('hidden');
        originalConnection.gain = 0.2;
        originalConnection.gater = originalGater;
        originalConnection.enabled = false;
        originalConnection.dcMask = 0;
        originalConnection.plasticityRate = 0.3;
        originalConnection.firstMoment = 0.4;
        originalConnection.eligibility = 0.6;
        originalConnection.previousDeltaWeight = 0.7;
        originalConnection.totalDeltaWeight = 0.8;
        originalConnection.xtrace.nodes.push(originalGater);
        originalConnection.xtrace.values.push(0.9);
        Connection.release(originalConnection);

        const nextFromNode = new Node('input');
        const nextToNode = new Node('output');

        // Act
        const reacquiredConnection = Connection.acquire(
          nextFromNode,
          nextToNode,
          0.95,
        );

        // Assert
        expect({
          dcMask: reacquiredConnection.dcMask,
          eligibility: reacquiredConnection.eligibility,
          enabled: reacquiredConnection.enabled,
          firstMoment: reacquiredConnection.firstMoment,
          from: reacquiredConnection.from,
          gain: reacquiredConnection.gain,
          gater: reacquiredConnection.gater,
          innovation: reacquiredConnection.innovation,
          plasticityRate: reacquiredConnection.plasticityRate,
          previousDeltaWeight: reacquiredConnection.previousDeltaWeight,
          reusedInstance: reacquiredConnection === originalConnection,
          to: reacquiredConnection.to,
          totalDeltaWeight: reacquiredConnection.totalDeltaWeight,
          weight: reacquiredConnection.weight,
          xtraceNodeCount: reacquiredConnection.xtrace.nodes.length,
          xtraceValueCount: reacquiredConnection.xtrace.values.length,
        }).toEqual({
          dcMask: 1,
          eligibility: 0,
          enabled: true,
          firstMoment: undefined,
          from: nextFromNode,
          gain: 1,
          gater: null,
          innovation: 2,
          plasticityRate: 0,
          previousDeltaWeight: 0,
          reusedInstance: true,
          to: nextToNode,
          totalDeltaWeight: 0,
          weight: 0.95,
          xtraceNodeCount: 0,
          xtraceValueCount: 0,
        });
      });

      it('reuses a clean pooled instance while generating a fresh default weight', () => {
        // Arrange
        const originalConnection = new Connection(fromNode, toNode, 0.5);
        Connection.release(originalConnection);
        const nextFromNode = new Node('input');
        const nextToNode = new Node('output');

        // Act
        const reacquiredConnection = Connection.acquire(
          nextFromNode,
          nextToNode,
        );

        // Assert
        expect({
          firstMoment: reacquiredConnection.firstMoment,
          gain: reacquiredConnection.gain,
          gater: reacquiredConnection.gater,
          plasticityRate: reacquiredConnection.plasticityRate,
          reusedInstance: reacquiredConnection === originalConnection,
          weightInRange:
            reacquiredConnection.weight >= -0.1 &&
            reacquiredConnection.weight <= 0.1,
        }).toEqual({
          firstMoment: undefined,
          gain: 1,
          gater: null,
          plasticityRate: 0,
          reusedInstance: true,
          weightInRange: true,
        });
      });
    });
  });

  describe('enabled', () => {
    describe('when gene expression is toggled off and back on', () => {
      it('tracks the enabled flag through the accessor', () => {
        // Arrange
        const connection = new Connection(fromNode, toNode, 0.5);

        // Act
        connection.enabled = false;
        const disabledState = connection.enabled;
        connection.enabled = true;

        // Assert
        expect({ disabledState, enabledState: connection.enabled }).toEqual({
          disabledState: false,
          enabledState: true,
        });
      });
    });
  });

  describe('gain', () => {
    describe('when the neutral gain is assigned before any custom gain exists', () => {
      it('keeps the connection at the implicit neutral gain', () => {
        // Arrange
        const connection = new Connection(fromNode, toNode, 0.5);

        // Act
        connection.gain = 1;

        // Assert
        expect(connection.gain).toBe(1);
      });
    });

    describe('when a non-neutral gain is later reset to one', () => {
      it('restores the neutral gain value through the accessor', () => {
        // Arrange
        const connection = new Connection(fromNode, toNode, 0.5);

        // Act
        connection.gain = 1.7;
        const nonNeutralGain = connection.gain;
        connection.gain = 1;

        // Assert
        expect({ nonNeutralGain, neutralGain: connection.gain }).toEqual({
          nonNeutralGain: 1.7,
          neutralGain: 1,
        });
      });
    });
  });

  describe('gater and hasGater', () => {
    describe('when a null gater is assigned before any real gater exists', () => {
      it('keeps the connection ungated', () => {
        // Arrange
        const connection = new Connection(fromNode, toNode, 0.5);

        // Act
        connection.gater = null;

        // Assert
        expect({
          gater: connection.gater,
          hasGater: connection.hasGater,
        }).toEqual({
          gater: null,
          hasGater: false,
        });
      });
    });

    describe('when a gater is assigned and then cleared', () => {
      it('tracks the presence flag alongside the stored gater reference', () => {
        // Arrange
        const connection = new Connection(fromNode, toNode, 0.5);
        const gatingNode = new Node('hidden');

        // Act
        connection.gater = gatingNode;
        const assignedState = {
          gater: connection.gater,
          hasGater: connection.hasGater,
        };
        connection.gater = null;

        // Assert
        expect({ assignedState, clearedState: connection.hasGater }).toEqual({
          assignedState: {
            gater: gatingNode,
            hasGater: true,
          },
          clearedState: false,
        });
      });
    });

    describe('when the gating flag is still set but the symbol-backed gater is missing', () => {
      it('falls back to a null gater value', () => {
        // Arrange
        const connection = new Connection(fromNode, toNode, 0.5);
        const gatingNode = new Node('hidden');
        connection.gater = gatingNode;
        const gaterSymbol = Object.getOwnPropertySymbols(connection).find(
          (symbolKey) => symbolKey.description === 'connGater',
        );
        if (!gaterSymbol) {
          throw new Error('Missing gater symbol');
        }

        // Act
        delete (connection as unknown as Record<symbol, unknown>)[gaterSymbol];

        // Assert
        expect(connection.gater).toBeNull();
      });

      it('clears the stale gating flag without requiring a symbol-backed gater', () => {
        // Arrange
        const connection = new Connection(fromNode, toNode, 0.5);
        const gatingNode = new Node('hidden');
        connection.gater = gatingNode;
        const gaterSymbol = Object.getOwnPropertySymbols(connection).find(
          (symbolKey) => symbolKey.description === 'connGater',
        );
        if (!gaterSymbol) {
          throw new Error('Missing gater symbol');
        }
        delete (connection as unknown as Record<symbol, unknown>)[gaterSymbol];

        // Act
        connection.gater = null;

        // Assert
        expect({
          gater: connection.gater,
          hasGater: connection.hasGater,
        }).toEqual({
          gater: null,
          hasGater: false,
        });
      });
    });
  });

  describe('dropconnect masks', () => {
    describe('when the alias setter is used to drop the connection', () => {
      it('updates both the alias view and the raw dcMask accessor', () => {
        // Arrange
        const connection = new Connection(fromNode, toNode, 0.5);

        // Act
        connection.dropConnectActiveMask = 0;

        // Assert
        expect({
          dcMask: connection.dcMask,
          dropConnectActiveMask: connection.dropConnectActiveMask,
        }).toEqual({
          dcMask: 0,
          dropConnectActiveMask: 0,
        });
      });
    });

    describe('when the raw setter is used to restore the active mask', () => {
      it('stores the active mask state', () => {
        // Arrange
        const connection = new Connection(fromNode, toNode, 0.5);
        connection.dcMask = 0;

        // Act
        connection.dcMask = 1;

        // Assert
        expect(connection.dcMask).toBe(1);
      });
    });
  });

  describe('plasticity', () => {
    describe('when the plastic flag is enabled directly', () => {
      it('reports the connection as plastic', () => {
        // Arrange
        const connection = new Connection(fromNode, toNode, 0.5);

        // Act
        connection.plastic = true;

        // Assert
        expect(connection.plastic).toBe(true);
      });
    });

    describe('when a plasticity rate is later reset to zero', () => {
      it('clears both the plastic flag and the stored rate', () => {
        // Arrange
        const connection = new Connection(fromNode, toNode, 0.5);

        // Act
        connection.plasticityRate = 0.25;
        const activeState = {
          plastic: connection.plastic,
          plasticityRate: connection.plasticityRate,
        };
        connection.plasticityRate = 0;

        // Assert
        expect({ activeState, resetState: connection.plasticityRate }).toEqual({
          activeState: {
            plastic: true,
            plasticityRate: 0.25,
          },
          resetState: 0,
        });
      });
    });

    describe('when plastic mode is disabled explicitly after a rate was assigned', () => {
      it('drops the stored plasticity rate', () => {
        // Arrange
        const connection = new Connection(fromNode, toNode, 0.5);
        connection.plasticityRate = 0.4;

        // Act
        connection.plastic = false;

        // Assert
        expect({
          plastic: connection.plastic,
          rate: connection.plasticityRate,
        }).toEqual({
          plastic: false,
          rate: 0,
        });
      });
    });

    describe('when the plasticity rate is cleared with an explicit undefined value', () => {
      it('removes the stored rate through the undefined branch as well', () => {
        // Arrange
        const connection = new Connection(fromNode, toNode, 0.5);
        connection.plasticityRate = 0.4;

        // Act
        connection.plasticityRate = undefined as unknown as number;

        // Assert
        expect({
          plastic: connection.plastic,
          rate: connection.plasticityRate,
        }).toEqual({
          plastic: false,
          rate: 0,
        });
      });
    });

    describe('when an undefined plasticity rate is assigned before any rate exists', () => {
      it('keeps the connection non-plastic with a zero rate', () => {
        // Arrange
        const connection = new Connection(fromNode, toNode, 0.5);

        // Act
        connection.plasticityRate = undefined as unknown as number;

        // Assert
        expect({
          plastic: connection.plastic,
          rate: connection.plasticityRate,
        }).toEqual({
          plastic: false,
          rate: 0,
        });
      });
    });
  });

  describe('optimizer accessors', () => {
    describe('when optimizer slots are assigned values', () => {
      it('exposes the stored optimizer values through each accessor', () => {
        // Arrange
        const connection = new Connection(fromNode, toNode, 0.5);

        // Act
        connection.firstMoment = 0.1;
        connection.secondMoment = 0.2;
        connection.gradientAccumulator = 0.3;
        connection.maxSecondMoment = 0.4;
        connection.infinityNorm = 0.5;
        connection.secondMomentum = 0.6;
        connection.lookaheadShadowWeight = 0.7;

        // Assert
        expect({
          firstMoment: connection.firstMoment,
          gradientAccumulator: connection.gradientAccumulator,
          infinityNorm: connection.infinityNorm,
          lookaheadShadowWeight: connection.lookaheadShadowWeight,
          maxSecondMoment: connection.maxSecondMoment,
          secondMoment: connection.secondMoment,
          secondMomentum: connection.secondMomentum,
        }).toEqual({
          firstMoment: 0.1,
          gradientAccumulator: 0.3,
          infinityNorm: 0.5,
          lookaheadShadowWeight: 0.7,
          maxSecondMoment: 0.4,
          secondMoment: 0.2,
          secondMomentum: 0.6,
        });
      });
    });

    describe('when one optimizer slot is cleared back to undefined', () => {
      it('removes that stored optimizer value', () => {
        // Arrange
        const connection = new Connection(fromNode, toNode, 0.5);
        connection.firstMoment = 0.1;

        // Act
        connection.firstMoment = undefined;

        // Assert
        expect(connection.firstMoment).toBeUndefined();
      });
    });

    describe('when an optimizer slot is cleared before the bag exists', () => {
      it('keeps the accessor undefined without allocating optimizer state', () => {
        // Arrange
        const connection = new Connection(fromNode, toNode, 0.5);

        // Act
        connection.firstMoment = undefined;

        // Assert
        expect(connection.firstMoment).toBeUndefined();
      });
    });
  });

  describe('toJSON()', () => {
    describe('when the endpoint indexes are missing', () => {
      it('serializes undefined endpoint indexes explicitly', () => {
        // Arrange
        const unindexedFromNode = new Node('input');
        const unindexedToNode = new Node('output');
        unindexedFromNode.index = undefined;
        unindexedToNode.index = undefined;
        const connection = new Connection(
          unindexedFromNode,
          unindexedToNode,
          0.4,
        );

        // Act
        const serializedConnection = connection.toJSON();

        // Assert
        expect({
          from: serializedConnection.from,
          to: serializedConnection.to,
        }).toEqual({
          from: undefined,
          to: undefined,
        });
      });
    });

    describe('when a gater exists but has no runtime index', () => {
      it('keeps the serialized gater field omitted', () => {
        // Arrange
        const connection = new Connection(fromNode, toNode, 0.4);
        const gatingNode = new Node('hidden');
        gatingNode.index = undefined;
        connection.gater = gatingNode;

        // Act
        const serializedConnection = connection.toJSON();

        // Assert
        expect(serializedConnection.gater).toBeUndefined();
      });
    });
  });
});
