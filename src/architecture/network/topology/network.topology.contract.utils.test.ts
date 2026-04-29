import Network from '../network';
import {
  hasFeedForwardTopologyContract,
  getTopologyIntent,
  setEnforceAcyclic,
  setTopologyIntent,
} from './network.topology.contract.utils';
import type { FeedForwardTopologyContractCarrier } from './network.topology.contract.utils';

jest.retryTimes(2, { logErrorsBeforeRetry: true });

describe('network topology contract utils chapter', () => {
  describe('hasFeedForwardTopologyContract()', () => {
    describe('given a carrier with a getTopologyIntent method returning feed-forward', () => {
      it('returns true when the intent accessor signals feed-forward', () => {
        // Arrange: carrier with getTopologyIntent that returns 'feed-forward'
        const carrier: FeedForwardTopologyContractCarrier = {
          getTopologyIntent: () => 'feed-forward',
        };

        // Act
        const result = hasFeedForwardTopologyContract(carrier);

        // Assert
        expect(result).toBe(true);
      });
    });

    describe('given a carrier with no getTopologyIntent but _topologyIntent set to feed-forward', () => {
      it('returns true when the runtime property carries feed-forward intent', () => {
        // Arrange: plain object with _topologyIntent, no method
        const carrier = {
          _topologyIntent: 'feed-forward',
        } as FeedForwardTopologyContractCarrier;

        // Act
        const result = hasFeedForwardTopologyContract(carrier);

        // Assert
        expect(result).toBe(true);
      });
    });

    describe('given a carrier with _enforceAcyclic set to true and no topology intent', () => {
      it('returns true because the low-level acyclic flag is honoured', () => {
        // Arrange: plain object with only _enforceAcyclic
        const carrier = {
          _enforceAcyclic: true,
        } as FeedForwardTopologyContractCarrier;

        // Act
        const result = hasFeedForwardTopologyContract(carrier);

        // Assert
        expect(result).toBe(true);
      });
    });

    describe('given a carrier with no topology intent and no acyclic flag', () => {
      it('returns false when neither feed-forward signal is present', () => {
        // Arrange: empty carrier
        const carrier: FeedForwardTopologyContractCarrier = {};

        // Act
        const result = hasFeedForwardTopologyContract(carrier);

        // Assert
        expect(result).toBe(false);
      });
    });
  });

  describe('getTopologyIntent()', () => {
    describe('given a network whose _topologyIntent has never been set', () => {
      it('falls back to unconstrained', () => {
        // Arrange: fresh Network has no _topologyIntent persisted
        const network = new Network(1, 1);
        (network as unknown as Record<string, unknown>)._topologyIntent =
          undefined;

        // Act
        const result = getTopologyIntent.call(network);

        // Assert: ?? 'unconstrained' fallback branch (line 34)
        expect(result).toBe('unconstrained');
      });
    });
  });

  describe('setEnforceAcyclic()', () => {
    describe('given false is passed to setEnforceAcyclic', () => {
      it('sets the topology intent to unconstrained', () => {
        // Arrange
        const network = new Network(1, 1);

        // Act: flag=false → 'unconstrained' arm
        setEnforceAcyclic.call(network, false);

        // Assert
        expect(getTopologyIntent.call(network)).toBe('unconstrained');
      });
    });

    describe('given true is passed to setEnforceAcyclic', () => {
      it('sets the topology intent to feed-forward', () => {
        // Arrange
        const network = new Network(1, 1);

        // Act: flag=true → 'feed-forward' arm (line 110 TRUE branch)
        setEnforceAcyclic.call(network, true);

        // Assert
        expect(getTopologyIntent.call(network)).toBe('feed-forward');
      });
    });
  });

  describe('setTopologyIntent()', () => {
    describe('given feed-forward is requested on a fresh network', () => {
      it('persists the feed-forward intent and marks the topology cache dirty', () => {
        // Arrange
        const network = new Network(1, 1);

        // Act
        setTopologyIntent.call(network, 'feed-forward');

        // Assert: public intent updated
        expect(getTopologyIntent.call(network)).toBe('feed-forward');
      });
    });
  });
});
