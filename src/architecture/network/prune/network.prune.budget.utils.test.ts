import mutation from '../../../methods/mutation/mutation';
import { config } from '../../../config';
import Network from '../network';
import {
  ensureGrowthBudget,
  getSparsityBudgetSnapshot,
} from './network.prune.budget.utils';

const originalWindowDescriptor = Object.getOwnPropertyDescriptor(
  globalThis,
  'window',
);
const originalPerformanceDescriptor = Object.getOwnPropertyDescriptor(
  globalThis,
  'performance',
);
const originalProcessDescriptor = Object.getOwnPropertyDescriptor(
  globalThis,
  'process',
);
const originalBrowserMemoryBudgetMB = config.browserMemoryBudgetMB;
const originalNodeHeapSoftLimitMB = config.nodeHeapSoftLimitMB;

type ErrorSnapshot = {
  message: string;
  name: string;
};

function captureErrorSnapshot(callback: () => void): ErrorSnapshot {
  try {
    callback();
  } catch (error) {
    if (error instanceof Error) {
      return {
        message: error.message,
        name: error.name,
      };
    }

    throw error;
  }

  return {
    message: 'Expected callback to throw',
    name: 'NoErrorThrown',
  };
}

describe('network prune budget utility chapter', () => {
  afterEach(() => {
    config.browserMemoryBudgetMB = originalBrowserMemoryBudgetMB;
    config.nodeHeapSoftLimitMB = originalNodeHeapSoftLimitMB;
    restoreGlobalProperty('window', originalWindowDescriptor);
    restoreGlobalProperty('performance', originalPerformanceDescriptor);
    restoreGlobalProperty('process', originalProcessDescriptor);
  });

  describe('configureSparsityBudget()', () => {
    describe('given maxConnections is less than one', () => {
      it('throws the shared max-connections range error', () => {
        // Arrange
        const network = new Network(1, 1, { seed: 10_260 });
        const runConfiguration = () =>
          network.configureSparsityBudget({ maxConnections: 0 });

        // Act
        const errorSnapshot = captureErrorSnapshot(runConfiguration);

        // Assert
        expect(errorSnapshot).toEqual({
          message: 'maxConnections must be an integer >= 1',
          name: 'NetworkPruneBudgetMaxConnectionsError',
        });
      });
    });

    describe('given growthGraceFraction is negative', () => {
      it('throws the shared growth-grace range error', () => {
        // Arrange
        const network = new Network(1, 1, { seed: 10_261 });
        const runConfiguration = () =>
          network.configureSparsityBudget({
            growthGraceFraction: -0.1,
            maxConnections: 2,
          });

        // Act
        const errorSnapshot = captureErrorSnapshot(runConfiguration);

        // Assert
        expect(errorSnapshot).toEqual({
          message: 'growthGraceFraction must be >= 0',
          name: 'NetworkPruneBudgetGrowthGraceFractionError',
        });
      });
    });
  });

  describe('getSparsityBudgetSnapshot()', () => {
    describe('given no budget decision has been recorded', () => {
      it('returns undefined', () => {
        // Arrange
        const network = new Network(1, 1, { seed: 10_262 });

        // Act
        const budgetSnapshot = getSparsityBudgetSnapshot(network);

        // Assert
        expect(budgetSnapshot).toBeUndefined();
      });
    });
  });

  describe('ensureGrowthBudget()', () => {
    describe('given an existing self-loop already consumes the last total-connection budget slot', () => {
      it('counts the self-loop before deciding whether growth must be denied', () => {
        // Arrange
        const network = new Network(1, 2, {
          seed: 10_272,
          enforceAcyclic: false,
        });
        network.connect(network.nodes[1], network.nodes[1]);
        network.configureSparsityBudget({ maxConnections: 3 });
        Reflect.set(network, 'disconnect', () => undefined);

        // Act
        const budgetAllowed = ensureGrowthBudget(network, 1);
        const budgetSnapshot = getSparsityBudgetSnapshot(network);

        // Assert
        expect({
          budgetAllowed,
          connectionCountBeforeDecision:
            budgetSnapshot?.connectionCountBeforeDecision,
          decision: budgetSnapshot?.decision,
        }).toEqual({
          budgetAllowed: false,
          connectionCountBeforeDecision: 3,
          decision: 'deny',
        });
      });
    });

    describe('given grace headroom extends the effective connection cap', () => {
      it('records an allow decision for growth that would otherwise hit the hard cap', () => {
        // Arrange
        const network = new Network(1, 1, { seed: 10_266 });
        network.configureSparsityBudget({
          growthGraceFraction: 1,
          maxConnections: 1,
        });

        // Act
        const budgetAllowed = ensureGrowthBudget(network, 1);
        const budgetSnapshot = getSparsityBudgetSnapshot(network);

        // Assert
        expect({
          allowedConnectionLimit: budgetSnapshot?.allowedConnectionLimit,
          budgetAllowed,
          decision: budgetSnapshot?.decision,
        }).toEqual({
          allowedConnectionLimit: 2,
          budgetAllowed: true,
          decision: 'allow',
        });
      });
    });

    describe('given projected growth stays within the configured connection limit', () => {
      it('records an allow decision without pruning', () => {
        // Arrange
        const network = new Network(1, 1, { seed: 10_263 });
        network.configureSparsityBudget({ maxConnections: 2 });

        // Act
        const budgetAllowed = ensureGrowthBudget(network, 1);
        const budgetSnapshot = getSparsityBudgetSnapshot(network);

        // Assert
        expect({
          budgetAllowed,
          connectionCount: network.connections.length,
          decision: budgetSnapshot?.decision,
        }).toEqual({
          budgetAllowed: true,
          connectionCount: 1,
          decision: 'allow',
        });
      });
    });

    describe('given one more connection would violate the hard minimum-remaining edge rule', () => {
      it('records a deny decision without pruning', () => {
        // Arrange
        const network = new Network(1, 1, { seed: 10_264 });
        network.configureSparsityBudget({ maxConnections: 1 });

        // Act
        const budgetAllowed = ensureGrowthBudget(network, 1);
        const budgetSnapshot = getSparsityBudgetSnapshot(network);

        // Assert
        expect({
          budgetAllowed,
          connectionCount: network.connections.length,
          decision: budgetSnapshot?.decision,
        }).toEqual({
          budgetAllowed: false,
          connectionCount: 1,
          decision: 'deny',
        });
      });
    });

    describe('given one removable connection can be pruned before the requested growth', () => {
      it('records a prune-then-allow decision and frees one connection slot', () => {
        // Arrange
        const network = new Network(1, 1, {
          enforceAcyclic: true,
          seed: 10_265,
        });
        network.mutate(mutation.ADD_NODE);
        network.connections[0].weight = 0.01;
        network.connections[1].weight = 0.9;
        network.configureSparsityBudget({ maxConnections: 2 });

        // Act
        const budgetAllowed = ensureGrowthBudget(network, 1);
        const budgetSnapshot = getSparsityBudgetSnapshot(network);

        // Assert
        expect({
          budgetAllowed,
          connectionCount: network.connections.length,
          decision: budgetSnapshot?.decision,
        }).toEqual({
          budgetAllowed: true,
          connectionCount: 1,
          decision: 'prune-then-allow',
        });
      });
    });

    describe('given pruning is selected but the network fails to free the planned slot', () => {
      it('records a deny decision after the unsuccessful prune attempt', () => {
        // Arrange
        const network = new Network(1, 1, {
          enforceAcyclic: true,
          seed: 10_267,
        });
        network.mutate(mutation.ADD_NODE);
        network.connections[0].weight = 0.01;
        network.connections[1].weight = 0.9;
        network.configureSparsityBudget({ maxConnections: 2 });
        Reflect.set(network, 'disconnect', () => undefined);

        // Act
        const budgetAllowed = ensureGrowthBudget(network, 1);
        const budgetSnapshot = getSparsityBudgetSnapshot(network);

        // Assert
        expect({
          budgetAllowed,
          connectionCount: network.connections.length,
          decision: budgetSnapshot?.decision,
        }).toEqual({
          budgetAllowed: false,
          connectionCount: 2,
          decision: 'deny',
        });
      });
    });

    describe('given the same deny state repeats immediately after an unsuccessful prune attempt', () => {
      it('skips the immediate retry instead of pruning twice', () => {
        // Arrange
        const network = new Network(1, 1, {
          enforceAcyclic: true,
          seed: 10_273,
        });
        network.mutate(mutation.ADD_NODE);
        network.connections[0].weight = 0.01;
        network.connections[1].weight = 0.9;
        network.configureSparsityBudget({ maxConnections: 2 });
        const disconnectSpy = jest.fn();
        Reflect.set(network, 'disconnect', disconnectSpy);

        // Act
        const firstBudgetAllowed = ensureGrowthBudget(network, 1);
        const firstBudgetSnapshot = getSparsityBudgetSnapshot(network);
        const secondBudgetAllowed = ensureGrowthBudget(network, 1);
        const secondBudgetSnapshot = getSparsityBudgetSnapshot(network);

        // Assert
        expect({
          disconnectCallCount: disconnectSpy.mock.calls.length,
          firstBudgetAllowed,
          firstDecision: firstBudgetSnapshot?.decision,
          secondBudgetAllowed,
          secondDecision: secondBudgetSnapshot?.decision,
        }).toEqual({
          disconnectCallCount: 1,
          firstBudgetAllowed: false,
          firstDecision: 'deny',
          secondBudgetAllowed: false,
          secondDecision: 'deny',
        });
      });
    });

    describe('given the repeated deny backoff window has elapsed under the same state', () => {
      it('retries the prune path on the next reevaluation attempt', () => {
        // Arrange
        const network = new Network(1, 1, {
          enforceAcyclic: true,
          seed: 10_274,
        });
        network.mutate(mutation.ADD_NODE);
        network.connections[0].weight = 0.01;
        network.connections[1].weight = 0.9;
        network.configureSparsityBudget({ maxConnections: 2 });
        const disconnectSpy = jest.fn();
        Reflect.set(network, 'disconnect', disconnectSpy);

        // Act
        ensureGrowthBudget(network, 1);
        ensureGrowthBudget(network, 1);
        const thirdBudgetAllowed = ensureGrowthBudget(network, 1);
        const thirdBudgetSnapshot = getSparsityBudgetSnapshot(network);

        // Assert
        expect({
          disconnectCallCount: disconnectSpy.mock.calls.length,
          thirdBudgetAllowed,
          thirdDecision: thirdBudgetSnapshot?.decision,
        }).toEqual({
          disconnectCallCount: 2,
          thirdBudgetAllowed: false,
          thirdDecision: 'deny',
        });
      });
    });

    describe('given the requested growth size changes after a denied prune attempt', () => {
      it('clears the stale backoff fingerprint and reevaluates immediately', () => {
        // Arrange
        const network = new Network(1, 2, {
          enforceAcyclic: false,
          seed: 10_275,
        });
        network.connect(network.nodes[1], network.nodes[1]);
        network.connections[0].weight = 0.01;
        network.connections[1].weight = 0.9;
        network.selfconns[0].weight = 0.2;
        network.configureSparsityBudget({ maxConnections: 3 });
        const disconnectSpy = jest.fn();
        Reflect.set(network, 'disconnect', disconnectSpy);

        // Act
        const firstBudgetAllowed = ensureGrowthBudget(network, 1);
        const secondBudgetAllowed = ensureGrowthBudget(network, 2);
        const secondBudgetSnapshot = getSparsityBudgetSnapshot(network);

        // Assert
        expect({
          disconnectCallCount: disconnectSpy.mock.calls.length,
          firstBudgetAllowed,
          secondBudgetAllowed,
          secondDecision: secondBudgetSnapshot?.decision,
        }).toEqual({
          disconnectCallCount: 3,
          firstBudgetAllowed: false,
          secondBudgetAllowed: false,
          secondDecision: 'deny',
        });
      });
    });

    describe('given node heap usage already exceeds the configured soft memory budget while the hard cap still has headroom', () => {
      it('prunes one connection before allowing the requested growth', () => {
        // Arrange
        const network = new Network(1, 1, {
          enforceAcyclic: true,
          seed: 10_268,
        });
        network.mutate(mutation.ADD_NODE);
        network.connections[0].weight = 0.01;
        network.connections[1].weight = 0.9;
        network.configureSparsityBudget({ maxConnections: 4 });
        config.nodeHeapSoftLimitMB = 1;
        setGlobalProperty('process', {
          memoryUsage: () =>
            ({
              arrayBuffers: 0,
              external: 0,
              heapTotal: 3 * 1024 * 1024,
              heapUsed: 2 * 1024 * 1024,
              rss: 4 * 1024 * 1024,
            }) as NodeJS.MemoryUsage,
        });

        // Act
        const budgetAllowed = ensureGrowthBudget(network, 1);
        const budgetSnapshot = getSparsityBudgetSnapshot(network);

        // Assert
        expect({
          budgetAllowed,
          connectionCount: network.connections.length,
          decision: budgetSnapshot?.decision,
          softBudgetEnvironment: budgetSnapshot?.softBudgetEnvironment,
          softBudgetTriggered: budgetSnapshot?.softBudgetTriggered,
        }).toEqual({
          budgetAllowed: true,
          connectionCount: 1,
          decision: 'prune-then-allow',
          softBudgetEnvironment: 'node',
          softBudgetTriggered: true,
        });
      });
    });

    describe('given browser heap usage already exceeds the configured soft memory budget while the hard cap still has headroom', () => {
      it('prunes one connection before allowing the requested growth', () => {
        // Arrange
        const network = new Network(1, 1, {
          enforceAcyclic: true,
          seed: 10_269,
        });
        network.mutate(mutation.ADD_NODE);
        network.connections[0].weight = 0.01;
        network.connections[1].weight = 0.9;
        network.configureSparsityBudget({ maxConnections: 4 });
        config.browserMemoryBudgetMB = 1;
        setGlobalProperty('window', {});
        setGlobalProperty('performance', {
          memory: {
            jsHeapSizeLimit: 4 * 1024 * 1024,
            totalJSHeapSize: 3 * 1024 * 1024,
            usedJSHeapSize: 2 * 1024 * 1024,
          },
        });
        setGlobalProperty('process', {});

        // Act
        const budgetAllowed = ensureGrowthBudget(network, 1);
        const budgetSnapshot = getSparsityBudgetSnapshot(network);

        // Assert
        expect({
          budgetAllowed,
          connectionCount: network.connections.length,
          decision: budgetSnapshot?.decision,
          softBudgetEnvironment: budgetSnapshot?.softBudgetEnvironment,
          softBudgetTriggered: budgetSnapshot?.softBudgetTriggered,
        }).toEqual({
          budgetAllowed: true,
          connectionCount: 1,
          decision: 'prune-then-allow',
          softBudgetEnvironment: 'browser',
          softBudgetTriggered: true,
        });
      });
    });

    describe('given browser heap usage stays below the configured soft memory budget', () => {
      it('keeps the growth decision on the non-soft-budget path', () => {
        // Arrange
        const network = new Network(1, 1, { seed: 10_276 });
        network.configureSparsityBudget({ maxConnections: 2 });
        config.browserMemoryBudgetMB = 4;
        setGlobalProperty('window', {});
        setGlobalProperty('performance', {
          memory: {
            jsHeapSizeLimit: 8 * 1024 * 1024,
            totalJSHeapSize: 3 * 1024 * 1024,
            usedJSHeapSize: 2 * 1024 * 1024,
          },
        });
        setGlobalProperty('process', {});

        // Act
        const budgetAllowed = ensureGrowthBudget(network, 1);
        const budgetSnapshot = getSparsityBudgetSnapshot(network);

        // Assert
        expect({
          budgetAllowed,
          decision: budgetSnapshot?.decision,
          softBudgetEnvironment: budgetSnapshot?.softBudgetEnvironment,
          softBudgetTriggered: budgetSnapshot?.softBudgetTriggered,
        }).toEqual({
          budgetAllowed: true,
          decision: 'allow',
          softBudgetEnvironment: undefined,
          softBudgetTriggered: false,
        });
      });
    });
  });
});

function setGlobalProperty(
  propertyName: 'window' | 'performance' | 'process',
  value: object,
): void {
  Object.defineProperty(globalThis, propertyName, {
    configurable: true,
    value,
  });
}

function restoreGlobalProperty(
  propertyName: 'window' | 'performance' | 'process',
  descriptor: PropertyDescriptor | undefined,
): void {
  if (descriptor) {
    Object.defineProperty(globalThis, propertyName, descriptor);
    return;
  }

  Reflect.deleteProperty(globalThis, propertyName);
}
