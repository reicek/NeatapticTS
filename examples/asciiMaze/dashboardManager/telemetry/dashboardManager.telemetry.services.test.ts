import Network from '../../../../src/architecture/network';
import Node from '../../../../src/architecture/node';
import type { DashboardManagerState } from '../dashboardManager.types';
import { createDetailedStatsSnapshot } from './dashboardManager.telemetry.services';

describe('createDetailedStatsSnapshot', () => {
  it('exports compact activation scheduling details from the current best network', () => {
    const network = new Network(1, 1, {
      seed: 901,
      enforceAcyclic: false,
    });
    const inputNode = network.nodes[0];
    const outputNode = network.nodes[1];
    const hiddenNode = new Node('hidden');

    network.nodes = [inputNode, hiddenNode, outputNode];
    network.connections.slice().forEach((connection) => {
      network.disconnect(connection.from, connection.to);
    });
    network.connect(inputNode, hiddenNode);
    network.connect(hiddenNode, hiddenNode);
    network.connect(hiddenNode, outputNode);
    network.activate([1]);

    const detailedStats = createDetailedStatsSnapshot(
      createDashboardState(network),
    );

    expect(detailedStats?.activationScheduling).toEqual({
      requestedMode: 'recurrent',
      executionPath: 'compiled-schedule',
      issue: null,
      stepCount: 3,
      recurrentComponentCount: 1,
    });
  });
});

function createDashboardState(network: Network): DashboardManagerState {
  return {
    solvedMazes: [],
    solvedMazeKeys: new Set(),
    currentBest: {
      result: {
        success: false,
        steps: 4,
        path: [
          [0, 0],
          [1, 0],
        ],
        fitness: 12.5,
        progress: 50,
      },
      network: network as unknown as import('../../interfaces').INetwork,
      generation: 7,
    },
    lastTelemetry: null,
    lastBestFitness: 12.5,
    histories: {
      bestFitness: [10, 12.5],
      complexityNodes: [],
      complexityConns: [],
      hypervolume: [],
      progress: [25, 50],
      speciesCount: [],
    },
    lastDetailedStats: null,
    runStartTs: null,
    perfStart: null,
    lastGeneration: 7,
    lastUpdateTs: null,
    scratch: {
      scores: [],
      speciesSizes: [],
      operatorStats: [],
      mutationEntries: [],
    },
  };
}
