import Network from '../../../../src/architecture/network/network';
import Node from '../../../../src/architecture/node';
import type { DashboardManagerState } from '../dashboardManager.types';
import {
  createDetailedStatsSnapshot,
  getDashboardLastTelemetry,
  updateTelemetryHistory,
} from './dashboardManager.telemetry.services';

const OPTIONAL_TELEMETRY_CASES: readonly {
  readonly label: string;
  readonly neatInstance: { getTelemetry?: () => unknown[] };
}[] = [
  { label: 'absent telemetry method', neatInstance: {} },
  { label: 'empty telemetry series', neatInstance: { getTelemetry: () => [] } },
];

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

describe('updateTelemetryHistory', () => {
  it.each(OPTIONAL_TELEMETRY_CASES)(
    'records current-best fitness with $label',
    ({ neatInstance }) => {
      const state = createDashboardState(new Network(1, 1));
      state.lastBestFitness = null;
      state.histories.bestFitness = [];

      updateTelemetryHistory(state, neatInstance);

      expect({
        bestFitness: getDashboardLastTelemetry(state).bestFitness,
        history: state.histories.bestFitness,
      }).toEqual({
        bestFitness: 12.5,
        history: [12.5],
      });
    },
  );
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
