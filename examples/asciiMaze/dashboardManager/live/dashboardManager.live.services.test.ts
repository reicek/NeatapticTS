import Network from '../../../../src/architecture/network';
import Node from '../../../../src/architecture/node';
import type {
  DashboardManagerContext,
  DashboardManagerState,
} from '../dashboardManager.types';
import { formatDashboardStat } from '../dashboardManager.utils';
import {
  redrawDashboard,
  resolveActivationSchedulingValue,
} from './dashboardManager.live.services';

const ANSI_ESCAPE_REGEX = new RegExp(
  `${String.fromCharCode(27)}\\[[0-9;]*m`,
  'g',
);

describe('resolveActivationSchedulingValue', () => {
  it('formats a compact scheduling summary for the live dashboard', () => {
    expect(
      resolveActivationSchedulingValue({
        activationScheduling: {
          requestedMode: 'recurrent',
          executionPath: 'compiled-schedule',
          issue: null,
          stepCount: 3,
          recurrentComponentCount: 1,
        },
      } as DashboardManagerState['lastDetailedStats']),
    ).toBe('recurrent via compiled-schedule');
  });
});

describe('redrawDashboard', () => {
  it('prints the scheduling summary in the live stats section when available', () => {
    const network = new Network(1, 1, {
      seed: 902,
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

    const loggedLines: string[] = [];
    const context: DashboardManagerContext = {
      state: createDashboardState(network),
      clearFn: jest.fn(),
      logFn: (...args: unknown[]) => {
        loggedLines.push(args.map((value) => String(value)).join(' '));
      },
      archiveFn: undefined,
      logBlank: () => {
        loggedLines.push('');
      },
      formatStat: formatDashboardStat,
    };

    redrawDashboard(context, ['S.E'], undefined);

    expect(stripAnsi(loggedLines.join('\n'))).toMatch(
      /Scheduling:\s+recurrent via compiled-schedule/,
    );
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
          [2, 0],
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

function stripAnsi(value: string): string {
  return value.replace(ANSI_ESCAPE_REGEX, '');
}
