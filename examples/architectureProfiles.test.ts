import {
  buildExampleArchitectureProfileNetwork,
  getApprovedExampleArchitectureProfiles,
  resolveExampleArchitectureProfile,
} from './architectureProfiles';
import { createGenomeFromNetwork } from '../src/neat/genome/genome';
import { FLAPPY_NETWORK_HIDDEN_LAYER_SIZES } from './flappy_bird/constants/constants.network';

describe('shared example architecture profiles', () => {
  describe('resolveExampleArchitectureProfile()', () => {
    it('keeps one shared MLP profile id while resolving demo-specific shapes', () => {
      const flappyProfile = resolveExampleArchitectureProfile('flappy-bird', 'mlp');
      const asciiProfile = resolveExampleArchitectureProfile('ascii-maze', 'mlp');

      expect({
        asciiConfiguration: asciiProfile.configuration,
        asciiFamily: asciiProfile.family,
        asciiApproved: asciiProfile.approvedForDemo,
        asciiId: asciiProfile.id,
        flappyConfiguration: flappyProfile.configuration,
        flappyFamily: flappyProfile.family,
        flappyApproved: flappyProfile.approvedForDemo,
        flappyId: flappyProfile.id,
      }).toEqual({
        asciiConfiguration: {
          family: 'MLP',
          hiddenLayerSizes: [6],
          input: 6,
          output: 4,
        },
        asciiFamily: 'MLP',
        asciiApproved: true,
        asciiId: 'mlp',
        flappyConfiguration: {
          family: 'MLP',
          hiddenLayerSizes: FLAPPY_NETWORK_HIDDEN_LAYER_SIZES,
          input: 12,
          output: 2,
        },
        flappyFamily: 'MLP',
        flappyApproved: true,
        flappyId: 'mlp',
      });
    });
  });

  describe('getApprovedExampleArchitectureProfiles()', () => {
    it('exposes the current approved profile set per demo without inventing demo-local names', () => {
      const approvedFlappyProfiles = getApprovedExampleArchitectureProfiles('flappy-bird');
      const approvedAsciiProfiles = getApprovedExampleArchitectureProfiles('ascii-maze');

      expect({
        ascii: approvedAsciiProfiles.map((profile) => profile.id),
        flappy: approvedFlappyProfiles.map((profile) => profile.id),
      }).toEqual({
        ascii: ['mlp'],
        flappy: ['mlp', 'random-sparse', 'narx', 'gru', 'lstm'],
      });
    });
  });

  describe('buildExampleArchitectureProfileNetwork()', () => {
    it('builds explicit-role feed-forward runtimes from the shared MLP contract', () => {
      const flappyNetwork = buildExampleArchitectureProfileNetwork(
        'flappy-bird',
        'mlp',
      );
      const asciiNetwork = buildExampleArchitectureProfileNetwork(
        'ascii-maze',
        'mlp',
      );

      expect({
        ascii: {
          inputNodeIds: asciiNetwork.inputNodeIds.length,
          outputNodeIds: asciiNetwork.outputNodeIds.length,
          topologyIntent: asciiNetwork.getTopologyIntent(),
        },
        flappy: {
          inputNodeIds: flappyNetwork.inputNodeIds.length,
          outputNodeIds: flappyNetwork.outputNodeIds.length,
          topologyIntent: flappyNetwork.getTopologyIntent(),
        },
      }).toEqual({
        ascii: {
          inputNodeIds: 6,
          outputNodeIds: 4,
          topologyIntent: 'feed-forward',
        },
        flappy: {
          inputNodeIds: 12,
          outputNodeIds: 2,
          topologyIntent: 'feed-forward',
        },
      });
    });

    it('enables the Flappy GRU direct readout shortcut so the browser profile can react to current-frame inputs immediately', () => {
      const flappyGruNetwork = buildExampleArchitectureProfileNetwork(
        'flappy-bird',
        'gru',
      );
      const directInputToOutputConnectionCount = flappyGruNetwork.connections.filter(
        (connection) =>
          connection.from.type === 'input' && connection.to.type === 'output',
      ).length;

      expect(directInputToOutputConnectionCount).toBe(24);
    });

    it('captures the shared Flappy NARX seed as a strict genome without leaking non-canonical node roles', () => {
      const flappyNarxNetwork = buildExampleArchitectureProfileNetwork(
        'flappy-bird',
        'narx',
      );

      expect(() => createGenomeFromNetwork(flappyNarxNetwork)).not.toThrow();
    });

    it.each(['gru', 'lstm'] as const)(
      'captures the shared Flappy %s seed as a strict genome before worker bootstrap',
      (architectureProfileId) => {
        const flappyRecurrentNetwork = buildExampleArchitectureProfileNetwork(
          'flappy-bird',
          architectureProfileId,
        );

        expect(() => createGenomeFromNetwork(flappyRecurrentNetwork)).not.toThrow();
      },
    );
  });
});