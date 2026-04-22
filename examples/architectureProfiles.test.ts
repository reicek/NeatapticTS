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
      // Arrange
      const flappyProfile = resolveExampleArchitectureProfile('flappy-bird', 'mlp');
      const asciiProfile = resolveExampleArchitectureProfile('ascii-maze', 'mlp');

      // Act
      const resolvedProfiles = {
        asciiConfiguration: asciiProfile.configuration,
        asciiFamily: asciiProfile.family,
        asciiApproved: asciiProfile.approvedForDemo,
        asciiId: asciiProfile.id,
        flappyConfiguration: flappyProfile.configuration,
        flappyFamily: flappyProfile.family,
        flappyApproved: flappyProfile.approvedForDemo,
        flappyId: flappyProfile.id,
      };

      // Assert
      expect({
        ...resolvedProfiles,
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
          input: 6,
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
      // Arrange
      const approvedFlappyProfiles = getApprovedExampleArchitectureProfiles('flappy-bird');
      const approvedAsciiProfiles = getApprovedExampleArchitectureProfiles('ascii-maze');

      // Act
      const approvedProfileIds = {
        ascii: approvedAsciiProfiles.map((profile) => profile.id),
        flappy: approvedFlappyProfiles.map((profile) => profile.id),
      };

      // Assert
      expect({
        ...approvedProfileIds,
      }).toEqual({
        ascii: ['mlp'],
        flappy: ['mlp', 'random-sparse', 'narx', 'gru', 'lstm'],
      });
    });
  });

  describe('buildExampleArchitectureProfileNetwork()', () => {
    it('builds explicit-role feed-forward runtimes from the shared MLP contract', () => {
      // Arrange
      const flappyNetwork = buildExampleArchitectureProfileNetwork(
        'flappy-bird',
        'mlp',
      );
      const asciiNetwork = buildExampleArchitectureProfileNetwork(
        'ascii-maze',
        'mlp',
      );

      // Act
      const networkShapes = {
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
      };

      // Assert
      expect({
        ...networkShapes,
      }).toEqual({
        ascii: {
          inputNodeIds: 6,
          outputNodeIds: 4,
          topologyIntent: 'feed-forward',
        },
        flappy: {
          inputNodeIds: 6,
          outputNodeIds: 2,
          topologyIntent: 'feed-forward',
        },
      });
    });

    describe('when the profile is the Flappy GRU preset', () => {
      it('adds one direct readout connection for each current-frame input and output pair', () => {
        // Arrange
        const flappyGruNetwork = buildExampleArchitectureProfileNetwork(
          'flappy-bird',
          'gru',
        );

        // Act
        const directInputToOutputConnectionCount = flappyGruNetwork.connections.filter(
          (connection) =>
            connection.from.type === 'input' && connection.to.type === 'output',
        ).length;

        // Assert
        expect(directInputToOutputConnectionCount).toBe(12);
      });
    });

    describe('when the profile is the Flappy NARX preset', () => {
      it('captures the seed as a strict genome without leaking non-canonical node roles', () => {
        // Arrange
        const flappyNarxNetwork = buildExampleArchitectureProfileNetwork(
          'flappy-bird',
          'narx',
        );

        // Act
        const captureStrictGenome = (): void => {
          createGenomeFromNetwork(flappyNarxNetwork);
        };

        // Assert
        expect(captureStrictGenome).not.toThrow();
      });
    });

    describe('when the profile is the Flappy GRU preset', () => {
      it('captures the seed as a strict genome before worker bootstrap', () => {
        // Arrange
        const flappyGruNetwork = buildExampleArchitectureProfileNetwork(
          'flappy-bird',
          'gru',
        );

        // Act
        const captureStrictGenome = (): void => {
          createGenomeFromNetwork(flappyGruNetwork);
        };

        // Assert
        expect(captureStrictGenome).not.toThrow();
      });
    });

    describe('when the profile is the Flappy LSTM preset', () => {
      it('captures the seed as a strict genome before worker bootstrap', () => {
        // Arrange
        const flappyLstmNetwork = buildExampleArchitectureProfileNetwork(
          'flappy-bird',
          'lstm',
        );

        // Act
        const captureStrictGenome = (): void => {
          createGenomeFromNetwork(flappyLstmNetwork);
        };

        // Assert
        expect(captureStrictGenome).not.toThrow();
      });
    });
  });
});