import {
  buildExampleArchitectureProfileNetwork,
  getApprovedExampleArchitectureProfiles,
  resolveExampleArchitectureProfile,
} from './architectureProfiles';

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
          hiddenLayerSizes: [16, 8, 4],
          input: 38,
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
        flappy: ['mlp'],
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
          inputNodeIds: 38,
          outputNodeIds: 2,
          topologyIntent: 'feed-forward',
        },
      });
    });
  });
});