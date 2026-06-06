import {
  addOpponentSnapshot,
  applyDecay,
  applyDiffusion,
  clearField,
  computeRoleDivergenceMetric,
  createCollectiveEvaluationContext,
  createOpponentSnapshotPool,
  createSharedField,
  createTeamFitnessEvaluator,
  createTwoPopulationHarness,
  readCell,
  resetCollectiveEvaluationState,
  runCollectiveEvaluationTick,
  runTwoTeamEvaluationTick,
  advanceTwoPopulations,
  writeCell,
} from './neat.nge-collective';

describe('neat.nge-collective (barrel)', () => {
  describe('shared-field exports', () => {
    it('exports createSharedField as a function', () => {
      expect(typeof createSharedField).toBe('function');
    });

    it('exports readCell as a function', () => {
      expect(typeof readCell).toBe('function');
    });

    it('exports writeCell as a function', () => {
      expect(typeof writeCell).toBe('function');
    });

    it('exports applyDecay as a function', () => {
      expect(typeof applyDecay).toBe('function');
    });

    it('exports applyDiffusion as a function', () => {
      expect(typeof applyDiffusion).toBe('function');
    });

    it('exports clearField as a function', () => {
      expect(typeof clearField).toBe('function');
    });
  });

  describe('evaluation exports', () => {
    it('exports createCollectiveEvaluationContext as a function', () => {
      expect(typeof createCollectiveEvaluationContext).toBe('function');
    });

    it('exports runCollectiveEvaluationTick as a function', () => {
      expect(typeof runCollectiveEvaluationTick).toBe('function');
    });

    it('exports resetCollectiveEvaluationState as a function', () => {
      expect(typeof resetCollectiveEvaluationState).toBe('function');
    });
  });

  describe('metrics exports', () => {
    it('exports computeRoleDivergenceMetric as a function', () => {
      expect(typeof computeRoleDivergenceMetric).toBe('function');
    });

    it('exports createOpponentSnapshotPool as a function', () => {
      expect(typeof createOpponentSnapshotPool).toBe('function');
    });

    it('exports addOpponentSnapshot as a function', () => {
      expect(typeof addOpponentSnapshot).toBe('function');
    });
  });

  describe('team-fitness exports', () => {
    it('exports createTeamFitnessEvaluator as a function', () => {
      expect(typeof createTeamFitnessEvaluator).toBe('function');
    });
  });

  describe('two-population exports', () => {
    it('exports createTwoPopulationHarness as a function', () => {
      expect(typeof createTwoPopulationHarness).toBe('function');
    });

    it('exports runTwoTeamEvaluationTick as a function', () => {
      expect(typeof runTwoTeamEvaluationTick).toBe('function');
    });

    it('exports advanceTwoPopulations as a function', () => {
      expect(typeof advanceTwoPopulations).toBe('function');
    });
  });
});
