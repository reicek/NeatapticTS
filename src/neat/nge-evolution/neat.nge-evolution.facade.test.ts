import ngeEvolution, {
  applyNgeEvolutionEpigeneticPrior,
  computeNgeEvolutionCompatibilityDistance,
  NGE_EVOLUTION_DEFAULT_ALPHA_TOPOLOGY,
  NGE_EVOLUTION_DEFAULT_COMPATIBILITY_DISTANCE_WEIGHTS,
  reproduceParthenogenesis,
  reproducePolyandric,
  reproduceSexual,
  NgeEvolution_BudgetError,
  NgeEvolution_ModeError,
} from './neat.nge-evolution';
import ngeEvolutionUtils, {
  ngeEvolutionCompatibilityUtils,
  ngeEvolutionConstants,
  ngeEvolutionEpigeneticUtils,
  ngeEvolutionErrors,
  ngeEvolutionReproductionUtils,
} from './neat.nge-evolution.utils';

/**
 * Owner-local tests for the NGE facade and helper shelves.
 */
describe('nge evolution facade shelf', () => {
  it('bundles the public runtime surface on one owner-local import path', () => {
    expect({
      alphaTopology: ngeEvolution.NGE_EVOLUTION_DEFAULT_ALPHA_TOPOLOGY,
      apply:
        ngeEvolution.applyNgeEvolutionEpigeneticPrior ===
        applyNgeEvolutionEpigeneticPrior,
      budgetError:
        ngeEvolution.NgeEvolution_BudgetError === NgeEvolution_BudgetError,
      compatibility:
        ngeEvolution.computeNgeEvolutionCompatibilityDistance ===
        computeNgeEvolutionCompatibilityDistance,
      defaultLifecycleWeight:
        NGE_EVOLUTION_DEFAULT_COMPATIBILITY_DISTANCE_WEIGHTS.lifecycle,
      modeErrorName: new NgeEvolution_ModeError('mode').name,
      parthenogenesis:
        ngeEvolution.reproduceParthenogenesis === reproduceParthenogenesis,
      polyandric: ngeEvolution.reproducePolyandric === reproducePolyandric,
      sexual: ngeEvolution.reproduceSexual === reproduceSexual,
    }).toEqual({
      alphaTopology: NGE_EVOLUTION_DEFAULT_ALPHA_TOPOLOGY,
      apply: true,
      budgetError: true,
      compatibility: true,
      defaultLifecycleWeight: 0.2,
      modeErrorName: 'NgeEvolution_ModeError',
      parthenogenesis: true,
      polyandric: true,
      sexual: true,
    });
  });
});

describe('nge evolution utils shelf', () => {
  it('groups helper chapters, defaults, and error classes under stable namespaces', () => {
    expect({
      compatibility:
        ngeEvolutionUtils.ngeEvolutionCompatibilityUtils ===
        ngeEvolutionCompatibilityUtils,
      decay: ngeEvolutionConstants.NGE_EVOLUTION_DEFAULT_EPIGENETIC_DECAY,
      epigenetic:
        ngeEvolutionUtils.ngeEvolutionEpigeneticUtils ===
        ngeEvolutionEpigeneticUtils,
      errorName: new ngeEvolutionErrors.NgeEvolution_BudgetError('budget').name,
      errors: ngeEvolutionUtils.ngeEvolutionErrors === ngeEvolutionErrors,
      reproduction:
        ngeEvolutionUtils.ngeEvolutionReproductionUtils ===
        ngeEvolutionReproductionUtils,
      reproductionKeys: Object.keys(ngeEvolutionReproductionUtils).toSorted(),
    }).toEqual({
      compatibility: true,
      decay: 0.05,
      epigenetic: true,
      errorName: 'NgeEvolution_BudgetError',
      errors: true,
      reproduction: true,
      reproductionKeys: [
        'reproduceParthenogenesis',
        'reproducePolyandric',
        'reproduceSexual',
      ],
    });
  });
});
