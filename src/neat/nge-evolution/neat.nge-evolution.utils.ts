import {
  NGE_EVOLUTION_DEFAULT_ALPHA_COMPUTATION,
  NGE_EVOLUTION_DEFAULT_ALPHA_LIFECYCLE,
  NGE_EVOLUTION_DEFAULT_ALPHA_MEMORY,
  NGE_EVOLUTION_DEFAULT_ALPHA_TOPOLOGY,
  NGE_EVOLUTION_DEFAULT_COMPATIBILITY_DISTANCE_WEIGHTS,
  NGE_EVOLUTION_DEFAULT_EPIGENETIC_DECAY,
  NGE_EVOLUTION_DEFAULT_POLYANDRIC_DRONE_CONTRIBUTION_FRACTION,
  NGE_EVOLUTION_DEFAULT_POLYANDRIC_QUEEN_BIAS,
} from './neat.nge-evolution.constants';
import { computeNgeEvolutionCompatibilityDistance } from './neat.nge-evolution.distance';
import { applyNgeEvolutionEpigeneticPrior } from './neat.nge-evolution.epigenetic';
import {
  NgeEvolution_BudgetError,
  NgeEvolution_ModeError,
  NgeEvolution_RegionError,
} from './neat.nge-evolution.errors';
import {
  reproduceParthenogenesis,
  reproducePolyandric,
  reproduceSexual,
} from './neat.nge-evolution.reproduction';

/**
 * Phase E compatibility-distance helpers grouped under one stable owner-local namespace object.
 */
export const ngeEvolutionCompatibilityUtils = {
  computeNgeEvolutionCompatibilityDistance,
};

/**
 * Birth-time epigenetic prior helper grouped under one stable owner-local namespace object.
 */
export const ngeEvolutionEpigeneticUtils = {
  applyNgeEvolutionEpigeneticPrior,
};

/**
 * Phase E reproduction-mode operator helpers grouped under one stable owner-local namespace.
 */
export const ngeEvolutionReproductionUtils = {
  reproduceParthenogenesis,
  reproducePolyandric,
  reproduceSexual,
};

/**
 * Default Phase E constants grouped under one stable owner-local namespace.
 */
export const ngeEvolutionConstants = {
  NGE_EVOLUTION_DEFAULT_ALPHA_TOPOLOGY,
  NGE_EVOLUTION_DEFAULT_ALPHA_COMPUTATION,
  NGE_EVOLUTION_DEFAULT_ALPHA_MEMORY,
  NGE_EVOLUTION_DEFAULT_ALPHA_LIFECYCLE,
  NGE_EVOLUTION_DEFAULT_COMPATIBILITY_DISTANCE_WEIGHTS,
  NGE_EVOLUTION_DEFAULT_EPIGENETIC_DECAY,
  NGE_EVOLUTION_DEFAULT_POLYANDRIC_DRONE_CONTRIBUTION_FRACTION,
  NGE_EVOLUTION_DEFAULT_POLYANDRIC_QUEEN_BIAS,
};

/**
 * Phase E evolution error classes grouped under one stable owner-local namespace object.
 */
export const ngeEvolutionErrors = {
  NgeEvolution_ModeError,
  NgeEvolution_RegionError,
  NgeEvolution_BudgetError,
};

/**
 * Default runtime helper bundle for the entire nge-evolution owner boundary module.
 */
const ngeEvolutionUtils = {
  ngeEvolutionCompatibilityUtils,
  ngeEvolutionEpigeneticUtils,
  ngeEvolutionReproductionUtils,
  ngeEvolutionConstants,
  ngeEvolutionErrors,
};

export default ngeEvolutionUtils;
