import {
  FLAPPY_BROWSER_ELITISM_COUNT,
  FLAPPY_BROWSER_POPULATION_SIZE,
} from '../../constants/constants';
import type { ExampleArchitectureProfileId } from '../../../architectureProfiles';

const FLAPPY_BROWSER_SPARSE_POPULATION_SIZE = 10;
const FLAPPY_BROWSER_SPARSE_ELITISM_COUNT = 4;
const FLAPPY_BROWSER_NARX_POPULATION_SIZE = 10;
const FLAPPY_BROWSER_NARX_ELITISM_COUNT = 2;
const FLAPPY_BROWSER_GRU_POPULATION_SIZE = 10;
const FLAPPY_BROWSER_GRU_ELITISM_COUNT = 2;
const FLAPPY_BROWSER_LSTM_POPULATION_SIZE = 10;
const FLAPPY_BROWSER_LSTM_ELITISM_COUNT = 2;

/**
 * Browser-local population budget for one architecture profile.
 *
 * The browser demo intentionally gives different profiles different flock sizes
 * because recurrent builders need more room than the lightweight MLP baseline,
 * while still staying responsive enough for an interactive page.
 */
export interface RuntimePopulationBudget {
  elitismCount: number;
  populationSize: number;
}

/**
 * Resolves the browser evolution budget for one architecture profile.
 *
 * These browser budgets are fixed performance caps rather than hardware-scaled
 * targets. The live page keeps MLP at its tiny baseline, gives Sparse a modest
 * 20-bird flock, and caps the heavier recurrent builders at 10 birds so
 * initialization and generation turnover stay responsive.
 *
 * @param architectureProfileId - Selected shared Flappy profile id.
 * @returns Browser-local population and elitism settings.
 */
export function resolveRuntimePopulationBudget(
  architectureProfileId: ExampleArchitectureProfileId,
): RuntimePopulationBudget {
  // Step 1: Keep Sparse modestly above the recurrent profiles without reintroducing heavy startup cost.
  if (architectureProfileId === 'random-sparse') {
    return {
      populationSize: FLAPPY_BROWSER_SPARSE_POPULATION_SIZE,
      elitismCount: FLAPPY_BROWSER_SPARSE_ELITISM_COUNT,
    };
  }

  // Step 2: Keep NARX intentionally tiny because browser warm startup is already expensive.
  if (architectureProfileId === 'narx') {
    return {
      populationSize: FLAPPY_BROWSER_NARX_POPULATION_SIZE,
      elitismCount: FLAPPY_BROWSER_NARX_ELITISM_COUNT,
    };
  }

  // Step 3: Keep GRU on the same tiny flock as NARX so initialization stays responsive.
  if (architectureProfileId === 'gru') {
    return {
      populationSize: FLAPPY_BROWSER_GRU_POPULATION_SIZE,
      elitismCount: FLAPPY_BROWSER_GRU_ELITISM_COUNT,
    };
  }

  // Step 4: Keep LSTM on the same tiny flock as the other heavy recurrent builders.
  if (architectureProfileId === 'lstm') {
    return {
      populationSize: FLAPPY_BROWSER_LSTM_POPULATION_SIZE,
      elitismCount: FLAPPY_BROWSER_LSTM_ELITISM_COUNT,
    };
  }

  // Step 5: Keep the shared lightweight browser baseline for MLP.
  return {
    populationSize: FLAPPY_BROWSER_POPULATION_SIZE,
    elitismCount: FLAPPY_BROWSER_ELITISM_COUNT,
  };
}
