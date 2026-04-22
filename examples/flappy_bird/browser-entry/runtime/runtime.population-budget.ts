import {
  FLAPPY_BROWSER_ELITISM_COUNT,
  FLAPPY_BROWSER_POPULATION_SIZE,
} from '../../constants/constants';
import type { ExampleArchitectureProfileId } from '../../../architectureProfiles';

const FLAPPY_BROWSER_SPARSE_POPULATION_SIZE = 30;
const FLAPPY_BROWSER_SPARSE_ELITISM_COUNT = 6;
const FLAPPY_BROWSER_NARX_POPULATION_SIZE = 40;
const FLAPPY_BROWSER_NARX_ELITISM_COUNT = 8;
const FLAPPY_BROWSER_GRU_POPULATION_SIZE = 18;
const FLAPPY_BROWSER_GRU_ELITISM_COUNT = 4;
const FLAPPY_BROWSER_LSTM_POPULATION_SIZE = 20;
const FLAPPY_BROWSER_LSTM_ELITISM_COUNT = 4;

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
 * Sparse and NARX keep wider browser budgets than the dense MLP baseline so
 * the interactive demo still has room to discover pipe-clearing behavior in a
 * small number of generations. GRU and LSTM stay smaller than NARX so the live
 * demo remains responsive, but LSTM keeps a broader flock than the old default
 * because the heavier gate stack needs more exploration headroom.
 *
 * @param architectureProfileId - Selected shared Flappy profile id.
 * @returns Browser-local population and elitism settings.
 */
export function resolveRuntimePopulationBudget(
  architectureProfileId: ExampleArchitectureProfileId,
): RuntimePopulationBudget {
  // Step 1: Widen lighter Sparse runs a bit because they stay comparatively cheap.
  if (architectureProfileId === 'random-sparse') {
    return {
      populationSize: FLAPPY_BROWSER_SPARSE_POPULATION_SIZE,
      elitismCount: FLAPPY_BROWSER_SPARSE_ELITISM_COUNT,
    };
  }

  // Step 2: Keep NARX broader than the baseline while trimming its browser cost a bit.
  if (architectureProfileId === 'narx') {
    return {
      populationSize: FLAPPY_BROWSER_NARX_POPULATION_SIZE,
      elitismCount: FLAPPY_BROWSER_NARX_ELITISM_COUNT,
    };
  }

  // Step 3: Keep GRU meaningfully above the MLP baseline without reintroducing visible stutter.
  if (architectureProfileId === 'gru') {
    return {
      populationSize: FLAPPY_BROWSER_GRU_POPULATION_SIZE,
      elitismCount: FLAPPY_BROWSER_GRU_ELITISM_COUNT,
    };
  }

  // Step 4: Give LSTM a mid-sized browser flock so it can discover stable pipe progress without matching NARX cost.
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