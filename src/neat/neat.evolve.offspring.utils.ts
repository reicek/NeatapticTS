import Network from '../architecture/network';
import {
  LINEAGE_BASE_DEPTH,
  LINEAGE_DEPTH_INCREMENT,
  OFFSPRING_FALLBACK_INDEX,
} from './neat.evolve.offspring.constants';

/**
 * Minimal surface needed for offspring generation.
 */
export interface OffspringContext {
  population: Network[];
  options: { equal?: boolean; reenableProb?: number };
  _getRNG: () => () => number;
  _nextGenomeId: number;
  _lineageEnabled?: boolean;
  _lastInbreedingCount?: number;
  ensureMinHiddenNodes?: (genome: Network) => void;
  ensureNoDeadEnds?: (genome: Network) => void;
}

/**
 * Create a child genome by crossing two parents selected via the provided callback.
 *
 * @param context - NEAT-like host containing population and options.
 * @param selectParent - Callback to select a parent genome.
 * @returns Newly created offspring genome.
 */
export function createOffspring(
  context: OffspringContext,
  selectParent: () => Network,
): Network {
  const parentOne = safelySelectParent(context, selectParent);
  const parentTwo = safelySelectParent(
    context,
    selectParent,
    context.population,
  );

  const offspring = Network.crossOver(
    parentOne,
    parentTwo,
    context.options.equal ?? false,
  );

  annotateOffspringMetadata(context, offspring, parentOne, parentTwo);
  enforceOffspringInvariants(context, offspring);

  return offspring;
}

function safelySelectParent(
  context: OffspringContext,
  selectParent: () => Network,
  populationFallback?: Network[],
): Network {
  try {
    return selectParent();
  } catch {
    const fallbackPopulation = populationFallback ?? context.population;
    const fallbackGenome =
      fallbackPopulation[OFFSPRING_FALLBACK_INDEX] ?? fallbackPopulation.at(0);
    if (fallbackGenome) return fallbackGenome;

    const rng = context._getRNG();
    const populationLength = fallbackPopulation.length;
    const randomIndex = Math.floor(rng() * Math.max(populationLength, 1));
    return (
      fallbackPopulation[randomIndex] ??
      fallbackPopulation[OFFSPRING_FALLBACK_INDEX]
    );
  }
}

function annotateOffspringMetadata(
  context: OffspringContext,
  offspring: Network,
  parentOne: Network,
  parentTwo: Network,
): void {
  (offspring as any)._reenableProb = context.options.reenableProb;
  (offspring as any)._id = context._nextGenomeId++;

  if (!context._lineageEnabled) return;

  const parentOneDepth = (parentOne as any)._depth ?? LINEAGE_BASE_DEPTH;
  const parentTwoDepth = (parentTwo as any)._depth ?? LINEAGE_BASE_DEPTH;

  (offspring as any)._parents = [
    (parentOne as any)._id,
    (parentTwo as any)._id,
  ];
  (offspring as any)._depth =
    LINEAGE_DEPTH_INCREMENT + Math.max(parentOneDepth, parentTwoDepth);

  if ((parentOne as any)._id === (parentTwo as any)._id) {
    context._lastInbreedingCount = (context._lastInbreedingCount ?? 0) + 1;
  }
}

function enforceOffspringInvariants(
  context: OffspringContext,
  offspring: Network,
): void {
  context.ensureMinHiddenNodes?.(offspring);
  context.ensureNoDeadEnds?.(offspring);
}
