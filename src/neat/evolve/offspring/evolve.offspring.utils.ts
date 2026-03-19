import Network from '../../../architecture/network';
import {
  LINEAGE_BASE_DEPTH,
  LINEAGE_DEPTH_INCREMENT,
  OFFSPRING_FALLBACK_INDEX,
} from './evolve.offspring.constants';

/**
 * Minimal surface needed for offspring generation.
 *
 * The context stays intentionally small so offspring creation can be reused by
 * the broader population-construction chapter without dragging the whole evolve
 * controller surface into this helper layer.
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

type OffspringMetadataCarrier = Network & {
  _reenableProb?: number;
  _id?: number;
  _parents?: Array<number | undefined>;
  _depth?: number;
};

/**
 * Create a child genome by crossing two parents selected via the provided callback.
 *
 * This helper is the compact crossover pipeline beneath the larger population
 * builder. It asks for two parents, tolerates selection failure through a small
 * fallback path, crosses the parents, annotates runtime metadata such as ids and
 * lineage depth, and finally reapplies minimum structural invariants so the new
 * child is ready for the rest of the evolve loop.
 *
 * @param context - NEAT-like host containing population and options.
 * @param selectParent - Callback to select a parent genome.
 * @returns A newly created offspring genome ready for later mutation and scoring.
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

/**
 * Select a parent with deterministic and random fallback recovery.
 *
 * Parent-selection helpers can fail during sparse populations, edge-case test
 * harnesses, or aggressive species filters. Instead of letting that failure
 * abort offspring creation immediately, this helper first falls back to a known
 * stable population index and then to a random population read when needed.
 *
 * @param context - NEAT-like host containing population and options.
 * @param selectParent - Callback to select a parent genome.
 * @param populationFallback - Optional alternate population to read from.
 * @returns A parent genome chosen from the preferred or fallback path.
 */
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

/**
 * Attach runtime metadata to a newly crossed child.
 *
 * The metadata step keeps offspring creation compatible with later lineage,
 * telemetry, and inbreeding reads without forcing the crossover call itself to
 * know about controller-level bookkeeping.
 *
 * @param context - NEAT-like host containing population and options.
 * @param offspring - Newly crossed child genome.
 * @param parentOne - First selected parent.
 * @param parentTwo - Second selected parent.
 * @returns Nothing.
 */
function annotateOffspringMetadata(
  context: OffspringContext,
  offspring: Network,
  parentOne: Network,
  parentTwo: Network,
): void {
  const offspringMetadata = offspring as OffspringMetadataCarrier;
  const parentOneMetadata = parentOne as OffspringMetadataCarrier;
  const parentTwoMetadata = parentTwo as OffspringMetadataCarrier;

  offspringMetadata._reenableProb = context.options.reenableProb;
  offspringMetadata._id = context._nextGenomeId++;

  if (!context._lineageEnabled) return;

  const parentOneDepth = parentOneMetadata._depth ?? LINEAGE_BASE_DEPTH;
  const parentTwoDepth = parentTwoMetadata._depth ?? LINEAGE_BASE_DEPTH;

  offspringMetadata._parents = [parentOneMetadata._id, parentTwoMetadata._id];
  offspringMetadata._depth =
    LINEAGE_DEPTH_INCREMENT + Math.max(parentOneDepth, parentTwoDepth);

  if (parentOneMetadata._id === parentTwoMetadata._id) {
    context._lastInbreedingCount = (context._lastInbreedingCount ?? 0) + 1;
  }
}

/**
 * Reapply minimum structural invariants after crossover.
 *
 * Crossover can produce a child that is technically valid for heredity but still
 * missing the controller's minimum hidden-node or dead-end guarantees. This
 * helper keeps that cleanup local to offspring creation so later population code
 * can treat returned children as already normalized.
 *
 * @param context - NEAT-like host containing population and options.
 * @param offspring - Newly crossed child genome.
 * @returns Nothing.
 */
function enforceOffspringInvariants(
  context: OffspringContext,
  offspring: Network,
): void {
  context.ensureMinHiddenNodes?.(offspring);
  context.ensureNoDeadEnds?.(offspring);
}
