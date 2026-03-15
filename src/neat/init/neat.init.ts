/**
 * Constructor bootstrap helpers for the shared NEAT controller facade.
 *
 * This chapter isolates the public `Neat` startup sequence so the main facade
 * can stay focused on the long-lived class surface while constructor-time
 * policy remains readable in one place. The helper deliberately preserves the
 * legacy ordering that existing tests rely on: apply defaults, prepare internal
 * controller state, attempt initial pool creation, then switch on lineage and
 * deterministic RNG access.
 */

import type Network from '../../architecture/network';
import * as methods from '../../methods/methods';
import { selection as selectionMethods } from '../../methods/selection';

interface NeatConstructorDefaults {
  populationSize: number;
  elitism: number;
  provenance: number;
  mutationRate: number;
  mutationAmount: number;
  compatibilityThreshold: number;
  maxNodes: number;
  maxConns: number;
  maxGates: number;
  excessCoeff: number;
  disjointCoeff: number;
  weightDiffCoeff: number;
  diversityPairSample: number;
  diversityGraphletSample: number;
  noveltyK: number;
}

interface NeatNoveltyOptions {
  enabled?: boolean;
  k?: number;
}

interface NeatDiversityMetricsOptions {
  enabled?: boolean;
  pairSample?: number;
  graphletSample?: number;
}

interface NeatMultiObjectiveOptions {
  enabled?: boolean;
  objectives?: unknown[];
}

interface NeatLineageOptions {
  enabled?: boolean;
}

interface NeatLineagePressureOptions {
  enabled?: boolean;
}

interface NeatInitializationOptions {
  popsize?: number;
  elitism?: number;
  provenance?: number;
  mutationRate?: number;
  mutationAmount?: number;
  fitnessPopulation?: boolean;
  clear?: boolean;
  equal?: boolean;
  compatibilityThreshold?: number;
  maxNodes?: number;
  maxConns?: number;
  maxGates?: number;
  excessCoeff?: number;
  disjointCoeff?: number;
  weightDiffCoeff?: number;
  mutation?: unknown[];
  selection?: unknown;
  crossover?: unknown;
  novelty?: NeatNoveltyOptions;
  diversityMetrics?: NeatDiversityMetricsOptions;
  fastMode?: boolean;
  speciation?: boolean;
  multiObjective?: NeatMultiObjectiveOptions;
  network?: Network | null;
  lineage?: NeatLineageOptions;
  lineageTracking?: boolean;
  lineagePressure?: NeatLineagePressureOptions;
  [key: string]: unknown;
}

interface NeatInitializationHost {
  options: NeatInitializationOptions;
  population: Network[];
  _lineageEnabled: boolean;
  _getRNG: () => () => number;
  createPool: (network: Network | null) => void;
  _noveltyArchive?: number[][];
  _nodeSplitInnovations?: Map<string, unknown>;
  _connInnovations?: Map<string, number>;
  _nextGlobalInnovation?: number;
  _species?: unknown[];
  _nextSpeciesId?: number;
  _speciesCreated?: Map<number, number>;
  _prevSpeciesMembers?: Map<number, Set<number>>;
  _speciesLastStats?: Map<number, unknown>;
  _objectiveAges?: Map<string, number>;
  _pendingObjectiveAdds?: unknown[];
  _pendingObjectiveRemoves?: unknown[];
  _objectiveStale?: Map<string, unknown>;
}

interface InitializeNeatConstructorRequest {
  optionBag: NeatInitializationOptions;
  rawOptions: NeatInitializationOptions;
  defaults: NeatConstructorDefaults;
}

/**
 * Apply the legacy constructor bootstrap sequence behind the public `Neat`
 * facade.
 *
 * This helper exists to keep [src/neat.ts](src/neat.ts) focused on the public
 * class surface while preserving the exact startup order that current tests and
 * migrated helpers rely on: mutate the caller-supplied options bag in place,
 * initialize controller state, optionally create the starting population, then
 * enable lineage tracking and bind the RNG accessor.
 *
 * @param host - `Neat` instance receiving constructor-time side effects.
 * @param request - Mutable options bag, raw constructor options, and public
 * default values exported by the facade.
 * @returns Nothing. The helper mutates `host` and `request.optionBag` in place.
 *
 * @example
 * ```ts
 * initializeNeatConstructor(this, {
 *   optionBag: this.options,
 *   rawOptions: options,
 *   defaults: publicDefaults,
 * });
 * ```
 */
export function initializeNeatConstructor(
  host: NeatInitializationHost,
  request: InitializeNeatConstructorRequest,
): void {
  const { optionBag, rawOptions, defaults } = request;

  // Step 1: Apply public defaults directly onto the caller-supplied options bag.
  applyOptionDefaults(host, optionBag, defaults);

  // Step 2: Ensure the runtime bookkeeping containers exist before pool creation.
  host.population = host.population || [];
  ensureInternalState(host);

  // Step 3: Preserve legacy best-effort pool bootstrapping semantics.
  bootstrapInitialPool(host, optionBag);

  // Step 4: Enable lineage tracking only after the startup pool attempt.
  enableLineageTracking(host, optionBag, rawOptions);

  // Step 5: Rebind the RNG accessor onto the live instance for migrated helpers.
  host._getRNG = host._getRNG.bind(host) as () => () => number;
}

function applyOptionDefaults(
  host: NeatInitializationHost,
  optionBag: NeatInitializationOptions,
  defaults: NeatConstructorDefaults,
): void {
  if (optionBag.popsize === undefined)
    optionBag.popsize = defaults.populationSize;
  if (optionBag.elitism === undefined) optionBag.elitism = defaults.elitism;
  if (optionBag.provenance === undefined)
    optionBag.provenance = defaults.provenance;
  if (optionBag.mutationRate === undefined)
    optionBag.mutationRate = defaults.mutationRate;
  if (optionBag.mutationAmount === undefined)
    optionBag.mutationAmount = defaults.mutationAmount;
  if (optionBag.fitnessPopulation === undefined)
    optionBag.fitnessPopulation = false;
  if (optionBag.clear === undefined) optionBag.clear = false;
  if (optionBag.equal === undefined) optionBag.equal = false;
  if (optionBag.compatibilityThreshold === undefined)
    optionBag.compatibilityThreshold = defaults.compatibilityThreshold;
  if (optionBag.maxNodes === undefined) optionBag.maxNodes = defaults.maxNodes;
  if (optionBag.maxConns === undefined) optionBag.maxConns = defaults.maxConns;
  if (optionBag.maxGates === undefined) optionBag.maxGates = defaults.maxGates;
  if (optionBag.excessCoeff === undefined)
    optionBag.excessCoeff = defaults.excessCoeff;
  if (optionBag.disjointCoeff === undefined)
    optionBag.disjointCoeff = defaults.disjointCoeff;
  if (optionBag.weightDiffCoeff === undefined)
    optionBag.weightDiffCoeff = defaults.weightDiffCoeff;
  if (optionBag.mutation === undefined)
    optionBag.mutation = buildDefaultMutationList();
  if (optionBag.selection === undefined)
    optionBag.selection = resolveDefaultSelection();
  if (optionBag.crossover === undefined)
    optionBag.crossover = methods.crossover
      ? methods.crossover.SINGLE_POINT
      : undefined;
  if (optionBag.novelty === undefined) optionBag.novelty = { enabled: false };
  if (optionBag.diversityMetrics === undefined)
    optionBag.diversityMetrics = { enabled: true };
  if (optionBag.fastMode && optionBag.diversityMetrics) {
    if (optionBag.diversityMetrics.pairSample == null)
      optionBag.diversityMetrics.pairSample = defaults.diversityPairSample;
    if (optionBag.diversityMetrics.graphletSample == null)
      optionBag.diversityMetrics.graphletSample =
        defaults.diversityGraphletSample;
    if (optionBag.novelty?.enabled && optionBag.novelty.k == null)
      optionBag.novelty.k = defaults.noveltyK;
  }

  host._noveltyArchive = [];

  if (optionBag.speciation === undefined) optionBag.speciation = false;
  if (
    optionBag.multiObjective?.enabled &&
    !Array.isArray(optionBag.multiObjective.objectives)
  ) {
    optionBag.multiObjective.objectives = [];
  }
}

function ensureInternalState(host: NeatInitializationHost): void {
  if (!host._nodeSplitInnovations) host._nodeSplitInnovations = new Map();
  if (!host._connInnovations) host._connInnovations = new Map();
  if (host._nextGlobalInnovation === undefined) host._nextGlobalInnovation = 0;
  if (!Array.isArray(host._species)) host._species = [];
  if (host._nextSpeciesId === undefined) host._nextSpeciesId = 1;
  if (!host._speciesCreated) host._speciesCreated = new Map();
  if (!host._prevSpeciesMembers) host._prevSpeciesMembers = new Map();
  if (!host._speciesLastStats) host._speciesLastStats = new Map();
  if (!host._objectiveAges) host._objectiveAges = new Map();
  if (!Array.isArray(host._pendingObjectiveAdds))
    host._pendingObjectiveAdds = [];
  if (!Array.isArray(host._pendingObjectiveRemoves))
    host._pendingObjectiveRemoves = [];
  if (!host._objectiveStale) host._objectiveStale = new Map();
}

function bootstrapInitialPool(
  host: NeatInitializationHost,
  optionBag: NeatInitializationOptions,
): void {
  try {
    if (optionBag.network !== undefined) host.createPool(optionBag.network);
    else if (optionBag.popsize) host.createPool(null);
  } catch {
    // Pool creation is best-effort; preserve constructor tolerance.
  }
}

function enableLineageTracking(
  host: NeatInitializationHost,
  optionBag: NeatInitializationOptions,
  rawOptions: NeatInitializationOptions,
): void {
  if (optionBag.lineage?.enabled || (optionBag.provenance ?? 0) > 0) {
    host._lineageEnabled = true;
  }
  if (optionBag.lineageTracking === true) {
    host._lineageEnabled = true;
  }
  if (rawOptions.lineagePressure?.enabled && host._lineageEnabled !== true) {
    host._lineageEnabled = true;
  }
}

function buildDefaultMutationList(): unknown[] {
  if (Array.isArray(methods.mutation.ALL)) {
    return methods.mutation.ALL.slice();
  }
  return methods.mutation.FFW ? [methods.mutation.FFW] : [];
}

function resolveDefaultSelection(): unknown {
  const methodCatalog = methods as typeof methods & {
    selection?: { TOURNAMENT?: unknown };
  };

  return (
    selectionMethods.TOURNAMENT ??
    methodCatalog.selection?.TOURNAMENT ??
    selectionMethods.FITNESS_PROPORTIONATE
  );
}
