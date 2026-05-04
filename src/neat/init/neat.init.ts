/**
 * Constructor bootstrap helpers for the shared NEAT controller facade.
 *
 * This chapter owns the one-time startup policy that would otherwise crowd the
 * public `Neat` class with constructor-only bookkeeping. The main facade stays
 * responsible for the long-lived API surface, while this file keeps the
 * generation-zero setup readable in one place: apply option defaults, prepare
 * controller-owned state, attempt the first population build, then enable the
 * features that depend on that initial shape.
 *
 * That ordering is deliberate rather than cosmetic. Existing tests, migration
 * seams, and several adjacent helper chapters assume the constructor behaves as
 * a stable bootstrap pipeline instead of a grab-bag of independent writes. In
 * particular, lineage tracking and RNG rebinding happen after the pool attempt
 * so they observe the same host state that older constructor paths expected.
 *
 * The controller-facing distinction to keep in mind is ownership over time.
 * `init/` owns generation-zero setup. It does not own evaluation, evolution,
 * telemetry, or persistence once the controller is already alive. That sounds
 * obvious, but startup logic is one of the easiest places for a mature library
 * to accumulate unrelated writes. When that happens, constructor behavior
 * becomes hard to explain and even harder to preserve during refactors.
 *
 * This root chapter keeps the bootstrap contract readable by asking a narrower
 * question: before the first generation can exist, what must become true about
 * the controller state, and in what order?

 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   options[Constructor options]:::base --> defaults[Apply public defaults]:::accent
 *   defaults --> state[Prepare controller-owned state]:::base
 *   state --> pool[Attempt generation-zero pool creation]:::base
 *   pool --> lineage[Enable lineage after startup pool attempt]:::base
 *   lineage --> rng[Rebind RNG access on live instance]:::base
 * ```

 * Bootstrap also sits beside later runtime chapters that answer different
 * questions:

 * ```mermaid
 * flowchart LR
 *   Startup[Constructor time] --> InitChapter[init: make the controller ready to exist]
 *   Runtime[After startup] --> Evaluate[evaluate: score the population]
 *   Runtime --> Evolve[evolve: breed and mutate]
 *   Runtime --> Telemetry[telemetry: inspect what the run is doing]
 * ```
 *
 * Read this file when you want to answer three startup questions:
 *
 * - which public defaults become concrete controller policy during
 *   construction,
 * - which internal state containers must exist before generation-zero work can
 *   proceed safely,
 * - why the constructor delegates here without turning this file into a second
 *   public facade.
 *
 * Historically, startup helpers like this appear once a controller grows large
 * enough that preserving constructor order becomes a compatibility problem, not
 * just a style preference. This chapter is the explicit record of that order.
 */

import type Network from '../../architecture/network/network';
import Connection from '../../architecture/connection/connection';
import * as methods from '../../methods/methods';
import { selection as selectionMethods } from '../../methods/selection/selection';
import { createInnovationTracker } from '../innovation-tracker/innovation-tracker';
import type { InnovationTracker } from '../innovation-tracker/innovation-tracker.types';

export interface NeatConstructorDefaults {
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

/**
 * Minimal constructor-time host contract required by the bootstrap helper.
 *
 * The boundary intentionally stays smaller than the full `Neat` class. This
 * helper needs write access to a few controller-owned fields and one pool
 * creation hook, but it does not own evaluation, evolution, or persistence.
 * Keeping the contract narrow prevents the init chapter from quietly becoming a
 * second facade.
 */
export interface NeatInitializationHost {
  options: NeatInitializationOptions;
  population: Network[];
  _lineageEnabled: boolean;
  _getRNG: () => () => number;
  createPool: (network: Network | null) => void;
  _noveltyArchive?: number[][];
  _innovationTracker?: InnovationTracker;
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

/**
 * Mutable constructor request packet consumed during bootstrap.
 *
 * `optionBag` is the live options object that the public facade keeps after the
 * constructor returns, `rawOptions` preserves the caller's original intent for
 * checks that should not be default-inflated, and `defaults` supplies the
 * public baseline constants exported by the surrounding NEAT surface.
 */
export interface InitializeNeatConstructorRequest {
  optionBag: NeatInitializationOptions;
  rawOptions: NeatInitializationOptions;
  defaults: NeatConstructorDefaults;
}

/**
 * Apply the legacy constructor bootstrap sequence behind the public `Neat`
 * facade.
 *
 * This helper exists to keep [src/neat.ts](src/neat.ts) focused on the public
 * class surface while preserving the exact startup order that current tests,
 * migrated helper chapters, and persistence-adjacent bootstrap paths already
 * rely on.
 *
 * The helper coordinates five constructor responsibilities without becoming a
 * second orchestration facade:
 *
 * 1. materialize public defaults onto the caller-owned options bag,
 * 2. prepare controller bookkeeping state and arrays,
 * 3. attempt generation-zero pool creation when a seed network or population
 *    size is available,
 * 4. enable lineage only after the initial pool attempt has settled,
 * 5. rebind RNG access so later helpers see the live controller instance.
 *
 * The key design constraint is order preservation. Several later reads assume
 * the host already owns concrete defaults and initialized internal state before
 * pool creation runs, while lineage and RNG access should only activate once
 * the bootstrap attempt has finished. Treat this helper as the constructor's
 * setup pipeline, not as a general-purpose runtime entrypoint.
 *
 * @param host - `Neat` instance receiving constructor-time side effects.
 * @param request - Mutable options bag, raw constructor intent, and public
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
 *
 * After the call returns, the instance has concrete startup policy, prepared
 * controller state, and either an attempted generation-zero pool or a safely
 * preserved empty population ready for later work.
 */
export function initializeNeatConstructor(
  host: NeatInitializationHost,
  request: InitializeNeatConstructorRequest,
): void {
  const { optionBag, rawOptions, defaults } = request;

  // Step 1: Apply public defaults directly onto the caller-supplied options bag.
  applyOptionDefaults(host, optionBag, defaults);

  // Step 2: Ensure the runtime bookkeeping containers exist before pool creation.
  host.population = host.population ?? [];
  ensureInternalState(host);

  // Step 3: Preserve legacy best-effort pool bootstrapping semantics.
  bootstrapInitialPool(host, optionBag);

  // Step 3b: Align the innovation tracker cursor above all innovation IDs
  // already assigned to connections in the initial population. Without this,
  // mutation-assigned IDs start at 0 and eventually collide with connection
  // IDs that were assigned by the Connection constructor counter.
  seedInnovationTrackerAboveConnectionCounter(host._innovationTracker);

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
  if (!host._innovationTracker)
    host._innovationTracker = createInnovationTracker();
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

/**
 * Align the innovation tracker cursor above all connection innovation IDs that
 * the Connection constructor already assigned during pool bootstrap.
 *
 * Without this step, the tracker starts at 0 and mutation-assigned innovations
 * eventually collide with the Connection-counter-assigned IDs in the initial
 * population, causing `assertValidGenomeContract` to reject a parent during
 * crossover with a duplicate-innovation error.
 *
 * @param tracker - Live innovation tracker to seed.
 * @returns Nothing.
 */
function seedInnovationTrackerAboveConnectionCounter(
  tracker: InnovationTracker | undefined,
): void {
  if (!tracker) return;
  // Connection._nextInnovation is one past the highest ID already assigned.
  const nextSafeInnovation = Connection.nextInnovation;
  if (tracker.nextInnovationId < nextSafeInnovation) {
    tracker.nextInnovationId = nextSafeInnovation;
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
