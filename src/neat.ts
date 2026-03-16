/*
 * ESLint configuration for intentional `any` usage in NEAT class
 *
 * This file uses `any` strategically for:
 * 1. Runtime metadata properties attached to genomes (_id, _parents, _depth, _reenableProb, etc.)
 *    - These are dynamically added during evolution and don't belong in the Network interface
 * 2. Dynamic options handling during initialization (opts: any)
 *    - Options are validated at runtime and come from user configuration
 * 3. Legacy compatibility for helper function delegation (this as any)
 *    - Maintains backward compatibility while refactored helpers use stricter types
 * 4. Type system limitations with cross-module interfaces
 *    - GenomeWithMetadata vs Network type bridging where runtime behavior is sound
 *
 * All `any` usage here is intentional, documented, and necessary for the architecture.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */

import Network from './architecture/network';
import type {
  ObjectiveDescriptor,
  SpeciesHistoryEntry,
  OperatorStatsRecord,
  SpeciesLike,
  TelemetryEntry,
  ParetoArchiveEntry,
  ObjectiveEvent,
} from './neat/shared/neat.shared.types';
// Static imports (post-migration from runtime require delegates)
import {
  selectMutationMethod,
  mutate,
  mutateAddNodeReuse,
  mutateAddConnReuse,
} from './neat/mutation/mutation';
import { evolve } from './neat/evolve/evolve';
import { evaluate } from './neat/evaluate/evaluate';
import {
  createPool,
  spawnFromParent,
  addGenome,
} from './neat/helpers/neat.helpers';
import { _getObjectives } from './neat/objectives/objectives';
import {
  computeDiversityStats,
  structuralEntropy,
} from './neat/diversity/diversity';
import type { DiversityStats } from './neat/diversity/diversity';
import { _fallbackInnov, _compatibilityDistance } from './neat/compat/compat';
import {
  _speciate,
  _applyFitnessSharing,
  _sortSpeciesMembers,
  _updateSpeciesStagnation,
} from './neat/speciation/speciation';
import { LINEAGE_SNAPSHOT_DEFAULT_LIMIT } from './neat/telemetry/accessors/telemetry.accessors';
import {
  DEFAULT_MAX_PARETO_FRONTS,
  DEFAULT_PARETO_ARCHIVE_JSONL_MAX,
  DEFAULT_PARETO_ARCHIVE_MAX_ENTRIES,
} from './neat/multiobjective/metrics/multiobjective.metrics';
import { SPECIES_HISTORY_JSONL_MAX_DEFAULT } from './neat/species/history/species.history';
import { getParent } from './neat/selection/selection';
import {
  exportPopulation,
  importPopulation,
  exportState,
  importStateImpl,
  toJSONImpl,
  fromJSONImpl,
} from './neat/export/neat.export';
import { getOrCreateRng, type RngHost } from './neat/rng/rng';
import { invalidateGenomeCaches } from './neat/cache/cache';
import { createOffspring } from './neat/evolve/offspring/evolve.offspring.utils';
import { warnIfNoBestGenome } from './neat/evolve/warnings/evolve.warnings.utils';
import { initializeNeatConstructor } from './neat/init/neat.init';
import type { NeatMaintenanceFacadeHost } from './neat/maintenance/facade/maintenance.facade';
import type { NeatPruningFacadeHost } from './neat/pruning/facade/pruning.facade';
import type { NeatRngFacadeHost } from './neat/rng/facade/rng.facade';
import type { NeatPopulationSummaryFacadeHost } from './neat/selection/facade/selection.facade';
import type { NeatTelemetryFacadeHost } from './neat/telemetry/facade/telemetry.facade';
import * as neatMaintenanceFacade from './neat/maintenance/facade/maintenance.facade';
import * as neatPruningFacade from './neat/pruning/facade/pruning.facade';
import * as neatRngFacade from './neat/rng/facade/rng.facade';
import * as neatPopulationSummaryFacade from './neat/selection/facade/selection.facade';
import * as neatTelemetryFacade from './neat/telemetry/facade/telemetry.facade';

/**
 * Root orchestration surface for NeuroEvolution of Augmenting Topologies (NEAT) in NeatapticTS.
 *
 * Within the broader `src` surface, this file is the public chapter map for the
 * library's evolutionary controller.
 * The heavy algorithmic work lives in focused `src/neat/**` modules, but this
 * root surface keeps the user-facing workflow readable: configure a population,
 * evaluate genomes, evolve the next generation, inspect telemetry, and persist
 * the state when a run becomes worth keeping.
 *
 * What this file covers:
 * - the high-level lifecycle of a NEAT run,
 * - the default knobs that shape population growth and speciation,
 * - reproducibility helpers such as seeded RNG snapshots and state export,
 * - educational inspection hooks for telemetry, species history, diversity, and Pareto fronts.
 *
 * What you can learn here:
 * - how the top-level controller stays orchestration-first even after the internal SOLID split,
 * - which public methods belong to setup, evolution, diagnostics, and persistence,
 * - which subsystem README to open next when you want the underlying mechanics.
 *
 * ```mermaid
 * flowchart LR
 *   Configure["Configure run<br/>sizes + fitness + options"] --> Seed["Seed or restore<br/>constructor / createPool / import"]
 *   Seed --> Score["Score population<br/>evaluate()"]
 *   Score --> Breed["Breed next generation<br/>evolve() / mutate()"]
 *   Breed --> Observe["Observe search pressure<br/>telemetry / species / diversity / Pareto"]
 *   Observe --> Persist["Persist or replay<br/>exportState() / toJSON() / RNG state"]
 *   Breed --> Score
 * ```
 *
 * Recommended reading after this root chapter:
 * - `./neat/evaluate/README.md` for scoring flow and objective handling
 * - `./neat/evolve/README.md` for reproduction orchestration
 * - `./neat/speciation/README.md` for compatibility distance and sharing
 * - `./neat/telemetry/README.md` for diagnostics and export surfaces
 *
 * @module neat
 */

/**
 * Internal permissive option bag backing the public `NeatOptions` alias.
 *
 * The root controller still accepts a wide option surface while the chaptered
 * implementation modules keep migrating toward narrower local contracts.
 */
type Options = { [k: string]: any };

/**
 * Public configuration bag for `Neat` evolutionary runs.
 *
 * `NeatOptions` collects the knobs that shape how search pressure is applied.
 * In practice, readers can think about the options in four teaching-friendly groups:
 *
 * - search size and tempo: `popsize`, `elitism`, `provenance`, `mutationRate`, `mutationAmount`
 * - species formation: compatibility threshold plus the excess, disjoint, and weight-difference coefficients
 * - observability: telemetry, lineage, diversity sampling, species history, Pareto archive controls
 * - reproducibility: `seed`, imported RNG state, and exported run state
 *
 * That organization matters because most experiment tuning questions are really
 * questions about pressure: how many candidates compete, how disruptive mutation
 * should feel, how aggressively genomes split into species, and how much evidence
 * you want to retain while the run is unfolding.
 *
 * ```ts
 * const options: NeatOptions = {
 *   popsize: 150,
 *   elitism: 5,
 *   mutationRate: 0.6,
 *   compatibilityThreshold: 3,
 *   fastMode: true,
 *   seed: 42,
 * };
 *
 * const neat = new Neat(3, 1, fitness, options);
 * ```
 *
 * This alias stays intentionally permissive for compatibility with legacy callers.
 * Prefer treating it as the stable front door and the narrower helper-level types in
 * `src/neat/**` as implementation detail.
 */
export type NeatOptions = Options;
/** Default population size when caller does not specify `popsize`. */
export const DEFAULT_POPULATION_SIZE = 50;
/** Default elitism count applied when unspecified. */
export const DEFAULT_ELITISM = 0;
/** Default provenance count applied when unspecified. */
export const DEFAULT_PROVENANCE = 0;
/** Default mutation rate tuned for test expectations. */
export const DEFAULT_MUTATION_RATE = 0.7;
/** Default number of mutation operations per genome. */
export const DEFAULT_MUTATION_AMOUNT = 1;
/** Default compatibility threshold controlling speciation distance. */
export const DEFAULT_COMPATIBILITY_THRESHOLD = 3;
/** Default maximum allowed nodes (Infinity = unbounded). */
export const DEFAULT_MAX_NODES = Infinity;
/** Default maximum allowed connections (Infinity = unbounded). */
export const DEFAULT_MAX_CONNS = Infinity;
/** Default maximum allowed gates (Infinity = unbounded). */
export const DEFAULT_MAX_GATES = Infinity;
/** Default excess coefficient for NEAT compatibility distance. */
export const DEFAULT_EXCESS_COEFF = 1;
/** Default disjoint coefficient for NEAT compatibility distance. */
export const DEFAULT_DISJOINT_COEFF = 1;
/** Default average weight difference coefficient for compatibility distance. */
export const DEFAULT_WEIGHT_DIFF_COEFF = 0.5;
/** Default pair-sample size used by diversity metrics in fast mode. */
export const DEFAULT_DIVERSITY_PAIR_SAMPLE = 20;
/** Default graphlet sample size used by diversity metrics in fast mode. */
export const DEFAULT_DIVERSITY_GRAPHLET_SAMPLE = 30;
/** Default neighbor count for novelty search when k is unspecified. */
export const DEFAULT_NOVELTY_K = 5;

/**
 * High-level NEAT controller that keeps the public workflow linear while the implementation stays chaptered.
 *
 * If you are learning the library, this is the class to read first. It owns the
 * practical experiment loop and answers the first questions most users ask:
 * how to seed a population, when to call `evaluate()` versus `evolve()`, how to
 * inspect species and telemetry, and how to export or replay a run deterministically.
 *
 * Design-wise, `Neat` is intentionally orchestration-first. Mutation operators,
 * speciation rules, telemetry formatting, archive management, cache invalidation,
 * and pruning policies all live in dedicated modules so this top-level surface can
 * stay readable even as the underlying algorithm becomes richer.
 *
 * Minimal workflow:
 *
 * ```ts
 * const neat = new Neat(2, 1, fitness, {
 *   popsize: 50,
 *   seed: 7,
 *   fastMode: true,
 * });
 *
 * await neat.evaluate();
 * const bestGenome = await neat.evolve();
 *
 * console.log(bestGenome.score);
 * console.log(neat.getTelemetry().at(-1));
 * ```
 */
class Neat {
  input: number;
  output: number;
  fitness: (network: Network) => number;
  options: Options;
  population: Network[] = [];
  generation: number = 0;
  /** Internal numeric state for the deterministic xorshift RNG when no user RNG is provided. */
  private _rngState?: number;
  /** Cached RNG function; created lazily and seeded from `_rngState` when used. */
  private _rng?: () => number;
  /** Operator statistics used by adaptive operator selection. */
  private _operatorStats: Map<string, OperatorStatsRecord> = new Map();
  /** Counter for assigning unique genome ids. */
  private _nextGenomeId: number = 1;
  /** Whether lineage metadata should be recorded on genomes. */
  private _lineageEnabled: boolean = false;
  /** Last observed count of inbreeding (used for detecting excessive cloning). */
  private _lastInbreedingCount: number = 0;
  /** Telemetry buffer storing diagnostic snapshots per generation. */
  private _telemetry: TelemetryEntry[] = [];
  /** Time-series history of species stats (for exports/telemetry). */
  private _speciesHistory: SpeciesHistoryEntry[] = [];
  /** Archive of Pareto front metadata for multi-objective tracking. */
  private _paretoArchive: ParetoArchiveEntry[] = [];
  /** Archive storing Pareto objectives snapshots. */
  private _paretoObjectivesArchive: number[][] = [];
  /** Novelty archive used by novelty search (behavior representatives). */
  private _noveltyArchive: number[][] = [];
  /** Queue of recent objective activation/deactivation events for telemetry. */
  private _objectiveEvents: ObjectiveEvent[] = [];
  /** Duration of the last evaluation run (ms). */
  private _lastEvalDuration?: number;
  /** Duration of the last evolve run (ms). */
  private _lastEvolveDuration?: number;
  /** Cached diversity metrics (computed lazily). */
  private _diversityStats?: DiversityStats;

  /**
   * Construct a new `Neat` controller around a fitness function and an option bag.
   *
   * The constructor does not just store values. It also normalizes the incoming
   * options, seeds deterministic randomness when requested, applies root defaults,
   * and prepares the controller so later chapter modules can assume a coherent host.
   * That makes construction the moment where experiment intent becomes runtime policy.
   *
   * @example
   * const neat = new Neat(3, 1, (network) => {
   *   const output = network.activate([0.2, 0.8, 1])[0];
   *   return 1 - Math.abs(output - 0.75);
   * }, {
   *   popsize: 80,
   *   mutationRate: 0.5,
   *   seed: 42,
   * });
   *
   * @param input Number of input neurons each genome should expose.
   * @param output Number of output neurons each genome should expose.
   * @param fitness Fitness function used during `evaluate()`.
   * @param options Optional run configuration overriding the built-in defaults.
   */
  constructor(
    input?: number,
    output?: number,
    fitness?: any,
    options: any = {},
  ) {
    this.input = input ?? 0;
    this.output = output ?? 0;
    this.fitness = fitness ?? (() => 0);
    this.options = options || {};

    initializeNeatConstructor(this as any, {
      optionBag: this.options as any,
      rawOptions: options as any,
      defaults: {
        populationSize: DEFAULT_POPULATION_SIZE,
        elitism: DEFAULT_ELITISM,
        provenance: DEFAULT_PROVENANCE,
        mutationRate: DEFAULT_MUTATION_RATE,
        mutationAmount: DEFAULT_MUTATION_AMOUNT,
        compatibilityThreshold: DEFAULT_COMPATIBILITY_THRESHOLD,
        maxNodes: DEFAULT_MAX_NODES,
        maxConns: DEFAULT_MAX_CONNS,
        maxGates: DEFAULT_MAX_GATES,
        excessCoeff: DEFAULT_EXCESS_COEFF,
        disjointCoeff: DEFAULT_DISJOINT_COEFF,
        weightDiffCoeff: DEFAULT_WEIGHT_DIFF_COEFF,
        diversityPairSample: DEFAULT_DIVERSITY_PAIR_SAMPLE,
        diversityGraphletSample: DEFAULT_DIVERSITY_GRAPHLET_SAMPLE,
        noveltyK: DEFAULT_NOVELTY_K,
      },
    });
  }

  // === Static factories ===
  /**
   * Restore a full evolutionary snapshot produced by `exportState()`.
   *
   * Use this when you want a paused experiment to resume with its controller
   * metadata, population, and archival context intact rather than rebuilding
   * only the bare genomes.
   *
   * @param bundle Serialized object with the shape `{ neat, population }`.
   * @param fitness Fitness function to attach to the restored controller.
   * @returns A `Neat` instance ready to continue evolution from the imported state.
   */
  static async importState(
    bundle: any,
    fitness: (n: Network) => number,
  ): Promise<Neat> {
    return (await importStateImpl.call(
      Neat as any,
      bundle,
      fitness as never,
    )) as unknown as Neat;
  }

  /**
   * Rebuild a `Neat` controller from serialized metadata without importing a population bundle.
   *
   * This is the lighter-weight sibling of `importState()`. It is useful when you
   * want controller defaults, innovation bookkeeping, or archive metadata back,
   * but you are handling genome population state separately.
   *
   * @param json Serialized controller metadata produced by `toJSON()`.
   * @param fitness Fitness function to attach to the reconstructed controller.
   * @returns Reconstructed `Neat` controller instance.
   */
  static fromJSON(json: any, fitness: (n: Network) => number): Neat {
    return fromJSONImpl.call(
      Neat as any,
      json,
      fitness as never,
    ) as unknown as Neat;
  }

  // === Population setup & RNG ===
  /**
   * Create the initial population pool, optionally cloning from a seed network.
   *
   * This is the explicit population bootstrap surface. Call it when you want to
   * start from a known architecture template instead of relying on whatever setup
   * a surrounding example or harness applies for you.
   *
   * @param network Optional template network copied into the initial pool.
   */
  createPool(network: Network | null): void {
    try {
      if (createPool && typeof createPool === 'function')
        createPool.call(this as any, network as never);
    } catch {
      // Pool creation is best-effort; swallow errors to preserve initialization.
    }
  }

  /**
   * Return the current opaque RNG numeric state used by the instance.
   * Useful for deterministic test replay and debugging.
   *
   * @returns Snapshot of the current controller RNG state.
   */
  snapshotRNGState() {
    return neatRngFacade.snapshotRNGState(this as unknown as NeatRngFacadeHost);
  }

  /**
   * Restore a previously-snapshotted RNG state. This restores the internal
   * seed but does not re-create the RNG function until next use.
   *
   * @param state Opaque numeric RNG state produced by `snapshotRNGState()`.
   * @returns Nothing. The controller will resume from the restored RNG state on next use.
   */
  restoreRNGState(state: any) {
    neatRngFacade.restoreRNGState(this as unknown as NeatRngFacadeHost, state);
  }

  /**
   * Import an RNG state (alias for restore; kept for compatibility).
   * @param state Numeric RNG state.
   * @returns Nothing. This is a compatibility alias for `restoreRNGState()`.
   */
  importRNGState(state: any) {
    neatRngFacade.importRNGState(this as unknown as NeatRngFacadeHost, state);
  }

  /**
   * Export the current RNG state for external persistence or tests.
   *
   * @returns Opaque RNG snapshot suitable for later replay.
   */
  exportRNGState() {
    return neatRngFacade.exportRNGState(this as unknown as NeatRngFacadeHost);
  }

  /**
   * Produce deterministic random samples using the instance RNG.
   *
   * @param sampleCount Number of random values to generate.
   * @returns Array of deterministic random samples.
   */
  sampleRandom(sampleCount: number): number[] {
    return neatRngFacade.sampleRandom(
      this as unknown as NeatRngFacadeHost,
      sampleCount,
    );
  }

  // === Evolution lifecycle ===
  /**
   * Advance the evolutionary loop by one generation.
   *
   * Conceptually, `evolve()` is the reproduction half of NEAT. It selects parents,
   * preserves elites and provenance when configured, applies structural and parametric
   * mutation, updates search bookkeeping, and returns the best genome observed for the step.
   * The heavy mechanics live in `src/neat/evolve/evolve.ts`; this method stays as the
   * readable front door to that orchestration.
   *
   * @example
   * // Score the current population first, then breed the next generation.
   * await neat.evaluate();
   * await neat.evolve();
   *
   * @returns Best genome selected by the evolution step.
   */
  async evolve(): Promise<Network> {
    return evolve.call(this as any);
  }

  /**
   * Evaluate the current population using the configured fitness function.
   * Delegates to the migrated evaluation helper to keep this class thin.
   *
   * In practice, this is the scoring half of the controller loop. It transforms a
   * population of candidate networks into evidence the rest of the algorithm can use:
   * fitness scores, objective values, telemetry, diversity statistics, and any derived
   * signals needed by selection or pruning.
   *
   * @returns Aggregated evaluation result (implementation specific).
   */
  async evaluate(): Promise<any> {
    return evaluate.call(this as any);
  }

  /**
   * Apply mutation pressure to the current population without advancing generation bookkeeping.
   *
   * This is the direct "variation" lever. Use it when you want to perturb the
   * current genomes in place for an experiment, a custom training loop, or a
   * test harness that separates mutation from the rest of `evolve()`.
   * In the normal NEAT workflow, `evolve()` is usually the better entry point
   * because it coordinates parent selection, elitism, offspring creation, and
   * mutation as one generation step.
   *
   * @returns Promise resolving once mutation has been applied to the current population.
   */
  async mutate(): Promise<void> {
    return mutate.call(this as any);
  }

  /**
   * Manually apply the configured generation-based pruning policy once.
   *
   * This is mainly useful when you are experimenting with pruning behavior and
   * want to trigger the controller's scheduled pruning logic outside the normal
   * evolve loop.
   *
   * @returns Promise resolving after the pruning policy has been evaluated.
   */
  async applyEvolutionPruning(): Promise<void> {
    return neatPruningFacade.applyEvolutionPruning(
      this as unknown as NeatPruningFacadeHost,
    );
  }

  /**
   * Run the adaptive pruning controller once using the controller's latest signals.
   *
   * Unlike scheduled pruning, this path reacts to the current search state,
   * such as stagnation or complexity pressure, instead of only looking at the
   * generation index.
   *
   * @returns Promise resolving after adaptive pruning has completed.
   */
  async applyAdaptivePruning(): Promise<void> {
    return neatPruningFacade.applyAdaptivePruning(
      this as unknown as NeatPruningFacadeHost,
    );
  }

  /** Emit a standardized warning when evolution loop finds no valid best genome (test hook). */
  _warnIfNoBestGenome() {
    warnIfNoBestGenome();
  }

  // === Reproduction & invariants ===
  /**
   * Select a parent genome using the controller's configured selection strategy.
   *
   * Read this as the "who gets to reproduce" hook. The exact policy depends on
   * the current selection configuration, but the intent is always the same:
   * convert the scored population into a plausible breeding candidate.
   *
   * @returns The selected parent genome.
   * @throws Error if tournament size exceeds population size.
   */
  getParent(): Network {
    return getParent.call(this as any) as unknown as Network;
  }

  /**
   * Build a child genome from parent selection and crossover.
   *
   * Use this when you want one reproduction event without running a full
   * generation step. The method delegates the parent choice to `getParent()` so
   * it still respects the controller's current breeding policy.
   *
   * @returns New network created from selected parent genomes.
   */
  getOffspring(): Network {
    return createOffspring(this as unknown as any, this.getParent.bind(this));
  }

  /**
   * Spawn a new genome derived from a single parent while preserving Neat bookkeeping.
   *
   * @param parent Parent genome to clone and mutate.
   * @param mutateCount Number of mutation passes to apply to the child.
   * @returns Child genome registered with the same bookkeeping conventions as normal evolution.
   */
  spawnFromParent(parent: Network, mutateCount: number = 1): Network {
    return spawnFromParent.call(
      this as any,
      parent as never,
      mutateCount,
    ) as unknown as Network;
  }

  /**
   * Register an externally-created genome into the `Neat` population.
   *
   * @param genome Genome to append into the population.
   * @param parents Optional lineage metadata recorded for teaching and telemetry.
   */
  addGenome(genome: Network, parents?: number[]): void {
    return addGenome.call(this as any, genome as any, parents as any);
  }

  /**
   * Selects a mutation method for a given genome based on constraints.
   *
   * @param genome Genome being considered for mutation.
   * @param rawReturnForTest Whether to expose raw selection output for test visibility.
   * @returns Selected mutation method or `null` when no valid method can be chosen.
   */
  selectMutationMethod(genome: Network, rawReturnForTest: boolean = true): any {
    try {
      return selectMutationMethod.call(
        this as any,
        genome as never,
        rawReturnForTest,
      );
    } catch {
      return null;
    }
  }

  /**
   * Ensure a network has the minimum number of hidden nodes according to configured policy.
   */
  ensureMinHiddenNodes(network: Network, multiplierOverride?: number) {
    return neatMaintenanceFacade.ensureMinHiddenNodes(
      this as unknown as NeatMaintenanceFacadeHost,
      network,
      multiplierOverride,
    );
  }

  /** Repair dead-end connectivity through the focused maintenance facade. */
  ensureNoDeadEnds(network: Network) {
    return neatMaintenanceFacade.ensureNoDeadEnds(
      this as unknown as NeatMaintenanceFacadeHost,
      network,
    );
  }

  /** Minimum hidden size considering explicit minHidden or multiplier policy. */
  getMinimumHiddenSize(multiplierOverride?: number): number {
    return neatMaintenanceFacade.getMinimumHiddenSize(
      this as unknown as NeatMaintenanceFacadeHost,
      multiplierOverride,
    );
  }

  // === Population stats & selection ===
  /**
   * Sorts the population in descending order of fitness scores.
   */
  sort(): void {
    return neatPopulationSummaryFacade.sort(
      this as unknown as NeatPopulationSummaryFacadeHost,
    );
  }

  /**
   * Retrieves the fittest genome from the population.
   */
  getFittest(): Network {
    return neatPopulationSummaryFacade.getFittest(
      this as unknown as NeatPopulationSummaryFacadeHost,
    );
  }

  /**
   * Calculates the average fitness score of the population.
   */
  getAverage(): number {
    return neatPopulationSummaryFacade.getAverage(
      this as unknown as NeatPopulationSummaryFacadeHost,
    );
  }

  // === Telemetry, objectives, and archives ===
  /** Public helper returning just the objective keys (tests rely on). */
  getObjectiveKeys(): string[] {
    return neatTelemetryFacade.getObjectiveKeys(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /**
   * Return the internal telemetry buffer.
   *
   * Telemetry is the controller's teaching surface for understanding why a run is
   * behaving a certain way. Instead of watching only the best score, you can inspect
   * species counts, diversity, objective events, evaluation timing, and other search signals.
   *
   * @returns Recorded telemetry entries in chronological order.
   */
  getTelemetry(): TelemetryEntry[] {
    return neatTelemetryFacade.getTelemetry(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /**
   * Export the telemetry buffer as JSON Lines.
   *
   * JSONL is the easiest format to append to files, stream into data tools, or
   * inspect generation-by-generation without loading a giant array into memory.
   *
   * @returns JSONL payload with one telemetry entry per line.
   */
  exportTelemetryJSONL(): string {
    return neatTelemetryFacade.exportTelemetryJSONL(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /**
   * Export recent telemetry entries as CSV.
   *
   * @param maxEntries Maximum number of recent telemetry entries to export.
   * @returns CSV string for quick spreadsheet or notebook analysis.
   */
  exportTelemetryCSV(maxEntries = 500): string {
    return neatTelemetryFacade.exportTelemetryCSV(
      this as unknown as NeatTelemetryFacadeHost,
      maxEntries,
    );
  }

  /**
   * Clear the recorded telemetry history.
   *
   * This does not reset the population or controller options. It only removes
   * the accumulated diagnostic snapshots so a new experiment phase can start
   * with a clean telemetry timeline.
   */
  clearTelemetry() {
    neatTelemetryFacade.clearTelemetry(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /**
   * Return a lightweight list of registered objective keys and their directions.
   *
   * @returns Objective descriptors currently active on the controller.
   */
  getObjectives(): { key: string; direction: 'max' | 'min' }[] {
    return neatTelemetryFacade.getObjectives(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /**
   * Register a custom objective for multi-objective optimization.
   *
   * Register objectives when a single scalar score is too narrow to express the
   * behavior you want. The controller can then reason about tradeoffs such as raw
   * score versus simplicity, novelty, or domain-specific constraints.
   *
   * @param key Stable objective identifier used in exports and telemetry.
   * @param direction Whether the objective should be minimized or maximized.
   * @param accessor Function extracting the objective value from a genome.
   */
  registerObjective(
    key: string,
    direction: 'min' | 'max',
    accessor: (g: any) => number,
  ) {
    return neatTelemetryFacade.registerTelemetryObjective(
      this as unknown as NeatTelemetryFacadeHost,
      key,
      direction,
      accessor,
    );
  }

  /**
   * Remove all custom objective registrations.
   *
   * Use this when a run is changing from one multi-objective regime to another
   * and you want the controller to forget the previous objective schema.
   */
  clearObjectives() {
    return neatTelemetryFacade.clearTelemetryObjectives(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /** Get recent objective add/remove events for telemetry exports and teaching. */
  getObjectiveEvents(): {
    gen: number;
    type: 'add' | 'remove';
    key: string;
  }[] {
    return neatTelemetryFacade.getObjectiveEvents(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /**
   * Return an array of {id, parents} for the first `limit` genomes in population.
   *
   * @param limit Maximum number of lineage records to return.
   * @returns Compact lineage snapshot for debugging and teaching inheritance flow.
   */
  getLineageSnapshot(
    limit: number = LINEAGE_SNAPSHOT_DEFAULT_LIMIT,
  ): { id: number; parents: number[] }[] {
    return neatTelemetryFacade.getLineageSnapshot(
      this as unknown as NeatTelemetryFacadeHost,
      limit,
    );
  }

  /**
   * Export recent species history as CSV.
   *
   * This is useful when you want to chart species growth, collapse, or
   * stagnation in a spreadsheet or notebook without writing a custom parser.
   *
   * @param maxEntries Maximum number of recent history entries to export.
   * @returns CSV payload representing recent species history snapshots.
   */
  exportSpeciesHistoryCSV(maxEntries = 200): string {
    return neatTelemetryFacade.exportSpeciesHistoryCSV(
      this as unknown as NeatTelemetryFacadeHost,
      maxEntries,
    );
  }

  /**
   * Export recent species history as JSON Lines.
   *
   * Choose this when you want machine-friendly archival output instead of the
   * flatter spreadsheet-oriented CSV export.
   *
   * @param maxEntries Maximum number of recent history entries to export.
   * @returns JSONL payload describing recent species-history entries.
   */
  exportSpeciesHistoryJSONL(
    maxEntries = SPECIES_HISTORY_JSONL_MAX_DEFAULT,
  ): string {
    return neatTelemetryFacade.exportSpeciesHistoryJSONL(
      this as unknown as NeatTelemetryFacadeHost,
      maxEntries,
    );
  }

  /**
   * Return a compact per-species summary for the current population snapshot.
   *
   * This is the quickest inspection surface when you want to know how many
   * niches currently exist, how large they are, and whether they have improved
   * recently.
   *
   * @returns One summary record per active species.
   */
  getSpeciesStats(): {
    id: number;
    size: number;
    bestScore: number;
    lastImproved: number;
  }[] {
    return neatTelemetryFacade.getSpeciesStats(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /**
   * Return the recorded species-history timeline.
   *
   * Unlike `getSpeciesStats()`, which only reflects the current generation,
   * this method exposes the historical view used for trend analysis.
   *
   * @returns Species history entries in recorded order.
   */
  getSpeciesHistory(): SpeciesHistoryEntry[] {
    return neatTelemetryFacade.getSpeciesHistory(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /**
   * Return the current novelty-archive size.
   *
   * This is a small diagnostic hook that tells you whether novelty search is
   * actively accumulating behavior representatives or staying mostly unused.
   *
   * @returns Number of archived novelty descriptors.
   */
  getNoveltyArchiveSize(): number {
    return neatTelemetryFacade.getNoveltyArchiveSize(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /**
   * Return compact multi-objective metrics for each genome in the current population.
   *
   * Use this when you want a flattened view of Pareto rank, crowding, raw score,
   * and structural size without reconstructing the full fronts yourself.
   *
   * @returns One compact metric record per genome.
   */
  getMultiObjectiveMetrics(): {
    rank: number;
    crowding: number;
    score: number;
    nodes: number;
    connections: number;
  }[] {
    return neatTelemetryFacade.getMultiObjectiveMetrics(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /**
   * Return mutation-operator success statistics.
   *
   * These numbers are useful when operator adaptation is enabled and you want
   * to inspect which mutation operators are being rewarded or ignored.
   *
   * @returns Per-operator attempt and success counters.
   */
  getOperatorStats(): { name: string; success: number; attempts: number }[] {
    return neatTelemetryFacade.getOperatorStats(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /**
   * Reconstruct Pareto fronts for the current population snapshot.
   *
   * @param maxFronts Maximum number of fronts to materialize.
   * @returns Fronts ordered from most to least dominant under the active objectives.
   */
  getParetoFronts(maxFronts = DEFAULT_MAX_PARETO_FRONTS): Network[][] {
    return neatTelemetryFacade.getParetoFronts(
      this as unknown as NeatTelemetryFacadeHost,
      maxFronts,
    );
  }

  /**
   * Return recent Pareto archive entries.
   *
   * This is the metadata-oriented archive view. Use it when you want to inspect
   * what front snapshots were retained over time without exporting the full JSONL
   * payload first.
   *
   * @param maxEntries Maximum number of recent archive entries to return.
   * @returns Recent Pareto archive metadata entries.
   */
  getParetoArchive(maxEntries = DEFAULT_PARETO_ARCHIVE_MAX_ENTRIES) {
    return neatTelemetryFacade.getParetoArchive(
      this as unknown as NeatTelemetryFacadeHost,
      maxEntries,
    );
  }

  /**
   * Export recent Pareto archive entries as JSON Lines.
   *
   * This is the easiest way to persist frontier history for offline analysis or
   * later replay in notebooks and visualization tools.
   *
   * @param maxEntries Maximum number of recent archive entries to export.
   * @returns JSONL payload for the requested Pareto archive window.
   */
  exportParetoFrontJSONL(
    maxEntries = DEFAULT_PARETO_ARCHIVE_JSONL_MAX,
  ): string {
    return neatTelemetryFacade.exportParetoFrontJSONL(
      this as unknown as NeatTelemetryFacadeHost,
      maxEntries,
    );
  }

  /**
   * Return timing statistics for the latest evaluation and evolution steps.
   *
   * This is a lightweight performance probe for experiments and benchmarks that
   * need to notice when scoring or breeding costs start drifting upward.
   *
   * @returns Recent runtime statistics for evaluation and evolution work.
   */
  getPerformanceStats() {
    return neatTelemetryFacade.getPerformanceStats(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /**
   * Return the latest cached diversity statistics.
   *
   * Diversity summaries answer a different question than raw fitness: whether
   * the population still explores varied structures or is collapsing toward a
   * narrower family of genomes.
   *
   * @returns Diversity metrics for the current population snapshot.
   */
  getDiversityStats(): DiversityStats {
    return neatTelemetryFacade.getDiversityStats(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /**
   * Reset the novelty archive.
   *
   * This is useful when you want to restart novelty pressure from a clean slate
   * without rebuilding the whole controller.
   */
  resetNoveltyArchive() {
    neatTelemetryFacade.resetNoveltyArchive(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /**
   * Clear the stored Pareto archive.
   *
   * Use this when a new phase of a run should stop comparing itself against the
   * previous archive history.
   */
  clearParetoArchive() {
    neatTelemetryFacade.clearParetoArchive(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  // === Export/import convenience ===
  /**
   * Export the current population as plain JSON objects.
   *
   * Choose this lighter snapshot when you only need the genomes themselves and
   * do not need generation counters, innovation maps, or other controller-level
   * state.
   *
   * @returns JSON-safe population snapshot.
   */
  export(): any[] {
    return exportPopulation.call(this as any);
  }

  /**
   * Replace the current population with serialized genomes.
   *
   * This is the population-only restore path. It keeps the current controller
   * instance, options, and metadata while swapping in a different genome set.
   * Use `importState()` when you want to restore controller metadata too.
   *
   * @param json Serialized population to import into the current controller.
   * @returns Promise resolving after the population is loaded.
   */
  async import(json: any[]): Promise<void> {
    return importPopulation.call(this as any, json as any);
  }

  /**
   * Export the full controller state, including metadata and population.
   *
   * This is the pause-and-resume snapshot. It is the best choice when you want
   * to continue the same run later with the same innovation history, generation
   * counter, and serialized genomes.
   *
   * @returns Full controller snapshot including metadata and population.
   */
  exportState(): any {
    return exportState.call(this as any);
  }

  /**
   * Serialize controller metadata without the concrete population.
   *
   * This is useful when you want to preserve run configuration and innovation
   * bookkeeping separately from genome payloads, or when the population will be
   * reconstructed by other means.
   *
   * @returns JSON-safe metadata snapshot useful for innovation-history persistence.
   */
  toJSON(): any {
    return toJSONImpl.call(this as any);
  }

  // === Private/internal helpers ===
  /**
   * Internal: return cached objective descriptors, building if stale.
   * @returns Cached or freshly built objective descriptors.
   */
  private _getObjectives(): ObjectiveDescriptor[] {
    return _getObjectives.call(this as any) as ObjectiveDescriptor[];
  }

  /**
   * Invalidate per-genome caches (compatibility distance, forward pass, etc.).
   * @param genome Genome instance whose caches should be cleared.
   */
  private _invalidateGenomeCaches(genome: any) {
    invalidateGenomeCaches(genome);
  }

  /**
   * Compute and cache diversity statistics used by telemetry and tests.
   * @returns Cached diversity statistics snapshot.
   */
  private _computeDiversityStats(): DiversityStats {
    const computedStats =
      computeDiversityStats(this.population, this) ??
      buildEmptyDiversityStats(this.population.length);
    this._diversityStats = computedStats;
    return computedStats;
  }

  /**
   * Compatibility wrapper retained for tests that reference (neat as any)._structuralEntropy.
   * @param genome Genome whose structural entropy is calculated.
   * @returns Structural entropy score for the genome.
   */
  private _structuralEntropy(genome: Network): number {
    return structuralEntropy(genome);
  }

  // Perform ADD_NODE honoring global innovation reuse mapping
  /**
   * Add-node mutation that reuses global innovation ids when possible.
   * @param genome Genome receiving the mutation.
   * @returns Mutated genome with added node.
   */
  private _mutateAddNodeReuse(genome: Network) {
    return mutateAddNodeReuse.call(this as any, genome as never);
  }

  /**
   * Add-connection mutation that reuses global innovation ids when possible.
   * @param genome Genome receiving the mutation.
   * @returns Mutated genome with added connection.
   */
  private _mutateAddConnReuse(genome: Network) {
    return mutateAddConnReuse.call(this as any, genome as never);
  }

  /**
   * Fallback innovation id resolver used when reuse mapping is absent.
   * @param conn Connection metadata used to derive the innovation id.
   * @returns Innovation id for the connection.
   */
  private _fallbackInnov(conn: any): number {
    return _fallbackInnov.call(this as any, conn);
  }

  /**
   * Compute compatibility distance between two networks (delegates to compat module).
   * @param netA First network for comparison.
   * @param netB Second network for comparison.
   * @returns Compatibility distance scalar.
   */
  _compatibilityDistance(netA: Network, netB: Network): number {
    return _compatibilityDistance.call(this as any, netA, netB);
  }

  /**
   * Partition population into species using configured compatibility metrics.
   * @returns Updated species assignments.
   */
  private _speciate() {
    return _speciate.call(this as any);
  }

  /**
   * Apply fitness sharing adjustments within each species.
   * @returns Adjusted species fitness data.
   */
  private _applyFitnessSharing() {
    return _applyFitnessSharing.call(this as any);
  }

  /**
   * Sort members within a species according to fitness and lineage rules.
   * @param sp Species whose members should be sorted.
   * @returns Sorted species members.
   */
  private _sortSpeciesMembers(sp: SpeciesLike) {
    return _sortSpeciesMembers.call(this as any, sp);
  }

  /**
   * Update stagnation metrics per species to inform pruning and selection.
   * @returns Updated stagnation state.
   */
  private _updateSpeciesStagnation() {
    return _updateSpeciesStagnation.call(this as any);
  }

  // Lightweight RNG accessor used throughout migrated modules
  /**
   * Provide a memoized RNG function, initializing from internal state if needed.
   * @returns RNG function bound to this instance.
   */
  private _getRNG(): () => number {
    return getOrCreateRng(this as unknown as RngHost);
  }
}

export default Neat;

/**
 * Build a zeroed diversity stats snapshot to use when no population metrics exist yet.
 * @param populationSize Population size used to populate the snapshot.
 * @returns DiversityStats with zeroed aggregates.
 */
function buildEmptyDiversityStats(populationSize: number): DiversityStats {
  return {
    lineageMeanDepth: 0,
    lineageMeanPairDist: 0,
    meanNodes: 0,
    meanConns: 0,
    nodeVar: 0,
    connVar: 0,
    meanCompat: 0,
    graphletEntropy: 0,
    population: populationSize,
  };
}
