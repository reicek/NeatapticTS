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
} from './neat/neat.types';
// Static imports (post-migration from runtime require delegates)
import {
  selectMutationMethod,
  mutate,
  mutateAddNodeReuse,
  mutateAddConnReuse,
} from './neat/neat.mutation';
import { evolve } from './neat/neat.evolve';
import { evaluate } from './neat/neat.evaluate';
import { createPool, spawnFromParent, addGenome } from './neat/neat.helpers';
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
import { LINEAGE_SNAPSHOT_DEFAULT_LIMIT } from './neat/neat.telemetry.accessors.utils';
import {
  DEFAULT_MAX_PARETO_FRONTS,
  DEFAULT_PARETO_ARCHIVE_JSONL_MAX,
  DEFAULT_PARETO_ARCHIVE_MAX_ENTRIES,
} from './neat/neat.multiobjective.metrics.utils';
import { SPECIES_HISTORY_JSONL_MAX_DEFAULT } from './neat/species/history/species.history';
import { getParent } from './neat/selection/selection';
import {
  exportPopulation,
  importPopulation,
  exportState,
  importStateImpl,
  toJSONImpl,
  fromJSONImpl,
} from './neat/neat.export';
import { getOrCreateRng, type RngHost } from './neat/rng/rng';
import { invalidateGenomeCaches } from './neat/cache/cache';
import { createOffspring } from './neat/neat.evolve.offspring.utils';
import { warnIfNoBestGenome } from './neat/neat.evolve.warnings.utils';
import { initializeNeatConstructor } from './neat/neat.init';
import type { NeatMaintenanceFacadeHost } from './neat/neat.maintenance.facade';
import type { NeatPopulationSummaryFacadeHost } from './neat/neat.population-summary.facade';
import type { NeatPruningFacadeHost } from './neat/pruning/facade/pruning.facade';
import type { NeatRngFacadeHost } from './neat/rng/facade/rng.facade';
import type { NeatTelemetryFacadeHost } from './neat/neat.telemetry.facade';
import * as neatMaintenanceFacade from './neat/neat.maintenance.facade';
import * as neatPopulationSummaryFacade from './neat/neat.population-summary.facade';
import * as neatPruningFacade from './neat/pruning/facade/pruning.facade';
import * as neatRngFacade from './neat/rng/facade/rng.facade';
import * as neatTelemetryFacade from './neat/neat.telemetry.facade';

/**
 * Configuration options for Neat evolutionary runs.
 *
 * Each property is optional and the class applies sensible defaults when a
 * field is not provided. Options control population size, mutation rates,
 * compatibility coefficients, selection strategy and other behavioral knobs.
 *
 * Example:
 * const opts: NeatOptions = { popsize: 100, mutationRate: 0.5 };
 * const neat = new Neat(3, 1, fitnessFn, opts);
 *
 * Note: this type is intentionally permissive to support staged migration and
 * legacy callers; prefer providing a typed options object where possible.
 */
type Options = { [k: string]: any };
// Public re-export for library consumers
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

export default class Neat {
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
   * Construct a new Neat instance.
   * Kept permissive during staged migration; accepts the same signature tests expect.
   *
   * @example
   * // Create a neat instance for 3 inputs and 1 output with default options
   * const neat = new Neat(3, 1, (net) => evaluateFitness(net));
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
   * Convenience: restore full evolutionary state previously produced by exportState().
   * @param bundle Object with shape { neat, population }
   * @param fitness Fitness function to attach
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

  static fromJSON(json: any, fitness: (n: Network) => number): Neat {
    return fromJSONImpl.call(
      Neat as any,
      json,
      fitness as never,
    ) as unknown as Neat;
  }

  // === Population setup & RNG ===
  /**
   * Create initial population pool. Delegates to helpers if present.
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
   */
  snapshotRNGState() {
    return neatRngFacade.snapshotRNGState(this as unknown as NeatRngFacadeHost);
  }

  /**
   * Restore a previously-snapshotted RNG state. This restores the internal
   * seed but does not re-create the RNG function until next use.
   *
   * @param state Opaque numeric RNG state produced by `snapshotRNGState()`.
   */
  restoreRNGState(state: any) {
    neatRngFacade.restoreRNGState(this as unknown as NeatRngFacadeHost, state);
  }

  /**
   * Import an RNG state (alias for restore; kept for compatibility).
   * @param state Numeric RNG state.
   */
  importRNGState(state: any) {
    neatRngFacade.importRNGState(this as unknown as NeatRngFacadeHost, state);
  }

  /**
   * Export the current RNG state for external persistence or tests.
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
   * Evolves the population by selecting, mutating, and breeding genomes.
   * This method is delegated to `src/neat/neat.evolve.ts` during the migration.
   *
   * @example
   * // Run a single evolution step (async)
   * await neat.evolve();
   */
  async evolve(): Promise<Network> {
    return evolve.call(this as any);
  }

  /**
   * Evaluate the current population using the configured fitness function.
   * Delegates to the migrated evaluation helper to keep this class thin.
   *
   * @returns Aggregated evaluation result (implementation specific).
   */
  async evaluate(): Promise<any> {
    return evaluate.call(this as any);
  }

  /**
   * Applies mutations to the population based on the mutation rate and amount.
   * Each genome is mutated using the selected mutation methods.
   * Slightly increases the chance of ADD_CONN mutation for more connectivity.
   */
  async mutate(): Promise<void> {
    return mutate.call(this as any);
  }

  /**
   * Manually apply evolution-time pruning once using the current generation
   * index and configuration in `options.evolutionPruning`.
   */
  async applyEvolutionPruning(): Promise<void> {
    return neatPruningFacade.applyEvolutionPruning(
      this as unknown as NeatPruningFacadeHost,
    );
  }

  /**
   * Run the adaptive pruning controller once.
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
   * Selects a parent genome for breeding based on the selection method.
   * Supports multiple selection strategies, including POWER, FITNESS_PROPORTIONATE, and TOURNAMENT.
   * @returns The selected parent genome.
   * @throws Error if tournament size exceeds population size.
   */
  getParent(): Network {
    return getParent.call(this as any) as unknown as Network;
  }

  /**
   * Generates an offspring by crossing over two parent networks.
   * Uses the crossover method described in the Instinct algorithm.
   * @returns A new network created from two parents.
   */
  getOffspring(): Network {
    return createOffspring(this as unknown as any, this.getParent.bind(this));
  }

  /**
   * Spawn a new genome derived from a single parent while preserving Neat bookkeeping.
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
   */
  addGenome(genome: Network, parents?: number[]): void {
    return addGenome.call(this as any, genome as any, parents as any);
  }

  /**
   * Selects a mutation method for a given genome based on constraints.
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
   */
  getTelemetry(): TelemetryEntry[] {
    return neatTelemetryFacade.getTelemetry(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /** Export telemetry as JSON Lines (one JSON object per line). */
  exportTelemetryJSONL(): string {
    return neatTelemetryFacade.exportTelemetryJSONL(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /**
   * Export recent telemetry entries as CSV.
   */
  exportTelemetryCSV(maxEntries = 500): string {
    return neatTelemetryFacade.exportTelemetryCSV(
      this as unknown as NeatTelemetryFacadeHost,
      maxEntries,
    );
  }

  /** Clear telemetry buffer and cached entries. */
  clearTelemetry() {
    neatTelemetryFacade.clearTelemetry(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /**
   * Return a lightweight list of registered objective keys and their directions.
   */
  getObjectives(): { key: string; direction: 'max' | 'min' }[] {
    return neatTelemetryFacade.getObjectives(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /**
   * Register a custom objective for multi-objective optimization.
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

  /** Clear all registered multi-objective objectives. */
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
   */
  getLineageSnapshot(
    limit: number = LINEAGE_SNAPSHOT_DEFAULT_LIMIT,
  ): { id: number; parents: number[] }[] {
    return neatTelemetryFacade.getLineageSnapshot(
      this as unknown as NeatTelemetryFacadeHost,
      limit,
    );
  }

  /** Export species history as CSV rows for offline inspection. */
  exportSpeciesHistoryCSV(maxEntries = 200): string {
    return neatTelemetryFacade.exportSpeciesHistoryCSV(
      this as unknown as NeatTelemetryFacadeHost,
      maxEntries,
    );
  }

  /** Export species history as JSON Lines for storage and analysis. */
  exportSpeciesHistoryJSONL(
    maxEntries = SPECIES_HISTORY_JSONL_MAX_DEFAULT,
  ): string {
    return neatTelemetryFacade.exportSpeciesHistoryJSONL(
      this as unknown as NeatTelemetryFacadeHost,
      maxEntries,
    );
  }

  /** Return a concise summary for each current species. */
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

  /** Returns the historical species statistics recorded each generation. */
  getSpeciesHistory(): SpeciesHistoryEntry[] {
    return neatTelemetryFacade.getSpeciesHistory(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /** Returns the number of entries currently stored in the novelty archive. */
  getNoveltyArchiveSize(): number {
    return neatTelemetryFacade.getNoveltyArchiveSize(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /** Returns compact multi-objective metrics for each genome in the current population. */
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

  /** Returns a summary of mutation/operator statistics used by operator adaptation. */
  getOperatorStats(): { name: string; success: number; attempts: number }[] {
    return neatTelemetryFacade.getOperatorStats(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /** Reconstruct Pareto fronts for the current population snapshot. */
  getParetoFronts(maxFronts = DEFAULT_MAX_PARETO_FRONTS): Network[][] {
    return neatTelemetryFacade.getParetoFronts(
      this as unknown as NeatTelemetryFacadeHost,
      maxFronts,
    );
  }

  /** Get recent Pareto archive entries (meta information about archived fronts). */
  getParetoArchive(maxEntries = DEFAULT_PARETO_ARCHIVE_MAX_ENTRIES) {
    return neatTelemetryFacade.getParetoArchive(
      this as unknown as NeatTelemetryFacadeHost,
      maxEntries,
    );
  }

  /** Export Pareto front archive as JSON Lines for external analysis. */
  exportParetoFrontJSONL(
    maxEntries = DEFAULT_PARETO_ARCHIVE_JSONL_MAX,
  ): string {
    return neatTelemetryFacade.exportParetoFrontJSONL(
      this as unknown as NeatTelemetryFacadeHost,
      maxEntries,
    );
  }

  /** Return recent performance statistics for the most recent evaluation and evolve operations. */
  getPerformanceStats() {
    return neatTelemetryFacade.getPerformanceStats(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /** Return the latest cached diversity statistics. */
  getDiversityStats(): DiversityStats {
    return neatTelemetryFacade.getDiversityStats(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /** Reset the novelty archive (clear entries). */
  resetNoveltyArchive() {
    neatTelemetryFacade.resetNoveltyArchive(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  /** Clear the Pareto archive. */
  clearParetoArchive() {
    neatTelemetryFacade.clearParetoArchive(
      this as unknown as NeatTelemetryFacadeHost,
    );
  }

  // === Export/import convenience ===
  /**
   * Exports the current population as an array of JSON objects.
   */
  export(): any[] {
    return exportPopulation.call(this as any);
  }

  /**
   * Imports a population from an array of JSON objects.
   */
  async import(json: any[]): Promise<void> {
    return importPopulation.call(this as any, json as any);
  }

  /**
   * Convenience: export full evolutionary state (meta + population genomes).
   */
  exportState(): any {
    return exportState.call(this as any);
  }

  /** Serialize NEAT meta (without population) for persistence of innovation history. */
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
