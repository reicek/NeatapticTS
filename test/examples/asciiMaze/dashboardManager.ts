// Rebuilt clean version with private fields
import { MazeUtils } from './mazeUtils';
import { MazeVisualization } from './mazeVisualization';
import { NetworkVisualization } from './networkVisualization';
import { colors } from './colors';
import { INetwork, IDashboardManager } from './interfaces';

// Region: Type Interfaces ----------------------------------------------------
/** Detailed stats structure produced inside the dashboard. */
interface AsciiMazeDetailedStats {
  generation: number;
  bestFitness: number | null;
  bestFitnessDelta: number | null;
  saturationFraction: number | null;
  actionEntropy: number | null;
  populationMean: number | null;
  populationMedian: number | null;
  enabledConnRatio: number | null;
  complexity: any;
  simplifyPhaseActive: boolean;
  perf: any;
  lineage: any;
  diversity: any;
  speciesCount: number | null;
  topSpeciesSizes: number[] | null;
  objectives: any;
  paretoFrontSizes: number[] | null;
  firstFrontSize: number;
  hypervolume: number | null;
  noveltyArchiveSize: number | null;
  operatorAcceptance: Array<{ name: string; acceptancePct: number }> | null;
  topMutations: Array<{ name: string; count: number }> | null;
  mutationStats: any;
  trends: {
    fitness: string | null;
    nodes: string | null;
    conns: string | null;
    hyper: string | null;
    progress: string | null;
    species: string | null;
  };
  histories: {
    bestFitness: number[];
    nodes: number[];
    conns: number[];
    hyper: number[];
    progress: number[];
    species: number[];
  };
  timestamp: number;
}

/** Public snapshot returned by getLastTelemetry(). */
interface AsciiMazeTelemetrySnapshot {
  generation: number;
  bestFitness: number | null;
  progress: number | null;
  speciesCount: number | null;
  gensPerSec: number;
  timestamp: number;
  details: AsciiMazeDetailedStats | null;
}

/**
 * DashboardManager
 *
 * ASCII dashboard for the maze NEAT example. Tracks current best genome,
 * bounded histories (fitness, complexity, hypervolume, progress, species),
 * archives solved mazes with efficiency stats, and emits telemetry events.
 *
 * @remarks
 * Not reentrant; create a new instance per evolution run. History buffers
 * are capped at `HISTORY_MAX` samples for predictable memory usage.
 */
export class DashboardManager implements IDashboardManager {
  #solvedMazes: Array<{
    maze: string[];
    result: any;
    network: INetwork;
    generation: number;
  }> = [];
  #solvedMazeKeys: Set<string> = new Set();
  #currentBest: {
    result: any;
    network: INetwork;
    generation: number;
  } | null = null;
  #lastTelemetry: any = null;
  #lastBestFitness: number | null = null;
  #bestFitnessHistory: number[] = [];
  #complexityNodesHistory: number[] = [];
  #complexityConnsHistory: number[] = [];
  #hypervolumeHistory: number[] = [];
  #progressHistory: number[] = [];
  #speciesCountHistory: number[] = [];
  #lastDetailedStats: AsciiMazeDetailedStats | null = null;
  #runStartTs: number | null = null;
  #perfStart: number | null = null;
  #lastGeneration: number | null = null;
  #lastUpdateTs: number | null = null;

  #logFn: (...args: any[]) => void;
  #clearFn: () => void;
  #archiveFn?: (...args: any[]) => void;

  static #HISTORY_MAX = 500;
  static #FRAME_INNER_WIDTH = 148;
  static #LEFT_PADDING = 7;
  static #RIGHT_PADDING = 1;
  static #STAT_LABEL_WIDTH = 28;
  static #ARCHIVE_SPARK_WIDTH = 64; // spark width in archive blocks
  static #GENERAL_SPARK_WIDTH = 64; // spark width in live panel
  static #SOLVED_LABEL_WIDTH = 22; // label width in archive stats
  static #HISTORY_EXPORT_WINDOW = 200; // samples exported in telemetry details
  static #SPARK_BLOCKS = Object.freeze([
    '▁',
    '▂',
    '▃',
    '▄',
    '▅',
    '▆',
    '▇',
    '█',
  ]);
  static #DELTA_EPSILON = 1e-9;
  static #TOP_OPERATOR_LIMIT = 6;
  static #TOP_MUTATION_LIMIT = 8;
  static #TOP_SPECIES_LIMIT = 5;
  static #LAYER_INFER_LOOP_MULTIPLIER = 4;
  static #LABEL_PATH_EFF = 'Path efficiency';
  static #LABEL_PATH_OVER = 'Path overhead';
  static #LABEL_UNIQUE = 'Unique cells visited';
  static #LABEL_REVISITS = 'Cells revisited';
  static #LABEL_STEPS = 'Steps';
  static #LABEL_FITNESS = 'Fitness';
  static #LABEL_ARCH = 'Architecture';
  static #FRAME_SINGLE_LINE_CHAR = '═';
  static #FRAME_BRIDGE_TOP = '╦════════════╦';
  static #FRAME_BRIDGE_BOTTOM = '╩════════════╩';
  static #EVOLVING_SECTION_LINE = '══════════════════════';
  static get FRAME_INNER_WIDTH() {
    return DashboardManager.#FRAME_INNER_WIDTH;
  }
  static get LEFT_PADDING() {
    return DashboardManager.#LEFT_PADDING;
  }
  static get RIGHT_PADDING() {
    return DashboardManager.#RIGHT_PADDING;
  }
  static get CONTENT_WIDTH() {
    return (
      DashboardManager.#FRAME_INNER_WIDTH -
      DashboardManager.#LEFT_PADDING -
      DashboardManager.#RIGHT_PADDING
    );
  }
  static get STAT_LABEL_WIDTH() {
    return DashboardManager.#STAT_LABEL_WIDTH;
  }
  static get HISTORY_MAX() {
    return DashboardManager.#HISTORY_MAX;
  }

  /**
   * Create a new DashboardManager.
   *
   * @param clearFn Function that clears the live dashboard region (terminal or DOM). Required.
   * @param logFn Function used for streaming live panel lines. Required.
   * @param archiveFn Optional function used to prepend/append solved-maze archive blocks (separate area / element).
   *
   * Defensive notes:
   * - Non-function arguments are coerced to no-ops to avoid runtime crashes in mixed environments (browser / node tests).
   * - All three functions are stored as private fields (#clearFn, #logFn, #archiveFn) for later reuse.
   */
  constructor(
    clearFn: () => void,
    logFn: (...args: any[]) => void,
    archiveFn?: (...args: any[]) => void
  ) {
    const noop = () => {};
    this.#clearFn = typeof clearFn === 'function' ? clearFn : noop;
    this.#logFn = typeof logFn === 'function' ? logFn : noop;
    this.#archiveFn = typeof archiveFn === 'function' ? archiveFn : undefined;
  }
  /** Emit a blank padded line inside the frame to avoid duplication. */
  #logBlank(): void {
    this.#logFn(
      `${colors.blueCore}║${NetworkVisualization.pad(
        ' ',
        DashboardManager.FRAME_INNER_WIDTH,
        ' '
      )}${colors.blueCore}║${colors.reset}`
    );
  }

  /**
   * Format a single statistic line (label + value) framed for the dashboard.
   *
   * Educational goals:
   * - Demonstrates consistent alignment via fixed label column width.
   * - Centralizes color application so other helpers (`#appendSolvedPathStats`, etc.) remain lean.
   * - Shows simple, allocation‑aware string building without external libs.
   *
   * Steps:
   * 1. Canonicalize label (ensure trailing colon) for uniform appearance.
   * 2. Pad label to `labelWidth` (left aligned) creating a fixed column.
   * 3. Normalize the value to string (numbers preserved; null/undefined become literal strings for transparency).
   * 4. Compose colored content segment (`label` + single space + `value`).
   * 5. Left/right pad inside the frame content width and wrap with vertical border glyphs.
   *
   * Performance notes:
   * - O(L) where L = composed string length; dominated by `padEnd` + `NetworkVisualization.pad`.
   * - Avoids template churn inside loops by keeping construction linear.
   * - No truncation: labels longer than `labelWidth` intentionally overflow to surface overly verbose labels during development.
   *
   * Determinism: Pure formatting; no external state or randomness.
   * Reentrancy: Safe; relies only on parameters and static sizing constants.
   * Edge cases: Empty label yields just a colon after canonicalization (":"); nullish values become "null" / "undefined" explicitly.
   *
   * @param label Descriptive metric label (colon appended if missing).
   * @param value Metric value (string or number) displayed after a space.
   * @param colorLabel ANSI / style token for the label portion.
   * @param colorValue ANSI / style token for the value portion.
   * @param labelWidth Fixed width for the label column (default derives from class constant).
   * @returns Fully framed, colorized line ready for logging.
   * @example
   * const line = (dashboard as any)["#formatStat"]("Fitness", 12.34);
   * // => "║  Fitness: 12.34  ... ║" (color codes omitted here)
   */
  #formatStat(
    label: string,
    value: string | number,
    colorLabel = colors.neonSilver,
    colorValue = colors.cyanNeon,
    labelWidth = DashboardManager.#STAT_LABEL_WIDTH
  ): string {
    // Step 1: Canonicalize label (ensure colon exactly once at end)
    const canonicalLabel = label.endsWith(':') ? label : `${label}:`;

    // Step 2: Fixed-width left-aligned label column
    const paddedLabel = canonicalLabel.padEnd(labelWidth, ' ');

    // Step 3: Normalize value to string (explicit String coercion for transparency)
    const valueString = typeof value === 'number' ? `${value}` : String(value);

    // Step 4: Compose colored inner content segment
    const coloredContent = `${colorLabel}${paddedLabel}${colorValue} ${valueString}${colors.reset}`;

    // Step 5: Wrap with frame borders & horizontal padding
    const leftPadSpaces = ' '.repeat(DashboardManager.LEFT_PADDING);
    const framed = `${
      colors.blueCore
    }║${leftPadSpaces}${NetworkVisualization.pad(
      coloredContent,
      DashboardManager.CONTENT_WIDTH,
      ' ',
      'left'
    )}${' '.repeat(DashboardManager.RIGHT_PADDING)}${colors.blueCore}║${
      colors.reset
    }`;
    return framed;
  }

  /**
   * Convert the tail of a numeric series into a compact Unicode sparkline.
   *
   * Educational intent: illustrates how a simple per-frame trend visualization
   * can be produced without external dependencies, while keeping allocation
   * costs minimal for frequent refreshes (every generation / UI frame).
   *
   * Steps:
   * 1. Slice the most recent `width` samples (via `MazeUtils.tail`) — bounded O(width).
   * 2. Filter out non‑finite samples (defensive; telemetry may contain NaN during warmup).
   * 3. Scan once to derive `minValue` / `maxValue` (range baseline).
   * 4. Map each sample to an index in the precomputed block ramp (#SPARK_BLOCKS).
   * 5. Append corresponding block characters into a single result string.
   *
   * Performance notes:
   * - Single pass min/max + single pass mapping: O(n) with n = min(series.length, width).
   * - No intermediate arrays beyond the tail slice (which reuses existing util) & final string builder.
   * - Uses descriptive local names to keep code educational; hot path is still trivial compared to rendering.
   * - Avoids `Math.min(...spread)` / `Array.prototype.map` to prevent temporary arrays & GC churn.
   *
   * Determinism: Pure function of input array slice (no randomness, no external state).
   * Reentrancy: Safe; no shared mutable scratch used.
   * Edge cases: Returns empty string for empty / all non‑finite input; collapses zero range to uniform block.
   *
   * @param series Numeric history (older -> newer) to visualize.
   * @param width Maximum number of most recent samples to encode (default 32); values <= 0 produce ''.
   * @returns Sparkline string (length <= width). Empty string when insufficient valid data.
   * @example
   * // Given recent fitness scores
   * const spark = dashboardManager["#buildSparkline"]([10,11,11.5,12,13], 4); // -> e.g. "▃▄▆█"
   */
  #buildSparkline(series: number[], width = 32): string {
    // Fast exits for invalid / trivial scenarios
    if (!Array.isArray(series) || !series.length || width <= 0) return '';

    // Step 1: Tail slice (bounded) — relies on existing utility for consistency
    const tailSlice = MazeUtils.tail<number>(series, width);
    const sampleCount = tailSlice.length;
    if (!sampleCount) return '';

    // Step 2: Filter non-finite values in-place by compaction to avoid new array
    let writeIndex = 0;
    for (let readIndex = 0; readIndex < sampleCount; readIndex++) {
      const sampleValue = tailSlice[readIndex];
      if (Number.isFinite(sampleValue)) {
        tailSlice[writeIndex++] = sampleValue;
      }
    }
    if (writeIndex === 0) return '';

    // Step 3: Compute min/max over the compacted prefix [0, writeIndex)
    let minValue = Infinity;
    let maxValue = -Infinity;
    for (let scanIndex = 0; scanIndex < writeIndex; scanIndex++) {
      const value = tailSlice[scanIndex];
      if (value < minValue) minValue = value;
      if (value > maxValue) maxValue = value;
    }
    // Use a small epsilon to guard against zero or near-zero ranges so
    // normalization remains stable (avoids divide-by-zero and huge
    // normalized values when min ~= max). Prefer the class constant for
    // easy tuning in one place.
    let valueRange = maxValue - minValue;
    if (Math.abs(valueRange) < DashboardManager.#DELTA_EPSILON) {
      valueRange = DashboardManager.#DELTA_EPSILON;
    }

    // Step 4: Map each sample to a block index
    const blocks = DashboardManager.#SPARK_BLOCKS;
    const blocksCount = blocks.length - 1; // highest ramp position index
    let sparkline = '';
    for (let encodeIndex = 0; encodeIndex < writeIndex; encodeIndex++) {
      const normalized = (tailSlice[encodeIndex] - minValue) / valueRange; // [0,1]
      const blockIndex = Math.min(
        blocksCount,
        Math.max(0, Math.floor(normalized * blocksCount))
      );
      sparkline += blocks[blockIndex];
    }
    return sparkline;
  }

  /** Create a lightweight key for a maze (dedupe solved mazes). */
  #getMazeKey(maze: string[]): string {
    return maze.join('');
  }

  /** Wrapper to append solved archive block (public logic retained from original). */
  #appendSolvedToArchive(
    solved: {
      maze: string[];
      result: any;
      network: INetwork;
      generation: number;
    },
    displayNumber: number
  ): void {
    if (!this.#archiveFn) return;
    const blockLines: string[] = [];
    this.#appendSolvedHeader(blockLines, solved, displayNumber);
    this.#appendSolvedSparklines(blockLines, solved.network);
    this.#appendSolvedMaze(blockLines, solved);
    this.#appendSolvedPathStats(blockLines, solved);
    // Architecture now included in sparklines section for consolidated solved summary (avoids duplication).
    this.#appendSolvedFooterAndEmit(blockLines);
  }

  /**
   * redraw
   *
   * Clear + repaint the live dashboard frame while updating rich stats snapshot.
   *
   * Steps (delegated to focused helpers for readability & GC awareness):
   * 1. beginFrameRefresh: clear terminal region & print static frame header.
   * 2. printCurrentBestSection: conditionally render evolving section (network, maze, stats, progress).
   * 3. updateDetailedStatsSnapshot: build/export metrics & sparklines using scratch arrays (bounded histories).
   * 4. Emit a spacer line to preserve the original layout rhythm.
   *
   * Performance considerations:
   * - Reuses `#scratch` arrays to avoid per-frame allocations when deriving top lists.
   * - Histories are already bounded (HISTORY_MAX) so sparkline work is O(width).
   * - Early exit when no telemetry & no current best yet.
   *
   * Determinism: Purely formatting & aggregation (no randomness).
   * Reentrancy: Not reentrant (mutates internal state and shared scratch buffers). One instance per run.
   * @param currentMaze Maze currently being evolved.
   * @param neat Optional NEAT implementation instance for population-level stats.
   */
  redraw(currentMaze: string[], neat?: any): void {
    // Update the high-resolution last-update timestamp when a redraw happens.
    this.#lastUpdateTs = globalThis.performance?.now?.() ?? Date.now();
    this.#beginFrameRefresh();
    if (this.#currentBest) this.#printCurrentBestSection(currentMaze);
    this.#updateDetailedStatsSnapshot(neat); // updates #lastDetailedStats (used by getLastTelemetry())
    this.#logBlank(); // spacer preserving legacy visual rhythm
  }

  /** Shared scratch allocations reused across redraw cycles to reduce GC churn. */
  #scratch: {
    scores: number[];
    speciesSizes: number[];
    operatorStats: any[];
    mutationEntries: [string, number][];
  } = { scores: [], speciesSizes: [], operatorStats: [], mutationEntries: [] };

  /** Clear & print static frame top. (Step 1 of redraw) */
  #beginFrameRefresh(): void {
    this.#clearFn();
    this.#printTopFrame();
  }

  /**
   * Build the rich detailed stats snapshot consumed by external telemetry observers.
   *
   * Educational overview:
   * This method aggregates multiple orthogonal evolution signals (fitness trends, structural complexity,
   * diversity, Pareto front geometry, operator acceptance, mutation frequencies, species distribution, etc.) into
   * one immutable plain object assigned to `#lastDetailedStats`. It is invoked once per redraw cycle (not per
   * individual genome evaluation) to amortize cost and keep UI refresh predictable.
   *
   * Steps (high‑level):
   * 1. Guard: if we have neither telemetry nor a current best candidate, skip work (no data yet).
   * 2. Destructure relevant sub-snapshots from the raw telemetry (complexity, perf, lineage, diversity, objectives...).
   * 3. Derive population statistics via `#computePopulationStats` (mean/median/species/enabled ratio) and patch gaps
   *    with the current best's fitness / species count as reasonable fallbacks for early generations.
   * 4. Generate sparkline trend strings for tracked bounded histories (fitness, nodes, conns, hypervolume, progress, species).
   * 5. Derive Pareto front size metrics & novelty archive size (defensive wrappers to tolerate optional APIs).
   * 6. Compute operator acceptance, top mutation operator counts, and largest species sizes (scratch-buffer reuse inside helpers).
   * 7. Compute best fitness delta (difference vs previous sample) for quick “is improving” signal.
   * 8. Assemble and assign a consolidated snapshot object with timestamps & derived boolean flags (e.g. simplifyPhaseActive).
   *
   * Performance notes:
   * - All history arrays are already bounded (HISTORY_MAX); sparkline generation is O(width) each.
   * - Sorting work (operator / mutation / species) is limited to top-N extraction with small fixed caps (config constants).
   * - Uses defensive optional chaining + nullish coalescing to avoid cascading throws; a single try/catch wraps overall build.
   * - Allocations: one snapshot object + a handful of small arrays (top lists). Histories are sliced lazily via helper.
   *
   * Determinism: Pure aggregation of previously captured deterministic data. No RNG usage.
   * Reentrancy: Not reentrant; mutates `#lastDetailedStats`. Acceptable because a single dashboard instance services one run.
   * Failure handling: Any unexpected error aborts this build silently (stats are opportunistic, UI remains functional).
   *
   * @param neat Optional NEAT engine instance (used for population stats, operator stats, novelty archive size, species sizes).
   */
  #updateDetailedStatsSnapshot(neat?: any): void {
    const telemetry = this.#lastTelemetry;
    // Step 1: Early guard when no data yet (avoids unnecessary object churn)
    if (!telemetry && !this.#currentBest) return;
    try {
      // Step 2: Pull out nested telemetry domains with safe optional access
      const complexitySnapshot = telemetry?.complexity;
      const perfSnapshot = telemetry?.perf;
      const lineageSnapshot = telemetry?.lineage;
      const diversitySnapshot = telemetry?.diversity;
      const rawFrontsArray = Array.isArray(telemetry?.fronts)
        ? telemetry.fronts
        : null;
      const objectivesSnapshot = telemetry?.objectives;
      const hypervolumeValue = telemetry?.hyper;
      const mutationStatsObj =
        telemetry?.mutationStats || telemetry?.mutation?.stats;

      // Current best scalar metrics (fitness + auxiliary run stats)
      const bestFitnessValue = this.#currentBest?.result?.fitness;
      const saturationFractionValue = (this.#currentBest as any)?.result
        ?.saturationFraction;
      const actionEntropyValue = (this.#currentBest as any)?.result
        ?.actionEntropy;

      // Step 3: Population-level summary (fills in early-run blanks with best fitness/species when needed)
      const populationStats = this.#computePopulationStats(neat);
      if (
        populationStats.mean == null &&
        typeof bestFitnessValue === 'number'
      ) {
        populationStats.mean = +bestFitnessValue.toFixed(2);
      }
      if (
        populationStats.median == null &&
        typeof bestFitnessValue === 'number'
      ) {
        populationStats.median = +bestFitnessValue.toFixed(2);
      }
      if (
        populationStats.speciesCount == null &&
        typeof telemetry?.species === 'number'
      ) {
        populationStats.speciesCount = telemetry.species;
      }

      // Step 4: Sparklines for bounded histories
      const sparkWidth = DashboardManager.#GENERAL_SPARK_WIDTH;
      const sparklines = {
        fitness:
          this.#buildSparkline(this.#bestFitnessHistory, sparkWidth) || null,
        nodes:
          this.#buildSparkline(this.#complexityNodesHistory, sparkWidth) ||
          null,
        conns:
          this.#buildSparkline(this.#complexityConnsHistory, sparkWidth) ||
          null,
        hyper:
          this.#buildSparkline(this.#hypervolumeHistory, sparkWidth) || null,
        progress:
          this.#buildSparkline(this.#progressHistory, sparkWidth) || null,
        species:
          this.#buildSparkline(this.#speciesCountHistory, sparkWidth) || null,
      } as const;

      // Step 5: Pareto + novelty archive metrics
      const firstFrontSize = rawFrontsArray?.[0]?.length || 0;
      const paretoFrontSizes = rawFrontsArray
        ? rawFrontsArray.map((front: any) => front?.length || 0)
        : null;
      const noveltyArchiveSize = this.#safeInvoke<number | null>(
        () =>
          neat?.getNoveltyArchiveSize ? neat.getNoveltyArchiveSize() : null,
        null
      );

      // Step 6: Operator acceptance, mutation frequencies, species distribution
      const operatorAcceptance = this.#computeOperatorAcceptance(neat);
      const topMutations = this.#computeTopMutations(mutationStatsObj);
      const topSpeciesSizes = this.#computeTopSpeciesSizes(neat);

      // Step 7: Best fitness delta (vs prior sample) — small improvement signal
      const bestFitnessDelta = (() => {
        if (typeof bestFitnessValue !== 'number') return null;
        const previousSample = this.#bestFitnessHistory.at(-2) ?? null;
        if (previousSample == null) return null;
        return +(bestFitnessValue - previousSample).toFixed(3);
      })();

      // Step 8: Consolidated snapshot assignment
      this.#lastDetailedStats = {
        generation: this.#currentBest?.generation || 0,
        bestFitness:
          typeof bestFitnessValue === 'number' ? bestFitnessValue : null,
        bestFitnessDelta,
        saturationFraction:
          typeof saturationFractionValue === 'number'
            ? saturationFractionValue
            : null,
        actionEntropy:
          typeof actionEntropyValue === 'number' ? actionEntropyValue : null,
        populationMean: populationStats.mean,
        populationMedian: populationStats.median,
        enabledConnRatio: populationStats.enabledRatio,
        complexity: complexitySnapshot || null,
        simplifyPhaseActive: !!(
          complexitySnapshot &&
          (complexitySnapshot.growthNodes < 0 ||
            complexitySnapshot.growthConns < 0)
        ),
        perf: perfSnapshot || null,
        lineage: lineageSnapshot || null,
        diversity: diversitySnapshot || null,
        speciesCount: populationStats.speciesCount,
        topSpeciesSizes,
        objectives: objectivesSnapshot || null,
        paretoFrontSizes,
        firstFrontSize,
        hypervolume:
          typeof hypervolumeValue === 'number' ? hypervolumeValue : null,
        noveltyArchiveSize,
        operatorAcceptance,
        topMutations,
        mutationStats: mutationStatsObj || null,
        trends: sparklines,
        histories: {
          bestFitness: this.#sliceHistoryForExport(this.#bestFitnessHistory),
          nodes: this.#sliceHistoryForExport(this.#complexityNodesHistory),
          conns: this.#sliceHistoryForExport(this.#complexityConnsHistory),
          hyper: this.#sliceHistoryForExport(this.#hypervolumeHistory),
          progress: this.#sliceHistoryForExport(this.#progressHistory),
          species: this.#sliceHistoryForExport(this.#speciesCountHistory),
        },
        timestamp: Date.now(),
      };
    } catch {
      // Snapshot production is optional; swallow to keep UI resilient.
    }
  }

  /**
   * Aggregate basic population-wide statistics (mean & median fitness, species count, enabled connection ratio).
   *
   * Educational intent:
   * Centralizes lightweight descriptive statistics needed for trend visualization and telemetry export
   * while demonstrating reuse of a shared scratch array to minimize per-generation allocations.
   *
   * Steps:
   * 1. Guard & early return if the provided engine instance lacks a population array.
   * 2. Reuse `#scratch.scores` (cleared in-place) to collect numeric fitness scores.
   * 3. Count total vs enabled connections across genomes to derive an enabled ratio (structural sparsity signal).
   * 4. Compute mean in a single pass over the scores scratch array.
   * 5. Clone & sort scores (ascending) to compute median (keeps original order intact for any other readers).
   * 6. Derive species count defensively (array length or null when absent).
   * 7. Return a small plain object (all numbers formatted to 2 decimals where derived) — consumers may patch
   *    missing values later (see fallback logic in `#updateDetailedStatsSnapshot`).
   *
   * Complexity:
   * - Score collection: O(G) with G = population size.
   * - Connection scan: O(E) with E = total number of connection entries (linear).
   * - Sorting for median: O(G log G) — acceptable for modest populations; if G became very large, a selection
   *   algorithm (nth_element style) could replace the sort (document trade-offs first if changed).
   *
   * Performance notes:
   * - Reuses a single scores scratch array (cleared via length reset) to avoid churn.
   * - Uses numeric formatting only at final aggregation (minimizes intermediate string creation).
   * - Avoids repeated optional chaining in inner loops by shallow local references.
   *
   * Determinism: Pure function of the provided `neat.population` snapshot (iteration order is respected).
   * Reentrancy: Safe; scratch array is instance-scoped but method is not expected to be invoked concurrently.
   * Edge cases: Empty / missing population returns all-null fields; division by zero guarded by connection count checks.
   *
   * @param neat NEAT-like engine instance exposing `population`, optional `species` collection.
   * @returns Object with `mean`, `median`, `speciesCount`, `enabledRatio` (each nullable when not derivable).
   */
  #computePopulationStats(
    neat?: any
  ): {
    mean: number | null;
    median: number | null;
    speciesCount: number | null;
    enabledRatio: number | null;
  } {
    // Step 1: Guard for absent / malformed population
    if (
      !neat ||
      !Array.isArray(neat.population) ||
      neat.population.length === 0
    ) {
      return {
        mean: null,
        median: null,
        speciesCount: null,
        enabledRatio: null,
      };
    }

    // Step 2: Reuse scratch array for fitness scores (clear via length assignment)
    const { scores } = this.#scratch;
    scores.length = 0;

    // Step 3: Scan genomes collecting scores & connection enablement stats
    let enabledConnectionsCount = 0;
    let totalConnectionsCount = 0;
    for (const genome of neat.population) {
      if (typeof genome?.score === 'number') scores.push(genome.score);
      const genomeConns = genome?.connections;
      if (Array.isArray(genomeConns)) {
        for (const connection of genomeConns) {
          totalConnectionsCount++;
          if (connection?.enabled !== false) enabledConnectionsCount++;
        }
      }
    }

    // Step 4 & 5: Mean and median computation
    let mean: number | null = null;
    let median: number | null = null;
    if (scores.length) {
      let sum = 0;
      for (let scoreIndex = 0; scoreIndex < scores.length; scoreIndex++) {
        sum += scores[scoreIndex];
      }
      mean = +(sum / scores.length).toFixed(2);

      // Clone before sort to preserve potential external reliance on original order (defensive)
      const sortedScores = [...scores].sort((a, b) => a - b);
      const middleIndex = Math.floor(sortedScores.length / 2);
      const medianRaw =
        sortedScores.length % 2 === 0
          ? (sortedScores[middleIndex - 1] + sortedScores[middleIndex]) / 2
          : sortedScores[middleIndex];
      median = +medianRaw.toFixed(2);
    }

    // Step 6: Enabled ratio (null when no connections observed)
    const enabledRatio = totalConnectionsCount
      ? +(enabledConnectionsCount / totalConnectionsCount).toFixed(2)
      : null;

    // Step 7: Species count (nullable)
    const speciesCount = Array.isArray(neat.species)
      ? neat.species.length
      : null;

    // Step 8: Return aggregate
    return { mean, median, speciesCount, enabledRatio };
  }

  /**
   * Derive a ranked list of operator acceptance percentages from the (optional) evolution engine.
   *
   * Educational focus:
   * - Demonstrates defensive integration with a loosely‑typed external API (`getOperatorStats`).
   * - Shows how to reuse an instance scratch buffer to avoid per‑refresh allocations.
   * - Illustrates compact ranking logic (copy + sort + slice) while preserving original raw snapshot.
   *
   * Acceptance definition:
   *   acceptancePct = (success / max(1, attempts)) * 100   (formatted to 2 decimals)
   *   A zero attempts count is clamped to 1 to avoid division by zero; this treats a (success>0, attempts==0)
   *   anomaly as full success rather than NaN — acceptable for a resilience‑biased dashboard.
   *
   * Steps:
   * 1. Guard: verify `neat.getOperatorStats` is a function (else return null to signal absence of data).
   * 2. Safe invoke inside try/catch (engines may throw while stats are initializing).
   * 3. Filter raw entries into `#scratch.operatorStats` keeping only { name:string, success:number, attempts:number }.
   * 4. Create a ranked copy sorted by descending acceptance ratio (stable for ties in modern JS engines).
   * 5. Map the top N (`#TOP_OPERATOR_LIMIT`) into a lightweight exported shape `{ name, acceptancePct }`.
   * 6. Return `null` when no valid entries remain after filtering (downstream rendering can simply skip the block).
   *
   * Complexity:
   * - Let K be the number of operator entries. Filtering O(K); sort O(K log K); slice/map O(min(N, K)).
   * - K is typically tiny (single digits), so the impact per redraw is negligible.
   *
   * Performance notes:
   * - Scratch buffer cleared via length reset (no new array each call).
   * - Only one extra array allocation (`rankedCopy`) for isolation of sort side‑effects.
   * - Formatting (toFixed) deferred until final mapping to limit transient string creation.
   *
   * Determinism: Pure given the operator stats snapshot (no randomness). Relies on stable `Array.prototype.sort` for tie ordering.
   * Reentrancy: Safe under single‑threaded assumption; scratch buffer is reused but not shared across concurrent calls.
   * Edge cases & error handling:
   * - Missing API / thrown error => null.
   * - Malformed entries (missing numeric fields) silently excluded.
   * - Division by zero avoided via denominator clamp.
   * - Empty post‑filter set => null (consistent sentinel).
   *
   * @param neat Optional engine exposing `getOperatorStats(): Array<{name:string, success:number, attempts:number}>`.
   * @returns Array of top operators with acceptance percentages or null when unavailable / no data.
   * @example
   * const acceptance = (dashboard as any)["#computeOperatorAcceptance"](neatInstance);
   * // => [ { name: 'mutateAddNode', acceptancePct: 62.5 }, ... ] or null
   */
  #computeOperatorAcceptance(
    neat?: any
  ): Array<{ name: string; acceptancePct: number }> | null {
    if (typeof neat?.getOperatorStats !== 'function') return null;

    let rawOperatorStats: any;
    try {
      rawOperatorStats = neat.getOperatorStats();
    } catch {
      return null; // Defensive: treat transient failures as absence of data.
    }
    if (!Array.isArray(rawOperatorStats) || rawOperatorStats.length === 0)
      return null;

    // Step 3: Populate scratch buffer with only well-formed entries.
    const scratchBuffer = this.#scratch.operatorStats;
    scratchBuffer.length = 0; // in-place clear
    for (const operatorStat of rawOperatorStats) {
      if (
        operatorStat &&
        typeof operatorStat.name === 'string' &&
        typeof operatorStat.success === 'number' &&
        typeof operatorStat.attempts === 'number'
      ) {
        scratchBuffer.push(operatorStat);
      }
    }
    if (!scratchBuffer.length) return null;

    // Step 4: Sort copy (preserve original ordering in scratch for potential reuse / future augmentation).
    const rankedCopy = [...scratchBuffer].sort((leftStat, rightStat) => {
      const leftAcceptance = leftStat.success / Math.max(1, leftStat.attempts);
      const rightAcceptance =
        rightStat.success / Math.max(1, rightStat.attempts);
      // Descending order; tie-break maintains stable relative ordering due to JS sort stability in modern engines.
      if (rightAcceptance !== leftAcceptance)
        return rightAcceptance - leftAcceptance;
      return 0;
    });

    // Step 5: Map top-N into exported simplified objects.
    const limit = Math.min(
      DashboardManager.#TOP_OPERATOR_LIMIT,
      rankedCopy.length
    );
    const acceptanceList: Array<{ name: string; acceptancePct: number }> = [];
    for (let rankIndex = 0; rankIndex < limit; rankIndex++) {
      const rankedStat = rankedCopy[rankIndex];
      const acceptancePct = +(
        (100 * rankedStat.success) /
        Math.max(1, rankedStat.attempts)
      ).toFixed(2);
      acceptanceList.push({ name: rankedStat.name, acceptancePct });
    }
    return acceptanceList.length ? acceptanceList : null;
  }

  /**
   * Produce a ranked list of the most frequent mutation operators observed so far.
   *
   * Educational focus:
   * - Demonstrates reuse of an in-place scratch tuple array to avoid allocation churn.
   * - Shows defensive extraction from a loosely-typed stats object (filtering only numeric counts).
   * - Illustrates a simple top-N selection pattern (sort + bounded slice) with explicit caps.
   *
   * Steps:
   * 1. Guard: return null if `mutationStats` is not a plain object.
   * 2. Clear and repopulate `#scratch.mutationEntries` with `[name, count]` tuples for numeric fields.
   * 3. Early return null when no valid entries collected (simplifies downstream rendering conditions).
   * 4. Sort the scratch array in-place by descending count (highest frequency first) using descriptive comparator param names.
   * 5. Take the top N (bounded by `#TOP_MUTATION_LIMIT`) and map to output objects `{ name, count }`.
   * 6. Return the resulting array (guaranteed non-empty) or null if absent.
   *
   * Complexity:
   * - Collection: O(K) with K = enumerable keys on `mutationStats`.
   * - Sort: O(K log K); K is typically modest (dozens at most) so overhead is negligible.
   * - Slice/map: O(min(N, K)).
   *
   * Performance notes:
   * - Scratch array is reused (length reset) preventing repeated allocation of tuple arrays each frame.
   * - In-place sort avoids cloning (`[...entries]`) found in earlier version, eliminating one transient array.
   * - Comparator accesses tuple indices directly, avoiding destructuring overhead in the hot call.
   *
   * Determinism: Pure transformation of the provided stats snapshot; no randomness.
   * Reentrancy: Safe for single-threaded invocation pattern; scratch state is not shared across instances.
   * Edge cases:
   * - Non-object or empty object => null.
   * - Non-numeric values silently skipped.
   * - Negative counts retained (still sorted numerically) under the assumption they signal net effects; could be filtered if undesired.
   *
   * @param mutationStats Arbitrary object mapping mutation operator names to numeric invocation counts.
   * @returns Array of top mutation operators (name + count) or null when no data.
   * @example
   * const top = (dashboard as any)["#computeTopMutations"]({ addNode: 42, addConn: 17 });
   * // => [ { name: 'addNode', count: 42 }, { name: 'addConn', count: 17 } ]
   */
  #computeTopMutations(
    mutationStats: any
  ): Array<{ name: string; count: number }> | null {
    // Step 1: Guard for invalid container
    if (!mutationStats || typeof mutationStats !== 'object') return null;

    // Step 2: Populate scratch with numeric entries only
    const mutationEntriesScratch = this.#scratch.mutationEntries;
    mutationEntriesScratch.length = 0;
    for (const mutationName of Object.keys(mutationStats)) {
      const occurrenceCount = mutationStats[mutationName];
      if (
        typeof occurrenceCount === 'number' &&
        Number.isFinite(occurrenceCount)
      ) {
        mutationEntriesScratch.push([mutationName, occurrenceCount]);
      }
    }

    // Step 3: Early return when no numeric stats present
    if (!mutationEntriesScratch.length) return null;

    // Step 4: In-place sort descending by count
    mutationEntriesScratch.sort(
      (leftEntry, rightEntry) => rightEntry[1] - leftEntry[1]
    );

    // Step 5: Map top-N to output objects
    const limit = Math.min(
      DashboardManager.#TOP_MUTATION_LIMIT,
      mutationEntriesScratch.length
    );
    const topMutations: Array<{ name: string; count: number }> = [];
    for (let rankIndex = 0; rankIndex < limit; rankIndex++) {
      const [mutationName, occurrenceCount] = mutationEntriesScratch[rankIndex];
      topMutations.push({ name: mutationName, count: occurrenceCount });
    }
    return topMutations;
  }

  /**
   * Compute the sizes (member counts) of the largest species (Top-N) in the current population snapshot.
   *
   * Educational focus:
   * - Demonstrates reuse of an integer scratch array to avoid new allocations every redraw.
   * - Highlights a simple pattern for extracting a Top-N ranking from a small set (in-place sort + bounded copy).
   * - Shows defensive handling of loosely-typed engine data (species objects may omit `members`).
   *
   * Steps:
   * 1. Guard: return null when `neat.species` is absent or empty.
   * 2. Repopulate `#scratch.speciesSizes` with numeric member counts (fallback 0 when ambiguous).
   * 3. In-place sort scratch array descending (largest first).
   * 4. Copy the first N (`#TOP_SPECIES_LIMIT`) values into a new output array for immutability to callers.
   * 5. Return the ranked sizes or null when no data.
   *
   * Complexity:
   * - Let S = species count. Population: O(S). Sort: O(S log S). Copy: O(min(S, N)). S is typically modest, so cost is trivial.
   *
   * Performance notes:
   * - Reuses a single scratch array (cleared via length assignment) to avoid allocation churn.
   * - In-place sort avoids creating an additional clone (`[...scratch]`), reducing temporary memory.
   * - Output array is sized at most `#TOP_SPECIES_LIMIT` (small, bounded allocation) for downstream display safety.
   *
   * Determinism: Pure function of the `neat.species` snapshot (ordering depends only on numerical counts; stable for equal sizes because JS sort is stable in modern engines but equal sizes preserve original order).
   * Reentrancy: Safe under single-threaded invocation pattern (scratch array reused but not shared concurrently).
   * Edge cases:
   * - Missing / non-array / empty species list => null.
   * - Species object missing `members` => treated as size 0.
   * - Negative member counts (unexpected) retained and sorted numerically; could be filtered if a real engine produced them.
   *
   * @param neat Optional NEAT-like engine instance exposing an array `species` with `members` arrays.
   * @returns Array of top species sizes (descending) or null when no species present.
   * @example
   * const sizes = (dashboard as any)["#computeTopSpeciesSizes"](neat);
   * // => [34, 21, 10] (up to 5 elements) or null
   */
  #computeTopSpeciesSizes(neat?: any): number[] | null {
    // Step 1: Guard for absence / emptiness
    if (!Array.isArray(neat?.species) || neat.species.length === 0) return null;

    // Step 2: Populate scratch with member counts
    const speciesSizesScratch = this.#scratch.speciesSizes;
    speciesSizesScratch.length = 0; // clear
    for (const speciesEntry of neat.species) {
      // Fallback to 0 when members array missing / non-array
      const sizeValue = Array.isArray(speciesEntry?.members)
        ? speciesEntry.members.length
        : 0;
      speciesSizesScratch.push(sizeValue);
    }
    if (!speciesSizesScratch.length) return null; // defensive (should not occur if earlier guard passed)

    // Step 3: In-place descending sort
    speciesSizesScratch.sort((leftSize, rightSize) => rightSize - leftSize);

    // Step 4: Bounded copy to output (immutability for consumers)
    const limit = Math.min(
      DashboardManager.#TOP_SPECIES_LIMIT,
      speciesSizesScratch.length
    );
    const topSpeciesSizes: number[] = [];
    for (let rankIndex = 0; rankIndex < limit; rankIndex++) {
      topSpeciesSizes.push(speciesSizesScratch[rankIndex]);
    }
    return topSpeciesSizes;
  }

  /** Safe invoke wrapper returning fallback on throw. */
  #safeInvoke<T>(fn: () => T, fallback: T): T {
    try {
      return fn();
    } catch {
      return fallback;
    }
  }

  /**
   * Append the top header lines for a solved maze archive block.
   *
   * Format mirrors other framed sections: a top border, a centered label line
   * identifying the solved ordinal and generation, and one spacer line to
   * visually separate from subsequent sparkline + maze content.
   *
   * We keep this lean: no dynamic width calculations beyond centering, and we
   * avoid extra temporary arrays (push directly into the provided accumulator).
   *
   * @param blockLines Accumulator array mutated by appending formatted lines.
   * @param solved Object containing result + generation metadata.
   * @param displayNumber 1-based solved maze index for user-friendly labeling.
   */
  #appendSolvedHeader(
    blockLines: string[],
    solved: {
      maze: string[];
      result: any;
      network: INetwork;
      generation: number;
    },
    displayNumber: number
  ): void {
    /**
     * Educational / formatting notes:
     * - Uses a fixed inner frame width to keep all archive sections visually aligned regardless of maze size.
     * - Centers a dynamic title string without allocating intermediate arrays (direct pushes to accumulator).
     * - Fitness value is formatted to two decimals only when finite; otherwise 'n/a' is displayed for clarity.
     * - Keeps allocation footprint minimal: a handful of short-lived strings (no joins over arrays).
     *
     * Steps:
     * 1. Resolve & validate sizing constants (frame inner width).
     * 2. Push a top border line (full width of heavy box characters).
     * 3. Build a descriptive centered title including solved ordinal, generation, and fitness.
     * 4. Compute left/right padding to center the title (favor left bias on odd extra space for stable layout).
     * 5. Push the centered title line with color accents.
     * 6. Push a spacer line to visually separate header from subsequent sparkline/stat/maze content.
     *
     * Determinism: Pure formatting based on provided parameters and static constants.
     * Reentrancy: Safe; only mutates the provided `blockLines` accumulator.
     * Edge cases:
     * - Extremely long title (e.g., unexpectedly large generation number) will be clipped visually by frame borders (allowed; signals anomaly).
     * - Non-numeric / NaN fitness gracefully downgrades to 'n/a'.
     */
    const innerWidth = DashboardManager.FRAME_INNER_WIDTH; // Step 1

    // Step 2: Top border line
    blockLines.push(
      `${colors.blueCore}╔${NetworkVisualization.pad(
        '═'.repeat(innerWidth),
        innerWidth,
        '═'
      )}╗${colors.reset}`
    );

    // Step 3: Title components (defensive numeric handling)
    const { result, generation } = solved;
    const rawFitness = result?.fitness;
    const formattedFitness =
      typeof rawFitness === 'number' && Number.isFinite(rawFitness)
        ? rawFitness.toFixed(2)
        : 'n/a';
    const title = ` SOLVED #${Math.max(
      1,
      displayNumber
    )} (GEN ${generation})  FITNESS ${formattedFitness} `;

    // Step 4: Centering math
    const leftPaddingSize = Math.max(
      0,
      Math.floor((innerWidth - title.length) / 2)
    );
    const rightPaddingSize = Math.max(
      0,
      innerWidth - title.length - leftPaddingSize
    );

    // Step 5: Centered title line
    blockLines.push(
      `${colors.blueCore}║${' '.repeat(leftPaddingSize)}${
        colors.orangeNeon
      }${title}${colors.blueCore}${' '.repeat(rightPaddingSize)}║${
        colors.reset
      }`
    );

    // Step 6: Spacer line
    blockLines.push(
      `${colors.blueCore}║${NetworkVisualization.pad(' ', innerWidth, ' ')}║${
        colors.reset
      }`
    );
  }

  /**
   * Append solved-run sparklines plus a one-line architecture summary.
   *
   * Additions over the original implementation:
   * - Includes current network architecture (layer sizes) using arrow formatting ("I <=> H1 <=> ... <=> O").
   * - Consolidates architecture here (removed separate later architecture line to avoid duplication).
   * - Retains aligned label formatting via shared `#formatStat` helper for consistency.
   *
   * Architecture derivation:
   * - Uses `#deriveArchitecture` (returns e.g. "6 - 6 - 5 - 4").
   * - Converts hyphen-delimited form to bi-directional arrow form replacing " - " with " <=> " for clearer layer transitions.
   * - Skips line when result is 'n/a'.
   *
   * Steps:
   * 1. Build architecture string (if derivable) and push as first line.
   * 2. For each tracked history series build a sparkline (bounded width) and push if non-empty.
   * 3. Emit a trailing blank framed line as a visual separator before maze rendering.
   *
   * Determinism: Pure formatting/read-only usage of snapshot histories & network.
   * Reentrancy: Safe; only mutates provided accumulator.
   * Edge cases: Empty histories yield omitted lines; architecture omitted when unknown.
   *
   * @param blockLines Accumulator mutated in place.
   * @param network Network whose architecture will be summarized; optional (can be nullish).
   */
  #appendSolvedSparklines(blockLines: string[], network?: INetwork): void {
    const solvedLabelWidth = DashboardManager.#SOLVED_LABEL_WIDTH;
    const solvedStat = (label: string, value: string) =>
      this.#formatStat(
        label,
        value,
        colors.neonSilver,
        colors.cyanNeon,
        solvedLabelWidth
      );
    const pushIf = (label: string, value: string | null | undefined) => {
      if (value) blockLines.push(solvedStat(label, value));
    };

    // Step 1: Architecture summary
    if (network) {
      let architectureRaw = 'n/a';
      try {
        architectureRaw = this.#deriveArchitecture(network as any);
      } catch {
        architectureRaw = 'n/a';
      }
      if (architectureRaw !== 'n/a') {
        const arrowArchitecture = architectureRaw
          .split(/\s*-\s*/)
          .join(' <=> ');
        pushIf(DashboardManager.#LABEL_ARCH, arrowArchitecture);
      }
    }

    // Step 2: Trend sparklines (bounded width)
    const archiveWidth = DashboardManager.#ARCHIVE_SPARK_WIDTH;
    pushIf(
      'Fitness trend',
      this.#buildSparkline(this.#bestFitnessHistory, archiveWidth)
    );
    pushIf(
      'Nodes trend',
      this.#buildSparkline(this.#complexityNodesHistory, archiveWidth)
    );
    pushIf(
      'Conns trend',
      this.#buildSparkline(this.#complexityConnsHistory, archiveWidth)
    );
    pushIf(
      'Hypervol trend',
      this.#buildSparkline(this.#hypervolumeHistory, archiveWidth)
    );
    pushIf(
      'Progress trend',
      this.#buildSparkline(this.#progressHistory, archiveWidth)
    );
    pushIf(
      'Species trend',
      this.#buildSparkline(this.#speciesCountHistory, archiveWidth)
    );

    // Step 3: Spacer line
    blockLines.push(
      `${colors.blueCore}║${NetworkVisualization.pad(
        ' ',
        DashboardManager.FRAME_INNER_WIDTH,
        ' '
      )}${colors.blueCore}║${colors.reset}`
    );
  }

  /**
   * Append a centered maze visualization for a newly solved maze.
   *
   * The visualization is produced by `MazeVisualization.visualizeMaze`, which
   * returns either a multi‑line string or an array of row strings. We normalize
   * the output to an array and then emit each row framed inside the dashboard
   * box. Rows are padded horizontally to the fixed `FRAME_INNER_WIDTH` so that
   * varying maze sizes (small corridors vs larger layouts) remain visually
   * centered and consistent with surrounding stats blocks.
   *
   * Steps (educational):
   * 1. Determine the terminal path position (last coordinate) – used to draw the agent end state.
   * 2. Generate a textual maze representation (string[] or string) including the path highlight.
   * 3. Normalize to an array of raw row strings (split on newlines if needed).
   * 4. Pad each row to the inner frame width (acts as horizontal centering) and push framed lines to `blockLines`.
   *
   * Performance & ES2023 notes:
   * - Uses `Array.prototype.at(-1)` for the final path coordinate (clearer than `path[path.length-1]`).
   * - Avoids the previous join/split round‑trip (now pads & pushes in a single pass), reducing temporary string allocations.
   * - Relies on local constants to minimize repeated property lookups (`innerWidth`).
   *
   * Determinism: purely formatting; does not mutate input arrays or rely on random state.
   * Reentrancy: safe (no shared scratch buffers used here).
   *
   * @param blockLines - Accumulated output lines for the solved maze archive block (mutated in place by appending framed rows).
   * @param solved - Object containing the raw `maze` character grid and a `result` with a `path` of `[x,y]` coordinates.
   * @remarks The `result.path` is expected to include the start cell; if empty, a fallback position `[0,0]` is used (rare edge case for defensive coding in examples).
   */
  #appendSolvedMaze(
    blockLines: string[],
    solved: {
      maze: string[];
      result: { path?: ReadonlyArray<readonly [number, number]> } & Record<
        string,
        any
      >;
    }
  ): void {
    // Step 1: Determine final position on the solved path (fallback to [0,0] if path missing)
    const pathCoordinates = solved.result.path as
      | ReadonlyArray<readonly [number, number]>
      | undefined;
    const endPosition = pathCoordinates?.at(-1) ?? [0, 0];

    // Step 2: Produce visualization (string or string[]). Pass the path so it can be highlighted.
    const visualization = MazeVisualization.visualizeMaze(
      solved.maze,
      endPosition as [number, number],
      (pathCoordinates ?? []) as [number, number][]
    );

    // Step 3: Normalize to array of lines.
    const rawLines: string[] = Array.isArray(visualization)
      ? visualization
      : (visualization as string).split('\n');

    // Step 4: Pad & frame each line (acts as centering); push directly to accumulator.
    const innerWidth = DashboardManager.FRAME_INNER_WIDTH; // local alias for micro-clarity
    for (const rawLine of rawLines) {
      const paddedRow = NetworkVisualization.pad(rawLine, innerWidth, ' ');
      blockLines.push(
        `${colors.blueCore}║${NetworkVisualization.pad(
          paddedRow,
          innerWidth,
          ' '
        )}${colors.blueCore}║${colors.reset}`
      );
    }
  }

  /**
   * Compute and append human‑readable path efficiency statistics for a solved maze.
   *
   * Metrics exposed (educational rationale):
   * - Path efficiency: optimal BFS distance vs actual traversed length – demonstrates how close evolution came to shortest path.
   * - Path overhead: percent longer than optimal – highlights wasted exploration after reaching a viable route.
   * - Unique cells visited / revisits: proxy for exploration vs dithering; useful to tune mutation operators.
   * - Steps: raw action count taken (often equals `pathLength`).
   * - Fitness: final scalar used for selection (displayed with two decimals for compactness).
   *
   * Steps:
   * 1. Derive aggregated path metrics via `#computePathMetrics` (encapsulates BFS optimal distance + visitation stats).
   * 2. Format each metric with consistent label width & colors using `formatStat` (keeps styling centralized).
   * 3. Push each formatted line to the `blockLines` accumulator.
   *
   * Performance notes:
   * - Single metrics object reused for string interpolation (no intermediate arrays created).
   * - Uses template literals directly; minimal extra allocations beyond the final output strings.
   * - Order is fixed to preserve snapshot diff stability for external log parsers.
   *
   * Determinism: relies on deterministic BFS + pure counting; no randomness.
   * Reentrancy: safe; no shared mutable scratch state.
   *
   * @param blockLines - Accumulator array mutated by appending formatted stat lines.
   * @param solved - Object holding the `maze` layout and `result` containing at least `path`, `steps`, and `fitness`.
   */
  #appendSolvedPathStats(
    blockLines: string[],
    solved: { maze: string[]; result: any }
  ): void {
    // Step 1: Derive metrics (single call encapsulates BFS + visitation stats)
    const metrics = this.#computePathMetrics(solved.maze, solved.result);

    // Local alias for consistent label width
    const labelWidth = DashboardManager.#SOLVED_LABEL_WIDTH;
    const solvedStat = (label: string, value: string) =>
      this.#formatStat(
        label,
        value,
        colors.neonSilver,
        colors.cyanNeon,
        labelWidth
      );

    // Step 2 & 3: Format and append in stable order
    blockLines.push(
      solvedStat(
        DashboardManager.#LABEL_PATH_EFF,
        `${metrics.optimalLength}/${metrics.pathLength} (${metrics.efficiencyPct}%)`
      )
    );
    blockLines.push(
      solvedStat(
        DashboardManager.#LABEL_PATH_OVER,
        `${metrics.overheadPct}% longer than optimal`
      )
    );
    blockLines.push(
      solvedStat(
        DashboardManager.#LABEL_UNIQUE,
        `${metrics.uniqueCellsVisited}`
      )
    );
    blockLines.push(
      solvedStat(
        DashboardManager.#LABEL_REVISITS,
        `${metrics.revisitedCells} times`
      )
    );
    blockLines.push(
      solvedStat(DashboardManager.#LABEL_STEPS, `${metrics.totalSteps}`)
    );
    blockLines.push(
      solvedStat(
        DashboardManager.#LABEL_FITNESS,
        `${metrics.fitnessValue.toFixed(2)}`
      )
    );
  }

  /** Emit footer & send archive block to logger. */
  static #CACHED_SOLVED_FOOTER_BORDER: string | null = null;

  /**
   * Append the solved-archive footer border and emit the accumulated block to the archive logger.
   *
   * Implementation notes:
   * - Reuses an internal cached bottom-border string to avoid recomputing the padded border on every solved maze.
   * - Emits the block as a single joined payload for efficiency; falls back to a line-wise append if the
   *   archive function throws or is not compatible with the single-string API.
   * - Clears the provided `blockLines` accumulator in-place after emission so callers (and tests) can reuse the
   *   same array as a scratch buffer, reducing GC churn in tight loops.
   *
   * Steps (inline):
   * 1. Ensure cached border exists (lazy-init).
   * 2. Append the bottom border to the provided accumulator.
   * 3. Attempt single-string emission with `{ prepend: true }`.
   * 4. On failure, fallback to line-by-line emission using the archive function or a no-op.
   * 5. Clear the accumulator for reuse.
   *
   * @param blockLines Mutable accumulator of framed lines representing a solved maze archive block. This
   *   function will append the closing border and emit the payload; the array will be emptied on return.
   * @example
   * const lines: string[] = [];
   * // ... various helpers push frame header, stats, maze rows into `lines` ...
   * (dashboard as any)["#appendSolvedFooterAndEmit"](lines);
   */
  #appendSolvedFooterAndEmit(blockLines: string[]): void {
    // Step 1: Lazy-initialize a cached bottom-border string to avoid repeated pad/repeat work.
    const innerFrameWidth = DashboardManager.FRAME_INNER_WIDTH;
    if (DashboardManager.#CACHED_SOLVED_FOOTER_BORDER === null) {
      // Build once and reuse; this avoids allocating a new long string on every solved maze.
      DashboardManager.#CACHED_SOLVED_FOOTER_BORDER = `${
        colors.blueCore
      }╚${NetworkVisualization.pad(
        '═'.repeat(innerFrameWidth),
        innerFrameWidth,
        '═'
      )}╝${colors.reset}`;
    }

    // Step 2: Append cached bottom border to the provided accumulator (no intermediate arrays).
    blockLines.push(DashboardManager.#CACHED_SOLVED_FOOTER_BORDER);

    // Step 3: Prefer a single-string emission for efficiency (smaller call overhead and fewer allocations).
    try {
      // Favor the original API shape: archiveFn(payload, { prepend: true }). Use a permissive any-cast
      // because test harnesses may provide different shapes.
      (this.#archiveFn as any)(blockLines.join('\n'), { prepend: true });

      // Step 5: Clear the accumulator in-place to allow caller reuse (reduces GC pressure in tests).
      blockLines.length = 0;
      return;
    } catch {
      // Step 4: Fallback to line-wise appends when the single-string API fails.
      const archiveAppend = this.#archiveFn ?? (() => {});

      // Use a conventional indexed loop with a descriptive iterator variable to avoid short-name warnings.
      for (let lineIndex = 0; lineIndex < blockLines.length; lineIndex++) {
        archiveAppend(blockLines[lineIndex]);
      }

      // Clear the accumulator for reuse by the caller.
      blockLines.length = 0;
    }
  }

  /**
   * Compute derived path metrics for a solved (or partially solved) maze run.
   *
   * Metrics returned (educational focus):
   * - optimalLength: Shortest possible path length (BFS over encoded maze). Provides a baseline for efficiency.
   * - pathLength: Actual traversed path length (steps between first & last coordinate). Used for overhead calculations.
   * - efficiencyPct: (optimal / actual * 100) clamped to 100%. Indicates how close the agent was to an optimal route.
   * - overheadPct: Percent the actual path exceeds optimal ((actual/optimal)*100 - 100). Negative values are clamped to 0 in practice by optimal <= path.
   * - uniqueCellsVisited: Distinct grid cells in the path – proxy for exploration breadth.
   * - revisitedCells: Times a cell coordinate was encountered after the first visit – proxy for dithering / loops.
   * - totalSteps: Reported step counter from the result object (may equal pathLength, but kept separate for clarity / future divergence like wait actions).
   * - fitnessValue: Raw fitness scalar copied through for convenience (avoids re-threading the original result where only metrics are needed).
   *
   * Steps:
   * 1. Locate start 'S' and exit 'E' positions in the maze (single pass each via MazeUtils helpers).
   * 2. Run BFS to obtain the optimal shortest path length between S and E (O(C) with C = cell count).
   * 3. Derive actual path length from provided coordinate list (defensive against empty / single-node path).
   * 4. Compute efficiency & overhead percentages with divide-by-zero guards (fallback to 0.0 when ambiguous).
   * 5. Count unique vs revisited cells in a single pass through the path (O(P) with P = pathLength+1 nodes).
   * 6. Return an immutable plain object used by formatting helpers.
   *
   * Complexity:
   * - BFS: O(C) where C = maze cell count.
   * - Path scan: O(P) where P = number of coordinates in path.
   * - Overall: O(C + P) per invocation, acceptable for archive-time formatting (not in a hot inner evolution loop).
   *
   * Determinism: Fully deterministic given identical maze + path (no randomness, stable BFS ordering assumed from MazeUtils implementation).
   * Reentrancy: Safe (allocates only local structures: Set + return object).
   * Memory: Extra allocations are bounded (Set size <= P). Suitable for occasional solved-maze archival.
   *
   * Edge cases handled:
   * - Empty or single-coordinate path: pathLength coerces to 0; efficiency & overhead emit '0.0'.
   * - Unreachable BFS (negative / non-positive optimalLength): treated as 0 for ratios (prevents NaN/Infinity).
   * - Division by zero avoided via guards; percentages formatted with one decimal place.
   *
   * @param maze Maze layout as array of row strings containing 'S' and 'E'.
   * @param result Evaluation result containing at least { path, steps, fitness }.
   * @returns Object with path + efficiency metrics (see description).
   * @example
   * const metrics = dashboard.#computePathMetrics(maze, { path, steps: path.length, fitness });
   * console.log(metrics.efficiencyPct); // e.g. '87.5'
   */
  #computePathMetrics(
    maze: string[],
    result: { path: [number, number][]; steps: number; fitness: number }
  ): {
    optimalLength: number;
    pathLength: number;
    efficiencyPct: string;
    overheadPct: string;
    uniqueCellsVisited: number;
    revisitedCells: number;
    totalSteps: number;
    fitnessValue: number;
  } {
    // Step 1: Resolve start & exit coordinates
    const startPosition = MazeUtils.findPosition(maze, 'S');
    const exitPosition = MazeUtils.findPosition(maze, 'E');

    // Step 2: Compute optimal shortest path via BFS (may return <=0 if unreachable)
    const bfsLength = MazeUtils.bfsDistance(
      MazeUtils.encodeMaze(maze),
      startPosition,
      exitPosition
    );
    const optimalLength = typeof bfsLength === 'number' ? bfsLength : 0;

    // Step 3: Derive actual path length (edges traversed); guard against empty path
    const rawPathLength = Math.max(0, result.path.length - 1);

    // Step 4: Efficiency & overhead (guard divide-by-zero and invalid optimal)
    let efficiencyPct = '0.0';
    let overheadPct = '0.0';
    if (rawPathLength > 0 && optimalLength > 0) {
      const efficiency = Math.min(1, optimalLength / rawPathLength) * 100;
      efficiencyPct = efficiency.toFixed(1);
      const overhead = (rawPathLength / optimalLength) * 100 - 100;
      overheadPct = overhead.toFixed(1);
    }

    // Step 5: Count unique vs revisits
    const uniqueCells = new Set<string>();
    let revisitedCells = 0;
    for (const [cellX, cellY] of result.path) {
      const key = `${cellX},${cellY}`;
      if (uniqueCells.has(key)) revisitedCells++;
      else uniqueCells.add(key);
    }

    // Step 6: Return immutable metrics object
    return {
      optimalLength,
      pathLength: rawPathLength,
      efficiencyPct,
      overheadPct,
      uniqueCellsVisited: uniqueCells.size,
      revisitedCells,
      totalSteps: result.steps,
      fitnessValue: result.fitness,
    };
  }

  /**
   * Infer a compact, human‑readable architecture string (e.g. "3 - 5 - 2" or "4 - 8 - 8 - 1").
   *
   * Supports several internal network representations encountered in examples:
   * 1. Layered form: `network.layers = [layer0, layer1, ...]` where each layer has `nodes` or is an array.
   * 2. Flat node list: `network.nodes = [...]` each node declaring a `type` ('input' | 'hidden' | 'output'). Hidden layers are
   *    approximated by a simple topological layering pass: we iteratively collect hidden nodes whose inbound connection sources
   *    are already assigned to earlier layers. Remaining nodes after no progress count as a single ambiguous layer (safety / cycle guard).
   * 3. Scalar fallback: numeric `input` & `output` counts (no hidden layers) -> returns "I - O".
   *
   * Steps:
   * 1. Early null/undefined guard.
   * 2. If a layered structure exists (>=2 layers) derive each layer size in order and return immediately (fast path).
   * 3. Else if a flat node list exists, split into input / hidden / output categories.
   * 4. If no hidden nodes: use explicit numeric counts (prefer explicit `input`/`output` props if present).
   * 5. Perform iterative hidden layer inference with a safety iteration cap to avoid infinite loops for malformed cyclic graphs.
   * 6. Assemble final size list: input size, inferred hidden sizes, output size.
   * 7. Fallback: if only scalar counts available, return them; otherwise 'n/a'.
   *
   * Algorithmic notes:
   * - Hidden layering pass is O(H * E_in) where H = hidden nodes, E_in = mean in-degree, acceptable for formatting/UI.
   * - The safety cap (`hiddenCount * LAYER_INFER_LOOP_MULTIPLIER`) prevents pathological spins on cyclic graphs lacking
   *   proper DAG layering; any leftover hidden nodes are grouped into a terminal bucket for transparency.
   * - We intentionally avoid mutating the original node objects (pure inspection) to keep side‑effects nil.
   *
   * Determinism: given a stable ordering of `network.nodes` and their connections, output is deterministic.
   * Reentrancy: safe; all state kept in local sets/arrays.
   *
   * @param networkInstance Arbitrary network-like object from examples or NEAT internals.
   * @returns Architecture string in the form "Input - Hidden... - Output" or 'n/a' if shape cannot be inferred.
   * @example
   * // Layered network
   * deriveArchitecture({ layers:[ {nodes:[1,2,3]}, {nodes:[4,5]}, {nodes:[6,7]} ] }) => "3 - 2 - 2"
   * @example
   * // Flat node list with inferred hidden tiers
   * deriveArchitecture({ nodes:[{type:'input'}, {type:'hidden'}, {type:'output'}] }) => "1 - 1 - 1"
   */
  #deriveArchitecture(networkInstance: any): string {
    // Step 1: Null/undefined quick exit
    if (!networkInstance) return 'n/a';

    // Step 2: Layered representation (fast path)
    const layerArray = networkInstance.layers;
    if (Array.isArray(layerArray) && layerArray.length >= 2) {
      const layerSizes: number[] = [];
      for (const layerRef of layerArray) {
        const size = Array.isArray(layerRef?.nodes)
          ? layerRef.nodes.length
          : Array.isArray(layerRef)
          ? layerRef.length
          : 0;
        layerSizes.push(size);
      }
      return layerSizes.join(' - ');
    }

    // Step 3: Flat node list representation
    const flatNodes = networkInstance.nodes;
    if (Array.isArray(flatNodes)) {
      const inputNodes = flatNodes.filter(
        (nodeItem: any) => nodeItem.type === 'input'
      );
      const outputNodes = flatNodes.filter(
        (nodeItem: any) => nodeItem.type === 'output'
      );
      const hiddenNodesAll = flatNodes.filter(
        (nodeItem: any) => nodeItem.type === 'hidden'
      );

      // Step 4: No hidden nodes -> simple case
      if (!hiddenNodesAll.length) {
        if (
          typeof networkInstance.input === 'number' &&
          typeof networkInstance.output === 'number'
        ) {
          return `${networkInstance.input} - ${networkInstance.output}`;
        }
        return `${inputNodes.length} - ${outputNodes.length}`;
      }

      // Step 5: Iterative hidden layer inference
      const assignedNodes = new Set<any>(inputNodes);
      let remainingHidden = hiddenNodesAll.slice();
      const inferredHiddenSizes: number[] = [];
      const safetyLimit =
        hiddenNodesAll.length * DashboardManager.#LAYER_INFER_LOOP_MULTIPLIER;
      let iterationCounter = 0;
      while (remainingHidden.length && iterationCounter < safetyLimit) {
        iterationCounter++;
        const currentLayer = remainingHidden.filter((hiddenNode: any) =>
          hiddenNode.connections?.in?.every((conn: any) =>
            assignedNodes.has(conn.from)
          )
        );
        if (!currentLayer.length) {
          // Group unresolved remainder into one bucket (cycles / malformed graph)
          inferredHiddenSizes.push(remainingHidden.length);
          break;
        }
        inferredHiddenSizes.push(currentLayer.length);
        for (const nodeRef of currentLayer) assignedNodes.add(nodeRef);
        remainingHidden = remainingHidden.filter(
          (nodeCandidate: any) => !assignedNodes.has(nodeCandidate)
        );
      }
      return [
        `${inputNodes.length}`,
        ...inferredHiddenSizes.map((hiddenSize) => `${hiddenSize}`),
        `${outputNodes.length}`,
      ].join(' - ');
    }

    // Step 7: Numeric scalar fallback
    if (
      typeof networkInstance.input === 'number' &&
      typeof networkInstance.output === 'number'
    ) {
      return `${networkInstance.input} - ${networkInstance.output}`;
    }
    return 'n/a';
  }

  /**
   * Ingest an evolution engine update (generation tick or improved candidate) and refresh live + archived displays.
   *
   * High‑level responsibilities:
   * 1. Lazy initialize timing anchors (wall clock + perf) on first call.
   * 2. Stash generation/time metadata for later telemetry sampling.
   * 3. Update the current best candidate reference.
   * 4. Archive a newly solved maze (once per unique layout) with rich stats.
   * 5. Pull the latest telemetry snapshot from the NEAT instance (if provided) and update bounded history buffers.
   * 6. Redraw the live ASCII dashboard (network summary, maze, stats, progress bar).
   * 7. Emit a structured telemetry payload (custom DOM event + postMessage + optional hook) for external consumers.
   *
   * Performance notes:
   * - History buffers are bounded (HISTORY_MAX) using push-with-trim helpers; memory growth is capped.
   * - Telemetry extraction takes only the last snapshot (`safeLast`) to minimize per-tick work.
   * - All formatting for the archive occurs only when a maze is first solved (amortized, infrequent).
   * - Uses `performance.now()` when available for higher‑resolution generation throughput metrics.
   *
   * Determinism: All state changes here are observational (no RNG). Ordering of history pushes is fixed.
   * Reentrancy: Not safe (instance maintains mutable internal single-run state). Use one instance per run.
   * Side‑effects: Console / logger output, optional DOM events (browser), optional parent frame messaging.
   *
   * @param maze - Current maze layout under evolution (array of row strings).
   * @param result - Candidate evaluation result (expects fields: fitness, path, success, progress, etc.).
   * @param network - Network associated with the candidate result.
   * @param generation - Current generation number reported by the engine.
   * @param neatInstance - Optional NEAT framework instance exposing `getTelemetry()` and optional stats helpers.
   * @example
   * dashboard.update(maze, evaluationResult, genome.network, generation, neat);
   */
  update(
    maze: string[],
    result: any,
    network: INetwork,
    generation: number,
    neatInstance?: any
  ): void {
    // Step 1: Lazy initialization of timing anchors
    if (this.#runStartTs == null) {
      this.#runStartTs = Date.now(); // wall‑clock anchor
      this.#perfStart = globalThis.performance?.now?.() ?? this.#runStartTs;
    }

    // Step 2: Record generation & update timestamp
    this.#lastUpdateTs = globalThis.performance?.now?.() ?? Date.now();
    this.#lastGeneration = generation;

    // Step 3: Update current best candidate reference
    this.#currentBest = { result, network, generation };

    // Step 4: Archive newly solved maze (once per unique layout)
    if (result?.success) {
      const solvedMazeKey = this.#getMazeKey(maze);
      if (!this.#solvedMazeKeys.has(solvedMazeKey)) {
        this.#solvedMazes.push({ maze, result, network, generation });
        this.#solvedMazeKeys.add(solvedMazeKey);
        const displayOrdinal = this.#solvedMazes.length; // 1-based position
        this.#appendSolvedToArchive(
          { maze, result, network, generation },
          displayOrdinal
        );
      }
    }

    // Step 5: Pull latest telemetry snapshot & update bounded histories
    const telemetrySeries = neatInstance?.getTelemetry?.();
    if (Array.isArray(telemetrySeries) && telemetrySeries.length) {
      this.#lastTelemetry = MazeUtils.safeLast(telemetrySeries as any[]);
      // Best fitness history (trend sparkline source)
      const latestFitness = this.#currentBest?.result?.fitness;
      if (typeof latestFitness === 'number') {
        this.#lastBestFitness = latestFitness;
        this.#bestFitnessHistory = MazeUtils.pushHistory(
          this.#bestFitnessHistory,
          latestFitness,
          DashboardManager.HISTORY_MAX
        );
      }
      // Complexity histories (mean nodes / connections)
      const complexitySnapshot = this.#lastTelemetry?.complexity;
      if (complexitySnapshot) {
        if (typeof complexitySnapshot.meanNodes === 'number') {
          this.#complexityNodesHistory = MazeUtils.pushHistory(
            this.#complexityNodesHistory,
            complexitySnapshot.meanNodes,
            DashboardManager.HISTORY_MAX
          );
        }
        if (typeof complexitySnapshot.meanConns === 'number') {
          this.#complexityConnsHistory = MazeUtils.pushHistory(
            this.#complexityConnsHistory,
            complexitySnapshot.meanConns,
            DashboardManager.HISTORY_MAX
          );
        }
      }
      // Hypervolume (multi‑objective front quality)
      const hyperVolumeLatest = this.#lastTelemetry?.hyper;
      if (typeof hyperVolumeLatest === 'number') {
        this.#hypervolumeHistory = MazeUtils.pushHistory(
          this.#hypervolumeHistory,
          hyperVolumeLatest,
          DashboardManager.HISTORY_MAX
        );
      }
      // Progress toward exit for current best
      const progressFraction = this.#currentBest?.result?.progress;
      if (typeof progressFraction === 'number') {
        this.#progressHistory = MazeUtils.pushHistory(
          this.#progressHistory,
          progressFraction,
          DashboardManager.HISTORY_MAX
        );
      }
      // Species count history
      const speciesCountSnapshot = this.#lastTelemetry?.species;
      if (typeof speciesCountSnapshot === 'number') {
        this.#speciesCountHistory = MazeUtils.pushHistory(
          this.#speciesCountHistory,
          speciesCountSnapshot,
          DashboardManager.HISTORY_MAX
        );
      }
    }

    // Step 6: Redraw live dashboard view
    this.redraw(maze, neatInstance);

    // Step 7: Emit external telemetry payload (event + postMessage + optional hook)
    try {
      const elapsedMs =
        this.#perfStart != null && globalThis.performance?.now
          ? globalThis.performance.now() - this.#perfStart
          : this.#runStartTs
          ? Date.now() - this.#runStartTs
          : 0;
      const generationsPerSecond =
        elapsedMs > 0 ? generation / (elapsedMs / 1000) : 0;
      const payload = {
        type: 'asciiMaze:telemetry',
        generation,
        bestFitness: this.#lastBestFitness,
        progress: this.#currentBest?.result?.progress ?? null,
        speciesCount: this.#speciesCountHistory.at(-1) ?? null,
        gensPerSec: +generationsPerSecond.toFixed(3),
        timestamp: Date.now(),
        details: this.#lastDetailedStats || null,
      };
      if (typeof window !== 'undefined') {
        try {
          window.dispatchEvent(
            new CustomEvent('asciiMazeTelemetry', { detail: payload })
          );
        } catch {}
        try {
          if (window.parent && window.parent !== window)
            window.parent.postMessage(payload, '*');
        } catch {}
        (window as any).asciiMazeLastTelemetry = payload; // polling surface
      }
      try {
        (this as any)._telemetryHook && (this as any)._telemetryHook(payload);
      } catch {}
    } catch {
      /* swallow telemetry emission errors */
    }
  }

  /**
   * Return the most recent telemetry snapshot including rich details.
   * Details may be null if not yet populated.
   * @returns Snapshot object with generation, fitness, progress and detail block.
   */
  getLastTelemetry(): AsciiMazeTelemetrySnapshot {
    const elapsedMs =
      this.#perfStart != null && typeof performance !== 'undefined'
        ? performance.now() - this.#perfStart
        : this.#runStartTs
        ? Date.now() - this.#runStartTs
        : 0;
    const generation = this.#lastGeneration ?? 0;
    const gensPerSec = elapsedMs > 0 ? generation / (elapsedMs / 1000) : 0;
    return {
      generation,
      bestFitness: this.#lastBestFitness,
      progress: this.#currentBest?.result?.progress ?? null,
      speciesCount: MazeUtils.safeLast(this.#speciesCountHistory) ?? null,
      gensPerSec: +gensPerSec.toFixed(3),
      // Expose the last update time if available; convert high-resolution perf time to
      // wall-clock ms when possible so consumers receive an absolute timestamp.
      timestamp: this.#resolveLastUpdateWallMs(),
      details: this.#lastDetailedStats || null,
    };
  }

  /**
   * Resolve the stored last-update timestamp to a wall-clock millisecond value.
   * If the stored value is a high-resolution perf.now() reading, convert it to
   * Date.now() anchored by the recorded `#runStartTs` / `#perfStart` pair. If no
   * last-update is available fall back to Date.now().
   */
  #resolveLastUpdateWallMs(): number {
    if (this.#lastUpdateTs == null) return Date.now();
    // If we have both perfStart and runStart anchors then #lastUpdateTs is likely a perf.now() value.
    if (
      this.#perfStart != null &&
      typeof globalThis.performance?.now === 'function' &&
      this.#runStartTs != null
    ) {
      return this.#runStartTs + (this.#lastUpdateTs - this.#perfStart);
    }
    // Otherwise #lastUpdateTs should already be a wall-clock timestamp.
    return this.#lastUpdateTs;
  }

  /**
   * Print the static top frame (dashboard title header) once at construction / first redraw.
   *
   * Educational focus:
   * - Demonstrates consistent frame construction (symmetric width) using shared constants.
   * - Shows explicit centering math for a colored title while measuring width from an uncolored template.
   * - Avoids ad‑hoc IIFEs: clearer sequential steps improve readability for newcomers.
   *
   * Steps:
   * 1. Emit a solid single‑line top border (full inner width).
   * 2. Emit a bridge line (visual taper) using preconfigured characters.
   * 3. Center and print the title "ASCII maze" with color accents, preserving frame alignment.
   * 4. Emit a lower bridge line to transition into evolving content sections.
   *
   * Centering approach:
   * - We compute visible width using an uncolored template string (box glyph + spaces + raw title + trailing glyph).
   * - Remaining horizontal space is split; a slight left‑bias (ceil on left) improves stability with odd widths.
   * - ANSI color codes are injected only after padding is determined so they don't skew calculations.
   *
   * Performance notes:
   * - Only a handful of short-lived strings are created; cost is negligible (runs once per session).
   * - Uses direct `this.#logFn` calls (no intermediate array joins) to keep GC pressure minimal.
   *
   * Determinism: Pure formatting (no randomness). Reentrancy not required (idempotent semantics acceptable).
   * Edge cases: If the frame width shrinks below the label length, padding clamps to zero and label is still printed.
   *
   * @example
   * (dashboard as any)["#printTopFrame"](); // Emits header block
   */
  #printTopFrame(): void {
    // Local alias for readability
    const innerWidth = DashboardManager.FRAME_INNER_WIDTH;

    // Step 1: Solid top border line
    this.#logFn(
      `${colors.blueCore}╔${NetworkVisualization.pad(
        DashboardManager.#FRAME_SINGLE_LINE_CHAR,
        innerWidth,
        DashboardManager.#FRAME_SINGLE_LINE_CHAR
      )}╗${colors.reset}`
    );

    // Step 2: Upper bridge line (visual accent)
    this.#logFn(
      `${colors.blueCore}╚${NetworkVisualization.pad(
        DashboardManager.#FRAME_BRIDGE_TOP,
        innerWidth,
        DashboardManager.#FRAME_SINGLE_LINE_CHAR
      )}╝${colors.reset}`
    );

    // Step 3: Centered colored title line
    const uncoloredTemplate = '║ ASCII maze ║'; // Used only for visible width calculation
    const templateLength = uncoloredTemplate.length;
    const remainingSpace = innerWidth - templateLength;
    const leftPaddingCount = Math.max(0, Math.ceil(remainingSpace / 2)) + 1; // slight left bias (mirrors prior behavior)
    const rightPaddingCount = Math.max(0, remainingSpace - leftPaddingCount);
    const coloredTitleSegment = `║ ${colors.neonYellow}ASCII maze${colors.blueCore} ║`;
    const centeredTitleLine = `${colors.blueCore}${' '.repeat(
      leftPaddingCount
    )}${coloredTitleSegment}${' '.repeat(rightPaddingCount)}${colors.reset}`;
    this.#logFn(centeredTitleLine);

    // Step 4: Lower bridge line framing transition to evolving section
    this.#logFn(
      `${colors.blueCore}╔${NetworkVisualization.pad(
        DashboardManager.#FRAME_BRIDGE_BOTTOM,
        innerWidth,
        DashboardManager.#FRAME_SINGLE_LINE_CHAR
      )}╗${colors.reset}`
    );
  }

  /** Orchestrate printing of evolving section (network + maze + stats + progress). */
  #printCurrentBestSection(currentMaze: string[]): void {
    const generation = this.#currentBest!.generation;
    // Section delim lines
    const sectionLine = DashboardManager.#EVOLVING_SECTION_LINE;
    this.#logFn(
      `${colors.blueCore}╠${NetworkVisualization.pad(
        sectionLine,
        DashboardManager.FRAME_INNER_WIDTH,
        '═'
      )}${colors.blueCore}╣${colors.reset}`
    );
    this.#logFn(
      `${colors.blueCore}║${NetworkVisualization.pad(
        `${colors.orangeNeon}EVOLVING (GEN ${generation})`,
        DashboardManager.FRAME_INNER_WIDTH,
        ' '
      )}${colors.blueCore}║${colors.reset}`
    );
    this.#logFn(
      `${colors.blueCore}╠${NetworkVisualization.pad(
        sectionLine,
        DashboardManager.FRAME_INNER_WIDTH,
        '═'
      )}${colors.blueCore}╣${colors.reset}`
    );
    this.#logBlank();
    this.#printNetworkSummary();
    this.#printLiveMaze(currentMaze);
    this.#printLiveStats(currentMaze);
    this.#printProgressBar();
  }

  /** Print network summary visualization. */
  #printNetworkSummary(): void {
    this.#logBlank();
    this.#logFn(
      NetworkVisualization.visualizeNetworkSummary(this.#currentBest!.network)
    );
    this.#logBlank();
  }

  /**
   * Render (and frame) the live maze for the current best candidate.
   *
   * Educational focus:
   * - Demonstrates safe use of `Array.prototype.at(-1)` for final coordinate extraction.
   * - Streams framed rows directly to the logger (avoids building one large joined string).
   * - Normalizes flexible visualization return types (string or string[]) into a unified iteration path.
   *
   * Steps:
   * 1. Resolve the agent's last path position (fallback `[0,0]` if path absent) for end‑marker highlighting.
   * 2. Ask `MazeVisualization.visualizeMaze` for a textual representation containing the path overlay.
   * 3. Normalize the result into an array of raw row strings (split on newlines when a single string is returned).
   * 4. For each row, pad to the fixed frame width and emit a framed line (borders + color) via `#logFn`.
   * 5. Surround the block with blank spacer lines for visual separation from adjacent sections.
   *
   * Performance notes:
   * - Avoids intermediate `.map().join()` allocation; writes each row immediately (lower peak memory for large mazes).
   * - Uses a local `innerWidth` alias to prevent repeated static property lookups in the hot loop.
   * - Only allocates one padded string per row (the framing template is assembled inline).
   *
   * Determinism: Pure formatting based on current maze + candidate path (no randomness).
   * Reentrancy: Not designed for concurrent invocation but method is self‑contained (no shared scratch mutation).
   * Edge cases:
   * - Empty visualization yields just spacer lines.
   * - Extremely long rows are hard-clipped visually by frame padding (consistent with rest of dashboard design).
   *
   * @param currentMaze Current maze layout (array of row strings) being evolved.
   */
  #printLiveMaze(currentMaze: string[]): void {
    // Step 1: Determine last path coordinate (agent end position)
    const pathCoordinates = this.#currentBest!.result.path as readonly [
      number,
      number
    ][];
    const endOfPathPosition = pathCoordinates?.at(-1) ?? [0, 0];

    // Step 2: Obtain visualization (may be string or string[])
    const rawVisualization = MazeVisualization.visualizeMaze(
      currentMaze,
      endOfPathPosition as readonly [number, number],
      pathCoordinates
    );

    // Step 3: Normalize to array of lines
    const visualizationLines: readonly string[] = Array.isArray(
      rawVisualization
    )
      ? rawVisualization
      : rawVisualization.split('\n');

    // Step 4: Emit framed & padded lines
    const innerWidth = DashboardManager.FRAME_INNER_WIDTH;
    this.#logBlank(); // leading spacer (Step 5a)
    for (const unpaddedRow of visualizationLines) {
      const paddedRow = NetworkVisualization.pad(unpaddedRow, innerWidth, ' ');
      this.#logFn(
        `${colors.blueCore}║${paddedRow}${colors.blueCore}║${colors.reset}`
      );
    }
    this.#logBlank(); // trailing spacer (Step 5b)
  }

  static #INT32_SCRATCH_POOL: Int32Array[] = [];

  /**
   * Render the live statistics block for the current best candidate.
   *
   * Enhancements over the original:
   * - Provides a concise JSDoc with parameter & example usage.
   * - Uses a small Int32Array pooling strategy for temporary numeric scratch space to reduce
   *   short-lived allocation churn during frequent redraws.
   * - Employs descriptive local variable names and step-level inline comments for clarity.
   *
   * Steps:
   * 1. Guard and emit a small spacer when no current best candidate exists.
   * 2. Rent a temporary typed-array buffer to hold derived numeric summary values.
   * 3. Populate the buffer with fitness, steps, and progress (scaled where appropriate).
   * 4. Emit a small, framed summary via existing `#formatStat` helper to preserve dashboard styling.
   * 5. Delegate the detailed printing to `MazeVisualization.printMazeStats` (keeps single-responsibility).
   * 6. Return the rented buffer to the internal pool and emit a trailing spacer.
   *
   * @param currentMaze Current maze layout used to compute/print maze-specific stats.
   * @example
   * // invoked internally by `update()` during redraw
   * (dashboard as any)["#printLiveStats"](maze);
   */
  #printLiveStats(currentMaze: string[]): void {
    // Step 1: Leading spacer for visual separation
    this.#logBlank();

    // Defensive guard: if there's no current best candidate, just emit spacer and exit.
    const currentBestCandidate = this.#currentBest;
    if (!currentBestCandidate) {
      this.#logBlank();
      return;
    }

    // Helper: rent a typed Int32Array of requested length from pool or allocate new.
    const rentInt32 = (requestedLength: number): Int32Array => {
      const pooled = DashboardManager.#INT32_SCRATCH_POOL.pop();
      if (pooled && pooled.length >= requestedLength)
        return pooled.subarray(0, requestedLength) as Int32Array;
      return new Int32Array(requestedLength);
    };

    // Helper: return buffer to pool (clear view references by pushing the underlying buffer view).
    const releaseInt32 = (buffer: Int32Array) => {
      // Keep pool bounded to avoid unbounded memory growth.
      if (DashboardManager.#INT32_SCRATCH_POOL.length < 8) {
        DashboardManager.#INT32_SCRATCH_POOL.push(buffer);
      }
    };

    // Step 2: Rent a small scratch buffer for numeric summaries: [fitnessScaled, steps, progressPct]
    const scratch = rentInt32(3);

    // Step 3: Populate numeric summary values defensively.
    const reportedFitness = currentBestCandidate.result?.fitness;
    scratch[0] =
      typeof reportedFitness === 'number' && Number.isFinite(reportedFitness)
        ? Math.round(reportedFitness * 100)
        : 0; // fitness * 100 as integer
    const reportedSteps = Number(currentBestCandidate.result?.steps ?? 0);
    scratch[1] = Number.isFinite(reportedSteps) ? reportedSteps : 0;
    const reportedProgress = Number(currentBestCandidate.result?.progress ?? 0);
    scratch[2] = Number.isFinite(reportedProgress)
      ? Math.round(reportedProgress * 100)
      : 0; // percent

    // Step 4: Emit a compact framed summary using existing formatting helper to keep consistent style.
    // We convert scaled integers back into user-friendly strings for display.
    const formattedFitness = (scratch[0] / 100).toFixed(2);
    const formattedSteps = `${scratch[1]}`;
    const formattedProgress = `${scratch[2]}%`;

    // Use the same label width as solved stats for alignment with other dashboard lines.
    const liveLabelWidth = DashboardManager.#SOLVED_LABEL_WIDTH;
    const liveStat = (label: string, value: string) =>
      this.#formatStat(
        label,
        value,
        colors.neonSilver,
        colors.cyanNeon,
        liveLabelWidth
      );

    this.#logFn(liveStat('Fitness', formattedFitness));
    this.#logFn(liveStat('Steps', formattedSteps));
    this.#logFn(liveStat('Progress', formattedProgress));

    // Step 5: Delegate the more detailed maze/stat printing to the existing visualization helper.
    MazeVisualization.printMazeStats(
      currentBestCandidate,
      currentMaze,
      this.#logFn
    );

    // Step 6: Release typed-array scratch buffer back to pool and trailing spacer for readability.
    releaseInt32(scratch);
    this.#logBlank();
  }

  /** Print progress bar lines. */
  #printProgressBar(): void {
    const padBlank = () =>
      this.#logFn(
        `${colors.blueCore}║${NetworkVisualization.pad(
          ' ',
          DashboardManager.FRAME_INNER_WIDTH,
          ' '
        )}${colors.blueCore}║${colors.reset}`
      );
    padBlank();
    const bar = `Progress to exit: ${MazeVisualization.displayProgressBar(
      this.#currentBest!.result.progress
    )}`;
    this.#logFn(
      `${colors.blueCore}║${NetworkVisualization.pad(
        ' ' + colors.neonSilver + bar + colors.reset,
        DashboardManager.FRAME_INNER_WIDTH,
        ' '
      )}${colors.blueCore}║${colors.reset}`
    );
    padBlank();
  }

  reset(): void {
    this.#solvedMazes = [];
    this.#solvedMazeKeys.clear();
    this.#currentBest = null;
  }

  /**
   * Produce an immutable tail slice of a bounded numeric history buffer.
   *
   * Educational focus:
   * - Encapsulates export window logic so callers don't duplicate clamp arithmetic.
   * - Demonstrates a micro-optimized manual copy for partial slices while using
   *   the native fast path (`Array.prototype.slice`) when returning the full buffer.
   * - Adds defensive guards for null / non-array input (returns empty array) to simplify callers.
   *
   * Steps:
   * 1. Guard: if the provided reference is not a non-empty array, return a new empty array.
   * 2. Compute the starting index for the export window (clamped to 0).
   * 3. If the window spans the entire history, return a shallow copy via `.slice()` (fast path).
   * 4. Allocate an output array sized exactly to the window length.
   * 5. Manually copy values (forward loop) to avoid creating an intermediate subarray before clone.
   * 6. Return the populated tail slice (caller receives an independent array).
   *
   * Complexity:
   * - Let N = history length, W = export window size (<= HISTORY_EXPORT_WINDOW).
   * - Computation: O(min(N, W)) element copies.
   * - Memory: O(min(N, W)) for the returned array.
   *
   * Performance notes:
   * - Manual copy avoids constructing a temporary array then cloning it; though engines optimize slice well,
   *   the explicit loop keeps intent clear and allows descriptive index naming for style compliance.
   * - Uses descriptive loop index (`offsetIndex`) instead of a terse variable to satisfy style guidelines.
   *
   * Determinism: Pure function of input array contents and static window constant.
   * Reentrancy: Safe (no shared mutable state). Input array is never mutated.
   * Edge cases:
   * - Null / undefined / non-array -> returns [].
   * - Empty array -> returns [].
   * - HISTORY_EXPORT_WINDOW >= history length -> returns shallow clone of entire history.
   *
   * @param history Source numeric history buffer (may be longer than export window).
   * @returns New array containing up to `HISTORY_EXPORT_WINDOW` most recent samples (oldest first inside the window).
   */
  #sliceHistoryForExport(history: number[] | undefined | null): number[] {
    // Step 1: Defensive null / type / emptiness guard
    if (!Array.isArray(history) || !history.length) return [];

    // Step 2: Compute window start index
    const startIndex = Math.max(
      0,
      history.length - DashboardManager.#HISTORY_EXPORT_WINDOW
    );

    // Step 3: Full-buffer fast path (return shallow clone)
    if (startIndex === 0) return history.slice();

    // Step 4: Allocate result sized to window length
    const windowLength = history.length - startIndex;
    const windowSlice = new Array<number>(windowLength);

    // Step 5: Manual forward copy
    for (let offsetIndex = 0; offsetIndex < windowLength; offsetIndex++) {
      windowSlice[offsetIndex] = history[startIndex + offsetIndex];
    }

    // Step 6: Return immutable tail window
    return windowSlice;
  }
}
