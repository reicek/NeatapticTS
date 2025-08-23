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
  #clearFn: () => void;
  #logFn: (...args: any[]) => void;
  #archiveFn?: (...args: any[]) => void;
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
  static #HISTORY_MAX = 500;
  static #FRAME_INNER_WIDTH = 148;
  static #LEFT_PADDING = 7;
  static #RIGHT_PADDING = 1;
  static #STAT_LABEL_WIDTH = 28;
  static #ARCHIVE_SPARK_WIDTH = 64; // spark width in archive blocks
  static #GENERAL_SPARK_WIDTH = 64; // spark width in live panel
  static #SOLVED_LABEL_WIDTH = 22; // label width in archive stats
  static #HISTORY_EXPORT_WINDOW = 200; // samples exported in telemetry details
  /** Unicode blocks used for sparklines (ascending). */
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
  /** Floating comparison epsilon for tiny deltas. */
  static #DELTA_EPSILON = 1e-9;
  /** Max operators listed in acceptance stats. */
  static #TOP_OPERATOR_LIMIT = 6;
  /** Max mutations listed. */
  static #TOP_MUTATION_LIMIT = 8;
  /** Max species sizes listed. */
  static #TOP_SPECIES_LIMIT = 5;
  /** Safety multiplier for hidden layer inference loop. */
  static #LAYER_INFER_LOOP_MULTIPLIER = 4;
  /** Label strings reused (dedup hidden classes). */
  static #LABEL_PATH_EFF = 'Path efficiency';
  static #LABEL_PATH_OVER = 'Path overhead';
  static #LABEL_UNIQUE = 'Unique cells visited';
  static #LABEL_REVISITS = 'Cells revisited';
  static #LABEL_STEPS = 'Steps';
  static #LABEL_FITNESS = 'Fitness';
  static #LABEL_ARCH = 'Architecture';
  /** Frame pattern segments reused in redraw. */
  static #FRAME_SINGLE_LINE_CHAR = '═';
  static #FRAME_BRIDGE_TOP = '╦════════════╦';
  static #FRAME_BRIDGE_BOTTOM = '╩════════════╩';
  static #EVOLVING_SECTION_LINE = '══════════════════════';
  // Public aliases for backwards compatibility while internals move to private fields
  static get HISTORY_MAX() {
    return DashboardManager.#HISTORY_MAX;
  }
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
  /** Emit a blank padded line inside the frame to avoid duplication. */
  private logBlank(): void {
    this.#logFn(
      `${colors.blueCore}║${NetworkVisualization.pad(
        ' ',
        DashboardManager.FRAME_INNER_WIDTH,
        ' '
      )}${colors.blueCore}║${colors.reset}`
    );
  }
  constructor(
    clearFn: () => void,
    logFn: (...a: any[]) => void,
    archiveFn?: (...a: any[]) => void
  ) {
    this.#clearFn = clearFn;
    this.#logFn = logFn;
    this.#archiveFn = archiveFn;
    (this as any).logFunction = (...args: any[]) => this.#logFn(...args);
  }

  /**
   * formatStat
   *
   * Small helper that returns a prettified line containing a label and value
   * with color codes applied. The resulting string fits into the dashboard
   * content width and includes frame padding.
   */
  private formatStat(
    label: string,
    value: string | number,
    colorLabel = colors.neonSilver,
    colorValue = colors.cyanNeon,
    labelWidth = DashboardManager.#STAT_LABEL_WIDTH
  ) {
    const lbl = label.endsWith(':') ? label : label + ':';
    const paddedLabel = lbl.padEnd(labelWidth, ' ');
    const composed = `${colorLabel}${paddedLabel}${colorValue} ${value}${colors.reset}`;
    return `${colors.blueCore}║${' '.repeat(
      DashboardManager.#LEFT_PADDING
    )}${NetworkVisualization.pad(
      composed,
      DashboardManager.CONTENT_WIDTH,
      ' ',
      'left'
    )}${' '.repeat(DashboardManager.#RIGHT_PADDING)}${colors.blueCore}║${
      colors.reset
    }`;
  }

  /**
   * buildSparkline
   *
   * Create a compact sparkline string (using block characters) from a numeric
   * series. The series is normalized to the block range and trimmed to the
   * requested width by taking the most recent values.
   */
  private buildSparkline(data: number[], width = 32) {
    if (!data.length) return '';
    const blocks = DashboardManager.#SPARK_BLOCKS;
    // Use shared small-tail helper from MazeUtils so intent is explicit and index math is centralized
    const tail = MazeUtils.tail<number>(data, width);
    let min = Infinity;
    let max = -Infinity;
    for (let tailIndex = 0; tailIndex < tail.length; tailIndex++) {
      const value = tail[tailIndex];
      if (value < min) min = value;
      if (value > max) max = value;
    }
    const range = max - min || 1;
    const out: string[] = [];
    for (let tailIndex = 0; tailIndex < tail.length; tailIndex++) {
      const value = tail[tailIndex];
      const blockIndex = Math.floor(
        ((value - min) / range) * (blocks.length - 1)
      );
      out.push(blocks[blockIndex]);
    }
    return out.join('');
  }

  /**
   * Return up to the last `n` items from `arr` as a new array.
   * Small utility to centralize tail-window extraction logic used across the dashboard.
   */
  // ...getTail removed in favor of MazeUtils.tail

  // Small helper to read the last element of an array safely.
  // Removed legacy #last helper; native Array.prototype.at(-1) is used directly (ES2023).

  /**
   * Push a numeric value onto a history buffer and keep it bounded to HISTORY_MAX.
   */
  // ...existing code...
  /**
   * Create a lightweight key for a maze (used to dedupe solved mazes).
   * The format is intentionally simple (concatenated rows) since the set
   * is only used for equality checks within a single run.
   */
  private getMazeKey(maze: string[]) {
    return maze.join('');
  }

  /**
   * appendSolvedToArchive
   *
   * When a maze is solved for the first time, format and append a boxed
   * representation of the solved maze to the provided `archiveLogFunction`.
   * The block includes a header, optional small trend sparklines, the
   * centered maze drawing, and several efficiency stats derived from the path.
   *
   * This function is careful to be a no-op if no archive logger was provided
   * during construction.
   *
   * @param solved - record containing maze, solution and generation
   * @param displayNumber - 1-based ordinal for the solved maze in the archive
   */
  private appendSolvedToArchive(
    solved: {
      maze: string[];
      result: any;
      network: INetwork;
      generation: number;
    },
    displayNumber: number
  ) {
    if (!this.#archiveFn) return;
    const blockLines: string[] = [];
    this.#appendSolvedHeader(blockLines, solved, displayNumber);
    this.#appendSolvedSparklines(blockLines);
    this.#appendSolvedMaze(blockLines, solved);
    this.#appendSolvedPathStats(blockLines, solved);
    this.#appendSolvedArchitecture(blockLines, solved.network);
    this.#appendSolvedFooterAndEmit(blockLines);
  }

  /** Append header/title/separator lines for a solved maze block. */
  #appendSolvedHeader(
    blockLines: string[],
    solved: { generation: number },
    displayNumber: number
  ): void {
    const header = `${colors.blueCore}╠${NetworkVisualization.pad(
      '═'.repeat(DashboardManager.FRAME_INNER_WIDTH),
      DashboardManager.FRAME_INNER_WIDTH,
      '═'
    )}╣${colors.reset}`;
    const title = `${colors.blueCore}║${NetworkVisualization.pad(
      `${colors.orangeNeon} SOLVED #${displayNumber} (Gen ${solved.generation})${colors.reset}${colors.blueCore}`,
      DashboardManager.FRAME_INNER_WIDTH,
      ' '
    )}║${colors.reset}`;
    const sep = `${colors.blueCore}╠${NetworkVisualization.pad(
      '─'.repeat(DashboardManager.FRAME_INNER_WIDTH),
      DashboardManager.FRAME_INNER_WIDTH,
      '─'
    )}╣${colors.reset}`;
    blockLines.push(header, title, sep);
  }

  /** Append trending sparklines for solved archive block. */
  #appendSolvedSparklines(blockLines: string[]): void {
    const solvedLabelWidth = DashboardManager.#SOLVED_LABEL_WIDTH;
    const solvedStat = (label: string, value: string) =>
      this.formatStat(
        label,
        value,
        colors.neonSilver,
        colors.cyanNeon,
        solvedLabelWidth
      );
    const pushIf = (label: string, value: string) =>
      value && blockLines.push(solvedStat(label, value));
    pushIf(
      'Fitness trend',
      this.buildSparkline(
        this.#bestFitnessHistory,
        DashboardManager.#ARCHIVE_SPARK_WIDTH
      )
    );
    pushIf(
      'Nodes trend',
      this.buildSparkline(
        this.#complexityNodesHistory,
        DashboardManager.#ARCHIVE_SPARK_WIDTH
      )
    );
    pushIf(
      'Conns trend',
      this.buildSparkline(
        this.#complexityConnsHistory,
        DashboardManager.#ARCHIVE_SPARK_WIDTH
      )
    );
    pushIf(
      'Hypervol trend',
      this.buildSparkline(
        this.#hypervolumeHistory,
        DashboardManager.#ARCHIVE_SPARK_WIDTH
      )
    );
    pushIf(
      'Progress trend',
      this.buildSparkline(
        this.#progressHistory,
        DashboardManager.#ARCHIVE_SPARK_WIDTH
      )
    );
    pushIf(
      'Species trend',
      this.buildSparkline(
        this.#speciesCountHistory,
        DashboardManager.#ARCHIVE_SPARK_WIDTH
      )
    );
    // Spacer
    blockLines.push(
      `${colors.blueCore}║${NetworkVisualization.pad(
        ' ',
        DashboardManager.FRAME_INNER_WIDTH,
        ' '
      )}${colors.blueCore}║${colors.reset}`
    );
  }

  /** Append centered solved maze drawing. */
  #appendSolvedMaze(
    blockLines: string[],
    solved: { maze: string[]; result: any }
  ): void {
    const endPosition = (solved.result.path as readonly [number, number][])?.at(
      -1
    );
    const visualization = MazeVisualization.visualizeMaze(
      solved.maze,
      ((endPosition as readonly [number, number]) ?? [0, 0]) as readonly [
        number,
        number
      ],
      solved.result.path as readonly [number, number][]
    );
    const lines = Array.isArray(visualization)
      ? visualization
      : visualization.split('\n');
    const centered = lines
      .map((line) =>
        NetworkVisualization.pad(line, DashboardManager.FRAME_INNER_WIDTH, ' ')
      )
      .join('\n');
    centered
      .split('\n')
      .forEach((mazeLine) =>
        blockLines.push(
          `${colors.blueCore}║${NetworkVisualization.pad(
            mazeLine,
            DashboardManager.FRAME_INNER_WIDTH,
            ' '
          )}${colors.blueCore}║${colors.reset}`
        )
      );
  }

  /** Compute path metrics & append stats lines. */
  #appendSolvedPathStats(
    blockLines: string[],
    solved: { maze: string[]; result: any }
  ): void {
    const solvedLabelWidth = DashboardManager.#SOLVED_LABEL_WIDTH;
    const solvedStat = (label: string, value: string) =>
      this.formatStat(
        label,
        value,
        colors.neonSilver,
        colors.cyanNeon,
        solvedLabelWidth
      );
    const metrics = this.#computePathMetrics(solved.maze, solved.result);
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

  /** Append derived architecture string. */
  #appendSolvedArchitecture(blockLines: string[], network: INetwork): void {
    const solvedLabelWidth = DashboardManager.#SOLVED_LABEL_WIDTH;
    const solvedStat = (label: string, value: string) =>
      this.formatStat(
        label,
        value,
        colors.neonSilver,
        colors.cyanNeon,
        solvedLabelWidth
      );
    let architecture = 'n/a';
    try {
      architecture = this.#deriveArchitecture(network as any);
    } catch {
      architecture = 'n/a';
    }
    blockLines.push(solvedStat(DashboardManager.#LABEL_ARCH, architecture));
  }

  /** Emit footer & send archive block to logger. */
  #appendSolvedFooterAndEmit(blockLines: string[]): void {
    blockLines.push(
      `${colors.blueCore}╚${NetworkVisualization.pad(
        '═'.repeat(DashboardManager.FRAME_INNER_WIDTH),
        DashboardManager.FRAME_INNER_WIDTH,
        '═'
      )}╝${colors.reset}`
    );
    try {
      (this.#archiveFn as any)(blockLines.join('\n'), { prepend: true });
    } catch {
      const append = this.#archiveFn ?? (() => {});
      blockLines.forEach((singleLine) => append(singleLine));
    }
  }

  /** Compute path metrics from maze + result object. */
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
    const startPosition = MazeUtils.findPosition(maze, 'S');
    const exitPosition = MazeUtils.findPosition(maze, 'E');
    const optimalLength = MazeUtils.bfsDistance(
      MazeUtils.encodeMaze(maze),
      startPosition,
      exitPosition
    );
    const pathLength = result.path.length - 1;
    const efficiencyPct = Math.min(
      100,
      Math.round((optimalLength / pathLength) * 100)
    ).toFixed(1);
    const overheadPct = ((pathLength / optimalLength) * 100 - 100).toFixed(1);
    const uniqueCells = new Set<string>();
    let revisitedCells = 0;
    for (const [cellX, cellY] of result.path) {
      const key = `${cellX},${cellY}`;
      if (uniqueCells.has(key)) revisitedCells++;
      else uniqueCells.add(key);
    }
    return {
      optimalLength,
      pathLength,
      efficiencyPct,
      overheadPct,
      uniqueCellsVisited: uniqueCells.size,
      revisitedCells,
      totalSteps: result.steps,
      fitnessValue: result.fitness,
    };
  }

  /** Derive architecture string (Input - Hidden... - Output). */
  #deriveArchitecture(networkInstance: any): string {
    if (!networkInstance) return 'n/a';
    if (
      Array.isArray(networkInstance.layers) &&
      networkInstance.layers.length >= 2
    ) {
      const sizes: number[] = [];
      for (
        let layerIndex = 0;
        layerIndex < networkInstance.layers.length;
        layerIndex++
      ) {
        const layerRef = networkInstance.layers[layerIndex];
        const size = Array.isArray(layerRef?.nodes)
          ? layerRef.nodes.length
          : Array.isArray(layerRef)
          ? layerRef.length
          : 0;
        sizes.push(size);
      }
      return sizes.join(' - ');
    }
    if (Array.isArray(networkInstance.nodes)) {
      const inputNodes = networkInstance.nodes.filter(
        (n: any) => n.type === 'input'
      );
      const outputNodes = networkInstance.nodes.filter(
        (n: any) => n.type === 'output'
      );
      const hiddenNodesAll = networkInstance.nodes.filter(
        (n: any) => n.type === 'hidden'
      );
      if (!hiddenNodesAll.length) {
        if (
          typeof networkInstance.input === 'number' &&
          typeof networkInstance.output === 'number'
        ) {
          return `${networkInstance.input} - ${networkInstance.output}`;
        }
        return `${inputNodes.length} - ${outputNodes.length}`;
      }
      const assignedSet = new Set<any>(inputNodes);
      let remainingHidden = hiddenNodesAll.slice();
      const hiddenSizes: number[] = [];
      const safetyLimit =
        hiddenNodesAll.length * DashboardManager.#LAYER_INFER_LOOP_MULTIPLIER;
      let iterationCounter = 0;
      while (remainingHidden.length && iterationCounter < safetyLimit) {
        iterationCounter++;
        const current = remainingHidden.filter((hiddenNode: any) =>
          hiddenNode.connections?.in?.every((conn: any) =>
            assignedSet.has(conn.from)
          )
        );
        if (!current.length) {
          hiddenSizes.push(remainingHidden.length);
          break;
        }
        hiddenSizes.push(current.length);
        for (const nodeRef of current) assignedSet.add(nodeRef);
        remainingHidden = remainingHidden.filter(
          (n: any) => !assignedSet.has(n)
        );
      }
      return [
        `${inputNodes.length}`,
        ...hiddenSizes.map((hs) => `${hs}`),
        `${outputNodes.length}`,
      ].join(' - ');
    }
    if (
      typeof networkInstance.input === 'number' &&
      typeof networkInstance.output === 'number'
    ) {
      return `${networkInstance.input} - ${networkInstance.output}`;
    }
    return 'n/a';
  }

  /**
   * update
   *
   * Called by the evolution engine to report the latest candidate solution
   * (or the current best). The dashboard will:
   * - update the currentBest reference used for the live view
   * - if the provided result is a successful solve and it's the first time
   *   we've seen this maze, append an archive block
   * - stash the latest telemetry values into small circular buffers for sparklines
   * - finally call `redraw` to update the live output
   */
  /**
   * Ingest a new evolution update (called by the engine each generation or candidate improvement).
   * @param maze Current maze layout being solved.
   * @param result Candidate evaluation result (includes fitness, path, progress, etc.).
   * @param network Neural network associated with the candidate.
   * @param generation Current generation number.
   * @param neatInstance Optional NEAT framework instance exposing telemetry helpers.
   * @returns void
   */
  update(
    maze: string[],
    result: any,
    network: INetwork,
    generation: number,
    neatInstance?: any
  ): void {
    if (this.#runStartTs == null) {
      this.#runStartTs = Date.now(); // wall-clock anchor
      this.#perfStart = globalThis.performance?.now?.() ?? this.#runStartTs;
    }
    this.#lastUpdateTs = globalThis.performance?.now?.() ?? Date.now();
    this.#lastGeneration = generation;
    // Update live candidate
    this.#currentBest = { result, network, generation };

    // If this run solved the maze and it's a new maze, add & archive it
    if (result.success) {
      const mazeKey = this.getMazeKey(maze);
      if (!this.#solvedMazeKeys.has(mazeKey)) {
        this.#solvedMazes.push({ maze, result, network, generation });
        this.#solvedMazeKeys.add(mazeKey);
        // Append to archive immediately when first solved
        const displayNumber = this.#solvedMazes.length; // 1-based
        this.appendSolvedToArchive(
          { maze, result, network, generation },
          displayNumber
        );
      }
    }

    // Pull the latest telemetry from the NEAT instance (if available)
    const telemetry = neatInstance?.getTelemetry?.();
    if (telemetry && telemetry.length) {
      // Keep only the most recent telemetry object
      // Capture latest telemetry snapshot (last element)
      this.#lastTelemetry = MazeUtils.safeLast(telemetry as any[]);

      // Record best fitness into a small history window for trend views
      const bestFit = this.#currentBest?.result?.fitness;
      if (typeof bestFit === 'number') {
        this.#lastBestFitness = bestFit;
        this.#bestFitnessHistory = MazeUtils.pushHistory(
          this.#bestFitnessHistory,
          bestFit,
          DashboardManager.HISTORY_MAX
        );
      }

      // Complexity telemetry: mean nodes/connectivity across population
      const complexityTelemetry = this.#lastTelemetry?.complexity;
      if (complexityTelemetry) {
        if (typeof complexityTelemetry.meanNodes === 'number') {
          this.#complexityNodesHistory = MazeUtils.pushHistory(
            this.#complexityNodesHistory,
            complexityTelemetry.meanNodes,
            DashboardManager.HISTORY_MAX
          );
        }
        if (typeof complexityTelemetry.meanConns === 'number') {
          this.#complexityConnsHistory = MazeUtils.pushHistory(
            this.#complexityConnsHistory,
            complexityTelemetry.meanConns,
            DashboardManager.HISTORY_MAX
          );
        }
      }

      // Hypervolume is used for multi-objective tracking
      const hyperVolumeValue = this.#lastTelemetry?.hyper;
      if (typeof hyperVolumeValue === 'number') {
        this.#hypervolumeHistory = MazeUtils.pushHistory(
          this.#hypervolumeHistory,
          hyperVolumeValue,
          DashboardManager.HISTORY_MAX
        );
      }

      // Progress: how close a candidate is to the exit
      const progressValue = this.#currentBest?.result?.progress;
      if (typeof progressValue === 'number') {
        this.#progressHistory = MazeUtils.pushHistory(
          this.#progressHistory,
          progressValue,
          DashboardManager.HISTORY_MAX
        );
      }

      // Species count history
      const speciesCountValue = this.#lastTelemetry?.species;
      if (typeof speciesCountValue === 'number') {
        this.#speciesCountHistory = MazeUtils.pushHistory(
          this.#speciesCountHistory,
          speciesCountValue,
          DashboardManager.HISTORY_MAX
        );
      }
    }

    // Render the live dashboard
    this.redraw(maze, neatInstance);

    // Emit telemetry payload (now includes rich details previously only rendered inline)
    try {
      const elapsedMs =
        this.#perfStart != null && globalThis.performance?.now
          ? globalThis.performance.now() - this.#perfStart
          : this.#runStartTs
          ? Date.now() - this.#runStartTs
          : 0;
      const gensPerSec = elapsedMs > 0 ? generation / (elapsedMs / 1000) : 0;
      const payload = {
        type: 'asciiMaze:telemetry',
        generation,
        bestFitness: this.#lastBestFitness,
        progress: this.#currentBest?.result?.progress ?? null,
        speciesCount: this.#speciesCountHistory.at(-1) ?? null,
        gensPerSec: +gensPerSec.toFixed(3),
        timestamp: Date.now(), // wall-clock timestamp for consumers
        details: this.#lastDetailedStats || null,
      };
      // Custom event inside iframe / page
      if (typeof window !== 'undefined') {
        try {
          window.dispatchEvent(
            new CustomEvent('asciiMazeTelemetry', { detail: payload })
          );
        } catch {}
        // postMessage to parent frame if different
        try {
          if (window.parent && window.parent !== window)
            window.parent.postMessage(payload, '*');
        } catch {}
        // Expose last telemetry on global for polling
        (window as any).asciiMazeLastTelemetry = payload;
      }
      // Optional userland callback hook (if injected)
      try {
        (this as any)._telemetryHook && (this as any)._telemetryHook(payload);
      } catch {}
    } catch {
      /* ignore telemetry errors */
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
      timestamp: Date.now(),
      details: this.#lastDetailedStats || null,
    };
  }

  /**
   * redraw
   *
   * Responsible for clearing the live area and printing a compact snapshot of
   * the current best candidate, a short network summary, the maze drawing and
   * several telemetry-derived stats. The function uses `logFunction` for all
   * output lines so the same renderer can be used both in Node and in the
   * browser (DOM adapter).
   * @param currentMaze The maze currently being evolved.
   * @param neat Optional NEAT implementation instance for population-level stats.
   * @returns void
   */
  redraw(currentMaze: string[], neat?: any): void {
    this.#clearFn();
    this.#printTopFrame();
    if (this.#currentBest) this.#printCurrentBestSection(currentMaze);

    // General stats area (telemetry-derived values). These are defensive reads
    // because telemetry may be missing early in the run.
    const last = this.#lastTelemetry;
    const complexity = last?.complexity;
    const perf = last?.perf;
    const lineage = last?.lineage;
    const fronts = Array.isArray(last?.fronts) ? last.fronts : null;
    const objectives = last?.objectives;
    const hyper = last?.hyper;
    const diversity = last?.diversity;
    const mutationStats = last?.mutationStats || last?.mutation?.stats;
    const bestFitness = this.#currentBest?.result?.fitness;

    // Small helpers used below when building the stats list
    const fmtNum = (numericValue: any, digits = 2) =>
      typeof numericValue === 'number' && isFinite(numericValue)
        ? numericValue.toFixed(digits)
        : '-';
    const deltaArrow = (curr?: number | null, prev?: number | null) => {
      if (curr == null || prev == null) return '';
      const diff = curr - prev;
      if (Math.abs(diff) < DashboardManager.#DELTA_EPSILON)
        return `${colors.neonSilver} (↔0)`;
      const color = diff > 0 ? colors.cyanNeon : colors.neonRed;
      const arrow = diff > 0 ? '↑' : '↓';
      return `${color} (${arrow}${diff.toFixed(2)})${colors.neonSilver}`;
    };

    // Derive some population-level stats if a NEAT instance is available
    let popMean: any = '-';
    let popMedian: any = '-';
    let speciesCount: any = '-';
    let enabledRatio: any = '-';
    if (neat && Array.isArray(neat.population)) {
      const scores: number[] = [];
      let enabled = 0,
        total = 0;
      neat.population.forEach((g: any) => {
        if (typeof g.score === 'number') scores.push(g.score);
        if (Array.isArray(g.connections)) {
          g.connections.forEach((connection: any) => {
            total++;
            if (connection.enabled !== false) enabled++;
          });
        }
      });
      if (scores.length) {
        const sum = scores.reduce(
          (runningTotal, scoreValue) => runningTotal + scoreValue,
          0
        );
        popMean = (sum / scores.length).toFixed(2);
        // Immutable copy before sort to avoid mutating original scores array
        const sorted = [...scores].sort(
          (leftScore, rightScore) => leftScore - rightScore
        );
        const mid = Math.floor(sorted.length / 2);
        popMedian = (sorted.length % 2 === 0
          ? (sorted[mid - 1] + sorted[mid]) / 2
          : sorted[mid]
        ).toFixed(2);
      }
      if (total) enabledRatio = (enabled / total).toFixed(2);
      speciesCount = Array.isArray(neat.species)
        ? neat.species.length.toString()
        : speciesCount;
    }

    // Build small sparklines used in the general stats area
    const firstFrontSize = fronts?.[0]?.length || 0;
    const SPARK_WIDTH = DashboardManager.#GENERAL_SPARK_WIDTH;
    const spark = this.buildSparkline(this.#bestFitnessHistory, SPARK_WIDTH);
    const sparkComplexityNodes = this.buildSparkline(
      this.#complexityNodesHistory,
      SPARK_WIDTH
    );
    const sparkComplexityConns = this.buildSparkline(
      this.#complexityConnsHistory,
      SPARK_WIDTH
    );
    const sparkHyper = this.buildSparkline(
      this.#hypervolumeHistory,
      SPARK_WIDTH
    );
    const sparkProgress = this.buildSparkline(
      this.#progressHistory,
      SPARK_WIDTH
    );
    const sparkSpecies = this.buildSparkline(
      this.#speciesCountHistory,
      SPARK_WIDTH
    );

    // Build externalized detailed stats object (formerly rendered inline)
    try {
      const satFrac = (this.#currentBest as any)?.result?.saturationFraction;
      const actEnt = (this.#currentBest as any)?.result?.actionEntropy;
      if (popMean === '-' && typeof bestFitness === 'number')
        popMean = bestFitness.toFixed(2);
      if (popMedian === '-' && typeof bestFitness === 'number')
        popMedian = bestFitness.toFixed(2);
      if (speciesCount === '-' && typeof last?.species === 'number')
        speciesCount = String(last.species);
      let noveltyArchiveSize: number | null = null;
      if (neat?.getNoveltyArchiveSize) {
        try {
          noveltyArchiveSize = neat.getNoveltyArchiveSize();
        } catch {}
      }
      let operatorAcceptance: Array<{
        name: string;
        acceptancePct: number;
      }> | null = null;
      if (neat?.getOperatorStats) {
        try {
          const ops = neat.getOperatorStats();
          if (Array.isArray(ops) && ops.length) {
            // Sort operators by success rate (immutable sort)
            const sortedOps = [...ops].sort(
              (leftOp: any, rightOp: any) =>
                rightOp.success / Math.max(1, rightOp.attempts) -
                leftOp.success / Math.max(1, leftOp.attempts)
            );
            operatorAcceptance = [];
            const operatorTake = Math.min(
              DashboardManager.#TOP_OPERATOR_LIMIT,
              sortedOps.length
            );
            for (
              let operatorIndex = 0;
              operatorIndex < operatorTake;
              operatorIndex++
            ) {
              const opStats = sortedOps[operatorIndex];
              operatorAcceptance.push({
                name: opStats.name,
                acceptancePct: +(
                  (100 * opStats.success) /
                  Math.max(1, opStats.attempts)
                ).toFixed(2),
              });
            }
          }
        } catch {}
      }
      let topMutations: Array<{ name: string; count: number }> | null = null;
      if (mutationStats && typeof mutationStats === 'object') {
        try {
          {
            const entries = Object.entries(mutationStats).filter(
              ([mutationKey, mutationValue]) =>
                typeof mutationValue === 'number'
            );
            const sortedEntries = [...(entries as [string, number][])]!.sort(
              (leftEntry, rightEntry) =>
                (rightEntry[1] as number) - (leftEntry[1] as number)
            );
            topMutations = [];
            const mutationTake = Math.min(
              DashboardManager.#TOP_MUTATION_LIMIT,
              sortedEntries.length
            );
            for (
              let mutationIndex = 0;
              mutationIndex < mutationTake;
              mutationIndex++
            ) {
              const [mutationName, mutationCount] = sortedEntries[
                mutationIndex
              ];
              topMutations.push({
                name: mutationName,
                count: mutationCount as number,
              });
            }
          }
        } catch {}
      }
      const topSpeciesSizes = Array.isArray(neat?.species)
        ? (() => {
            const sizes = neat.species.map((s: any) => s.members?.length || 0);
            const sortedSizes = [...sizes].sort(
              (sizeA: number, sizeB: number) => sizeB - sizeA
            );
            const outArray: number[] = [];
            const speciesTake = Math.min(
              DashboardManager.#TOP_SPECIES_LIMIT,
              sortedSizes.length
            );
            for (
              let speciesIndex = 0;
              speciesIndex < speciesTake;
              speciesIndex++
            ) {
              outArray.push(sortedSizes[speciesIndex]);
            }
            return outArray;
          })()
        : null;
      const paretoFrontSizes = fronts
        ? fronts.map((f: any) => f?.length || 0)
        : null;
      this.#lastDetailedStats = {
        generation: this.#currentBest?.generation || 0,
        bestFitness: typeof bestFitness === 'number' ? bestFitness : null,
        bestFitnessDelta: ((): number | null => {
          if (typeof bestFitness !== 'number') return null;
          const prev = this.#bestFitnessHistory.at(-2) ?? null;
          if (prev == null) return null;
          const diff = bestFitness - prev;
          return +diff.toFixed(3);
        })(),
        saturationFraction: typeof satFrac === 'number' ? satFrac : null,
        actionEntropy: typeof actEnt === 'number' ? actEnt : null,
        populationMean: popMean === '-' ? null : +popMean,
        populationMedian: popMedian === '-' ? null : +popMedian,
        enabledConnRatio: enabledRatio === '-' ? null : +enabledRatio,
        complexity: complexity || null,
        simplifyPhaseActive: !!(
          complexity &&
          (complexity.growthNodes < 0 || complexity.growthConns < 0)
        ),
        perf: perf || null,
        lineage: lineage || null,
        diversity: diversity || null,
        speciesCount: speciesCount === '-' ? null : +speciesCount,
        topSpeciesSizes,
        objectives: objectives || null,
        paretoFrontSizes,
        firstFrontSize,
        hypervolume: typeof hyper === 'number' ? hyper : null,
        noveltyArchiveSize,
        operatorAcceptance,
        topMutations,
        mutationStats: mutationStats || null,
        trends: {
          fitness: spark || null,
          nodes: sparkComplexityNodes || null,
          conns: sparkComplexityConns || null,
          hyper: sparkHyper || null,
          progress: sparkProgress || null,
          species: sparkSpecies || null,
        },
        histories: {
          bestFitness: this.sliceHistoryForExport(this.#bestFitnessHistory),
          nodes: this.sliceHistoryForExport(this.#complexityNodesHistory),
          conns: this.sliceHistoryForExport(this.#complexityConnsHistory),
          hyper: this.sliceHistoryForExport(this.#hypervolumeHistory),
          progress: this.sliceHistoryForExport(this.#progressHistory),
          species: this.sliceHistoryForExport(this.#speciesCountHistory),
        },
        timestamp: Date.now(),
      };
    } catch {
      // Fail silently – details are optional
    }

    // Replace old verbose stats area with a minimal spacer line (keeps frame aesthetics)
    this.logBlank();
  }

  /** Print the static top frame (title header). */
  #printTopFrame(): void {
    this.#logFn(
      `${colors.blueCore}╔${NetworkVisualization.pad(
        DashboardManager.#FRAME_SINGLE_LINE_CHAR,
        DashboardManager.FRAME_INNER_WIDTH,
        DashboardManager.#FRAME_SINGLE_LINE_CHAR
      )}${colors.blueCore}╗${colors.reset}`
    );
    this.#logFn(
      `${colors.blueCore}╚${NetworkVisualization.pad(
        DashboardManager.#FRAME_BRIDGE_TOP,
        DashboardManager.FRAME_INNER_WIDTH,
        DashboardManager.#FRAME_SINGLE_LINE_CHAR
      )}${colors.blueCore}╝${colors.reset}`
    );
    this.#logFn(
      (() => {
        const rawLabel = '║ ASCII maze ║';
        const rawLength = rawLabel.length; // visible columns
        const width = DashboardManager.FRAME_INNER_WIDTH;
        const remaining = width - rawLength;
        const leftPad = Math.max(0, Math.ceil(remaining / 2)) + 1; // bias right by giving extra space to left side
        const rightPad = Math.max(0, remaining - leftPad);
        const coloredLabel = `║ ${colors.neonYellow}ASCII maze${colors.blueCore} ║`;
        return `${colors.blueCore}${' '.repeat(
          leftPad
        )}${coloredLabel}${' '.repeat(rightPad)}${colors.reset}`;
      })()
    );
    this.#logFn(
      `${colors.blueCore}╔${NetworkVisualization.pad(
        DashboardManager.#FRAME_BRIDGE_BOTTOM,
        DashboardManager.FRAME_INNER_WIDTH,
        DashboardManager.#FRAME_SINGLE_LINE_CHAR
      )}${colors.blueCore}╗${colors.reset}`
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
    this.logBlank();
    this.#printNetworkSummary();
    this.#printLiveMaze(currentMaze);
    this.#printLiveStats(currentMaze);
    this.#printProgressBar();
  }

  /** Print network summary visualization. */
  #printNetworkSummary(): void {
    this.logBlank();
    this.#logFn(
      NetworkVisualization.visualizeNetworkSummary(this.#currentBest!.network)
    );
    this.logBlank();
  }

  /** Print the maze visualization for current best. */
  #printLiveMaze(currentMaze: string[]): void {
    const lastPosition = (this.#currentBest!.result.path as readonly [
      number,
      number
    ][])?.at(-1) ?? [0, 0];
    const visualization = MazeVisualization.visualizeMaze(
      currentMaze,
      (lastPosition as readonly [number, number]) ?? [0, 0],
      this.#currentBest!.result.path as readonly [number, number][]
    );
    const lines = Array.isArray(visualization)
      ? visualization
      : visualization.split('\n');
    const centered = lines
      .map(
        (line) =>
          `${colors.blueCore}║${NetworkVisualization.pad(
            line,
            DashboardManager.FRAME_INNER_WIDTH,
            ' '
          )}${colors.blueCore}║`
      )
      .join('\n');
    this.logBlank();
    this.#logFn(centered);
    this.logBlank();
  }

  /** Print current best maze stats. */
  #printLiveStats(currentMaze: string[]): void {
    this.logBlank();
    MazeVisualization.printMazeStats(
      this.#currentBest!,
      currentMaze,
      this.#logFn
    );
    this.logBlank();
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
   * Return the last export window slice of a numeric history (immutable copy).
   * @param history Source numeric history buffer.
   * @returns New array containing up to HISTORY_EXPORT_WINDOW most recent samples.
   */
  private sliceHistoryForExport(history: number[]): number[] {
    if (!history.length) return [];
    const startIndex = Math.max(
      0,
      history.length - DashboardManager.#HISTORY_EXPORT_WINDOW
    );
    // Fast path: if the window covers entire array, return a shallow copy.
    if (startIndex === 0) return history.slice();
    const outLength = history.length - startIndex;
    const out = new Array(outLength);
    for (let i = 0; i < outLength; i++) {
      out[i] = history[startIndex + i];
    }
    return out;
  }
}
