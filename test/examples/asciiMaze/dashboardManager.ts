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
  static #CONTENT_WIDTH =
    DashboardManager.#FRAME_INNER_WIDTH -
    DashboardManager.#LEFT_PADDING -
    DashboardManager.#RIGHT_PADDING;
  static #STAT_LABEL_WIDTH = 28;
  static #ARCHIVE_SPARK_WIDTH = 64; // spark width in archive blocks
  static #GENERAL_SPARK_WIDTH = 64; // spark width in live panel
  static #SOLVED_LABEL_WIDTH = 22; // label width in archive stats
  static #HISTORY_EXPORT_WINDOW = 200; // samples exported in telemetry details
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
    return DashboardManager.#CONTENT_WIDTH;
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
      DashboardManager.#CONTENT_WIDTH,
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
    const blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    // Use shared small-tail helper from MazeUtils so intent is explicit and index math is centralized
    const tail = MazeUtils.tail<number>(data, width);
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < tail.length; i++) {
      const value = tail[i];
      if (value < min) min = value;
      if (value > max) max = value;
    }
    const range = max - min || 1;
    const out: string[] = [];
    for (let i = 0; i < tail.length; i++) {
      const value = tail[i];
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

    // Render solved maze visualization using the MazeVisualization helper
    // Use modern Array.prototype.at for last element access
    const endPos = (solved.result.path as readonly [number, number][])?.at(-1);
    const solvedMazeVisualization = MazeVisualization.visualizeMaze(
      solved.maze,
      ((endPos as readonly [number, number]) ?? [0, 0]) as readonly [
        number,
        number
      ],
      solved.result.path as readonly [number, number][]
    );
    const solvedMazeLines = Array.isArray(solvedMazeVisualization)
      ? solvedMazeVisualization
      : solvedMazeVisualization.split('\n');

    // Center each maze line to the frame width
    const centeredSolvedMaze = solvedMazeLines
      .map((line) =>
        NetworkVisualization.pad(line, DashboardManager.FRAME_INNER_WIDTH, ' ')
      )
      .join('\n');

    // Create boxed header / title / separator lines consistent with the dashboard frame
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

    // Build the entire boxed block as one string so we can prepend it to the
    // archive (newest-first). Building a single string also reduces DOM churn.
    const blockLines: string[] = [];
    blockLines.push(header);
    blockLines.push(title);
    blockLines.push(sep);

    // Optional trending sparklines derived from stored history windows
    const solvedLabelWidth = DashboardManager.#SOLVED_LABEL_WIDTH;
    const solvedStat = (label: string, value: string) =>
      this.formatStat(
        label,
        value,
        colors.neonSilver,
        colors.cyanNeon,
        solvedLabelWidth
      );

    const spark = this.buildSparkline(
      this.#bestFitnessHistory,
      DashboardManager.#ARCHIVE_SPARK_WIDTH
    );
    const sparkComplexityNodes = this.buildSparkline(
      this.#complexityNodesHistory,
      DashboardManager.#ARCHIVE_SPARK_WIDTH
    );
    const sparkComplexityConns = this.buildSparkline(
      this.#complexityConnsHistory,
      DashboardManager.#ARCHIVE_SPARK_WIDTH
    );
    const sparkHyper = this.buildSparkline(
      this.#hypervolumeHistory,
      DashboardManager.#ARCHIVE_SPARK_WIDTH
    );
    const sparkProgress = this.buildSparkline(
      this.#progressHistory,
      DashboardManager.#ARCHIVE_SPARK_WIDTH
    );
    const sparkSpecies = this.buildSparkline(
      this.#speciesCountHistory,
      DashboardManager.#ARCHIVE_SPARK_WIDTH
    );

    if (spark) blockLines.push(solvedStat('Fitness trend', spark));
    if (sparkComplexityNodes)
      blockLines.push(solvedStat('Nodes trend', sparkComplexityNodes));
    if (sparkComplexityConns)
      blockLines.push(solvedStat('Conns trend', sparkComplexityConns));
    if (sparkHyper) blockLines.push(solvedStat('Hypervol trend', sparkHyper));
    if (sparkProgress)
      blockLines.push(solvedStat('Progress trend', sparkProgress));
    if (sparkSpecies)
      blockLines.push(solvedStat('Species trend', sparkSpecies));

    // Blank spacer line inside the box
    blockLines.push(
      `${colors.blueCore}║${NetworkVisualization.pad(
        ' ',
        DashboardManager.FRAME_INNER_WIDTH,
        ' '
      )}${colors.blueCore}║${colors.reset}`
    );

    // Append the centered maze drawing, row by row (each padded into the frame)
    centeredSolvedMaze
      .split('\n')
      .forEach((l) =>
        blockLines.push(
          `${colors.blueCore}║${NetworkVisualization.pad(
            l,
            DashboardManager.FRAME_INNER_WIDTH,
            ' '
          )}${colors.blueCore}║${colors.reset}`
        )
      );

    // Compute a few path efficiency stats: optimal length (BFS), actual path length, revisit counts
    const startPos = MazeUtils.findPosition(solved.maze, 'S');
    const exitPos = MazeUtils.findPosition(solved.maze, 'E');
    const optimalLength = MazeUtils.bfsDistance(
      MazeUtils.encodeMaze(solved.maze),
      startPos,
      exitPos
    );
    const pathLength = solved.result.path.length - 1;
    const efficiency = Math.min(
      100,
      Math.round((optimalLength / pathLength) * 100)
    ).toFixed(1);
    const overhead = ((pathLength / optimalLength) * 100 - 100).toFixed(1);

    // Count unique vs revisited cells along the path
    const uniqueCells = new Set<string>();
    let revisitedCells = 0;
    for (const [x, y] of solved.result.path) {
      const cellKey = `${x},${y}`;
      if (uniqueCells.has(cellKey)) revisitedCells++;
      else uniqueCells.add(cellKey);
    }

    // Append efficiency & fitness stats
    blockLines.push(
      solvedStat(
        'Path efficiency',
        `${optimalLength}/${pathLength} (${efficiency}%)`
      )
    );
    blockLines.push(
      solvedStat('Path overhead', `${overhead}% longer than optimal`)
    );
    blockLines.push(solvedStat('Unique cells visited', `${uniqueCells.size}`));
    blockLines.push(solvedStat('Cells revisited', `${revisitedCells} times`));
    blockLines.push(solvedStat('Steps', `${solved.result.steps}`));
    blockLines.push(
      solvedStat('Fitness', `${solved.result.fitness.toFixed(2)}`)
    );

    // Bottom border of the boxed block
    blockLines.push(
      `${colors.blueCore}╚${NetworkVisualization.pad(
        '═'.repeat(DashboardManager.FRAME_INNER_WIDTH),
        DashboardManager.FRAME_INNER_WIDTH,
        '═'
      )}╝${colors.reset}`
    );

    // Finally, emit the entire block using the archive logger. Pass the `{ prepend: true }`
    // option so the logger places the newest block at the top of the archive.
    try {
      (this.#archiveFn as any)(blockLines.join('\n'), { prepend: true });
    } catch {
      const append = this.#archiveFn ?? (() => {});
      blockLines.forEach((l) => append(l));
    }
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
    // Clear the live area (archive is untouched)
    this.#clearFn();

    // Header: top frame lines
    this.#logFn(
      `${colors.blueCore}╔${NetworkVisualization.pad(
        '═',
        DashboardManager.FRAME_INNER_WIDTH,
        '═'
      )}${colors.blueCore}╗${colors.reset}`
    );
    this.#logFn(
      `${colors.blueCore}╚${NetworkVisualization.pad(
        '╦════════════╦',
        DashboardManager.FRAME_INNER_WIDTH,
        '═'
      )}${colors.blueCore}╝${colors.reset}`
    );
    this.#logFn(
      `${colors.blueCore}${NetworkVisualization.pad(
        `║ ${colors.neonYellow}ASCII maze${colors.blueCore} ║`,
        150,
        ' '
      )}${colors.reset}`
    );
    this.#logFn(
      `${colors.blueCore}╔${NetworkVisualization.pad(
        '╩════════════╩',
        DashboardManager.FRAME_INNER_WIDTH,
        '═'
      )}${colors.blueCore}╗${colors.reset}`
    );

    // Print current best for active maze if available
    if (this.#currentBest) {
      this.#logFn(
        `${colors.blueCore}╠${NetworkVisualization.pad(
          '══════════════════════',
          DashboardManager.FRAME_INNER_WIDTH,
          '═'
        )}${colors.blueCore}╣${colors.reset}`
      );
      this.#logFn(
        `${colors.blueCore}║${NetworkVisualization.pad(
          `${colors.orangeNeon}EVOLVING (GEN ${this.#currentBest.generation})`,
          DashboardManager.FRAME_INNER_WIDTH,
          ' '
        )}${colors.blueCore}║${colors.reset}`
      );
      this.#logFn(
        `${colors.blueCore}╠${NetworkVisualization.pad(
          '══════════════════════',
          DashboardManager.FRAME_INNER_WIDTH,
          '═'
        )}${colors.blueCore}╣${colors.reset}`
      );
      this.logBlank();

      // Network summary (compact visualization)
      this.logBlank();
      this.#logFn(
        NetworkVisualization.visualizeNetworkSummary(this.#currentBest.network)
      );
      this.logBlank();

      // Maze visualization for the live candidate
      const lastPos = (this.#currentBest.result.path as readonly [
        number,
        number
      ][])?.at(-1) ?? [0, 0];
      const currentMazeVisualization = MazeVisualization.visualizeMaze(
        currentMaze,
        (lastPos as readonly [number, number]) ?? [0, 0],
        this.#currentBest.result.path as readonly [number, number][]
      );
      const currentMazeLines = Array.isArray(currentMazeVisualization)
        ? currentMazeVisualization
        : currentMazeVisualization.split('\n');
      const centeredCurrentMaze = currentMazeLines
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
      this.#logFn(centeredCurrentMaze);
      this.logBlank();

      // Print stats for the current best solution (delegates to MazeVisualization)
      this.logBlank();
      MazeVisualization.printMazeStats(
        this.#currentBest,
        currentMaze,
        this.#logFn
      );
      this.logBlank();

      // Progress bar for current candidate
      this.#logFn(
        `${colors.blueCore}║${NetworkVisualization.pad(
          ' ',
          DashboardManager.FRAME_INNER_WIDTH,
          ' '
        )}${colors.blueCore}║${colors.reset}`
      );
      this.#logFn(
        (() => {
          const bar = `Progress to exit: ${MazeVisualization.displayProgressBar(
            this.#currentBest.result.progress
          )}`;
          return `${colors.blueCore}║${NetworkVisualization.pad(
            ' ' + colors.neonSilver + bar + colors.reset,
            DashboardManager.FRAME_INNER_WIDTH,
            ' '
          )}${colors.blueCore}║${colors.reset}`;
        })()
      );
      this.#logFn(
        `${colors.blueCore}║${NetworkVisualization.pad(
          ' ',
          DashboardManager.FRAME_INNER_WIDTH,
          ' '
        )}${colors.blueCore}║${colors.reset}`
      );
    }

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
      if (Math.abs(diff) < 1e-9) return `${colors.neonSilver} (↔0)`;
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
            const take = Math.min(6, sortedOps.length);
            for (let i = 0; i < take; i++) {
              const opStats = sortedOps[i];
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
            const takeM = Math.min(8, sortedEntries.length);
            for (let i = 0; i < takeM; i++) {
              const [mutationName, mutationCount] = sortedEntries[i];
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
            const out: number[] = [];
            const takeS = Math.min(5, sortedSizes.length);
            for (let i = 0; i < takeS; i++) out.push(sortedSizes[i]);
            return out;
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
