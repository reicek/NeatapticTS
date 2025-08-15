/**
 * Dashboard Manager - Handles the visualization dashboard
 *
 * This module contains the DashboardManager class, which manages the
 * state of the dynamic terminal dashboard that displays maze solving progress
 * and optionally appends solved mazes to an archive area. The implementation
 * is designed to be readable for educational purposes: it gathers telemetry
 * from the running NEAT instance, keeps small historical series for sparklines,
 * and renders both a live view (cleared and redrawn each frame) and an
 * archive view (appended once per solved maze).
 */
import { Network } from '../../../src/neataptic';
import { MazeUtils } from './mazeUtils';
import { MazeVisualization } from './mazeVisualization';
import { NetworkVisualization } from './networkVisualization';
import { colors } from './colors';
import { INetwork, IDashboardManager } from './interfaces';

/**
 * DashboardManager: manages solved mazes, current best, and terminal output.
 * Supports an optional archive logger to which solved-maze blocks are appended
 * while the live logger is used for the active maze redraws.
 */
/**
 * DashboardManager
 *
 * Responsibilities:
 * - Keep track of solved mazes and avoid duplicates.
 * - Maintain the current best solution (for live rendering).
 * - Collect small telemetry histories used to render sparklines.
 * - Render a compact terminal-style dashboard to `logFunction` and
 *   append formatted solved-maze blocks to `archiveLogFunction` when present.
 *
 * Constructor parameters are function references so the dashboard can remain
 * agnostic about where output actually goes (Node console, browser DOM, etc.).
 */
export class DashboardManager implements IDashboardManager {
  // List of solved maze records (keeps full maze + solution for archival display)
  private solvedMazes: Array<{
    maze: string[];
    result: any;
    network: INetwork;
    generation: number;
  }> = [];

  // Set of maze keys we've already archived to avoid duplicate entries
  private solvedMazeKeys: Set<string> = new Set<string>();

  // Currently evolving/best candidate for the active maze (live view)
  private currentBest: {
    result: any;
    network: INetwork;
    generation: number;
  } | null = null;

  // Functions supplied by the embedding environment. Keep dashboard I/O pluggable.
  private clearFunction: () => void;
  private logFunction: (...args: any[]) => void;
  private archiveLogFunction?: (...args: any[]) => void;

  // Telemetry and small history windows used for rendering trends/sparklines
  private _lastTelemetry: any = null;
  private _lastBestFitness: number | null = null;
  private _bestFitnessHistory: number[] = [];
  private _complexityNodesHistory: number[] = [];
  private _complexityConnsHistory: number[] = [];
  private _hypervolumeHistory: number[] = [];
  private _progressHistory: number[] = [];
  private _speciesCountHistory: number[] = [];

  // Layout constants for the ASCII-art framed display
  private static readonly FRAME_INNER_WIDTH = 148;
  private static readonly LEFT_PADDING = 7;
  private static readonly RIGHT_PADDING = 1;
  private static readonly CONTENT_WIDTH =
    DashboardManager.FRAME_INNER_WIDTH -
    DashboardManager.LEFT_PADDING -
    DashboardManager.RIGHT_PADDING;
  private static readonly STAT_LABEL_WIDTH = 28;
  private static opennessLegend =
    'Openness: 1=best, (0,1)=longer improving, 0.001=only backtrack, 0=wall/dead/non-improving';

  /**
   * Construct a new DashboardManager
   *
   * @param clearFn - function that clears the "live" output area (no-op for archive)
   * @param logFn - function that accepts strings to render the live dashboard
   * @param archiveLogFn - optional function to which solved-maze archive blocks are appended
   */
  constructor(
    clearFn: () => void,
    logFn: (...args: any[]) => void,
    archiveLogFn?: (...args: any[]) => void
  ) {
    this.clearFunction = clearFn;
    this.logFunction = logFn;
    this.archiveLogFunction = archiveLogFn;
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
    labelWidth = DashboardManager.STAT_LABEL_WIDTH
  ) {
    // Ensure label ends with ':' and pad to labelWidth for column alignment
    const lbl = label.endsWith(':') ? label : label + ':';
    const paddedLabel = lbl.padEnd(labelWidth, ' ');

    // Compose colored label + value, then pad/truncate to content width
    const composed = `${colorLabel}${paddedLabel}${colorValue} ${value}${colors.reset}`;
    return `${colors.blueCore}║${' '.repeat(
      DashboardManager.LEFT_PADDING
    )}${NetworkVisualization.pad(
      composed,
      DashboardManager.CONTENT_WIDTH,
      ' ',
      'left'
    )}${' '.repeat(DashboardManager.RIGHT_PADDING)}${colors.blueCore}║${
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
  private buildSparkline(data: number[], width = 32): string {
    if (!data || !data.length) return '';
    const blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    const slice = data.slice(-width);
    const min = Math.min(...slice);
    const max = Math.max(...slice);
    // Avoid division by zero
    const range = max - min || 1;
    return slice
      .map((v) => {
        // Map value into block index
        const idx = Math.floor(((v - min) / range) * (blocks.length - 1));
        return blocks[idx];
      })
      .join('');
  }

  /**
   * getMazeKey
   *
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
    // If the embedder did not supply an archive logger, there's nothing to do
    if (!this.archiveLogFunction) return;

    // Render solved maze visualization using the MazeVisualization helper
    const endPos = solved.result.path[solved.result.path.length - 1];
    const solvedMazeVisualization = MazeVisualization.visualizeMaze(
      solved.maze,
      endPos,
      solved.result.path
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
    const solvedLabelWidth = 22;
    const solvedStat = (label: string, value: string) =>
      this.formatStat(
        label,
        value,
        colors.neonSilver,
        colors.cyanNeon,
        solvedLabelWidth
      );

    const spark = this.buildSparkline(this._bestFitnessHistory, 64);
    const sparkComplexityNodes = this.buildSparkline(
      this._complexityNodesHistory,
      64
    );
    const sparkComplexityConns = this.buildSparkline(
      this._complexityConnsHistory,
      64
    );
    const sparkHyper = this.buildSparkline(this._hypervolumeHistory, 64);
    const sparkProgress = this.buildSparkline(this._progressHistory, 64);
    const sparkSpecies = this.buildSparkline(this._speciesCountHistory, 64);

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
      (this.archiveLogFunction as any)(blockLines.join('\n'), {
        prepend: true,
      });
    } catch {
      // Fallback: if the archive logger doesn't accept options, just append each line
      const append = this.archiveLogFunction ?? (() => {});
      blockLines.forEach((ln) => append(ln));
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
  update(
    maze: string[],
    result: any,
    network: INetwork,
    generation: number,
    neatInstance?: any
  ): void {
    // Update live candidate
    this.currentBest = { result, network, generation };

    // If this run solved the maze and it's a new maze, add & archive it
    if (result.success) {
      const mazeKey = this.getMazeKey(maze);
      if (!this.solvedMazeKeys.has(mazeKey)) {
        this.solvedMazes.push({ maze, result, network, generation });
        this.solvedMazeKeys.add(mazeKey);
        // Append to archive immediately when first solved
        const displayNumber = this.solvedMazes.length; // 1-based
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
      this._lastTelemetry = telemetry[telemetry.length - 1];

      // Record best fitness into a small history window for trend views
      const bestFit = this.currentBest?.result?.fitness;
      if (typeof bestFit === 'number') {
        this._lastBestFitness = bestFit;
        this._bestFitnessHistory.push(bestFit);
        if (this._bestFitnessHistory.length > 500)
          this._bestFitnessHistory.shift();
      }

      // Complexity telemetry: mean nodes/connectivity across population
      const c = this._lastTelemetry?.complexity;
      if (c) {
        if (typeof c.meanNodes === 'number') {
          this._complexityNodesHistory.push(c.meanNodes);
          if (this._complexityNodesHistory.length > 500)
            this._complexityNodesHistory.shift();
        }
        if (typeof c.meanConns === 'number') {
          this._complexityConnsHistory.push(c.meanConns);
          if (this._complexityConnsHistory.length > 500)
            this._complexityConnsHistory.shift();
        }
      }

      // Hypervolume is used for multi-objective tracking
      const h = this._lastTelemetry?.hyper;
      if (typeof h === 'number') {
        this._hypervolumeHistory.push(h);
        if (this._hypervolumeHistory.length > 500)
          this._hypervolumeHistory.shift();
      }

      // Progress: how close a candidate is to the exit
      const prog = this.currentBest?.result?.progress;
      if (typeof prog === 'number') {
        this._progressHistory.push(prog);
        if (this._progressHistory.length > 500) this._progressHistory.shift();
      }

      // Species count history
      const sc = this._lastTelemetry?.species;
      if (typeof sc === 'number') {
        this._speciesCountHistory.push(sc);
        if (this._speciesCountHistory.length > 500)
          this._speciesCountHistory.shift();
      }
    }

    // Render the live dashboard
    this.redraw(maze, neatInstance);
  }

  /**
   * redraw
   *
   * Responsible for clearing the live area and printing a compact snapshot of
   * the current best candidate, a short network summary, the maze drawing and
   * several telemetry-derived stats. The function uses `logFunction` for all
   * output lines so the same renderer can be used both in Node and in the
   * browser (DOM adapter).
   */
  redraw(currentMaze: string[], neat?: any): void {
    // Clear the live area (archive is untouched)
    this.clearFunction();

    // Header: top frame lines
    this.logFunction(
      `${colors.blueCore}╔${NetworkVisualization.pad(
        '═',
        DashboardManager.FRAME_INNER_WIDTH,
        '═'
      )}${colors.blueCore}╗${colors.reset}`
    );
    this.logFunction(
      `${colors.blueCore}╚${NetworkVisualization.pad(
        '╦════════════╦',
        DashboardManager.FRAME_INNER_WIDTH,
        '═'
      )}${colors.blueCore}╝${colors.reset}`
    );
    this.logFunction(
      `${colors.blueCore}${NetworkVisualization.pad(
        `║ ${colors.neonYellow}ASCII maze${colors.blueCore} ║`,
        150,
        ' '
      )}${colors.reset}`
    );
    this.logFunction(
      `${colors.blueCore}╔${NetworkVisualization.pad(
        '╩════════════╩',
        DashboardManager.FRAME_INNER_WIDTH,
        '═'
      )}${colors.blueCore}╗${colors.reset}`
    );

    // Print current best for active maze if available
    if (this.currentBest) {
      this.logFunction(
        `${colors.blueCore}╠${NetworkVisualization.pad(
          '══════════════════════',
          DashboardManager.FRAME_INNER_WIDTH,
          '═'
        )}${colors.blueCore}╣${colors.reset}`
      );
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(
          `${colors.orangeNeon}EVOLVING (GEN ${this.currentBest.generation})`,
          DashboardManager.FRAME_INNER_WIDTH,
          ' '
        )}${colors.blueCore}║${colors.reset}`
      );
      this.logFunction(
        `${colors.blueCore}╠${NetworkVisualization.pad(
          '══════════════════════',
          DashboardManager.FRAME_INNER_WIDTH,
          '═'
        )}${colors.blueCore}╣${colors.reset}`
      );
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(
          ' ',
          DashboardManager.FRAME_INNER_WIDTH,
          ' '
        )}${colors.blueCore}║${colors.reset}`
      );

      // Network summary (compact visualization)
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(
          ' ',
          DashboardManager.FRAME_INNER_WIDTH,
          ' '
        )}${colors.blueCore}║${colors.reset}`
      );
      this.logFunction(
        NetworkVisualization.visualizeNetworkSummary(this.currentBest.network)
      );
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(
          ' ',
          DashboardManager.FRAME_INNER_WIDTH,
          ' '
        )}${colors.blueCore}║${colors.reset}`
      );

      // Maze visualization for the live candidate
      const lastPos = this.currentBest.result.path[
        this.currentBest.result.path.length - 1
      ];
      const currentMazeVisualization = MazeVisualization.visualizeMaze(
        currentMaze,
        lastPos,
        this.currentBest.result.path
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
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(
          ' ',
          DashboardManager.FRAME_INNER_WIDTH,
          ' '
        )}${colors.blueCore}║${colors.reset}`
      );
      this.logFunction(centeredCurrentMaze);
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(
          ' ',
          DashboardManager.FRAME_INNER_WIDTH,
          ' '
        )}${colors.blueCore}║${colors.reset}`
      );

      // Print stats for the current best solution (delegates to MazeVisualization)
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(
          ' ',
          DashboardManager.FRAME_INNER_WIDTH,
          ' '
        )}${colors.blueCore}║${colors.reset}`
      );
      MazeVisualization.printMazeStats(
        this.currentBest,
        currentMaze,
        this.logFunction
      );
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(
          ' ',
          DashboardManager.FRAME_INNER_WIDTH,
          ' '
        )}${colors.blueCore}║${colors.reset}`
      );

      // Progress bar for current candidate
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(
          ' ',
          DashboardManager.FRAME_INNER_WIDTH,
          ' '
        )}${colors.blueCore}║${colors.reset}`
      );
      this.logFunction(
        (() => {
          const bar = `Progress to exit: ${MazeVisualization.displayProgressBar(
            this.currentBest.result.progress
          )}`;
          return `${colors.blueCore}║${NetworkVisualization.pad(
            ' ' + colors.neonSilver + bar + colors.reset,
            DashboardManager.FRAME_INNER_WIDTH,
            ' '
          )}${colors.blueCore}║${colors.reset}`;
        })()
      );
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(
          ' ',
          DashboardManager.FRAME_INNER_WIDTH,
          ' '
        )}${colors.blueCore}║${colors.reset}`
      );
    }

    // General stats area (telemetry-derived values). These are defensive reads
    // because telemetry may be missing early in the run.
    const last = this._lastTelemetry;
    const complexity = last?.complexity;
    const perf = last?.perf;
    const lineage = last?.lineage;
    const fronts = Array.isArray(last?.fronts) ? last.fronts : null;
    const objectives = last?.objectives;
    const hyper = last?.hyper;
    const diversity = last?.diversity;
    const mutationStats = last?.mutationStats || last?.mutation?.stats;
    const bestFitness = this.currentBest?.result?.fitness;

    // Small helpers used below when building the stats list
    const fmtNum = (v: any, digits = 2) =>
      typeof v === 'number' && isFinite(v) ? v.toFixed(digits) : '-';
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
          g.connections.forEach((c: any) => {
            total++;
            if (c.enabled !== false) enabled++;
          });
        }
      });
      if (scores.length) {
        const sum = scores.reduce((a, b) => a + b, 0);
        popMean = (sum / scores.length).toFixed(2);
        const sorted = scores.slice().sort((a, b) => a - b);
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
    const SPARK_WIDTH = 64;
    const spark = this.buildSparkline(this._bestFitnessHistory, SPARK_WIDTH);
    const sparkComplexityNodes = this.buildSparkline(
      this._complexityNodesHistory,
      SPARK_WIDTH
    );
    const sparkComplexityConns = this.buildSparkline(
      this._complexityConnsHistory,
      SPARK_WIDTH
    );
    const sparkHyper = this.buildSparkline(
      this._hypervolumeHistory,
      SPARK_WIDTH
    );
    const sparkProgress = this.buildSparkline(
      this._progressHistory,
      SPARK_WIDTH
    );
    const sparkSpecies = this.buildSparkline(
      this._speciesCountHistory,
      SPARK_WIDTH
    );

    // Collect stat lines into an array then print via the logFunction so the
    // ordering is explicit and easy to modify for learners exploring the code.
    const statsLines: string[] = [];
    statsLines.push(
      this.formatStat(
        'Current generation',
        `${this.currentBest?.generation || 0}`
      )
    );
    if (typeof bestFitness === 'number')
      statsLines.push(
        this.formatStat(
          'Best fitness',
          `${bestFitness.toFixed(2)}${deltaArrow(
            bestFitness,
            this._bestFitnessHistory.length > 1
              ? this._bestFitnessHistory[this._bestFitnessHistory.length - 2]
              : null
          )}`
        )
      );
    const satFrac = (this.currentBest as any)?.result?.saturationFraction;
    if (typeof satFrac === 'number')
      statsLines.push(
        this.formatStat('Saturation fraction', satFrac.toFixed(3))
      );
    const actEnt = (this.currentBest as any)?.result?.actionEntropy;
    if (typeof actEnt === 'number')
      statsLines.push(
        this.formatStat('Action entropy (path)', actEnt.toFixed(3))
      );
    if (popMean === '-' && typeof bestFitness === 'number')
      popMean = bestFitness.toFixed(2);
    if (popMedian === '-' && typeof bestFitness === 'number')
      popMedian = bestFitness.toFixed(2);
    statsLines.push(this.formatStat('Population mean', popMean));
    statsLines.push(this.formatStat('Population median', popMedian));
    if (complexity)
      statsLines.push(
        this.formatStat(
          'Complexity mean n/c',
          `${fmtNum(complexity.meanNodes, 2)}/${fmtNum(
            complexity.meanConns,
            2
          )}  max ${fmtNum(complexity.maxNodes, 0)}/${fmtNum(
            complexity.maxConns,
            0
          )}`,
          colors.neonSilver,
          colors.orangeNeon
        )
      );
    if (
      complexity &&
      (complexity.growthNodes < 0 || complexity.growthConns < 0)
    )
      statsLines.push(
        this.formatStat(
          'Simplify phase',
          'active',
          colors.neonSilver,
          colors.neonGreen
        )
      );
    if (sparkComplexityNodes)
      statsLines.push(
        this.formatStat(
          'Nodes trend',
          sparkComplexityNodes,
          colors.neonSilver,
          colors.neonYellow
        )
      );
    if (sparkComplexityConns)
      statsLines.push(
        this.formatStat(
          'Conns trend',
          sparkComplexityConns,
          colors.neonSilver,
          colors.neonYellow
        )
      );
    statsLines.push(this.formatStat('Enabled conn ratio', enabledRatio));
    if (perf && (perf.evalMs != null || perf.evolveMs != null))
      statsLines.push(
        this.formatStat(
          'Perf eval/evolve ms',
          `${fmtNum(perf.evalMs, 1)}/${fmtNum(perf.evolveMs, 1)}`
        )
      );
    if (lineage)
      statsLines.push(
        this.formatStat(
          'Lineage depth b/mean',
          `${lineage.depthBest}/${fmtNum(lineage.meanDepth, 2)}`
        )
      );
    if (lineage?.inbreeding != null)
      statsLines.push(
        this.formatStat('Inbreeding', fmtNum(lineage.inbreeding, 3))
      );
    if (speciesCount === '-' && typeof last?.species === 'number')
      speciesCount = String(last.species);
    statsLines.push(this.formatStat('Species count', speciesCount));
    if (diversity?.structuralVar != null)
      statsLines.push(
        this.formatStat(
          'Structural variance',
          fmtNum(diversity.structuralVar, 3)
        )
      );
    if (diversity?.objectiveSpread != null)
      statsLines.push(
        this.formatStat(
          'Objective spread',
          fmtNum(diversity.objectiveSpread, 3)
        )
      );
    if (Array.isArray(neat?.species) && neat.species.length) {
      const sizes = neat.species
        .map((s: any) => s.members?.length || 0)
        .sort((a: number, b: number) => b - a);
      const top3 = sizes.slice(0, 3).join('/') || '-';
      statsLines.push(this.formatStat('Top species sizes', top3));
    }
    if (fronts)
      statsLines.push(
        this.formatStat(
          'Pareto fronts',
          `${fronts.map((f: any) => f?.length || 0).join('/')}`
        )
      );
    statsLines.push(
      this.formatStat('First front size', firstFrontSize.toString())
    );
    if (objectives)
      statsLines.push(
        this.formatStat(
          'Objectives',
          objectives.join(', '),
          colors.neonSilver,
          colors.neonIndigo
        )
      );
    if (hyper !== undefined)
      statsLines.push(this.formatStat('Hypervolume', fmtNum(hyper, 4)));
    if (sparkHyper)
      statsLines.push(
        this.formatStat(
          'Hypervolume trend',
          sparkHyper,
          colors.neonSilver,
          colors.neonGreen
        )
      );
    if (spark)
      statsLines.push(
        this.formatStat(
          'Fitness trend',
          spark,
          colors.neonSilver,
          colors.neonYellow
        )
      );
    if (sparkProgress)
      statsLines.push(
        this.formatStat(
          'Progress trend',
          sparkProgress,
          colors.neonSilver,
          colors.cyanNeon
        )
      );
    if (sparkSpecies)
      statsLines.push(
        this.formatStat(
          'Species trend',
          sparkSpecies,
          colors.neonSilver,
          colors.neonIndigo
        )
      );
    if (neat?.getNoveltyArchiveSize) {
      try {
        const nov = neat.getNoveltyArchiveSize();
        statsLines.push(this.formatStat('Novelty archive', `${nov}`));
      } catch {}
    }
    if (neat?.getOperatorStats) {
      try {
        const ops = neat.getOperatorStats();
        if (Array.isArray(ops) && ops.length) {
          const top = ops
            .slice()
            .sort(
              (a: any, b: any) =>
                b.success / Math.max(1, b.attempts) -
                a.success / Math.max(1, a.attempts)
            )
            .slice(0, 4)
            .map(
              (o: any) =>
                `${o.name}:${(
                  (100 * o.success) /
                  Math.max(1, o.attempts)
                ).toFixed(0)}%`
            )
            .join(' ');
          if (top)
            statsLines.push(
              this.formatStat(
                'Op acceptance',
                top,
                colors.neonSilver,
                colors.neonGreen
              )
            );
        }
      } catch {}
    }
    if (mutationStats && typeof mutationStats === 'object') {
      const entries = Object.entries(mutationStats)
        .filter(([k, v]) => typeof v === 'number')
        .sort((a, b) => (b[1] as number) - (a[1] as number))
        .slice(0, 5)
        .map(([k, v]) => `${k}:${(v as number).toFixed(0)}`)
        .join(' ');
      if (entries)
        statsLines.push(
          this.formatStat(
            'Top mutations',
            entries,
            colors.neonSilver,
            colors.neonGreen
          )
        );
    }

    // Emit collected stat lines using the supplied log function
    statsLines.forEach((ln) => this.logFunction(ln));
    this.logFunction(
      `${colors.blueCore}║${NetworkVisualization.pad(
        ' ',
        DashboardManager.FRAME_INNER_WIDTH,
        ' '
      )}${colors.blueCore}║${colors.reset}`
    );
  }

  reset(): void {
    this.solvedMazes = [];
    this.solvedMazeKeys.clear();
    this.currentBest = null;
  }
}
