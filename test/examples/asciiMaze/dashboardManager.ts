/**
 * Dashboard Manager - Handles the visualization dashboard
 *
 * This module contains the DashboardManager class, which manages the
 * state of the dynamic terminal dashboard that displays maze solving progress.
 * It tracks solved mazes, current best solutions, and handles all terminal output
 * for visualizing the agent's progress, network, and statistics.
 */

import { Network } from '../../../src/neataptic';
import { MazeUtils } from './mazeUtils';
import { MazeVisualization } from './mazeVisualization';
import { NetworkVisualization } from './networkVisualization';
import { colors } from './colors';
import { INetwork, IDashboardManager } from './interfaces'; // Added INetwork

/**
 * Stores successful maze solutions and current best networks,
 * and manages all dashboard output and visualization.
 */
export class DashboardManager implements IDashboardManager {
  // Array to store successful solutions for each unique maze
  private solvedMazes: Array<{
    maze: string[];
    result: any;
    network: INetwork; // Changed Network to INetwork
    generation: number;
  }> = [];

  // Tracks which mazes have already been solved to prevent duplicates
  private solvedMazeKeys: Set<string> = new Set<string>();

  // Current best solution for the active maze
  private currentBest: {
    result: any;
    network: INetwork; // Changed Network to INetwork
    generation: number;
  } | null = null;

  // Terminal clearing function
  private clearFunction: () => void;
  // Terminal logging function
  private logFunction: (...args: any[]) => void;
  // Track previous telemetry snapshot for delta display
  private _lastTelemetry: any = null;
  private _lastBestFitness: number | null = null;
  private _bestFitnessHistory: number[] = [];
  private _complexityNodesHistory: number[] = [];
  private _complexityConnsHistory: number[] = [];
  private _hypervolumeHistory: number[] = [];
  private _progressHistory: number[] = []; // best progress per generation for current maze
  private _speciesCountHistory: number[] = [];
  private static readonly FRAME_INNER_WIDTH = 148; // characters between the two vertical borders
  private static readonly LEFT_PADDING = 7; // spaces after left border inside stats lines
  private static readonly RIGHT_PADDING = 1; // space before right border
  private static readonly CONTENT_WIDTH =
    DashboardManager.FRAME_INNER_WIDTH -
    DashboardManager.LEFT_PADDING -
    DashboardManager.RIGHT_PADDING; // width given to padded content (excludes borders and paddings)
  private static readonly STAT_LABEL_WIDTH = 28; // default label width for main stats
  private static opennessLegend =
    'Openness: 1=best, (0,1)=longer improving, 0.001=only backtrack, 0=wall/dead/non-improving';

  // Generic stat formatter (also used for solved maze stats) ensuring consistent width math
  private formatStat(
    label: string,
    value: string,
    colorLabel = colors.neonSilver,
    colorValue = colors.cyanNeon,
    labelWidth = DashboardManager.STAT_LABEL_WIDTH
  ) {
    const lbl = label.endsWith(':') ? label : label + ':';
    const paddedLabel = lbl.padEnd(labelWidth, ' ');
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

  private buildSparkline(data: number[], width = 32): string {
    if (!data.length) return '';
    const blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    const slice = data.slice(-width);
    const min = Math.min(...slice);
    const max = Math.max(...slice);
    const range = max - min || 1;
    return slice
      .map((v) => {
        const idx = Math.floor(((v - min) / range) * (blocks.length - 1));
        return blocks[idx];
      })
      .join('');
  }

  /**
   * Constructs a DashboardManager.
   * @param clearFn - Function to clear the terminal.
   * @param logFn - Function to log output to the terminal.
   */
  constructor(clearFn: () => void, logFn: (...args: any[]) => void) {
    this.clearFunction = clearFn;
    this.logFunction = logFn;
  }

  /**
   * Creates a unique key for a maze to prevent duplicate solutions.
   * @param maze - The maze array to create a key for.
   * @returns A string that uniquely identifies this maze.
   */
  private getMazeKey(maze: string[]): string {
    // Create a simple hash of the maze by joining all rows
    // This gives us a unique identifier for each unique maze layout
    return maze.join('');
  }

  /**
   * Updates the dashboard with new results.
   * Tracks the current best solution and records solved mazes.
   * @param maze - The current maze being solved.
   * @param result - The result of the latest attempt.
   * @param network - The neural network that produced the result.
   * @param generation - Current generation number.
   */
  update(
    maze: string[],
    result: any,
    network: INetwork,
    generation: number,
    neatInstance?: any
  ): void {
    // Changed Network to INetwork
    // Save this as current best
    this.currentBest = {
      result,
      network,
      generation,
    };

    // If maze was solved and we haven't solved this specific maze before, add to solved mazes list
    if (result.success) {
      const mazeKey = this.getMazeKey(maze);

      if (!this.solvedMazeKeys.has(mazeKey)) {
        // This is a new solved maze, add it to our records
        this.solvedMazes.push({
          maze,
          result,
          network,
          generation,
        });
        this.solvedMazeKeys.add(mazeKey);
      }
    }

    // Redraw the dashboard
    // Grab latest telemetry (if provided) before redraw
    const telemetry = neatInstance?.getTelemetry?.();
    if (telemetry && telemetry.length) {
      this._lastTelemetry = telemetry[telemetry.length - 1];
      // Track best fitness
      const bestFit = this.currentBest?.result?.fitness;
      if (typeof bestFit === 'number') {
        this._lastBestFitness = bestFit;
        this._bestFitnessHistory.push(bestFit);
        if (this._bestFitnessHistory.length > 500)
          this._bestFitnessHistory.shift();
      }
      // Track complexity trends
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
      // Track hypervolume
      const h = this._lastTelemetry?.hyper;
      if (typeof h === 'number') {
        this._hypervolumeHistory.push(h);
        if (this._hypervolumeHistory.length > 500)
          this._hypervolumeHistory.shift();
      }
      // Track progress of current best
      const prog = this.currentBest?.result?.progress;
      if (typeof prog === 'number') {
        this._progressHistory.push(prog);
        if (this._progressHistory.length > 500) this._progressHistory.shift();
      }
      const sc = this._lastTelemetry?.species;
      if (typeof sc === 'number') {
        this._speciesCountHistory.push(sc);
        if (this._speciesCountHistory.length > 500)
          this._speciesCountHistory.shift();
      }
    }
    this.redraw(maze, neatInstance);
  }

  /**
   * Clears the terminal and redraws the dashboard with all content.
   * Displays current best solution, solved mazes, and statistics.
   * @param currentMaze - The maze currently being solved.
   */
  redraw(currentMaze: string[], neat?: any): void {
    // Clear the screen
    this.clearFunction();

    // Draw dashboard header
    this.logFunction(
      `${colors.blueCore}╔${NetworkVisualization.pad('═', 148, '═')}╗${
        colors.reset
      }`
    );
    this.logFunction(
      `${colors.blueCore}╚${NetworkVisualization.pad(
        '╦════════════╦',
        148,
        '═'
      )}╝${colors.reset}`
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
        148,
        '═'
      )}╗${colors.reset}`
    );

    // Print current best for active maze
    if (this.currentBest) {
      this.logFunction(
        `${colors.blueCore}╠${NetworkVisualization.pad(
          '══════════════════════',
          148,
          '═'
        )}╣${colors.reset}`
      );
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(
          `${colors.orangeNeon}EVOLVING (GEN ${this.currentBest.generation})`,
          148,
          ' '
        )}${colors.blueCore}║${colors.reset}`
      );
      this.logFunction(
        `${colors.blueCore}╠${NetworkVisualization.pad(
          '══════════════════════',
          148,
          '═'
        )}╣${colors.reset}`
      );
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
          colors.reset
        }`
      );

      // Visualize the network
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
          colors.reset
        }`
      );
      this.logFunction(
        NetworkVisualization.visualizeNetworkSummary(this.currentBest.network)
      );
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
          colors.reset
        }`
      );

      // Show maze visualization with agent's path
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
            `${colors.blueCore}║${NetworkVisualization.pad(line, 148, ' ')}${
              colors.blueCore
            }║`
        )
        .join('\n');
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
          colors.reset
        }`
      );
      this.logFunction(centeredCurrentMaze);
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
          colors.reset
        }`
      );

      // Print stats for the current best solution
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
          colors.reset
        }`
      );
      MazeVisualization.printMazeStats(
        this.currentBest,
        currentMaze,
        this.logFunction
      );
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
          colors.reset
        }`
      );

      // Show progress bar for agent's progress to exit
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
          colors.reset
        }`
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
        `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
          colors.reset
        }`
      );
    }

    this.logFunction(
      `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
        colors.reset
      }`
    );
    const last = this._lastTelemetry;
    const complexity = last?.complexity;
    const perf = last?.perf;
    const lineage = last?.lineage;
    const fronts = Array.isArray(last?.fronts) ? last.fronts : null;
    const objectives = last?.objectives;
    const hyper = last?.hyper;
    const diversity = last?.diversity;
    const mutationStats = last?.mutationStats || last?.mutation?.stats; // attempt to detect mutation operator stats if library exposes
    const bestFitness = this.currentBest?.result?.fitness;

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

    // Derive population stats if neat available
    let popMean = '-';
    let popMedian = '-';
    let speciesCount = '-';
    let enabledRatio = '-';
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

    const firstFrontSize = fronts?.[0]?.length || 0;
    const SPARK_WIDTH = 64; // doubled length for clearer trends
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
    // Additional maze-run metrics (present on current best result)
    const satFrac = (this.currentBest as any)?.result?.saturationFraction;
    if (typeof satFrac === 'number') {
      statsLines.push(
        this.formatStat('Saturation fraction', satFrac.toFixed(3))
      );
    }
    const actEnt = (this.currentBest as any)?.result?.actionEntropy;
    if (typeof actEnt === 'number') {
      statsLines.push(
        this.formatStat('Action entropy (path)', actEnt.toFixed(3))
      );
    }
    // Fallback: if still '-' but we have best fitness, show it so panel not empty early gens
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
    ) {
      statsLines.push(
        this.formatStat(
          'Simplify phase',
          'active',
          colors.neonSilver,
          colors.neonGreen
        )
      );
      // Strategy not directly available; placeholder could be extended by telemetry injection later
    }
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
    // If speciation array not exposed, fall back to telemetry snapshot
    if (speciesCount === '-' && typeof last?.species === 'number') {
      speciesCount = String(last.species);
    }
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
      // Show top 3 species sizes
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
      // Summarize activation counts if structure like {ADD_NODE: count, ...}
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

    statsLines.forEach((ln) => this.logFunction(ln));
    this.logFunction(
      `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
        colors.reset
      }`
    );
    if (this.solvedMazes.length > 0) {
      this.logFunction(
        `${colors.blueCore}╠${NetworkVisualization.pad(
          '╦══════════════╦',
          148,
          '═'
        )}╣${colors.reset}`
      );
      this.logFunction(
        `${colors.blueCore}╠${NetworkVisualization.pad(
          `╣ ${colors.orangeNeon}SOLVED MAZES${colors.blueCore} ╠`,
          148,
          '═'
        )}╣${colors.reset}`
      );
      this.logFunction(
        `${colors.blueCore}╠${NetworkVisualization.pad(
          '╩══════════════╩',
          148,
          '═'
        )}╣${colors.reset}`
      );
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
          colors.reset
        }`
      );
    }

    // Print all solved mazes with their statistics and network summaries
    if (this.solvedMazes.length > 0) {
      // Loop through solved mazes in reverse order (last solved first)
      for (let i = this.solvedMazes.length - 1; i >= 0; i--) {
        const solved = this.solvedMazes[i];
        const endPos = solved.result.path[solved.result.path.length - 1];
        // Calculate display number (last solved is #1)
        const displayNumber = this.solvedMazes.length - i;
        const solvedMazeVisualization = MazeVisualization.visualizeMaze(
          solved.maze,
          endPos,
          solved.result.path
        );
        const solvedMazeLines = Array.isArray(solvedMazeVisualization)
          ? solvedMazeVisualization
          : solvedMazeVisualization.split('\n');
        const centeredSolvedMaze = solvedMazeLines
          .map((line) =>
            NetworkVisualization.pad(
              line,
              DashboardManager.FRAME_INNER_WIDTH,
              ' '
            )
          )
          .join('\n');
        // Entry header
        this.logFunction(
          `${colors.blueCore}╠${NetworkVisualization.pad(
            '═'.repeat(DashboardManager.FRAME_INNER_WIDTH),
            DashboardManager.FRAME_INNER_WIDTH,
            '═'
          )}╣${colors.reset}`
        );
        this.logFunction(
          `${colors.blueCore}║${NetworkVisualization.pad(
            `${colors.orangeNeon} SOLVED #${displayNumber} (Gen ${solved.generation})${colors.reset}${colors.blueCore}`,
            DashboardManager.FRAME_INNER_WIDTH,
            ' '
          )}║${colors.reset}`
        );
        this.logFunction(
          `${colors.blueCore}╠${NetworkVisualization.pad(
            '─'.repeat(DashboardManager.FRAME_INNER_WIDTH),
            DashboardManager.FRAME_INNER_WIDTH,
            '─'
          )}╣${colors.reset}`
        );
        // Optional trend stats inside each solved block (reuse global sparklines)
        const solvedLabelWidth = 22; // narrower label width for solved maze stats & trends
        const solvedStat = (label: string, value: string) =>
          this.formatStat(
            label,
            value,
            colors.neonSilver,
            colors.cyanNeon,
            solvedLabelWidth
          );
        if (spark) this.logFunction(solvedStat('Fitness trend', spark));
        if (sparkComplexityNodes)
          this.logFunction(solvedStat('Nodes trend', sparkComplexityNodes));
        if (sparkComplexityConns)
          this.logFunction(solvedStat('Conns trend', sparkComplexityConns));
        if (sparkHyper)
          this.logFunction(solvedStat('Hypervol trend', sparkHyper));
        if (sparkProgress)
          this.logFunction(solvedStat('Progress trend', sparkProgress));
        if (sparkSpecies)
          this.logFunction(solvedStat('Species trend', sparkSpecies));
        // Blank spacer
        this.logFunction(
          `${colors.blueCore}║${NetworkVisualization.pad(
            ' ',
            DashboardManager.FRAME_INNER_WIDTH,
            ' '
          )}║${colors.reset}`
        );
        // Render maze lines ensuring blue color re-applied before right border
        centeredSolvedMaze
          .split('\n')
          .forEach((l) =>
            this.logFunction(
              `${colors.blueCore}║${NetworkVisualization.pad(
                l,
                DashboardManager.FRAME_INNER_WIDTH,
                ' '
              )}${colors.blueCore}║${colors.reset}`
            )
          );

        // Print efficiency and other stats for the solved maze
        const startPos = MazeUtils.findPosition(solved.maze, 'S');
        const exitPos = MazeUtils.findPosition(solved.maze, 'E');
        const optimalLength = MazeUtils.bfsDistance(
          MazeUtils.encodeMaze(solved.maze),
          startPos,
          exitPos
        );
        const pathLength = solved.result.path.length - 1;
        // Efficiency is the percentage of optimal length to actual (lower = more roundabout path)
        const efficiency = Math.min(
          100,
          Math.round((optimalLength / pathLength) * 100)
        ).toFixed(1);
        // Overhead is how much longer than optimal the path is (100% means twice as long as optimal)
        const overhead = ((pathLength / optimalLength) * 100 - 100).toFixed(1);

        // Calculate unique cells visited vs revisited cells
        const uniqueCells = new Set<string>();
        let revisitedCells = 0;
        for (const [x, y] of solved.result.path) {
          const cellKey = `${x},${y}`;
          if (uniqueCells.has(cellKey)) {
            revisitedCells++;
          } else {
            uniqueCells.add(cellKey);
          }
        }

        // Reuse solvedStat helper (already defined earlier in this loop)
        this.logFunction(
          solvedStat(
            'Path efficiency',
            `${optimalLength}/${pathLength} (${efficiency}%)`
          )
        );
        this.logFunction(
          solvedStat('Path overhead', `${overhead}% longer than optimal`)
        );
        this.logFunction(
          solvedStat('Unique cells visited', `${uniqueCells.size}`)
        );
        this.logFunction(
          solvedStat('Cells revisited', `${revisitedCells} times`)
        );
        this.logFunction(solvedStat('Steps', `${solved.result.steps}`));
        this.logFunction(
          solvedStat('Fitness', `${solved.result.fitness.toFixed(2)}`)
        );
        if (i === 0) {
          // Final bottom border once after last (oldest) solved maze
          this.logFunction(
            `${colors.blueCore}╚${NetworkVisualization.pad(
              '═'.repeat(148),
              148,
              '═'
            )}╝${colors.reset}`
          );
        }
        // Optionally, print detailed network structure for debugging
        // if (this.currentBest?.network) printNetworkStructure(this.currentBest.network);
      }
    }
  }

  /**
   * Clears all saved state, including solved mazes and current best.
   */
  reset(): void {
    this.solvedMazes = [];
    this.solvedMazeKeys.clear();
    this.currentBest = null;
  }
}
