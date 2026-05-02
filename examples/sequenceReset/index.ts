import { Architect } from '../../src/browser-entry.ts';

/** Input size for the LSTM used in this example. */
const SEQUENCE_RESET_LSTM_INPUT_SIZE = 1;

/** Number of LSTM hidden units. */
const SEQUENCE_RESET_LSTM_HIDDEN_SIZE = 4;

/** Output size for the LSTM used in this example. */
const SEQUENCE_RESET_LSTM_OUTPUT_SIZE = 1;

/**
 * Fixed weight applied to every connection after construction.
 *
 * Pinning weights makes the walkthrough fully reproducible and separates
 * the sequence-state story from random initialization effects.
 */
const SEQUENCE_RESET_UNIFORM_WEIGHT = 0.3;

/**
 * Fixed bias applied to every non-input node after construction.
 *
 * A small non-zero bias prevents degenerate all-zero gate outputs so the
 * LSTM actually accumulates state across time steps.
 */
const SEQUENCE_RESET_NON_INPUT_BIAS = 0.05;

/**
 * Five-step input sequence fed to the network in each run.
 *
 * The variation (0.1 → 0.7 → 0.2) is intentional: differing inputs across
 * steps force the LSTM gates to do real work, so state accumulation is
 * clearly visible in the output trajectory.
 */
const SEQUENCE_RESET_INPUT_SEQUENCE = [0.1, 0.3, 0.7, 0.5, 0.2] as const;

/** Decimal places used when rounding output values in the result. */
const SEQUENCE_RESET_OUTPUT_DECIMAL_PLACES = 5;

/** Architecture metadata surfaced by the sequence reset example. */
export interface SequenceResetArchitectureSummary {
  /** Number of input nodes. */
  readonly inputCount: number;
  /** Number of LSTM hidden units (not total hidden nodes). */
  readonly lstmHiddenSize: number;
  /** Number of output nodes. */
  readonly outputCount: number;
  /** Topology intent string from the public network API. */
  readonly topologyType: string;
}

/**
 * Public summary returned by the sequence reset starter example.
 *
 * The three output arrays allow side-by-side comparison of:
 * - state accumulation (baseline fresh run),
 * - determinism after `clear()` (should equal baseline),
 * - carryover when `clear()` is skipped (should differ from baseline).
 */
export interface SequenceResetExampleResult {
  /** Architecture shape used in this walkthrough. */
  readonly architecture: SequenceResetArchitectureSummary;
  /** Five-step input sequence fed in each run. */
  readonly inputSequence: number[];
  /**
   * Outputs from the first run starting from a fresh (zero) state.
   *
   * Because the LSTM carries state across time steps, each output depends
   * on all previous inputs — not just the current one.
   */
  readonly stateAccumulationOutputs: number[];
  /**
   * Outputs from a second run after `network.clear()` resets all state.
   *
   * Must exactly equal `stateAccumulationOutputs`: same inputs from the same
   * zero state must produce the same outputs.
   */
  readonly afterResetOutputs: number[];
  /**
   * Outputs from a third run where `clear()` was NOT called first.
   *
   * State accumulated by the second run carries into this run, changing the
   * initial conditions and producing different outputs from the baseline.
   */
  readonly carryoverOutputs: number[];
}

/**
 * Runs the sequence reset starter example.
 *
 * The example teaches three concrete behaviors of a recurrent LSTM network:
 *
 * 1. **State accumulation** — each step's output depends on the full input
 *    history because the LSTM memory cells carry state forward in time.
 *
 * 2. **Reset determinism** — `network.clear()` zeroes all node states so the
 *    next sequence replay from the same starting point produces byte-identical
 *    outputs.
 *
 * 3. **Carryover effect** — skipping `clear()` between sequences changes the
 *    initial conditions for the next run, producing different outputs even for
 *    identical inputs.
 *
 * All weights and biases are pinned to fixed values after construction so the
 * walkthrough is fully reproducible across runs and environments.
 *
 * @returns Structured summary of three sequence runs showing the reset semantics.
 *
 * @example
 * ```ts
 * import { runSequenceResetExample } from './index';
 *
 * const result = runSequenceResetExample();
 * const identical = result.afterResetOutputs.every(
 *   (value, index) => value === result.stateAccumulationOutputs[index],
 * );
 * console.log('clear() restores fresh state:', identical); // true
 * ```
 */
export function runSequenceResetExample(): SequenceResetExampleResult {
  const network = buildFixedWeightLstm();
  const inputSequence = [...SEQUENCE_RESET_INPUT_SEQUENCE];

  // Step 1: Run from a fresh zero state — establishes the baseline trajectory.
  const stateAccumulationOutputs = runSequencePass(network, inputSequence);

  // Step 2: Reset all accumulated state, then replay the same sequence.
  network.clear();
  const afterResetOutputs = runSequencePass(network, inputSequence);

  // Step 3: Run the sequence once more WITHOUT clearing.
  //         State from step 2 carries in, changing the initial conditions.
  const carryoverOutputs = runSequencePass(network, inputSequence);

  return {
    architecture: {
      inputCount: network.inputNodeIds.length,
      lstmHiddenSize: SEQUENCE_RESET_LSTM_HIDDEN_SIZE,
      outputCount: network.outputNodeIds.length,
      topologyType: network.getTopologyIntent(),
    },
    inputSequence,
    stateAccumulationOutputs,
    afterResetOutputs,
    carryoverOutputs,
  };

  /**
   * Builds an LSTM network with uniform fixed weights for a deterministic demo.
   *
   * Pinning every connection weight and non-input bias to the same small value
   * ensures the memory cells accumulate non-trivial state across time steps
   * while keeping the walkthrough fully reproducible.
   *
   * @returns Freshly constructed LSTM with pinned parameters.
   */
  function buildFixedWeightLstm() {
    const lstmNetwork = Architect.lstm(
      SEQUENCE_RESET_LSTM_INPUT_SIZE,
      SEQUENCE_RESET_LSTM_HIDDEN_SIZE,
      SEQUENCE_RESET_LSTM_OUTPUT_SIZE,
    );

    // Pin connection weights for deterministic output trajectories.
    for (const connection of lstmNetwork.connections) {
      connection.weight = SEQUENCE_RESET_UNIFORM_WEIGHT;
    }
    for (const connection of lstmNetwork.selfconns) {
      connection.weight = SEQUENCE_RESET_UNIFORM_WEIGHT;
    }

    // Pin biases on all non-input nodes.
    for (const node of lstmNetwork.nodes) {
      if (node.type !== 'input') {
        node.bias = SEQUENCE_RESET_NON_INPUT_BIAS;
      }
    }

    return lstmNetwork;
  }

  /**
   * Activates the network once per step and collects rounded outputs.
   *
   * @param targetNetwork - Recurrent network to activate.
   * @param sequence - Input values, one per time step.
   * @returns Per-step output values rounded to a stable decimal precision.
   */
  function runSequencePass(
    targetNetwork: ReturnType<typeof buildFixedWeightLstm>,
    sequence: number[],
  ): number[] {
    return sequence.map((stepInput) =>
      roundToStablePrecision(targetNetwork.activate([stepInput])[0]),
    );
  }

  /**
   * Rounds an output value to the configured decimal precision.
   *
   * @param rawValue - Raw activation output.
   * @returns Value rounded to `SEQUENCE_RESET_OUTPUT_DECIMAL_PLACES` places.
   */
  function roundToStablePrecision(rawValue: number): number {
    return parseFloat(rawValue.toFixed(SEQUENCE_RESET_OUTPUT_DECIMAL_PLACES));
  }
}

/**
 * Formats the sequence reset result into a human-readable multi-line summary.
 *
 * The table shows all three runs side-by-side so the reset and carryover
 * effects are directly visible without any extra computation.
 *
 * @param result - Structured result from `runSequenceResetExample`.
 * @returns Multi-line string summary suitable for console output or display.
 *
 * @example
 * ```ts
 * import { formatSequenceResetExampleResult, runSequenceResetExample } from './index';
 *
 * console.log(formatSequenceResetExampleResult(runSequenceResetExample()));
 * ```
 */
export function formatSequenceResetExampleResult(
  result: SequenceResetExampleResult,
): string {
  const headerLine = `Sequence Reset (LSTM ${result.architecture.inputCount} → ${result.architecture.lstmHiddenSize} → ${result.architecture.outputCount})`;
  const columnHeader = `${'Step'.padStart(4)}  ${'Input'.padStart(7)}  ${'Run 1'.padStart(9)}  ${'Run 2'.padStart(9)}  ${'Run 3'.padStart(9)}`;
  const separator = '─'.repeat(columnHeader.length);

  const tableRows = result.inputSequence
    .map((inputValue, stepIndex) =>
      formatTableRow(result, inputValue, stepIndex),
    )
    .join('\n');

  const resetMatches = result.afterResetOutputs.every(
    (outputValue, stepIndex) =>
      outputValue === result.stateAccumulationOutputs[stepIndex],
  );
  const carryoverDiffers = result.carryoverOutputs.some(
    (outputValue, stepIndex) =>
      outputValue !== result.stateAccumulationOutputs[stepIndex],
  );

  const legendLines = [
    'Run 1: fresh state — baseline sequence pass',
    'Run 2: clear() before run — replays identically to Run 1',
    'Run 3: no clear() before run — accumulated state changes outputs',
    '',
    `clear() restores fresh-start behavior: ${String(resetMatches)}`,
    `Without clear(), carryover changes outputs: ${String(carryoverDiffers)}`,
  ].join('\n');

  return [
    headerLine,
    '',
    columnHeader,
    separator,
    tableRows,
    '',
    legendLines,
  ].join('\n');

  /**
   * Formats one row in the three-column output table.
   *
   * @param exampleResult - Full result for column lookups.
   * @param inputValue - Input value for this step.
   * @param stepIndex - Zero-based step index.
   * @returns Formatted row string.
   */
  function formatTableRow(
    exampleResult: SequenceResetExampleResult,
    inputValue: number,
    stepIndex: number,
  ): string {
    const stepLabel = String(stepIndex + 1).padStart(4);
    const inputLabel = inputValue.toFixed(3).padStart(7);
    const run1Label = exampleResult.stateAccumulationOutputs[stepIndex]
      .toFixed(5)
      .padStart(9);
    const run2Label = exampleResult.afterResetOutputs[stepIndex]
      .toFixed(5)
      .padStart(9);
    const run3Label = exampleResult.carryoverOutputs[stepIndex]
      .toFixed(5)
      .padStart(9);
    return `${stepLabel}  ${inputLabel}  ${run1Label}  ${run2Label}  ${run3Label}`;
  }
}
