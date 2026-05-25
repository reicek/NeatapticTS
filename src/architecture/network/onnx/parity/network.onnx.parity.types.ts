import type Network from '../../network';
import type { OnnxExportOptions } from '../network.onnx';

/**
 * Named Phase 9 runtime-parity lanes for the current ONNX execution boundary.
 *
 * Phase 9A freezes this list before golden parity cases widen coverage. Each
 * lane names one exporter-owned subset whose binary `.onnx` artifact can later
 * be compared against `Network.activate()` with an explicit tolerance packet.
 *
 * @example
 * ```ts
 * const baselineLane: OnnxRuntimeParityLane = 'baseline-float32-dense';
 * ```
 */
export type OnnxRuntimeParityLane =
  | 'baseline-float32-dense'
  | 'storage-fp16-dense'
  | 'static-8bit-dense-qlinear'
  | 'static-8bit-conv-qlinear'
  | 'dynamic-uint8-dense-guidance';

/**
 * Execution state for one Phase 9 runtime-parity fixture.
 *
 * `execute` means the current harness should export, run, and compare the
 * binary model immediately. `skip` keeps the lane in the frozen inventory but
 * requires an explicit follow-up tranche before execution begins.
 */
export type OnnxRuntimeParityExecutionMode = 'execute' | 'skip';

/**
 * Explicit tolerance packet for one runtime-parity lane.
 *
 * Phase 9 keeps tolerances named and lane-specific so reduced-precision cases
 * cannot silently inherit looser thresholds from unrelated fixtures.
 */
export type OnnxRuntimeParityTolerancePacket = {
  maximumMeanSquaredError: number;
  maximumAbsoluteDifference: number;
};

/**
 * Seeded randomized-parity options for one Phase 9C sample run.
 *
 * Phase 9C keeps the runtime subset narrow, but broadens evidence from named
 * golden fixtures into reproducible randomized samples. The seed and sample
 * count make failures replayable.
 */
export type OnnxRuntimeParityRandomizedRunOptions = {
  seed: number;
  sampleCount: number;
};

/**
 * Deterministic fixture descriptor for one Phase 9 runtime-parity lane.
 *
 * Each descriptor couples a stable network factory, the binary export options,
 * one native runtime sample, the ONNX Runtime feed shape, and the tolerance
 * packet that later golden or randomized passes must honor.
 */
export type OnnxRuntimeParityFixtureDescriptor = {
  id: OnnxRuntimeParityLane;
  lane: OnnxRuntimeParityLane;
  executionMode: OnnxRuntimeParityExecutionMode;
  skipReason?: string;
  tolerance: OnnxRuntimeParityTolerancePacket;
  exportOptions?: OnnxExportOptions;
  nativeInputValues: readonly number[];
  runtimeInputValues?: readonly number[];
  runtimeInputDimensions?: readonly number[];
  createNetwork: () => Network;
};

/**
 * Executable fixture descriptor for runtime-parity lanes that are already approved.
 *
 * This narrows the generic fixture descriptor to the executed subset so callers
 * such as the Phase 9C randomized runner do not need unreachable skipped-result
 * guards after they create lane-approved samples.
 */
export type OnnxRuntimeParityExecutedFixtureDescriptor =
  OnnxRuntimeParityFixtureDescriptor & {
    executionMode: 'execute';
    skipReason?: undefined;
  };

/**
 * Result for one executed Phase 9 runtime-parity fixture.
 *
 * The harness reports both native and ONNX Runtime outputs so later golden and
 * randomized passes can reuse the same comparison packet without changing the
 * execution seam.
 */
export type OnnxRuntimeParityExecutedResult = {
  fixture: OnnxRuntimeParityFixtureDescriptor;
  skipped: false;
  inputNames: string[];
  outputNames: string[];
  nativeOutput: number[];
  runtimeOutput: number[];
  meanSquaredError: number;
  maxAbsoluteDifference: number;
  isWithinTolerance: boolean;
};

/**
 * Result for one inventory-only Phase 9 runtime-parity fixture.
 */
export type OnnxRuntimeParitySkippedResult = {
  fixture: OnnxRuntimeParityFixtureDescriptor;
  skipped: true;
  skipReason: string;
};

/**
 * Runtime-parity result union for the Phase 9 harness seam.
 */
export type OnnxRuntimeParityResult =
  | OnnxRuntimeParityExecutedResult
  | OnnxRuntimeParitySkippedResult;

/**
 * Flattened result for one seeded randomized parity sample.
 *
 * The Phase 9C runner returns a compact packet per sample so tests can freeze
 * deterministic summaries without reimplementing the execution seam.
 */
export type OnnxRuntimeParityRandomizedSampleResult = {
  fixtureId: OnnxRuntimeParityLane;
  lane: OnnxRuntimeParityLane;
  sampleIndex: number;
  nativeInputValues: number[];
  runtimeInputValues: number[];
  runtimeInputDimensions?: readonly number[];
  nativeOutput: number[];
  runtimeOutput: number[];
  meanSquaredError: number;
  maxAbsoluteDifference: number;
  isWithinTolerance: boolean;
};
