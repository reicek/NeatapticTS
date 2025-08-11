/**
 * File: network.train.ts
 * ----------------------------------------------------
 * Houses the full gradient-based training pipeline extracted from the Network class.
 * Exposes stateless helper functions that operate on a supplied Network instance.
 * The primary exported entrypoints are:
 *  - applyGradientClippingImpl: Performs several gradient clipping strategies.
 *  - trainSetImpl: Executes a forward/backward pass across a dataset slice (micro-batch aware).
 *  - trainImpl: Orchestrates multi-iteration training with smoothing, early stop, checkpoints, optimizers, mixed precision, etc.
 *
 * Design Goals:
 *  - Separation of concerns: Keep Network lean; isolate training complexity here.
 *  - Composability: Each helper does one thing (clipping, data pass, orchestration).
 *  - Transparency: Rich inline commentary & JSDoc on all steps / variables.
 *  - Extensibility: New optimizers / smoothing strategies can be injected with minimal refactor.
 */
import * as methods from '../methods/methods';
import { config } from '../config';
import type Network from './network';

// Internal helper: gradient clipping implementations
/**
 * Apply gradient clipping to a network's accumulated gradients before the optimizer step.
 *
 * Modes:
 *  - norm: global L2 norm clipping
 *  - percentile: clamp magnitudes beyond a percentile threshold
 *  - layerwiseNorm: per-layer (or per logical grouping) L2 clipping
 *  - layerwisePercentile: per-layer percentile clipping
 *
 * Implementation notes:
 *  - Gradients are gathered into groups (entire net or per layer) to compute norms / percentiles.
 *  - We intentionally operate on raw accumulated fields (totalDeltaWeight / totalDeltaBias).
 *  - Clipping modifies deltas in-place ensuring downstream optimizer sees scaled gradients.
 */
export function applyGradientClippingImpl(net: Network, cfg: { mode: 'norm'|'percentile'|'layerwiseNorm'|'layerwisePercentile'; maxNorm?: number; percentile?: number }) {
  /** Back-reference to network with internal (non public) fields */
  const nany = net as any;
  /** Collect gradient values grouped by whole-network or per-layer depending on mode */
  const gather = () => {
    /** Aggregated gradient groups (each group is a flat list of weight/bias deltas) */
    const groups: number[][] = [];
    if (cfg.mode.startsWith('layerwise')) {
      if ((net as any).layers && (net as any).layers.length > 0) {
        for (let li = 0; li < (net as any).layers.length; li++) {
          const layer = (net as any).layers[li];
            if (!layer || !layer.nodes) continue;
            const g: number[] = [];
            layer.nodes.forEach((n: any) => {
              if (!n || n.type === 'input') return;
              n.connections.in.forEach((c: any) => { if (typeof c.totalDeltaWeight === 'number') g.push(c.totalDeltaWeight); });
              n.connections.self.forEach((c: any) => { if (typeof c.totalDeltaWeight === 'number') g.push(c.totalDeltaWeight); });
              if (typeof n.totalDeltaBias === 'number') g.push(n.totalDeltaBias);
            });
            if (g.length) groups.push(g);
        }
      } else {
        net.nodes.forEach(n => {
          if (n.type === 'input') return;
          const g: number[] = [];
          (n as any).connections.in.forEach((c: any) => { if (typeof c.totalDeltaWeight === 'number') g.push(c.totalDeltaWeight); });
          (n as any).connections.self.forEach((c: any) => { if (typeof c.totalDeltaWeight === 'number') g.push(c.totalDeltaWeight); });
          if (typeof (n as any).totalDeltaBias === 'number') g.push((n as any).totalDeltaBias);
          if (g.length) groups.push(g);
        });
      }
    } else {
      const g: number[] = [];
      net.nodes.forEach(n => {
        (n as any).connections.in.forEach((c: any) => { if (typeof c.totalDeltaWeight === 'number') g.push(c.totalDeltaWeight); });
        (n as any).connections.self.forEach((c: any) => { if (typeof c.totalDeltaWeight === 'number') g.push(c.totalDeltaWeight); });
        if (typeof (n as any).totalDeltaBias === 'number') g.push((n as any).totalDeltaBias);
      });
      if (g.length) groups.push(g);
    }
    return groups;
  };
  /** Gradient groups for scaling decisions */
  const groups = gather();
  nany._lastGradClipGroupCount = groups.length;
  /** Compute absolute-value percentile threshold within an array of signed gradient values */
  const percentile = (arr: number[], p: number) => {
    if (!arr.length) return 0;
    const sorted = [...arr].sort((a,b)=>Math.abs(a)-Math.abs(b));
    const rank = Math.min(sorted.length - 1, Math.max(0, Math.floor((p/100)*sorted.length - 1)));
    return Math.abs(sorted[rank]);
  };
  /** Apply scaling / clipping transform to all collected gradients */
  const applyScale = (scaleFn: (current: number, group: number[]) => number) => {
    let gi = 0;
    net.nodes.forEach(n => {
      if (cfg.mode.startsWith('layerwise') && n.type === 'input') return;
      const group = cfg.mode.startsWith('layerwise') ? groups[gi++] : groups[0];
      (n as any).connections.in.forEach((c: any) => { if (typeof c.totalDeltaWeight === 'number') c.totalDeltaWeight = scaleFn(c.totalDeltaWeight, group); });
      (n as any).connections.self.forEach((c: any) => { if (typeof c.totalDeltaWeight === 'number') c.totalDeltaWeight = scaleFn(c.totalDeltaWeight, group); });
      if (typeof (n as any).totalDeltaBias === 'number') (n as any).totalDeltaBias = scaleFn((n as any).totalDeltaBias, group);
    });
  };
  if (cfg.mode === 'norm' || cfg.mode === 'layerwiseNorm') {
    /** Target maximum allowed L2 norm (group wise) */
    const maxN = cfg.maxNorm || 1;
    groups.forEach(g => {
      const norm = Math.sqrt(g.reduce((s,v)=>s+v*v,0));
      if (norm > maxN && norm > 0) {
        const scale = maxN / norm;
        applyScale((cur, group)=> group === g ? cur * scale : cur);
      }
    });
  } else if (cfg.mode === 'percentile' || cfg.mode === 'layerwisePercentile') {
    /** Percentile (0-100) used to derive clipping magnitude threshold */
    const p = cfg.percentile || 99;
    groups.forEach(g => {
      const thresh = percentile(g, p);
      if (thresh <= 0) return;
      applyScale((cur, group)=> group === g && Math.abs(cur) > thresh ? (thresh * Math.sign(cur)) : cur);
    });
  }
}

/**
 * Execute forward + backward passes over the provided dataset (one logical iteration / epoch).
 * Supports micro-batching / gradient accumulation, multiple optimizer variants, mixed precision guard rails.
 *
 * Returns the averaged error across processed samples (skipping invalid dimension samples gracefully).
 */
export function trainSetImpl(
  net: Network,
  set: { input: number[]; output: number[] }[],
  batchSize: number,
  accumulationSteps: number,
  currentRate: number,
  momentum: number,
  regularization: any,
  costFunction: (target: number[], output: number[]) => number,
  optimizer?: any
): number {
  /** Network instance with internal helper fields */
  const nany = net as any;
  /** Cumulative (unsmoothed) error across all correctly-dimensioned samples */
  let errorSum = 0;
  /** Counter for samples processed in the current micro-batch (resets after optimizer step boundary) */
  let processedSamplesInBatch = 0;
  /** Number of micro-batches whose gradients have been accumulated since last optimizer step */
  nany._gradAccumMicroBatches = 0;
  /** Total number of valid samples processed (dimension‑mismatched samples are skipped) */
  let totalProcessedSamples = 0;
  /** Cached list of output layer nodes (avoids repeated filtering in inner loop) */
  const outputNodes = net.nodes.filter(n => n.type === 'output');
  /** Resolved cost function (function wrapper over user supplied variant or built-in) */
  let computeError: (t:number[], o:number[])=>number;
  if (typeof costFunction === 'function') computeError = costFunction as any; else if ((costFunction as any) && typeof (costFunction as any).fn === 'function') computeError = (costFunction as any).fn; else if ((costFunction as any) && typeof (costFunction as any).calculate === 'function') computeError = (costFunction as any).calculate; else computeError = () => 0;

  for (let i = 0; i < set.length; i++) {
  /** Current training example (input/output pair) */
  const dataPoint = set[i];
  /** Feature vector for current sample */
  const input = dataPoint.input;
  /** Ground-truth target vector for current sample */
  const target = dataPoint.output;
    if (input.length !== net.input || target.length !== net.output) {
      if (config.warnings) console.warn(`Data point ${i} has incorrect dimensions (input: ${input.length}/${net.input}, output: ${target.length}/${net.output}), skipping.`);
      continue;
    }
    try {
  /** Network prediction for current sample (forward pass under training mode) */
  const output = (net as any).activate(input, true); // training forward
      if (optimizer && optimizer.type && optimizer.type !== 'sgd') {
        for (let o = 0; o < outputNodes.length; o++) {
          (outputNodes[o] as any).propagate(currentRate, momentum, false, regularization, target[o]);
        }
        for (let r = net.nodes.length - 1; r >= 0; r--) {
          const node = net.nodes[r];
            if (node.type === 'output' || node.type === 'input') continue;
            (node as any).propagate(currentRate, momentum, false, regularization);
        }
      } else {
        for (let o = 0; o < outputNodes.length; o++) {
          (outputNodes[o] as any).propagate(currentRate, momentum, true, regularization, target[o]);
        }
        for (let r = net.nodes.length - 1; r >= 0; r--) {
          const node = net.nodes[r];
            if (node.type === 'output' || node.type === 'input') continue;
            (node as any).propagate(currentRate, momentum, true, regularization);
        }
      }
      errorSum += computeError(target, output);
      processedSamplesInBatch++;
      totalProcessedSamples++;
    } catch (e: any) {
      if (config.warnings) console.warn(`Error processing data point ${i} (input: ${JSON.stringify(input)}): ${e.message}. Skipping.`);
    }
    if (processedSamplesInBatch > 0 && ((((i + 1) % batchSize) === 0) || i === set.length - 1)) {
      // End of a micro-batch
      if (optimizer && optimizer.type && optimizer.type !== 'sgd') {
        nany._gradAccumMicroBatches++;
        /** Whether accumulated micro-batches reach threshold for an optimizer step */
        const readyForStep = (nany._gradAccumMicroBatches % accumulationSteps === 0) || i === set.length - 1;
        if (readyForStep) {
          // Increment global optimizer step (used by adaptive optimizers & telemetry)
          nany._optimizerStep = (nany._optimizerStep || 0) + 1;
          // Mixed precision overflow detection & optimizer step
          /** Flag set when numeric overflow (Inf/NaN) is detected under mixed precision */
          let overflowDetected = false;
          if (nany._mixedPrecision.enabled) {
            if (nany._forceNextOverflow) { overflowDetected = true; nany._forceNextOverflow = false; }
            else {
              net.nodes.forEach(node => {
                if ((node as any)._fp32Bias !== undefined) {
                  if (!Number.isFinite((node as any).bias)) overflowDetected = true;
                }
              });
            }
          }
          if (overflowDetected) {
            // Zero accumulated gradients & update overflow telemetry
            net.nodes.forEach(node => {
              (node as any).connections.in.forEach((c: any) => { c.totalDeltaWeight = 0; });
              (node as any).connections.self.forEach((c: any) => { c.totalDeltaWeight = 0; });
              if (typeof (node as any).totalDeltaBias === 'number') (node as any).totalDeltaBias = 0;
              (node as any).previousDeltaBias = 0;
            });
            if (nany._mixedPrecision.enabled) {
              nany._mixedPrecisionState.badSteps++;
              nany._mixedPrecisionState.goodSteps = 0;
              nany._mixedPrecision.lossScale = Math.max(nany._mixedPrecisionState.minLossScale, Math.floor(nany._mixedPrecision.lossScale / 2) || 1);
              nany._mixedPrecisionState.overflowCount = (nany._mixedPrecisionState.overflowCount||0)+1;
              nany._mixedPrecisionState.scaleDownEvents = (nany._mixedPrecisionState.scaleDownEvents||0)+1;
              nany._lastOverflowStep = nany._optimizerStep;
            }
            nany._lastGradNorm = 0;
          } else {
            if (nany._currentGradClip) applyGradientClippingImpl(net, nany._currentGradClip);
            // Average accumulated gradients if requested
            if (accumulationSteps > 1 && nany._accumulationReduction === 'average') {
              net.nodes.forEach(node => {
                (node as any).connections.in.forEach((c: any) => { if (typeof c.totalDeltaWeight === 'number') c.totalDeltaWeight /= accumulationSteps; });
                (node as any).connections.self.forEach((c: any) => { if (typeof c.totalDeltaWeight === 'number') c.totalDeltaWeight /= accumulationSteps; });
                if (typeof (node as any).totalDeltaBias === 'number') (node as any).totalDeltaBias /= accumulationSteps;
              });
            }
            /** Sum of squared last applied weight deltas (for reporting gradient norm) */
            let sumSq = 0;
            net.nodes.forEach(node => {
              if (node.type === 'input') return;
              (node as any).applyBatchUpdatesWithOptimizer({
                type: optimizer.type,
                baseType: optimizer.baseType,
                beta1: optimizer.beta1,
                beta2: optimizer.beta2,
                eps: optimizer.eps,
                weightDecay: optimizer.weightDecay,
                momentum: optimizer.momentum ?? momentum,
                lrScale: currentRate,
                t: nany._optimizerStep,
                la_k: optimizer.la_k,
                la_alpha: optimizer.la_alpha
              });
              (node as any).connections.in.forEach((c: any) => { if (typeof c.previousDeltaWeight === 'number') sumSq += c.previousDeltaWeight * c.previousDeltaWeight; });
              (node as any).connections.self.forEach((c: any) => { if (typeof c.previousDeltaWeight === 'number') sumSq += c.previousDeltaWeight * c.previousDeltaWeight; });
            });
            if (nany._mixedPrecision.enabled) {
              nany._mixedPrecisionState.goodSteps++;
              /** Steps of stability required before attempting lossScale growth */
              const incEvery = (nany)._mpIncreaseEvery || 200;
              if (nany._mixedPrecisionState.goodSteps >= incEvery && nany._mixedPrecision.lossScale < nany._mixedPrecisionState.maxLossScale) {
                nany._mixedPrecision.lossScale *= 2;
                nany._mixedPrecisionState.goodSteps = 0;
                nany._mixedPrecisionState.scaleUpEvents = (nany._mixedPrecisionState.scaleUpEvents||0)+1;
              }
            }
            nany._lastGradNorm = Math.sqrt(sumSq);
          }
        }
        processedSamplesInBatch = 0;
      }
    }
  }
  if (nany._lastGradNorm == null) nany._lastGradNorm = 0;
  return totalProcessedSamples > 0 ? errorSum / totalProcessedSamples : 0;
}

// Main train implementation (formerly Network.train)
/**
 * High-level training orchestrator implementing:
 *  - Input validation
 *  - Option normalization / defaults
 *  - Optimizer config (including lookahead composition)
 *  - Mixed precision scaling & dynamic loss scale adjustments
 *  - Gradient clipping (global / layerwise / percentile)
 *  - Gradient accumulation (average or sum semantics)
 *  - Multi smoothing strategies (EMA, WMA, median, trimmed mean, gaussian, adaptive-ema)
 *  - Early stopping by patience & min delta
 *  - Plateau smoothing (independent series) for potential schedulers
 *  - Checkpoint hooks (last & best)
 *  - Metrics hook emission per iteration
 *  - Final error + iterations + wall-clock time summary
 */
export function trainImpl(net: Network, set: { input: number[]; output: number[] }[], options: any): { error: number; iterations: number; time: number } {
  const nany = net as any;
  if (!set || set.length === 0 || set[0].input.length !== net.input || set[0].output.length !== net.output) {
    throw new Error('Dataset is invalid or dimensions do not match network input/output size!');
  }
  options = options || {};
  // Validate stopping criteria (tests expect warning + throw when both missing)
  if (typeof options.iterations === 'undefined' && typeof options.error === 'undefined') {
    if (config.warnings) console.warn('Missing `iterations` or `error` option.');
    throw new Error('Missing `iterations` or `error` option. Training requires a stopping condition.');
  }
  if (config.warnings) {
    if (typeof options.rate === 'undefined') {
      console.warn('Missing `rate` option');
      console.warn('Missing `rate` option, using default learning rate 0.3.');
    }
    if (typeof options.iterations === 'undefined') {
      console.warn('Missing `iterations` option. Training will run potentially indefinitely until `error` threshold is met.');
    }
  }
  /** Target error threshold for early exit (if absent, run fixed iterations only) */
  let targetError = options.error ?? -Infinity; // if undefined we only run fixed iterations
  /** User supplied or default cost function (accepts function object with fn / calculate) */
  const cost = options.cost || methods.Cost.mse;
  if (typeof cost !== 'function' && !(typeof cost === 'object' && (typeof (cost as any).fn === 'function' || typeof (cost as any).calculate === 'function'))) {
    throw new Error('Invalid cost function provided to Network.train.');
  }
  /** Base learning rate (may be modulated by external schedule callback) */
  const baseRate = options.rate ?? 0.3; // base learning rate (may be externally scheduled)
  /** Dropout probability applied at node activation time during training */
  const dropout = options.dropout || 0;
  if (dropout < 0 || dropout >= 1) throw new Error('dropout must be in [0,1)');
  /** Momentum coefficient for optimizers supporting classical momentum semantics */
  const momentum = options.momentum || 0;
  /** Micro-batch size (number of samples after which gradients are locally aggregated) */
  const batchSize = options.batchSize || 1; // micro-batch size for weight update granularity
  if (batchSize > set.length) throw new Error('Batch size cannot be larger than the dataset length.');
  /** Number of micro-batches whose gradients are accumulated before applying an optimizer step */
  const accumulationSteps = options.accumulationSteps || 1; // number of micro-batches to accumulate before optimizer step
  nany._accumulationReduction = (options.accumulationReduction === 'sum') ? 'sum' : 'average';
  if (accumulationSteps < 1 || !Number.isFinite(accumulationSteps)) throw new Error('accumulationSteps must be >=1');
  if (options.gradientClip) {
    const gc = options.gradientClip;
    if (gc.mode) {
      nany._currentGradClip = { mode: gc.mode, maxNorm: gc.maxNorm, percentile: gc.percentile } as any;
    } else if (typeof gc.maxNorm === 'number') {
      nany._currentGradClip = { mode: 'norm', maxNorm: gc.maxNorm };
    } else if (typeof gc.percentile === 'number') {
      nany._currentGradClip = { mode: 'percentile', percentile: gc.percentile } as any;
    }
    nany._gradClipSeparateBias = !!gc.separateBias;
  } else {
    nany._currentGradClip = undefined;
    nany._gradClipSeparateBias = false;
  }
  if (options.mixedPrecision) {
    const mp = options.mixedPrecision === true ? { lossScale: 1024 } : options.mixedPrecision;
    nany._mixedPrecision.enabled = true;
    nany._mixedPrecision.lossScale = mp.lossScale || 1024;
    const dyn = mp.dynamic || {};
    nany._mixedPrecisionState.minLossScale = dyn.minScale || 1;
    nany._mixedPrecisionState.maxLossScale = dyn.maxScale || 65536;
    nany._mpIncreaseEvery = dyn.increaseEvery || dyn.stableStepsForIncrease || 200;
    net.connections.forEach(c => { (c as any)._fp32Weight = c.weight; });
    net.nodes.forEach(n => { if (n.type !== 'input') (n as any)._fp32Bias = n.bias; });
  } else {
    nany._mixedPrecision.enabled = false;
    nany._mixedPrecision.lossScale = 1;
    nany._mpIncreaseEvery = 200;
  }
  /** Supported optimizer type identifiers (validation whitelist) */
  const allowedOptimizers = new Set(['sgd','rmsprop','adagrad','adam','adamw','amsgrad','adamax','nadam','radam','lion','adabelief','lookahead']); // supported optimizers
  /** Normalized optimizer configuration (may wrap lookahead base optimizer) */
  let optimizerConfig: any = undefined;
  if (typeof options.optimizer !== 'undefined') {
    if (typeof options.optimizer === 'string') optimizerConfig = { type: options.optimizer.toLowerCase() }; else if (typeof options.optimizer === 'object' && options.optimizer !== null) { optimizerConfig = { ...options.optimizer }; if (typeof optimizerConfig.type === 'string') optimizerConfig.type = optimizerConfig.type.toLowerCase(); } else { throw new Error('Invalid optimizer option; must be string or object'); }
    if (!allowedOptimizers.has(optimizerConfig.type)) throw new Error(`Unknown optimizer type: ${optimizerConfig.type}`);
    if (optimizerConfig.type === 'lookahead') {
      if (!optimizerConfig.baseType) optimizerConfig.baseType = 'adam';
      if (optimizerConfig.baseType === 'lookahead') throw new Error('Nested lookahead (baseType lookahead) is not supported');
      if (!allowedOptimizers.has(optimizerConfig.baseType)) throw new Error(`Unknown baseType for lookahead: ${optimizerConfig.baseType}`);
      optimizerConfig.la_k = optimizerConfig.la_k || 5;
      optimizerConfig.la_alpha = optimizerConfig.la_alpha ?? 0.5;
    }
  }
  /** Maximum iterations to perform (capped by patience / target error if provided) */
  const iterations = options.iterations ?? Number.MAX_SAFE_INTEGER; // open-ended if only error threshold is provided
  /** Wall clock start time (ms) for elapsed time reporting */
  const start = Date.now();
  /** Final monitored (smoothed) error recorded upon termination */
  let finalError = Infinity;
  // Early-stop smoothing (primary monitored series)
  /** Window size for primary smoothing statistic (ignored for pure EMA if 1 + ema specified) */
  const movingAverageWindow = Math.max(1, options.movingAverageWindow || 1); // smoothing window length
  /** Smoothing strategy for early stopping monitor (sma|ema|adaptive-ema|median|gaussian|trimmed|wma) */
  const movingAverageType = options.movingAverageType || 'sma'; // smoothing strategy
  /** Effective alpha for standard EMA (derived if not explicitly provided) */
  const emaAlpha = (() => {
    if (movingAverageType !== 'ema') return undefined;
    if (options.emaAlpha && options.emaAlpha > 0 && options.emaAlpha <= 1) return options.emaAlpha;
    // Default EMA alpha heuristic
    return 2 / (movingAverageWindow + 1);
  })();
  // Plateau smoothing (independent from early-stop smoothing if provided)
  /** Independent smoothing window for plateau detection (may differ from main window) */
  const plateauWindow = Math.max(1, options.plateauMovingAverageWindow || movingAverageWindow); // independent plateau tracking window
  /** Smoothing algorithm for plateau metric (defaults to monitored smoothing type) */
  const plateauType = options.plateauMovingAverageType || movingAverageType;
  /** Alpha used if plateau smoothing selects EMA */
  const plateauEmaAlpha = (() => {
    if (plateauType !== 'ema') return undefined;
    if (options.plateauEmaAlpha && options.plateauEmaAlpha > 0 && options.plateauEmaAlpha <= 1) return options.plateauEmaAlpha;
    return 2 / (plateauWindow + 1);
  })();
  /** Number of consecutive non-improving iterations tolerated before early stop (undefined disables) */
  const earlyStopPatience = options.earlyStopPatience;
  /** Minimum required improvement (absolute error decrease) to reset patience counter */
  const earlyStopMinDelta = options.earlyStopMinDelta || 0;
  /** Best (lowest) monitored error observed so far */
  let bestError = Infinity;
  /** Count of consecutive iterations without sufficient improvement */
  let noImproveCount = 0;
  /** Circular buffer capacity for primary smoothing */
  const recentErrorsCapacity = movingAverageWindow;
  /** Underlying storage for recent raw errors (circular). Index points to next write position. */
  const recentErrorsBuf: number[] = new Array(recentErrorsCapacity);
  /** Number of valid entries currently stored (<= capacity). */
  let recentErrorsCount = 0;
  /** Next write index (wraps modulo capacity). */
  let recentErrorsWriteIdx = 0;
  /** Insert a new raw error sample into the circular buffer. */
  const recentErrorsPush = (value: number) => {
    if (recentErrorsCapacity === 1) { recentErrorsBuf[0] = value; recentErrorsCount = 1; recentErrorsWriteIdx = 0; return; }
    recentErrorsBuf[recentErrorsWriteIdx] = value;
    recentErrorsWriteIdx = (recentErrorsWriteIdx + 1) % recentErrorsCapacity;
    if (recentErrorsCount < recentErrorsCapacity) recentErrorsCount++;
  };
  /** Materialize chronological (oldest->newest) array view (allocates new array of size recentErrorsCount). */
  const recentErrorsChrono = (): number[] => {
    if (recentErrorsCount === 0) return [];
    if (recentErrorsCount < recentErrorsCapacity) return recentErrorsBuf.slice(0, recentErrorsCount);
    const out = new Array(recentErrorsCount);
    const start = recentErrorsWriteIdx; // oldest element
    for (let i=0;i<recentErrorsCount;i++) out[i] = recentErrorsBuf[(start + i) % recentErrorsCapacity];
    return out;
  };
  /** Last EMA value (primary monitor) when EMA strategy is active */
  let emaValue: number | undefined = undefined; // for plain EMA
  /** Baseline EMA for adaptive-ema strategy */
  let adaptiveBaseEmaValue: number | undefined = undefined;
  /** Adaptive EMA (variance-scaled alpha) for adaptive-ema strategy */
  let adaptiveEmaValue: number | undefined = undefined;
  /** Circular buffer capacity for plateau smoothing */
  const plateauCapacity = plateauWindow;
  /** Plateau error circular buffer storage */
  const plateauBuf: number[] = new Array(plateauCapacity);
  let plateauCount = 0;
  let plateauWriteIdx = 0;
  const plateauPush = (value: number) => {
    if (plateauCapacity === 1) { plateauBuf[0] = value; plateauCount = 1; plateauWriteIdx = 0; return; }
    plateauBuf[plateauWriteIdx] = value;
    plateauWriteIdx = (plateauWriteIdx + 1) % plateauCapacity;
    if (plateauCount < plateauCapacity) plateauCount++;
  };
  const plateauChrono = (): number[] => {
    if (plateauCount === 0) return [];
    if (plateauCount < plateauCapacity) return plateauBuf.slice(0, plateauCount);
    const out = new Array(plateauCount);
    const start = plateauWriteIdx;
    for (let i=0;i<plateauCount;i++) out[i] = plateauBuf[(start + i) % plateauCapacity];
    return out;
  };
  /** Last plateau EMA value (if plateau uses EMA) */
  let plateauEmaValue: number | undefined = undefined;
  // Set network-level dropout so activation path handles masks (tests spy on activate)
  net.dropout = dropout;
  /** Actual iterations executed (may be < requested iterations due to early exit criteria) */
  let performedIterations = 0;
  for (let iter=1; iter<=iterations; iter++) {
  // Pruning schedule hook
  if ((net as any)._maybePrune) { (net as any)._maybePrune((nany._globalEpoch||0) + iter); }
  /** Raw (unsmoothed) training error returned by this iteration's dataset pass */
  const trainError = trainSetImpl(net, set, batchSize, accumulationSteps, baseRate, momentum, {}, cost as any, optimizerConfig);
    performedIterations = iter;
  recentErrorsPush(trainError);
    // Early-stop smoothing compute
  /** Smoothed / monitored error value after applying selected smoothing strategy */
  let monitored = trainError;
    if (movingAverageWindow > 1 || movingAverageType === 'ema' || movingAverageType === 'adaptive-ema') {
      const recentArr = recentErrorsChrono();
      if (movingAverageType === 'median') {
        const sorted = [...recentArr].sort((a,b)=>a-b);
        const mid = Math.floor(sorted.length/2);
        monitored = sorted.length % 2 ? sorted[mid] : (sorted[mid-1]+sorted[mid])/2;
      } else if (movingAverageType === 'ema') {
        if (emaValue == null) emaValue = trainError; else emaValue = emaValue + (emaAlpha!)*(trainError - emaValue);
        monitored = emaValue;
      } else if (movingAverageType === 'adaptive-ema') {
        // Compute variance-scaled alpha; also retain a baseline EMA and take the more responsive (lower) reading.
        const mean = recentArr.reduce((a,b)=>a+b,0)/recentArr.length;
        const variance = recentArr.reduce((a,b)=>a+(b-mean)*(b-mean),0)/recentArr.length;
        const baseAlpha = emaAlpha || (2/(movingAverageWindow+1));
        const varScaled = variance / (Math.max(mean*mean, 1e-8));
        // Increase responsiveness more aggressively; cap to 0.95
        const adaptAlpha = Math.min(0.95, Math.max(baseAlpha, baseAlpha * (1 + 2*varScaled)));
        if (adaptiveBaseEmaValue == null) {
          adaptiveBaseEmaValue = trainError;
          adaptiveEmaValue = trainError;
        } else {
          adaptiveBaseEmaValue = adaptiveBaseEmaValue + baseAlpha*(trainError - adaptiveBaseEmaValue);
          adaptiveEmaValue = adaptiveEmaValue! + adaptAlpha*(trainError - adaptiveEmaValue!);
        }
        // Guarantee we never do worse than baseline EMA; pick min
        monitored = Math.min(adaptiveEmaValue!, adaptiveBaseEmaValue!);
      } else if (movingAverageType === 'gaussian') {
        const arr = recentArr;
        const n = arr.length;
        const sigma = (movingAverageWindow/3) || 1;
        let sumW = 0, acc = 0;
        for (let i=0;i<n;i++) {
          const w = Math.exp(-0.5 * Math.pow((i - (n-1))/sigma,2));
            sumW += w; acc += w * arr[i];
        }
        monitored = acc / (sumW||1);
      } else if (movingAverageType === 'trimmed') {
        const ratio = Math.min(0.49, Math.max(0, options.trimmedRatio||0.1));
        const sorted = [...recentArr].sort((a,b)=>a-b);
        const drop = Math.floor(sorted.length * ratio);
        const trimmed = sorted.slice(drop, sorted.length - drop);
        monitored = trimmed.reduce((a,b)=>a+b,0)/(trimmed.length||1);
      } else if (movingAverageType === 'wma') {
        let wSum = 0, acc=0; for (let i=0;i<recentArr.length;i++){ const w = i+1; wSum += w; acc += w*recentArr[i]; } monitored = acc/(wSum||1);
      } else { // sma default
        monitored = recentArr.reduce((a,b)=>a+b,0) / recentArr.length;
      }
    } else if (movingAverageType === 'ema') { // window=1 but EMA requested
      if (emaValue == null) emaValue = trainError; else emaValue = emaValue + (emaAlpha!)*(trainError - emaValue);
      monitored = emaValue;
    } else if (movingAverageType === 'adaptive-ema') {
      const baseAlpha = emaAlpha || (2/(movingAverageWindow+1));
      const adaptAlpha = Math.min(0.95, Math.max(baseAlpha, baseAlpha * 1.5));
      if (adaptiveBaseEmaValue == null) {
        adaptiveBaseEmaValue = trainError; adaptiveEmaValue = trainError;
      } else {
        adaptiveBaseEmaValue = adaptiveBaseEmaValue + baseAlpha*(trainError - adaptiveBaseEmaValue);
        adaptiveEmaValue = adaptiveEmaValue! + adaptAlpha*(trainError - adaptiveEmaValue!);
      }
      monitored = Math.min(adaptiveEmaValue!, adaptiveBaseEmaValue!);
    }
    finalError = monitored;
    // Plateau smoothing compute (used only for metricsHook exposure & potential rate policies)
  plateauPush(trainError);
  /** Plateau-tracked error after applying plateau smoothing strategy */
  let plateauError: number | undefined = trainError;
    if (plateauWindow > 1 || plateauType === 'ema') {
      if (plateauType === 'median') {
        const sorted = [...plateauChrono()].sort((a,b)=>a-b);
        const mid = Math.floor(sorted.length/2);
        plateauError = sorted.length % 2 ? sorted[mid] : (sorted[mid-1]+sorted[mid])/2;
      } else if (plateauType === 'ema') {
        if (plateauEmaValue == null) plateauEmaValue = trainError; else plateauEmaValue = plateauEmaValue + (plateauEmaAlpha!)*(trainError - plateauEmaValue);
        plateauError = plateauEmaValue;
      } else { // sma
        const arr = plateauChrono();
        plateauError = arr.reduce((a,b)=>a+b,0) / arr.length;
      }
    }
    // metricsHook
    if (typeof options.metricsHook === 'function') {
      try { options.metricsHook({ iteration: iter, error: finalError, plateauError, gradNorm: nany._lastGradNorm ?? 0 }); } catch {}
    }
    // checkpoint
    if (options.checkpoint && typeof options.checkpoint.save === 'function') {
      if (options.checkpoint.last) { try { options.checkpoint.save({ type:'last', iteration: iter, error: finalError, network: net.toJSON() }); } catch {} }
      if (options.checkpoint.best) {
        if (finalError < (net as any)._checkpointBestError || (net as any)._checkpointBestError == null) {
          (net as any)._checkpointBestError = finalError;
          try { options.checkpoint.save({ type:'best', iteration: iter, error: finalError, network: net.toJSON() }); } catch {}
        }
      }
    }
    // schedule function
    if (options.schedule && options.schedule.iterations && iter % options.schedule.iterations === 0) {
      try { options.schedule.function({ error: finalError, iteration: iter }); } catch {}
    }
    // Early stopping patience
    if (finalError < bestError - earlyStopMinDelta) { bestError = finalError; noImproveCount = 0; } else if (earlyStopPatience) { noImproveCount++; }
    if (earlyStopPatience && noImproveCount >= earlyStopPatience) {
      break;
    }
    if (finalError <= targetError) { break; }
  }
  // Reset dropout masks & property post training
  net.nodes.forEach(n => { if (n.type === 'hidden') n.mask = 1; });
  net.dropout = 0;
  // Track global epoch counter for features needing cumulative iteration reference (e.g. pruning schedule)
  (nany._globalEpoch = (nany._globalEpoch||0) + performedIterations);
  return { error: finalError, iterations: performedIterations, time: Date.now() - start };
}
