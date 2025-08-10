### This library is being updated to TypeScript

<img src="https://cdn-images-1.medium.com/max/800/1*THG2__H9YHxYIt2sulzlTw.png" width="100%"/>

# NeatapticTS

NeatapticTS offers flexible neural networks; neurons and synapses can be removed with a single line of code. No fixed architecture is required for neural networks to function at all. This flexibility allows networks to be shaped for your dataset through neuro-evolution, which is done using multiple threads.

# Network Constructor Update

The `Network` class constructor now supports an optional third parameter for configuration:

```ts
new Network(input: number, output: number, options?: { minHidden?: number })
```
- `input`: Number of input nodes (required)
- `output`: Number of output nodes (required)
- `options.minHidden`: (optional) If set, enforces a minimum number of hidden nodes. If omitted or 0, no minimum is enforced. This allows true 1-1 (input-output only) networks.

**Example:**
```ts
// Standard 1-1 network (no hidden nodes)
const net = new Network(1, 1);

// Enforce at least 3 hidden nodes
const netWithHidden = new Network(2, 1, { minHidden: 3 });
```

# Neat Evolution minHidden Option

The `minHidden` option can also be passed to the `Neat` class to enforce a minimum number of hidden nodes in all evolved networks:

```ts
import Neat from './src/neat';
const neat = new Neat(2, 1, fitnessFn, { popsize: 50, minHidden: 5 });
```
- All networks created by the evolutionary process will have at least 5 hidden nodes.
- This is useful for ensuring a minimum network complexity during neuro-evolution.

See tests in `test/neat.ts` for usage and verification.

---

# ONNX Import/Export

NeatapticTS now supports exporting to and importing from ONNX format for strictly layered MLPs. This allows interoperability with other machine learning frameworks.

- Use `exportToONNX(network)` to export a network to ONNX.
- Use `importFromONNX(onnxModel)` to import a compatible ONNX model as a `Network` instance.

See tests in `test/network/onnx.export.test.ts` and `test/network/onnx.import.test.ts` for usage examples.

---

### Further notices

NeatapticTS is based on [Neataptic](https://github.com/wagenaartje/neataptic). Parts of [Synaptic](https://github.com/cazala/synaptic) were used to develop Neataptic.

The neuro-evolution algorithm used is the [Instinct](https://medium.com/@ThomasWagenaar/neuro-evolution-on-steroids-82bd14ddc2f6) algorithm.

##### Original [repository](https://github.com/wagenaartje/neataptic) in now [unmaintained](https://github.com/wagenaartje/neataptic/issues/112)

---

## Added Training Features

- Learning rate schedulers: fixed, step, exp, inv, cosine annealing, cosine annealing w/ warm restarts, linear warmup+decay, reduce-on-plateau.
- Regularization (L1, L2, custom function), dropout, DropConnect.
- Per-iteration `metricsHook` exposing `{ iteration, error, gradNorm }`.
- Checkpointing (`best`, `last`) via `checkpoint.save` callback.
- Advanced optimizers: `sgd`, `rmsprop`, `adagrad`, `adam`, `adamw`, `amsgrad`, `adamax`, `nadam`, `radam`, `lion`, `adabelief`, and `lookahead` wrapper.
- Gradient improvements: per-call gradient clipping (global / layerwise, norm or percentile), micro-batch gradient accumulation (`accumulationSteps`) independent of data `batchSize`, optional mixed precision (loss-scaled with dynamic scaling) training.

### Gradient Improvements

Gradient clipping (optional):
```ts
net.train(data, { iterations:500, rate:0.01, optimizer:'adam', gradientClip:{ mode:'norm', maxNorm:1 } });
net.train(data, { iterations:500, rate:0.01, optimizer:'adam', gradientClip:{ mode:'percentile', percentile:99 } });
// Layerwise variants: 'layerwiseNorm' | 'layerwisePercentile'
```

Micro-batch accumulation (simulate larger effective batch without increasing memory):
```ts
// Process 1 sample at a time, accumulate 8 micro-batches, then apply one optimizer step
net.train(data, { iterations:100, rate:0.005, batchSize:1, accumulationSteps:8, optimizer:'adam' });
```
If `accumulationSteps > 1`, gradients are averaged before the optimizer step so results match a single larger batch (deterministic given same sample order).

Mixed precision (simulated FP16 gradients with FP32 master weights + dynamic loss scaling):
```ts
net.train(data, { iterations:300, rate:0.01, optimizer:'adam', mixedPrecision:{ lossScale:1024 } });
```
Behavior:
* Stores master FP32 copies of weights/biases (`_fp32Weight`, `_fp32Bias`).
* Scales gradients during accumulation; unscales before clipping / optimizer update; adjusts `lossScale` down on overflow (NaN/Inf), attempts periodic doubling after sustained stable steps (configurable via `mixedPrecision.dynamic`).
* Raw gradient norm (pre-optimizer, post-scaling/clipping) exposed via metrics hook as `gradNormRaw` (legacy post-update norm still `gradNorm`).
* Pure JS numbers remain 64-bit; this is a functional simulation for stability and future WASM/GPU backends.

Clipping modes:
| mode | scope | description |
|------|-------|-------------|
| norm | global | Clip global L2 gradient norm to `maxNorm`. |
| percentile | global | Clamp individual gradients above given percentile magnitude. |
| layerwiseNorm | per layer | Apply norm clipping per architectural layer (fallback per node if no layer info). Optionally splits weight vs bias groups via `gradientClip.separateBias`. |
| layerwisePercentile | per layer | Percentile clamp per architectural layer (fallback per node). Supports `separateBias`. |

Notes:
* Provide either `{ mode, maxNorm? , percentile? }` or shorthand `{ maxNorm }` / `{ percentile }`.
* Percentile ranking is magnitude-based.
* Accumulation averages gradients; to sum instead (rare) scale `rate` accordingly.
* Dynamic loss scaling heuristics: halves on detected overflow, doubles after configurable stable steps (default 200) within bounds `[minScale,maxScale]`.
* Config: `mixedPrecision:{ lossScale:1024, dynamic:{ minScale:1, maxScale:131072, increaseEvery:300 } }`.
* Accumulation reduction: default averages gradients; specify `accumulationReduction:'sum'` to sum instead (then adjust learning rate manually, e.g. multiply by 1/accumulationSteps if you want equivalent averaging semantics).
* Layerwise clipping: set `gradientClip:{ mode:'layerwiseNorm', maxNorm:1, separateBias:true }` to treat biases separately from weights.
* Two gradient norms tracked: raw (pre-update) and legacy (post-update deltas). Future APIs may expose both formally.
* Access stats: `net.getTrainingStats()` -> `{ gradNorm, gradNormRaw, lossScale, optimizerStep, mp:{ good, bad, overflowCount, scaleUps, scaleDowns, lastOverflowStep } }`.
* Test hook (not for production): `net.testForceOverflow()` forces the next mixed-precision step to register an overflow (used in unit tests to validate telemetry paths).
* Gradient clip grouping count: `net.getLastGradClipGroupCount()` (useful to verify separateBias effect).
* Rate helper for accumulation: `const adjRate = Network.adjustRateForAccumulation(rate, accumulationSteps, accumulationReduction)`.
* Deterministic seeding: `new Network(4,2,{ seed:123 })` or later `net.setSeed(123)` ensures reproducible initial weights, biases, connection order, and mutation randomness for training (excluding certain static reconstruction paths). For NEAT evolution pass `seed` in `new Neat(...,{ seed:999 })`.
* Overflow telemetry: during mixed precision training, `overflowCount` increments on detected NaN/Inf when unscaling gradients; `scaleUps` / `scaleDowns` count dynamic loss scale adjustments.

### Learning Rate Scheduler Usage

```ts
import methods from './src/methods/methods';
const net = new Network(2,1);
const ratePolicy = methods.Rate.cosineAnnealingWarmRestarts(200, 1e-5, 2);
net.train(data, { iterations: 1000, rate: 0.1, ratePolicy });
```

Reduce-on-plateau automatically receives current error because `train` detects a 3-arg scheduler:
```ts
const rop = methods.Rate.reduceOnPlateau({ patience: 20, factor: 0.5, minRate: 1e-5 });
net.train(data, { iterations: 5000, rate: 0.05, ratePolicy: rop });
```

#### Scheduler Reference

| Scheduler | Factory Call | Key Params | Behavior |
|-----------|--------------|------------|----------|
| fixed | `Rate.fixed()` | – | Constant learning rate. |
| step | `Rate.step(gamma?, stepSize?)` | gamma (default 0.9), stepSize (default 100) | Multiplies rate by `gamma` every `stepSize` iterations. |
| exp | `Rate.exp(gamma?)` | gamma (default 0.999) | Exponential decay: `rate * gamma^t`. |
| inv | `Rate.inv(gamma?, power?)` | gamma (0.001), power (2) | Inverse time decay: `rate / (1 + γ * t^p)`. |
| cosine annealing | `Rate.cosineAnnealing(period?, minRate?)` | period (1000), minRate (0) | Cosine decay from base to `minRate` each period. |
| cosine warm restarts | `Rate.cosineAnnealingWarmRestarts(initialPeriod, minRate?, tMult?)` | initialPeriod, minRate (0), tMult (2) | Cosine cycles with period multiplied by `tMult` after each restart. |
| linear warmup + decay | `Rate.linearWarmupDecay(totalSteps, warmupSteps?, endRate?)` | totalSteps, warmupSteps (auto 10%), endRate (0) | Linear ramp to base, then linear decay to `endRate`. |
| reduce on plateau | `Rate.reduceOnPlateau(opts)` | patience, factor, minRate, threshold | Monitors error; reduces current rate by `factor` after `patience` non-improving iterations. |

Notes:
* All scheduler factories return a function `(baseRate, iteration)` except `reduceOnPlateau`, which returns `(baseRate, iteration, error)`; `train` auto-detects and supplies `error` if the function arity is 3.
* You can wrap or compose schedulers—see below for a composition pattern.

#### Composing Schedulers (Warmup then Plateau)

You can combine policies by writing a small delegator that switches logic after warmup completes and still passes error when required:

```ts
import methods from './src/methods/methods';

const warmup = methods.Rate.linearWarmupDecay(500, 100, 0.1); // only use warmup phase portion
const plateau = methods.Rate.reduceOnPlateau({ patience: 15, factor: 0.5, minRate: 1e-5 });

// Hybrid policy: first 100 steps use linear ramp; afterward delegate to plateau (needs error)
const hybrid = (base: number, t: number, err?: number) => {
	if (t <= 100) return warmup(base, t); // ignore decay tail by cutting early
	// plateau expects error (3-arg); train will pass it because we define length >= 3 when we use 'err'
	return plateau(base, t - 100, err!); // shift iteration so plateau's patience focuses on post-warmup
};

net.train(data, { iterations: 2000, rate: 0.05, ratePolicy: hybrid });
```

For more elaborate chaining (e.g., staged cosine cycles then plateau), follow the same pattern: evaluate `t`, decide which inner policy to call, adjust `t` relative to that stage, and pass along `error` if the target policy needs it.


### Metrics Hook & Checkpoints

```ts
net.train(data, {
	iterations: 800,
	rate: 0.05,
	metricsHook: ({ iteration, error, gradNorm }) => console.log(iteration, error, gradNorm),
	checkpoint: {
		best: true,
		last: true,
		save: ({ type, iteration, error, network }) => {/* persist */}
	}
});
```

### DropConnect

```ts
net.enableDropConnect(0.3);
net.train(data, { iterations: 300, rate: 0.05 });
net.disableDropConnect();
```

### Advanced Optimizers

Supply `optimizer` to `train` as a simple string (uses defaults) or a config object.

Basic (defaults):
```ts
net.train(data, { iterations: 200, rate: 0.01, optimizer: 'adam' });
```

Custom AdamW:
```ts
net.train(data, {
	iterations: 500,
	rate: 0.005,
	optimizer: { type: 'adamw', beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01 }
});
```

Lion (sign-based update):
```ts
net.train(data, { iterations: 300, rate: 0.001, optimizer: { type: 'lion', beta1: 0.9, beta2: 0.99 } });
```

Adamax (robust to sparse large gradients):
```ts
net.train(data, { iterations: 300, rate: 0.002, optimizer: { type: 'adamax', beta1: 0.9, beta2: 0.999 } });
```

NAdam (Nesterov momentum style lookahead on first moment):
```ts
net.train(data, { iterations: 300, rate: 0.001, optimizer: { type: 'nadam', beta1: 0.9, beta2: 0.999 } });
```

RAdam (more stable early training variance):
```ts
net.train(data, { iterations: 300, rate: 0.001, optimizer: { type: 'radam', beta1: 0.9, beta2: 0.999 } });
```

AdaBelief (faster convergence / better generalization via surprise-based variance):
```ts
net.train(data, { iterations: 300, rate: 0.001, optimizer: { type: 'adabelief', beta1: 0.9, beta2: 0.999 } });
```

Lookahead (wraps a base optimizer; performs k fast steps then interpolates):
```ts
net.train(data, {
	iterations: 400,
	rate: 0.01,
	optimizer: { type: 'lookahead', baseType: 'adam', la_k: 5, la_alpha: 0.5 }
});
```

Optimizer reference:

| Optimizer   | Key Params (in object) | Notes |
|-------------|------------------------|-------|
| sgd         | momentum               | Nesterov momentum internally in propagate when update=true |
| rmsprop     | eps                    | Uses fixed decay 0.9 / 0.1 split for cache |
| adagrad     | eps                    | Cache accumulates squared grads (monotonic) |
| adam        | beta1, beta2, eps      | Standard bias correction |
| adamw       | beta1, beta2, eps, weightDecay | Decoupled weight decay applied after adaptive step |
| amsgrad     | beta1, beta2, eps      | Maintains max second-moment vhat |
| adamax      | beta1, beta2, eps      | Infinity norm (u) instead of v |
| nadam       | beta1, beta2, eps      | Nesterov variant of Adam |
| radam       | beta1, beta2, eps      | Rectifies variance early in training |
| lion        | beta1, beta2           | Direction = sign(beta1*m + beta2*m2) |
| adabelief   | beta1, beta2, eps      | Second moment of (g - m) (gradient surprise) |
| lookahead   | baseType, la_k, la_alpha | Interpolates toward slow weights every k steps |

General notes:
* Step counter resets each `train` call (t starts at 1) for reproducibility.
* `lookahead.baseType` defaults to `adam` and cannot itself be `lookahead`.
* Only AdamW applies decoupled `weightDecay`; for others combine regularization if needed.
* Adamax may help with sparse or bursty gradients (uses infinity norm).\
* NAdam can yield slightly faster early progress due to lookahead on m.\
* RAdam mitigates the need for warmup; behaves like Adam after variance rectification threshold.\
* AdaBelief can reduce over-adaptation to noisy gradients by modeling belief deviation.\
* Lion performs well in some large-scale settings due to sign-based memory efficiency.

### Label Smoothing

Cross-entropy with label smoothing discourages over-confident predictions.

```ts
import methods from './src/methods/methods';
const loss = methods.Cost.labelSmoothing([1,0,0],[0.8,0.1,0.1],0.1);
```

### Weight Noise

Adds zero-mean Gaussian noise to weights on each *training* forward pass (inference unaffected). Original weights are restored immediately after the pass (noise is ephemeral / non-destructive).

Basic usage:
```ts
net.enableWeightNoise(0.05); // global stdDev = 0.05
net.train(data, { iterations: 100, rate:0.05 });
net.disableWeightNoise();
```

Per-hidden-layer std deviations (layered networks only):
```ts
const layered = Architect.perceptron(4, 32, 16, 8, 2); // input, 3 hidden, output
layered.enableWeightNoise({ perHiddenLayer: [0.05, 0.02, 0.0] }); // third hidden layer noiseless
```

Dynamic schedule (e.g. cosine decay) for global std:
```ts
net.enableWeightNoise(0.1); // initial value (will be overridden by schedule each step)
net.setWeightNoiseSchedule(step => 0.1 * Math.cos(step / 500));
```

Deterministic seeding / custom RNG (affects dropout, dropconnect, stochastic depth, weight noise sampling):
```ts
net.setSeed(42); // reproducible stochastic regularization
// or provide a custom RNG
net.setRandom(() => myDeterministicGenerator.next());
```

Diagnostics:
* Each connection temporarily stores last noise in `connection._wnLast` (for test / inspection).
* Global training forward pass count: `net.trainingStep`.
* Last skipped layers (stochastic depth): `net.lastSkippedLayers`.
* Regularization statistics (after any forward): `net.getRegularizationStats()` returns
	`{ droppedHiddenNodes, totalHiddenNodes, droppedConnections, totalConnections, skippedLayers, weightNoise: { count, sumAbs, maxAbs, meanAbs } }`.
* RNG state: `net.snapshotRNG()` -> `{ step, state }`, restore seed/state via `net.setSeed(seed)` then `net.setRNGState(state)` for replay; or replace generator with `net.restoreRNG(fn)`.
* To combine with DropConnect / Dropout the sampling is independent (noise applied before masking).

Gotchas:
* Per-layer noise ignores global schedule (schedule currently applies only when using a single global std). If you need both, emulate by updating `enableWeightNoise` each epoch.
* Very large noise (> weight scale) can destabilize gradients; start small (e.g. 1-10% of typical weight magnitude).

### Stochastic Depth (Layer Drop)

Randomly skip (drop) entire hidden layers during training for deeper layered networks to reduce effective depth and encourage resilient representations.

```ts
const deep = Architect.perceptron(8,16,16,16,4,2); // input + 4 hidden + output
deep.setSeed(123); // reproducible skipping
deep.setStochasticDepth([0.9,0.85,0.8,0.75]); // survival probabilities per hidden layer
deep.train(data, { iterations:500, rate:0.01 });
deep.disableStochasticDepth(); // inference uses full depth
```

Runtime info:
* Recently skipped layer indices available via `(deep as any)._lastSkippedLayers` (for test / debugging).
* Surviving layer outputs are scaled by `1/p` to preserve expectation (like inverted dropout).
* Dynamic scheduling: `deep.setStochasticDepthSchedule((step, current) => current.map((p,i)=> Math.max(0.5, p - 0.0005*step)))` to slowly reduce survival probabilities (example).

Rules:
* Provide exactly one probability per hidden layer (input & output excluded).
* Probabilities must be in (0,1]. Use 1.0 to always keep a layer.
* A layer only skips if its input activation vector size equals its own size (simple identity passthrough). Otherwise it is forced to run to avoid shape mismatch.
* Stochastic depth and node-level dropout can co-exist; skipping occurs before dropout at deeper layers.
* Clear schedule: `deep.clearStochasticDepthSchedule()`.

Example combining schedule + stats capture:
```ts
deep.setSeed(2025);
deep.setStochasticDepth([0.9,0.85,0.8]);
deep.setStochasticDepthSchedule((step, probs) => probs.map(p => Math.max(0.7, p - 0.0001*step)));
for (let i=0;i<10;i++) {
	deep.activate(sampleInput, true);
	console.log(deep.getRegularizationStats());
}
```

### DropConnect (recap)
Already supported: randomly drops individual connections per training pass.

```ts
net.enableDropConnect(0.2);
net.train(data, { iterations:200, rate:0.02 });
net.disableDropConnect();
```
