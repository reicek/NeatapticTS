# neat/hybrid

Hybrid evaluation joins deterministic parameter vectors, isolated gradient
updates, and policy-aware scoring into one explicit workflow.

This boundary is where you decide whether learning is only a temporary aid
for scoring or a change that should propagate back into the population. In
this module, "Lamarckian" means trained weights are written back to the live
candidate, either by calling `fromParameterVector(...)` yourself or by using
`persistTrainedWeights: true` with `evaluateCandidate(...)`. Fitness-only
scoring is closer to the Baldwin effect: learning changes the phenotype that
gets scored, but the genotype that remains in the population is unchanged.
See Wikipedia contributors,
[Lamarckism](https://en.wikipedia.org/wiki/Lamarckism), and Wikipedia
contributors,
[Baldwin effect](https://en.wikipedia.org/wiki/Baldwin_effect), for the
historical vocabulary behind that distinction.

The determinism promise is intentionally laddered instead of absolute:

1. Same-runtime ordered deterministic when topology, dataset order, training
   settings, and an explicit `seed` all match.
2. Best-effort reproducible when dataset order and settings match but no
   `seed` is supplied, so stochastic training features may still drift.
3. Cross-runtime exact replay is out of scope for this lane; matching
   worker, browser, and Node results still depends on broader replay
   contracts.

Recommended progression:

1. Start with `fineTune: 'never'` to measure the pure evolutionary
   baseline.
2. Move to `fineTune: 'always'` with `persistTrainedWeights: false` when you
   want exploratory scoring without mutating the population.
3. Opt into `persistTrainedWeights: true` only when Lamarckian
   carry-forward is an explicit design choice.

`fineTune: 'conditional'` remains blocked until the evaluation boundary
exposes a deterministic ranking or tie-break rule. Without that ordering
contract, "train only the best candidates" is ambiguous.

Example:

```ts
import {
  Network,
  evaluateCandidate,
  fineTuneVector,
  fromParameterVector,
  toParameterVector,
} from 'neataptic';

const candidate = new Network(2, 1, { seed: 42 });
const dataset = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];
const fineTuneOptions = { steps: 50, learningRate: 0.05, seed: 123 };

// Inspect a detached trained vector first.
const baselineVector = toParameterVector(candidate);
const { trainedVector, metrics } = fineTuneVector(
  candidate,
  baselineVector,
  dataset,
  fineTuneOptions,
);
console.log('training error:', metrics?.error);

// Materialize the trained variant on a clone so the live candidate stays untouched.
const inspectedCandidate = candidate.clone();
fromParameterVector(inspectedCandidate, trainedVector);

// Score the trained variant without persisting it back to the population.
const fitnessOnly = await evaluateCandidate(candidate, dataset, {
  policy: { fineTune: 'always', persistTrainedWeights: false },
  fineTuneOptions,
  scoreNetwork: (network) => -network.test(dataset, { cost: 'mse' }).error,
});
console.log('fitness-only score:', fitnessOnly.fitness);

// Opt into Lamarckian persistence only when you intentionally want it.
if (fitnessOnly.fitness > -0.05) {
  fromParameterVector(candidate, trainedVector);
}
```

## neat/hybrid/neat.hybrid.types.ts

### EvaluateCandidateOptions

Inputs for one standalone hybrid candidate evaluation.

`fineTuneOptions` is required whenever `policy.fineTune !== 'never'`
because this helper delegates training to `fineTuneVector(...)` rather than
guessing learning settings.

### HybridEvaluationPolicy

Explicit hybrid policy for one candidate evaluation.

Recommended progression keeps the policy easy to reason about: start with
`fineTune: 'never'` to measure the evolutionary baseline, move to
`fineTune: 'always'` with `persistTrainedWeights: false` for fitness-only
Baldwin-style scoring, and opt into `persistTrainedWeights: true` only when
deliberate Lamarckian carry-forward is part of the experiment. `conditional`
remains blocked until the caller can supply a deterministic ranking or
tie-break contract.

Persistence stays separate from the fine-tune trigger so fitness-only
scoring remains the safe default. Callers should keep
`persistTrainedWeights` false unless they explicitly want Lamarckian
persistence after a successful score.

### HybridEvaluationResult

Result from one hybrid candidate evaluation pass.

`trainedNetwork` is present only when fine-tuning runs. Fitness-only callers
can inspect the detached trained variant without mutating the canonical
candidate, while Lamarckian callers receive the same trained snapshot that
was scored before explicit persistence is applied.

Typical downstream uses include: forwarding `fitness` to the NEAT population
score, comparing `trainedNetwork` weights against the original candidate to
measure fine-tune delta, checkpointing the trained snapshot, or discarding
the result entirely when only the fitness score matters.

### HybridFineTuneMode

Explicit fine-tune trigger for one hybrid evaluation pass.

- `never` evaluates the live candidate as-is without calling any training helper.
- `always` fine-tunes a detached clone before scoring; whether trained weights
  persist back is controlled separately by `HybridEvaluationPolicy.persistTrainedWeights`.
- `conditional` requires a deterministic ranking or tie-break contract at the
  evaluation boundary before it can be unblocked. Passing `conditional` to
  `evaluateCandidate` throws at runtime until that surface exists.

### HybridScoreNetwork

```ts
HybridScoreNetwork(
  candidate: default,
): number | Promise<number>
```

Scoring callback used after the helper resolves the network state to score.

The callback may be synchronous or async, but it should treat the supplied
network as the exact candidate state selected by the hybrid policy.

## neat/hybrid/neat.hybrid.ts

### evaluateCandidate

```ts
evaluateCandidate(
  network: default,
  dataset: TrainingSample[],
  options: EvaluateCandidateOptions,
): Promise<HybridEvaluationResult>
```

Evaluate one candidate under an explicit hybrid fine-tune policy.

The helper keeps policy decisions visible: `fineTune` controls whether
training runs, `scoreNetwork` decides how the selected network state is
scored, and `persistTrainedWeights` controls whether the trained vector is
applied back to the original candidate after a successful score.
`persistTrainedWeights: false` is the safe default because the trained
variant is otherwise discarded after scoring.

`conditional` remains blocked until the NEAT evaluation surface exposes a
deterministic ranking or tie-break contract. This helper inherits the
same-runtime ordered determinism limits of `fineTuneVector(...)`; it does
not claim cross-runtime exact replay.

```ts
import { Network, evaluateCandidate } from 'neataptic';

const network = new Network(2, 1);
const dataset = [
  { input: [0, 0], output: [0] },
  { input: [1, 1], output: [0] },
  { input: [1, 0], output: [1] },
  { input: [0, 1], output: [1] },
];

// Fitness-only: fine-tune a detached clone, score it, discard trained weights.
const fitnessOnly = await evaluateCandidate(network, dataset, {
  policy: { fineTune: 'always', persistTrainedWeights: false },
  fineTuneOptions: { steps: 50, learningRate: 0.01, seed: 42 },
  scoreNetwork: (candidate) => -candidate.test(dataset, { cost: 'mse' }).error,
});
console.log('fitness:', fitnessOnly.fitness); // original candidate unchanged

// Lamarckian: apply trained weights back to the candidate on explicit opt-in.
const lamarckian = await evaluateCandidate(network, dataset, {
  policy: { fineTune: 'always', persistTrainedWeights: true },
  fineTuneOptions: { steps: 50, learningRate: 0.01, seed: 42 },
  scoreNetwork: (candidate) => -candidate.test(dataset, { cost: 'mse' }).error,
});
console.log('fitness after persistence:', lamarckian.fitness);
```

Parameters:

- `network` - Live candidate selected by the caller's fitness delegate.
- `dataset` - Ordered training samples passed to `fineTuneVector(...)` when training runs.
- `options` - Explicit policy, training settings, and scoring callback.

Returns: Fitness plus the detached trained network when fine-tuning runs.
