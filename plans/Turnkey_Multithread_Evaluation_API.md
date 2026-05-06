# Turnkey Multithread Evaluation API Plan (Node + Browser Workers)

**Status:** [PLANNED]

## Purpose

Provide a first-class API to evaluate many networks **in parallel** using workers, targeting the core use case:

- evaluate a population each generation
- compute fitness scores
- keep results deterministic and reproducible

This plan builds on worker-friendly inference payloads.

## Goals

- G1: Simple, safe API to evaluate a batch of networks concurrently.
- G2: Support both Node (`worker_threads`) and browser (Web Workers).
- G3: Deterministic evaluation given a seed and fixed dataset.
- G4: Graceful fallback to single-thread evaluation.

## Non-goals

- Running arbitrary user functions inside workers by string-eval.
- Perfect “auto parallelization” of every NEAT workflow; focus on evaluation.

## Core design

### Separation of responsibilities

- Main thread:
  - chooses which networks to evaluate
  - serializes them to payloads
  - schedules work + collects fitnesses
- Worker:
  - receives a payload
  - runs `predict()`
  - computes and returns a numeric fitness (and optional metadata)

### Dataset handling

Two supported patterns:

1. **Dataset shipped once** at worker init (preferred)
2. Dataset shipped per task (simple but slower)

## Proposed public API

```ts
export interface WorkerPoolOptions {
  workerCount?: number;
  maxQueueSize?: number;
  idleTerminateMs?: number;
}

export interface EvaluationJobOptions {
  payloadMode?: 'json' | 'typed';
  numericPrecision?: 'full' | 'f32';
}

export interface FitnessResult {
  fitness: number;
  meta?: Record<string, number | string | boolean>;
}

export interface BatchEvaluationResult {
  results: FitnessResult[];
  elapsedMs: number;
}

export interface WorkerEvaluatorSpec {
  kind: 'node' | 'browser';
  workerEntry: string; // module path / URL
  initMessage?: unknown;
}

export async function evaluateInWorkers(
  networks: Network[],
  evaluator: WorkerEvaluatorSpec,
  options?: WorkerPoolOptions & EvaluationJobOptions,
): Promise<BatchEvaluationResult>;
```

Notes:

- `workerEntry` is user-provided to avoid bundling complexities and to avoid unsafe code generation.
- The worker script imports NeatapticTS’s worker predictor helper and a user fitness function.

## Worker-side protocol (v1)

Messages:

- `init`: optional; send dataset/config
- `task`: `{ taskId, payload }`
- `result`: `{ taskId, fitnessResult }`
- `error`: `{ taskId, message, stack? }`

Determinism requirements:

- If randomness is used inside fitness, it must be seeded from a deterministic seed passed in `initMessage`.

## Downstream unlock — NEATchat follow-up

This plan is one of the hard blockers for `plans/NEATchat.plans.md`.
NEATchat should not start worker-backed chat inference, candidate scoring, or
background adaptation until this plan exposes an execution surface that is more
concrete than a future pool outline.

For NEATchat, this plan is considered ready only when all of the following are
true:

1. Step 1 and Step 2 are complete, so both Node and browser worker pools exist
   with matching ordering semantics.
2. Results are guaranteed to return in input order with deterministic task
   scheduling or documented tie-breaking, so candidate comparison is
   replayable.
3. The API exposes a clear single-thread fallback path, because NEATchat must
   stay truthful in environments where workers are unavailable or disabled.
4. Step 3 is complete, so the worker template and init-task-result protocol are
   documented well enough for downstream chat jobs to reuse without inventing a
   NEATchat-specific worker contract.

This plan depends on
`plans/Worker_Friendly_Network_Serialization_Fastpath.md` for the payload and
predictor substrate. It does not replace checkpointing or parameter-vector
contracts, which remain separate NEATchat blockers.

## Recommended agent + skill combo by step

- Step 1 — `Evaluation Pool Scout` + `multithread-evaluation`
- Step 2 — `Evaluation Pool Scout` + `multithread-evaluation`
- Step 3 — `Evaluation Pool Scout` + `multithread-evaluation`
- Step 4 — `Evaluation Pool Scout` + `multithread-evaluation`

## Implementation steps

### Step 1 — Node worker pool

- Implement a small worker pool for Node.
- Support queueing tasks, backpressure, and shutdown.

Acceptance:

- Evaluates a batch and returns results in input order.

### Step 2 — Browser worker pool

- Implement browser worker pool with similar semantics.
- Keep API shape consistent.

Acceptance:

- Works in a minimal browser demo.

### Step 3 — Fitness worker template + docs

- Provide a documented “worker template” file users can copy:
  - imports `createWorkerPredictor`
  - loads dataset in init
  - handles tasks

Acceptance:

- Example works with a tiny XOR-style dataset.

### Step 4 — Integration with NEAT evolution loop (optional helper)

- Provide a helper that evaluates a population and assigns `score`/`fitness`.

Acceptance:

- NEAT loop can opt-in with minimal code.

## Testing strategy

- Unit tests:
  - pool schedules tasks correctly
  - results are stable and ordered
- Integration tests:
  - Node worker evaluation for a small batch
  - Browser worker evaluation (if test infra supports it; otherwise keep as docs-only initially)

## Risks and mitigations

- Risk: bundler friction for worker entry modules.
  - Mitigation: document patterns for Vite/Webpack; keep worker entry as user-controlled.
- Risk: sending large datasets repeatedly.
  - Mitigation: init-time dataset broadcast; consider SharedArrayBuffer later.

## Success criteria

- Users can parallelize evaluation with a small amount of glue code.
- Works in both Node and browser environments.
- Determinism story is clear and enforceable.
