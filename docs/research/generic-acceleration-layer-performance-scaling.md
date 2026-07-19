# Generic Acceleration Layer — Performance/Scaling Review Evidence

## Question

Is the `plans/Generic_Acceleration_Layer.plans.md` performance/scaling design concrete enough to implement, and is it consistent with the existing GPU/worker infrastructure it builds on? In particular, does the async variant evaluator contract contain everything needed to actually dispatch and score weight variants across GPU batch, worker pool, and CPU backends?

Originating step: Phase 1 Step 04 performance/scaling review of `plans/Generic_Acceleration_Layer.plans.md`.

## Evidence

### 1. Current variant evaluator is synchronous and never dispatches to GPU/workers
- **Source:** `src/performance/nge/nge.acceleration.variants.ts`, lines 16–19, 256, 345, 369–370.
- **Strength:** Static source code; JSDoc explicitly admits the limitation.
- **Finding:** The function always evaluates variants on CPU and reports `gpuUsed: false`, `workersUsed: false`. This validates the plan's stated problem and the need for `evaluateWeightVariantsAsync`.

### 2. Existing worker pool already supports ordered batch evaluation and slot reuse
- **Source:** `src/architecture/network/worker-payload/network.worker-payload.pool.ts`, lines 66–206.
- **Strength:** Static source code.
- **Finding:** `ParallelInferencePool.evaluateOrderedBatch` is async, uses reference-identity slot reuse (`slot.assignedPayload === payload`), and supports configurable worker count. The plan's centralization goal can reuse this primitive.

### 3. GPU buffer pool cap is currently hard-coupled to NGE constants
- **Source:** `src/architecture/network/gpu/network.gpu.buffer-set-pool.ts`, line 25, lines 131–133.
- **Strength:** Static source code.
- **Finding:** The pool imports `NGE_GROW_STABILIZE_BUFFER_POOL_MAX_POOLED_BYTES` from NGE constants. The plan's claim that the cap is NGE-coupled and needs decoupling is correct.

### 4. GPU eligibility predicate and dispatch seam already exist
- **Source:** `src/architecture/network/gpu/network.gpu.fallback.ts`, lines 58–68, 98–109.
- **Strength:** Static source code.
- **Finding:** `isGPUEligible` and `dispatchActivation` give the generic layer a real predicate to reuse. The plan's auto-enable policy can build on this.

### 5. Scaling tiers, config defaults, and cost/benefit formulas are concrete
- **Source:** `plans/Generic_Acceleration_Layer.plans.md`, lines 2131–2167, 2187–2238.
- **Strength:** Plan documentation.
- **Finding:** The plan lists concrete numeric defaults, override paths, a scaling-tier table, lifecycle-stage mapping, and explicit `gpuCost`/`workerCost`/`cpuCost` formulas. These are implementable as written.

### 6. Async variant evaluator contract lacks the evaluation task
- **Source:** `plans/Generic_Acceleration_Layer.plans.md`, lines 238–279; current `nge.acceleration.variants.ts` signature and `examples/racing_curriculum/controller/runtime.adaptation.ts` lines 353–379.
- **Strength:** Static source code + plan documentation; conflict between what the current implementation needs and what the new API declares.
- **Finding:** The plan's `WeightVariant` only contains `weightIndex` and `delta`. The `VariantEvaluator` / `evaluateWeightVariantsAsync` signatures accept a network, variants, seed, config, and observer, but they do **not** accept the input samples, target/reference output, or a scoring function. The current NGE callback in the racing demo constructs `inputs` and `target` locally and calls `evaluateWeightVariants(network, inputs, target, variantCount, options)`. The current implementation also needs a scorer to compute `bestScore` / `scores`. Without these parameters, the generic function cannot compute scores.
- **Conflict:** Plan AC-005 says the implementation "actually routes evaluation to the GPU batch path" and returns `bestScore`/`scores`, but the API surface does not provide the data or scoring logic required to produce those values.
- **Tie-break:** Static source code of the existing evaluator and its real call site take precedence over the plan's type declarations, because they represent the actual runtime contract that must be preserved.

### 7. GPU-side scoring kernel does not currently exist
- **Source:** `src/architecture/network/gpu/network.gpu.activate.ts` and `src/architecture/network/gpu/network.gpu.batched.ts`.
- **Strength:** Static source code.
- **Finding:** The existing GPU kernels compute network activations only. The plan's strategy mentions "run scoring on GPU and read back only the scalar score" as an optimization, but no GPU scoring kernel is described or present. This is not a functional blocker because CPU scoring after readback can be used, but it is a residual risk for the "avoid readback" optimization.

## Decision

The performance/scaling design is **largely concrete and consistent** with existing infrastructure, but it is **not ready for implementation** because the async variant evaluator API is incomplete.

- **APPROVED areas:** GPU auto-enable policy, worker pool centralization, dynamic buffer pool scaling, scaling tiers and dispatch heuristics, GPU readback/resident-memory strategy, and config-overridable cost/benefit model.
- **BLOCKING area:** `evaluateWeightVariantsAsync` and `VariantEvaluator` must add the evaluation task surface (inputs, target/reference, and a scoring function) before Phase 5 can implement AC-005 safely. The NGE adapter (`src/neat/nge-juvenile/neat.nge-juvenile.variants.ts`) must then consume that surface from the lifecycle context.

Recommended fix:
1. Add `inputs: number[][]` (or a matrix/encoder abstraction), `target?: number[]`, and `score: (actual: number[] | Float32Array, target?: number[]) => number` to `VariantEvaluator` and `evaluateWeightVariantsAsync`.
2. Update `WeightVariantResult` and `VariantEvaluationNetwork` documentation to describe how scoring is applied.
3. Provide a default scorer (e.g., negative MSE against `target`) so the API remains usable for library callers without NGE.
4. Update the NGE adapter to supply `inputs`, `target`, and the lifecycle scorer to the generic call.

## Risks

| Risk | Owner | Mitigation |
|---|---|---|
| GPU-side scoring kernel does not exist, so the "avoid readback" optimization may require a follow-up GPU compute kernel. | Phase 7 / 01-planning | Keep Phase 5 readback path on CPU scoring; document Phase 7 kernel as future enhancement. |
| NGE adapter may re-introduce NGE-coupled inputs/target/scorer construction if the generic API does not expose them. | Phase 5 implementation | Add explicit `inputs`/`target`/`score` parameters to the generic contract. |
| Dynamic buffer pool cap uses `device.limits.maxBufferSize / 4`, which is a per-buffer limit, not a memory budget. | Phase 7 implementation | Validate on real devices; provide `maxPooledBytes` override. |
| Worker pool centralization changes ownership; premature disposal during long NGE runs could hurt performance. | Phase 5 implementation / tests | Add lifecycle ownership tests and racing demo E2E. |

