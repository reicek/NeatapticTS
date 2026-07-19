/**
 * Generic network acceleration layer for NeatapticTS.
 *
 * Evolutionary training spends most of its time evaluating genomes against a
 * task. The acceleration layer turns that evaluation bottleneck into a portable,
 * observable policy decision: run on CPU, dispatch to WebGPU, or spread across
 * worker threads, depending on network size, batch shape, and environment.
 *
 * The layer is intentionally split into small, side-effect-free pieces:
 * - `acceleration.types` — the portable vocabulary of modes, capabilities, and
 *   configuration.
 * - `acceleration.constants` — conservative default thresholds and caps,
 *   including the dynamic buffer-pool cap heuristic
 *   {@link resolveBufferPoolMaxPooledBytes} used by the GPU buffer-set pool.
 * - `acceleration.config` — `resolveAccelerationConfig()` fills in defaults
 *   without touching the runtime.
 * - `acceleration.detect` — `detectAcceleration()` probes the host for WebGPU
 *   and worker availability.
 * - `acceleration.resolve` — `resolveAccelerationMode()` turns capability
 *   reports into an active backend choice.
 * - `acceleration.policy` — `AccelerationPolicy` and
 *   `LifecycleAccelerationPolicy` encode reusable decision rules.
 * - `acceleration.observer` — injectable telemetry so callers can track backend
 *   transitions, fallbacks, and timing.
 * - `acceleration.gpu` — `autoEnableGpu()` and `shouldAutoEnableGpu()` decide
 *   when a network is large enough to justify WebGPU offload.
 * - `acceleration.gpu.device` — `requestGPUDevice()` and `isDeviceReady()`
 *   bootstrap and monitor the shared WebGPU device used by GPU acceleration.
 * - `acceleration.workers` — `autoEnableWorker()` and `shouldAutoEnableWorker()`
 *   decide when worker-thread evaluation is the right fit.
 * - `acceleration.variants` — `evaluateWeightVariantsAsync()` scores candidate
 *   weight perturbations on any network surface without mutating the original
 *   weights.
 * - `acceleration.orchestrator` — `autoEnableAcceleration()` combines the GPU
 *   and worker probes into a single backend choice.
 * - `acceleration.manager` — `AccelerationManager` owns the acceleration
 *   lifecycle for one network: init, enable, disable, re-evaluate, and teardown.
 * - `workerPoolLifecycle` — `createWorkerPoolLifecycle()` creates, reuses, and
 *   tears down a scoped pool of worker threads. Higher-level callers (for
 *   example the worker-payload dispatcher) import this handle instead of
 *   spawning workers directly, keeping the dependency direction clean.
 * - `acceleration.benchmark` — `runRegressionBenchmark()` runs a deterministic
 *   CPU vs GPU micro-benchmark and blacklists the GPU backend when it is
 *   slower than CPU by a configured ratio. This module is internal to
 *   `src/acceleration/` and is not exported from the public barrel.
 *
 * The public API follows a predictable pipeline: a partial config is resolved to
 * full defaults, the environment is probed, the active mode is chosen, and a
 * policy produces the final backend decision. CPU is always available as a
 * safe fallback, and every unavailable backend reports a human-readable reason.
 * The regression guard (`runRegressionBenchmark`) can blacklist the GPU backend
 * when the current environment's GPU path is slower than CPU for the reference
 * workload, keeping the default `auto` path safe on machines where GPU setup
 * overhead dominates. When the chosen backend is `worker`,
 * `createWorkerPoolLifecycle()` provides the actual pool of threads that the
 * inference layer broadcasts to.
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *   classDef pool fill:#0a1f2e,stroke:#00d4aa,color:#e0fff7,stroke-width:1.5px;
 *
 *   Config[Partial user config]:::base --> ResolveConfig[resolveAccelerationConfig]:::accent
 *   ResolveConfig --> Detect[detectAcceleration]:::accent
 *   Detect --> Status{Capability reports}
 *   Status --> GPU{{GPU available?}}
 *   Status --> Worker{{Workers available?}}
 *   Status --> CPU[CPU fallback]:::base
 *   GPU --> ResolveMode[resolveAccelerationMode]:::accent
 *   Worker --> ResolveMode
 *   CPU --> ResolveMode
 *   ResolveMode --> Policy[AccelerationPolicy]:::accent
 *   Policy --> Active[Active backend]:::base
 *   Active --> AutoEnable[autoEnableAcceleration]:::accent
 *   AutoEnable --> Manager[AccelerationManager]:::accent
 *   Manager --> Observer[AccelerationObserver telemetry]:::base
 *   Active -->|backend: worker| Pool[createWorkerPoolLifecycle]:::pool
 *   Pool --> WorkerSet[Worker pool handle]
 *   WorkerSet --> Broadcast[broadcast / terminate]
 * ```
 *
 * @example
 * Probe the environment and inspect the chosen backend:
 * ```ts
 * import {
 *   detectAcceleration,
 *   resolveAccelerationMode,
 *   AccelerationPolicy,
 *   LifecycleAccelerationPolicy,
 *   NoopAccelerationObserver,
 *   type AccelerationObserver,
 * } from '../acceleration';
 *
 * const detected = detectAcceleration({}, 2048);
 * const resolved = resolveAccelerationMode(detected, { backend: 'auto' });
 * const decision = new AccelerationPolicy().decide(resolved);
 * console.log(decision.backend); // 'gpu', 'worker', or 'cpu'
 * ```
 *
 * @example
 * Track backend transitions with an observer:
 * ```ts
 * const observer: AccelerationObserver = {
 *   onBackendChange: (event) =>
 *     console.log(`${event.previous} -> ${event.current}: ${event.reason}`),
 * };
 * const policy = new LifecycleAccelerationPolicy({ backend: 'auto' });
 * policy.onMutated();
 * const status = policy.evaluate({ backend: 'auto' }, 2048, observer);
 * ```
 *
 * @example
 * Let the library auto-select a backend for a 2048-node network:
 * ```ts
 * import { autoEnableAcceleration } from '../acceleration';
 *
 * const status = await autoEnableAcceleration({ nodeCount: 2048 });
 * console.log(status.mode); // 'gpu', 'worker', or 'cpu'
 * console.log(status.gpu.reason); // human-readable GPU decision
 * ```
 *
 * @example
 * Own the acceleration lifecycle with a manager:
 * ```ts
 * import { AccelerationManager } from '../acceleration';
 *
 * const manager = new AccelerationManager({ config: { backend: 'auto' } });
 * await manager.init(2048);
 * console.log(manager.getStatus().mode);
 * await manager.reEvaluate(4096); // re-check after a topology mutation
 * await manager.teardown();
 * ```
 *
 * @example
 * Create and reuse a scoped worker pool:
 * ```ts
 * import {
 *   createWorkerPoolLifecycle,
 *   type WorkerPoolHandle,
 * } from '../acceleration';
 *
 * const lifecycle = createWorkerPoolLifecycle({ maxWorkers: 4 });
 * const pool: WorkerPoolHandle = await lifecycle.create();
 * console.log(lifecycle.activeCount()); // up to 4
 * pool.broadcast({ kind: 'ping' });
 * await pool.terminate();
 * ```
 */

export * from './acceleration.types';
export * from './acceleration.constants';
export * from './acceleration.variants';
export * from './acceleration.config';
export * from './acceleration.observer';
export * from './acceleration.detect';
export * from './acceleration.resolve';
export * from './acceleration.policy';
export * from './acceleration.gpu';
export * from './acceleration.gpu.device';
export * from './acceleration.workers';
export * from './acceleration.orchestrator';
export * from './acceleration.manager';
export * from './workerPoolLifecycle';
