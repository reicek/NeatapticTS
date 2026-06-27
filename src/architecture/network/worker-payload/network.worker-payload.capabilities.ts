/**
 * Capability snapshot for the current worker-backed inference host.
 *
 * This probe answers a narrow question: which worker transport tiers are
 * honestly usable in the current runtime before one evaluation pool or batch
 * helper commits to a transport strategy?
 *
 * @example
 * ```ts
 * const capabilities = detectInferenceWorkerCapabilities({
 *   crossOriginIsolated: globalThis.crossOriginIsolated === true,
 *   hasChannelWorker: true,
 *   hasSharedWorker: true,
 *   runtime: 'browser',
 * });
 * const transport = resolveAutoInferenceTransport(capabilities);
 * ```
 */
export interface InferenceWorkerCapabilities {
  /** Whether one persistent channel worker can be opened for repeated inference calls. */
  readonly inferenceChannel: boolean;
  /** Human-readable explanations for unavailable higher transport tiers. */
  readonly reasons: string[];
  /** Host runtime resolved for the capability probe. */
  readonly runtime: 'browser' | 'node';
  /** Whether the shared-memory worker tier is honestly usable. */
  readonly sharedMemory: boolean;
  /** Whether typed-array payload export remains available as the universal fallback. */
  readonly transferable: true;
}

/**
 * Inputs used to probe worker-backed inference transport support.
 *
 * Keep delivery facts explicit here. Step 1 only answers capability and auto
 * selection; Step 2 will own the generic worker-entry and URL resolution
 * helpers that feed these booleans.
 *
 * @example
 * ```ts
 * const capabilities = detectInferenceWorkerCapabilities({
 *   crossOriginIsolated: false,
 *   hasChannelWorker: true,
 *   hasSharedWorker: true,
 *   runtime: 'browser',
 * });
 * ```
 */
export interface InferenceWorkerCapabilityOptions {
  /** Browser-only delivery fact for one channel worker entry. */
  readonly hasChannelWorker?: boolean;
  /** Browser-only delivery fact for one shared-memory worker entry. */
  readonly hasSharedWorker?: boolean;
  /** Browser-only cross-origin isolation status for SharedArrayBuffer use. */
  readonly crossOriginIsolated?: boolean;
  /** Optional runtime override for tests or mixed-host probes. */
  readonly runtime?: 'auto' | 'browser' | 'node';
  /** Optional SharedArrayBuffer availability override. */
  readonly sharedArrayBufferAvailable?: boolean;
  /** Optional Worker constructor availability override for browser hosts. */
  readonly workerConstructorAvailable?: boolean;
}

/**
 * Automatically selected worker-backed inference transport tier for the current host.
 *
 * - `'shared-memory'` — SharedArrayBuffer transfer path when cross-origin isolation is active.
 * - `'channel'` — Persistent channel worker when a channel-worker script was delivered.
 * - `'transferable'` — Universal typed-array fallback when higher tiers are unavailable.
 */
export type AutoInferenceTransport =
  'channel' | 'shared-memory' | 'transferable';

/**
 * Detect the usable worker-backed inference transport tiers for one host.
 *
 * The probe stays intentionally lightweight. It does not open a worker or fetch
 * a script; it only combines runtime facts and delivery availability so callers
 * can decide whether to use shared-memory, channel, or transferable fallback.
 *
 * @param options - Runtime and delivery facts for the current host.
 * @returns Capability snapshot describing the usable transport tiers.
 * @example
 * ```ts
 * const capabilities = detectInferenceWorkerCapabilities({
 *   hasChannelWorker: true,
 *   hasSharedWorker: true,
 *   runtime: 'browser',
 * });
 * ```
 */
export function detectInferenceWorkerCapabilities(
  options: InferenceWorkerCapabilityOptions = {},
): InferenceWorkerCapabilities {
  const runtime = resolveWorkerCapabilityRuntime(options.runtime);

  return runtime === 'browser'
    ? detectBrowserInferenceWorkerCapabilities(options)
    : detectNodeInferenceWorkerCapabilities(options);
}

/**
 * Resolve the honest automatic transport choice for one capability snapshot.
 *
 * The priority order matches the current transport ladder: shared-memory first,
 * then persistent channels, then transferable payload fallback.
 *
 * @param capabilities - Capability snapshot returned by the probe.
 * @returns Best automatic transport choice for the current host.
 * @example
 * ```ts
 * const capabilities = detectInferenceWorkerCapabilities({
 *   hasChannelWorker: true,
 *   runtime: 'browser',
 * });
 * const transport = resolveAutoInferenceTransport(capabilities);
 * ```
 */
export function resolveAutoInferenceTransport(
  capabilities: InferenceWorkerCapabilities,
): AutoInferenceTransport {
  // Step 1: Prefer the highest transport tier the host can use honestly.
  if (capabilities.sharedMemory) {
    return 'shared-memory';
  }

  // Step 2: Reuse one warm worker channel when shared memory is unavailable.
  if (capabilities.inferenceChannel) {
    return 'channel';
  }

  // Step 3: Keep the typed-array payload fallback explicit.
  return 'transferable';
}

function detectBrowserInferenceWorkerCapabilities(
  options: InferenceWorkerCapabilityOptions,
): InferenceWorkerCapabilities {
  const reasons: string[] = [];
  const workerConstructorAvailable =
    options.workerConstructorAvailable ??
    typeof globalThis.Worker === 'function';
  const hasChannelWorker = options.hasChannelWorker ?? false;
  const hasSharedWorker = options.hasSharedWorker ?? false;
  const sharedArrayBufferAvailable =
    options.sharedArrayBufferAvailable ??
    typeof globalThis.SharedArrayBuffer === 'function';
  const crossOriginIsolated =
    options.crossOriginIsolated ?? globalThis.crossOriginIsolated === true;
  const inferenceChannel = workerConstructorAvailable && hasChannelWorker;
  const sharedMemory =
    workerConstructorAvailable &&
    sharedArrayBufferAvailable &&
    crossOriginIsolated &&
    hasSharedWorker;

  if (!inferenceChannel) {
    pushReason(
      reasons,
      workerConstructorAvailable
        ? 'InferenceChannel browser transport needs one worker delivery path.'
        : 'InferenceChannel browser transport requires Worker support in the host runtime.',
    );
  }

  if (!sharedMemory) {
    pushReason(
      reasons,
      resolveSharedMemoryUnavailableReason({
        crossOriginIsolated,
        hasSharedWorker,
        sharedArrayBufferAvailable,
        workerConstructorAvailable,
      }),
    );
  }

  return {
    inferenceChannel,
    reasons,
    runtime: 'browser',
    sharedMemory,
    transferable: true,
  };
}

function detectNodeInferenceWorkerCapabilities(
  options: InferenceWorkerCapabilityOptions,
): InferenceWorkerCapabilities {
  const reasons: string[] = [];
  const sharedArrayBufferAvailable =
    options.sharedArrayBufferAvailable ??
    typeof globalThis.SharedArrayBuffer === 'function';

  if (!sharedArrayBufferAvailable) {
    pushReason(
      reasons,
      'Shared-memory inference requires SharedArrayBuffer support in the host runtime.',
    );
  }

  return {
    inferenceChannel: true,
    reasons,
    runtime: 'node',
    sharedMemory: sharedArrayBufferAvailable,
    transferable: true,
  };
}

function resolveWorkerCapabilityRuntime(
  runtime: InferenceWorkerCapabilityOptions['runtime'] = 'auto',
): 'browser' | 'node' {
  if (runtime === 'browser' || runtime === 'node') {
    return runtime;
  }

  return typeof process !== 'undefined' && typeof process.cwd === 'function'
    ? 'node'
    : 'browser';
}

function resolveSharedMemoryUnavailableReason(options: {
  crossOriginIsolated: boolean;
  hasSharedWorker: boolean;
  sharedArrayBufferAvailable: boolean;
  workerConstructorAvailable: boolean;
}): string {
  if (!options.workerConstructorAvailable) {
    return 'Shared-memory inference requires Worker support in the host runtime.';
  }

  if (!options.sharedArrayBufferAvailable) {
    return 'Shared-memory inference requires SharedArrayBuffer support in the host runtime.';
  }

  if (!options.crossOriginIsolated) {
    return 'Shared-memory inference requires crossOriginIsolated=true in browser hosts.';
  }

  return 'Shared-memory inference needs one shared worker delivery path in this host.';
}

function pushReason(reasons: string[], reason: string): void {
  reasons.push(reason);
}
