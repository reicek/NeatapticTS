# Worker Transport Reference Notes

This file stores paraphrased implementation notes for the
`worker-inference-transport` skill.

These notes intentionally summarize upstream documentation instead of copying it
verbatim. Follow the upstream links for canonical wording and full API details.

## Why These Sources

Phase 4 worker transport needs three kinds of external knowledge:

- web worker lifecycle and delivery constraints,
- message or memory transport semantics,
- Node-specific worker-thread differences that are easy to miss when browser
  and Node APIs look similar.

## Source Map

### 1. MDN Web Workers guide

- URL:
  https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers
- Authors: MDN contributors
- License note: MDN content is published under a Creative Commons attribution
  and share-alike license. See MDN's license page:
  https://developer.mozilla.org/docs/MDN/Writing_guidelines/Attrib_copyright_license
- Why it matters:
  - Workers run in a separate global scope and cannot touch the DOM directly.
  - Messages are cloned by default, transferred when explicitly moved, and only
    shared for explicit shared-memory objects.
  - Blob or data-url workers inherit CSP differently from separately hosted
    worker scripts, which matters for the `workerUrl` override and inline blob
    defaults.
  - Workers are useful for CPU-bound work, not UI work.

### 2. MDN MessageChannel reference

- URL: https://developer.mozilla.org/en-US/docs/Web/API/MessageChannel
- Authors: MDN contributors
- License note: same MDN license page as above.
- Why it matters:
  - `MessageChannel` creates a dedicated two-port channel that is widely
    available in browsers and Node.
  - A channel gives a stable protocol surface that is cleaner than overloading
    the default worker-global message channel.
  - Port transfer is explicit, which makes ownership and warm predictor setup
    easier to reason about.

### 3. Node.js worker_threads API

- URL: https://nodejs.org/api/worker_threads.html
- Upstream code and docs source: https://github.com/nodejs/node
- License note: consult the upstream repository for the current project
  license and terms before reusing source text.
- Why it matters:
  - Node worker threads use structured clone semantics similar to the web, but
    Node exposes extra failure cases around `Buffer` pooling and transfer lists.
  - `MessageChannel` and `MessagePort` exist in Node and should be preferred for
    custom protocols instead of overloading the default worker channel.
  - `markAsUntransferable(...)` documents a real pitfall: not every backing
    buffer should be transferred.
  - `worker.performance.eventLoopUtilization()` and worker lifecycle events are
    useful for focused diagnostics and transport benchmarks.

## Implementation Notes

### Structured clone vs transfer vs shared memory

- Clone:
  - simplest and most portable,
  - preserves data values, not prototypes or accessors,
  - cost grows with payload size.
- Transfer:
  - moves ownership of an `ArrayBuffer`,
  - sender-side views become unusable after transfer,
  - safest when the code owns the exact typed-array storage.
- Shared memory:
  - both sides can observe the same bytes,
  - requires explicit synchronization and feature gating,
  - should remain opt-in with a lower-tier fallback.

### Node-specific transport cautions

- Do not assume a `Buffer` owns its backing memory.
- Avoid transferring storage that may come from a pooled `Buffer`.
- Treat `'messageerror'` as a real validation case, not a theoretical one.
- Prefer pool or channel reuse over one-worker-per-request when benchmarking or
  shipping repeated CPU-bound inference.

### Browser-specific delivery cautions

- Worker creation should be compatible with bundler-controlled URLs or an
  explicit `workerUrl` override.
- Blob workers are useful, but CSP must be considered explicitly.
- Same-origin rules apply to spawned worker scripts and subworkers.
- Shared memory paths need cross-origin isolation before they can be treated as
  available.

## Working Heuristics For This Repo

- Keep `PortableInferencePayload` as the truth-preserving fallback.
- Use `TransferableInferencePayload` when benchmarking shows copy cost is worth
  paying with extra ownership discipline.
- Use `InferenceChannel` when warm worker-side state or repeated requests matter
  more than one-off startup simplicity.
- Use `SharedInferenceWorker` only after the feature gate, fallback path, and
  synchronization protocol are all explicit in the public contract.