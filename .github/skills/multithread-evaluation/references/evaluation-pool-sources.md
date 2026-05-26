# Evaluation Pool Reference Notes

This file stores paraphrased reference notes for the
`multithread-evaluation` skill.

These notes summarize upstream material rather than copying it verbatim. Use the
linked sources for canonical wording and full API details.

## Source Map

### 1. Node.js async context and `AsyncResource`

- URL:
  https://nodejs.org/api/async_context.html#using-asyncresource-for-a-worker-thread-pool
- Upstream repo: https://github.com/nodejs/node
- Why it matters:
  - Node explicitly documents a worker-pool pattern where each scheduled task is
    represented by an `AsyncResource` so callback correlation follows task
    lifetime rather than worker lifetime.
  - This is the right mental model for task-level observability in a pool.
  - It also reinforces that pool reuse is preferable to spawning one worker per
    short task.

### 2. MDN Web Workers guide

- URL:
  https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers
- Authors: MDN contributors
- License note: MDN content is available under a Creative Commons attribution
  and share-alike license. See:
  https://developer.mozilla.org/docs/MDN/Writing_guidelines/Attrib_copyright_license
- Why it matters:
  - Browser workers are best for CPU-heavy background work.
  - Feature detection and worker URL delivery patterns matter for browser pool
    portability.
  - Workers communicate by message passing, so dataset and result movement costs
    must be acknowledged in the API.

### 3. Wikipedia overview of work stealing

- URL: https://en.wikipedia.org/wiki/Work_stealing
- Authors: Wikipedia contributors
- License note: Wikipedia text is available under CC BY-SA 4.0. See the page
  footer and Wikimedia terms.
- Why it matters:
  - Work stealing is a useful conceptual reference for advanced scheduling, but
    it is not automatically the right first implementation for this repo.
  - The fork-join theory highlights why simple ordered FIFO baselines are often
    easier to validate before more dynamic queue policies are attempted.

## Practical Notes

### Ordered result collection

- Worker completion order and public result order should be separated.
- The simplest stable strategy is to assign each input a logical index and write
  the worker result back to that index regardless of completion time.

### Worker count

- For CPU-bound work, a fixed-size pool near available parallelism is a good
  starting point.
- For small batches, startup and messaging overhead can dominate any speedup.

### Dataset strategy

- Init-time dataset broadcast is usually the right default when the same dataset
  is reused across many evaluations.
- Per-task shipping is simpler but turns transport overhead into the hidden
  bottleneck.

### Why not default to work stealing immediately

- Work stealing is powerful but adds queue-state complexity.
- The initial repo need is a truthful, deterministic, cross-environment worker
  pool, not a research scheduler.
- Start with a predictable queue and only adopt richer policies if benchmarks
  show they are worth the complexity.

## Working Heuristics For This Repo

- Prefer simple queue semantics first, dynamic queueing later.
- Preserve input ordering in public outputs.
- Keep the single-thread fallback explicit and tested.
- Use task-level correlation tools in Node when pool observability matters.
- Treat browser and Node parity as an API contract, even when the internal pool
  mechanics differ.
