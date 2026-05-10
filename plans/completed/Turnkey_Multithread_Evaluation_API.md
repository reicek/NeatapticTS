# Turnkey Multithread Evaluation API Plan

**Status:** [DONE]

## Scope

- Extract the reusable worker-evaluation ergonomics that Astro Bird proved valuable into public library APIs.
- Keep `Network.activate(...)` synchronous and honest while exposing explicit async helpers for ordered parallel evaluation.
- Close the proof-of-concept loop only after Flappy installed the shared helpers and the ownership story matched the code and docs.

## Final state

- The library now exposes `detectInferenceWorkerCapabilities(...)`, `resolveAutoInferenceTransport(...)`, `resolveBrowserWorkerAssetUrl(...)`, `ParallelInferencePool`, `evaluateInWorkers(...)`, and `createNeatParallelPopulationEvaluator(...)`.
- The reusable worker-helper ladder is exported through `src/architecture/network/worker-payload/`, `src/architecture/network.ts`, and `src/neataptic.ts`.
- Astro Bird installs the full helper ladder on its worker and trainer proof surfaces while keeping example-specific runtime policy above the shared transport boundary.
- Public docs and Flappy docs were aligned at closure so the transport substrate, turnkey helpers, and example ownership model tell the same story.

## Audit summary

- All six extraction targets closed with focused tests, TypeScript validation, docs regeneration, and repo-wide validation during the active workstream.
- The helper surface stayed boolean-first and async-only on explicit worker helpers rather than leaking hidden async behavior into synchronous activation APIs.
- The workstream is terminally closed for current scope and should reopen only if a new reusable worker-evaluation API gap appears or the Flappy proof surface drifts away from the shared helper ownership model.

## Reopen conditions

- A new Node or browser worker-evaluation API gap appears that does not fit the archived six-helper shelf.
- The Flappy proof surface stops matching the documented helper ownership story.
- A future worker or checkpointing lane needs a genuinely new reusable evaluation helper instead of a local integration pass.

## Audit log

- See [Turnkey_Multithread_Evaluation_API.logs.md](Turnkey_Multithread_Evaluation_API.logs.md).