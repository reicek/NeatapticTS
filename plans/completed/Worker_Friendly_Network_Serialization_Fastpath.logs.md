# Worker Inference Transport — Four Progressive Strategies Log

**Status:** [DONE]

## Audit scope

- Objective: close the Phase 4 worker-transport substrate lane after all four inference transport strategies were delivered, the Flappy proof surface was validated, and the turnkey extraction handoff was finished.
- Coverage included the shared inference IR, portable and transferable payloads, persistent channel workers, shared-memory workers, Astro Bird proof-surface integration, and the final browser-host acceptance decision.

## Durable milestones

### [DONE] Shared inference IR and payload substrate

- Added the deterministic inference IR plus `PortableInferencePayload` and `TransferableInferencePayload` export paths.
- Kept runtime-significant fields such as activation schedule, response, mask, recurrent state, and output ordering explicit across the worker transport surface.

### [DONE] Persistent channel and shared-memory worker transport

- Added `InferenceChannel` and `SharedInferenceWorker` plus their worker-side receivers.
- Validated the reusable transport substrate independently from Astro Bird policy decisions.

### [DONE] Astro Bird proof-of-concept and browser acceptance closure

- Integrated the bounded shared-memory pool proof surface and browser runtime budget policy into Astro Bird.
- Closed the Step 4.8 host acceptance gate after confirming the live recurrent browser path no longer stalls in `Status initializing` on the isolated webpack host.
- Recorded the accepted runtime outcome: the shared-memory path remains a reusable transport proof surface, while recurrent browser evaluation stays worker-local for the live Flappy runtime because the frame-by-frame rollout path does not amortize the shared-worker roundtrip cost.

## Controls and evidence

- Focused Jest slices for the worker-payload substrate, Flappy evaluation worker-pool boundary, and Flappy worker runtime behavior.
- TypeScript validation via `npx tsc --noEmit -p tsconfig.json`.
- Worker bundle regeneration via `npm run build:flappy-worker`.
- Cross-origin-isolated local host validation on the Astro Bird browser surface.

## Reopen triggers

- Re-enable the recurrent shared-memory browser path only if a future implementation proves it is actually viable on the live host.
- Reopen if a new transport tier, IR revision, or proof-surface ownership change invalidates the archived substrate baseline.
