# Worker Inference Transport — Four Progressive Strategies

**Status:** [DONE]

## Scope

- Define and ship the worker-friendly inference transport substrate shared by portable, transferable, channel, and shared-memory execution modes.
- Prove each transport on Astro Bird before treating the substrate as closed for current scope.
- Hand off reusable ergonomics to the turnkey multithread evaluation lane once the transport boundary was stable enough to support public helper extraction.

## Final state

- `PortableInferencePayload`, `TransferableInferencePayload`, `InferenceChannel`, and `SharedInferenceWorker` are implemented and exported through the public network worker-payload shelf.
- The shared inference IR, predictor construction path, worker receivers, and transport helpers are covered by owner-local tests and documented as the durable transport substrate.
- Astro Bird remains the proof surface for the transport stack, but the accepted browser runtime policy is now explicit: recurrent browser evaluation stays on the worker-local path on the isolated host, while the shared-memory worker path remains a reusable transport substrate rather than the active recurrent browser default.
- The turnkey helper extraction brief is closed and archived separately, so this plan now stands as the substrate baseline and reopen point.

## Audit summary

- All four transport phases closed and the Astro Bird Step 4.8 user-host acceptance gate is complete.
- The live recurrent browser stall was resolved by keeping the example’s recurrent browser profiles on the worker-local evaluator path while preserving the shared-memory substrate as a reusable lower-level worker transport.
- The generic Flappy worker-pool boundary also retained a bounded default worker-count cap so browser hosts do not fan out to full hardware concurrency accidentally.

## Reopen conditions

- A future pass makes the recurrent browser shared-memory path viable enough to become the active Flappy runtime policy again.
- The transport substrate ownership story changes or one of the public transport tiers needs a new IR or runtime contract version.
- A future flagship proof surface exposes a substrate gap that cannot be handled within the archived four-strategy boundary.

## Audit log

- See [Worker_Friendly_Network_Serialization_Fastpath.logs.md](Worker_Friendly_Network_Serialization_Fastpath.logs.md).