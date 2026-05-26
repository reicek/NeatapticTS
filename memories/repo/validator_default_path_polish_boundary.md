# Validator Default Path Polish — Learning Event

**Date:** 2026-05-20  
**Pass:** validator-path polish (POLISH mode, low-severity gap)

## Gap closed

`validate-plan-sync.mjs` and `validate-plan-phase-packets.mjs` used a stale
default `--plan` path pointing to an active tracker location.  
After the tracker was archived to `plans/completed/`, both scripts failed with
their old hardcoded default.

## Fix applied

- Updated default path in both scripts (and shared `customization-utils.mjs`)
  to `plans/completed/Agentic_Workflow_Architecture.plans.md`.
- Both validators now return `PASS` with zero issues using the archived default.

## Invariant to preserve

When a `.plans.md` tracker is compressed and moved to `plans/completed/`,
update the `--plan` default in any validator scripts that reference it by
path (especially `validate-plan-sync.mjs` and `validate-plan-phase-packets.mjs`)
in the same commit/pass.
