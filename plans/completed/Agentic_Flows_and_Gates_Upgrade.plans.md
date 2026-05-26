# Agentic Flows and Gates Upgrade

**Status:** [DONE]

## Scope

This workstream upgraded the NeatapticTS customization system from nested delegation to a named-flow, deterministic-gate, and universal-helper model. The bounded surface stayed within `.github/agents/`, `.github/skills/`, `.github/flows/`, `scripts/agent-customization/`, MCP workflow or validation servers, tracker files, and local learning-event evidence. No `src/` changes belong to this archived lane.

## Final state

- All six migration phases (A-F) are closed, and the repo now uses named flow YAMLs with explicit exit gates and post-phase fan-out across the numbered SDLC agents.
- The final delivered baseline includes 30 named flows, 12 gates, 3 registered local MCP servers, and a universal `00-helping` escalation path with learning-event visibility.
- Representative validation closed green at `61/61` gate tests passing, `validate-agent-graph.mjs` clean, and no owned regressions in the workflow infrastructure surfaces touched by Phase F.
- The active tracker has been replaced by this compressed archive. Future work should reopen from this completed baseline and its matching log rather than restoring the deleted active packet.

## Audit summary

- Phase A: established the flow schema, pilot `04-implementing` flows, Tier-1 gates, and the first gate-aware MCP surface.
- Phase B: added Tier-2 gates, gate-exception recording, and the three-consecutive-failures escalation path.
- Phase C: rolled out flow coverage for `01-planning`, `02-researching`, and `03-red-testing`, including MCP-resource-first reconnaissance where appropriate.
- Phase D: rolled out flow coverage for `05-green-testing`, `06-documenting`, and `07-logging`, completing the eight-agent flow inventory.
- Phase E: added workflow-gap analytics so gate failures, escalations, and flow drift can be audited from local session evidence.
- Phase F: updated the repo instructions and core workflow skills so flow-aware routing, gate contracts, and tracker closure rules are documented in the durable customization surfaces.

## Reopen conditions

- The numbered agent workflow needs another architectural change to its flow inventory, gate contract, or universal-helper escalation behavior.
- MCP workflow, validation, or gate-server contracts change in a way that invalidates the closed flow-aware baseline recorded here.
- Future work needs to revise the documented flow-aware routing rules or rebuild the gate catalog beyond the accepted 30-flow, 12-gate baseline.

## Audit log

- Durable completion notes now live in [Agentic_Flows_and_Gates_Upgrade.logs.md](Agentic_Flows_and_Gates_Upgrade.logs.md).
