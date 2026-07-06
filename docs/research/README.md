# Research Artifacts

This directory contains pre-design research artifacts produced by `02-researching`.
Each artifact captures what was unknown before a design decision, the evidence
used to resolve it, and the residual risks that downstream steps must account for.

## When to write a research artifact

`02-researching` must materialize any resolved unknowns as a research artifact
before handing off to the next step. If the research phase only confirmed facts
that were already certain, no artifact is needed. If the phase resolved
genuine unknowns, the artifact is mandatory.

## Naming convention

Name each artifact `<feature-slug>.md` using the feature or decision slug from
the active plan. Examples:

- `worker-transport-strategy.md`
- `gpu-fast-path-boundaries.md`
- `nge-memory-tier-policy.md`

## Required sections

Every research artifact must include the following sections:

### Question

The precise unknown that the research phase was asked to resolve. One or two
sentences. Link back to the originating plan step when possible.

### Evidence

Source-grounded findings in strict order of authority: runtime/validation results,
static source code, comments or documentation, then external references. For
each finding, note the source and its strength. Record conflicting evidence
explicitly with the tie-break rule used.

### Decision

The decision that the evidence supports, stated as a single actionable
conclusion. Include the rationale and any alternatives that were rejected.

### Risks

Residual risks, open questions, or assumptions that remain after the decision.
Name the owner phase or step that must validate or retire each risk.

## Linking from a step packet

When a research artifact is produced, the `02-researching` step packet must link
to it. Use the `research_artifact` field in the step-packet YAML, or reference
the artifact path in the handoff summary so the next step can locate the evidence.
