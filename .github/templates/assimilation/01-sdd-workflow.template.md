# 01 — SDD / Workflow

## External Development Loop

Describe the tool's end-to-end workflow from prompt/idea to shipped artifact.
Use a small diagram if the original repo documents one.

## Commands and Entry Points

| Command / Script | Purpose | Nearest NeatapticTS Equivalent |
| ---------------- | ------- | ------------------------------ |
| `...`            | ...     | `...`                          |

## Artifact Lifecycle

What artifacts are produced at each stage? How are they tracked, validated,
and handed off?

## Comparison with NeatapticTS Phase Handoff

| Stage             | External Tool | NeatapticTS Phase Handoff                     |
| ----------------- | ------------- | --------------------------------------------- |
| Planning / triage | <mechanism>   | `01-planning` → step packet YAML              |
| Implementation    | <mechanism>   | `04-implementing` + `implementation-executor` |
| Green validation  | <mechanism>   | `05-green-testing` + `coverage-guard`         |
| Documentation     | <mechanism>   | `06-documenting` + JSDoc/README generation    |
| Compression / log | <mechanism>   | `07-logging` + phase compression              |

## Cherry-Pick Candidates

- <Workflow element worth adopting and why>
- <Workflow element to reject and why>
