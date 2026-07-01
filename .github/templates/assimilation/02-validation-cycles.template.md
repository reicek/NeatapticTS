# 02 — Validation Cycles

## External Quality Gates

List the external tool's validation gates: tests, lints, audits, coverage,
formatting, security scans.

## Gate Contracts

| Gate           | Tool / Command | Pass Criteria | Auto-block? |
| -------------- | -------------- | ------------- | ----------- |
| Build          | `...`          | `...`         | <yes/no>    |
| Lint           | `...`          | `...`         | <yes/no>    |
| Tests          | `...`          | `...`         | <yes/no>    |
| Coverage       | `...`          | `...`         | <yes/no>    |
| Docs freshness | `...`          | `...`         | <yes/no>    |

## Comparison with NeatapticTS Gates

| Concern           | External Tool | NeatapticTS Gate                                 |
| ----------------- | ------------- | ------------------------------------------------ |
| Plan sync         | <mechanism>   | `plan-sync`, `step-packet`, `validate-plan-sync` |
| Agent graph       | <mechanism>   | `agent-graph`, `tier-enforcement`                |
| Routing freshness | <mechanism>   | `routing-table-freshness`                        |
| Coverage guard    | <mechanism>   | `coverage-guard` + 100% on changed `src/` files  |

## Cherry-Pick Candidates

- <Validation idea worth adopting and why>
- <Validation idea to reject and why>
