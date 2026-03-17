---
name: test-fix-workflow
description: 'Systematically fix multiple test failures by planning first, applying all fixes before broad test execution, validating types early, and only running the full suite at the end.'
argument-hint: 'Describe the failing surface, available failure output, whether the issue is type-level, runtime, or mixed, and any known plan file or validation constraints.'
user-invocable: true
disable-model-invocation: false
---

# Test Fix Workflow

Use this skill when multiple test failures need a disciplined, low-distraction
repair pass.

This skill is the canonical workflow for multi-failure test repair in this
repo. It owns the durable planning sequence, validation cadence, and the rule
that broad test execution happens only after the planned fixes are applied.

## When to Use

- The user asks to fix multiple test failures.
- The failure output spans more than one file, subsystem, or failure class.
- The work needs a durable plan rather than ad hoc trial-and-error.
- TypeScript errors, runtime issues, and test assertions are mixed together.

Do not invoke this skill for a single obvious failing assertion unless the task
is likely to expand into a broader failure-repair pass.

## Task Packet

Pass a compact packet that includes:

- failing surface or package area,
- available failure output or prior diagnosis,
- whether the failures are type-level, runtime, assertion-level, or mixed,
- existing plan path if one already exists,
- whether broad test execution is currently blocked or expensive,
- required final validations.

Compact example:

```text
Use test-fix-workflow for failing Flappy Bird trainer tests.
Available output: ts errors in trainer/evaluation plus 6 runtime test failures.
Type: mixed TypeScript + runtime.
Plan: plans/TestsFix.md.
Final validation: npx tsc --noEmit -p tsconfig.test.json, then npm test.
```

## Required Workflow

1. Create or update a durable fix plan before changing code.
2. Group failures by class.
   - Typical buckets: TypeScript compilation blockers, runtime logic, async or
     sequencing issues, assertion drift, and investigation-required failures.
3. Prioritize the plan.
   - Prefer: blocking type errors first, then cheap/high-confidence fixes, then
     deeper investigation items.
4. Apply all planned fixes systematically before running broad tests.
5. Do not run `npm test`, `npm run test:silent`, or partial failure scans during
   the main fix phase.
6. TypeScript-only validation is allowed during the fix phase when it helps
   confirm compile-time repairs.
7. After the planned fixes are complete, run the final broad validation.
8. Analyze any remaining failures and update the plan rather than switching to
   unstructured iteration.

## Guardrails

- Do not prepend specific calendar dates to durable fix-plan headings, status
   logs, or handoff sections. Use stable undated labels so the plan can be
   revised cleanly across sessions.
- Do not bounce between test execution and partial fixes when the workflow is
  still in the main repair phase.
- Do not treat partial reruns as a substitute for a durable plan.
- Do not improvise a new order once the plan is in motion unless new evidence
  forces a reprioritization.
- Do not run the full suite before the planned fixes are actually in place.

## Validation Rules

Preferred validation cadence:

- During the fix phase: `npx tsc --noEmit -p tsconfig.test.json` when needed.
- After all planned fixes: `npm test` or `npm run test:silent`.

If the task is compile-heavy rather than runtime-heavy, file- or package-level
diagnostics may be enough before the final suite run.

## Expected Final Output

A strong run should report:

- the plan file used or created,
- the failure categories addressed,
- which validations were intentionally deferred until the end,
- final validation results,
- any remaining failures or follow-up items.