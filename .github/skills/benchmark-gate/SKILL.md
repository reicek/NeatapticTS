---
name: benchmark-gate
description: 'Use when: validating that a change preserves performance benchmarks before green-testing closure.'
argument-hint: 'Describe the benchmark target, baseline, and tolerance threshold to enforce.'
user-invocable: false
disable-model-invocation: false
skills:
  - execute
---

# benchmark-gate

## Purpose

A performance regression gate. Runs the configured benchmark target against a
baseline and enforces a tolerance threshold before green-testing closure.
Reports pass/fail with the measured delta.

## Use when

- `05-green-testing` needs to confirm a change did not regress the configured
  benchmark beyond the allowed tolerance.
