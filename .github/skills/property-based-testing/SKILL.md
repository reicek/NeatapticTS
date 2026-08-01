---
name: property-based-testing
description: 'Use when: designing red-phase property/fuzz tests that assert invariants over generated input.'
argument-hint: 'Describe the invariant, input generator, shrinking strategy, and validation command.'
user-invocable: false
disable-model-invocation: false
skills:
  - execute
  - creating-unit-tests
---

# property-based-testing

## Purpose

Design property-based and fuzz tests that assert invariants over generated
input rather than single examples. Produces red-phase contracts with input
generators and shrinking for minimal counterexamples.

## Use when

- `03-red-testing` needs invariant/property coverage for a pure function or
  state transition that example tests under-explore.
