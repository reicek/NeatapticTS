---
name: security-review
description: 'Use when: auditing changed source for exploitable security vulnerabilities before validation handoff.'
argument-hint: 'Describe the change set (files/diff), threat surface, and the validation gate to satisfy.'
user-invocable: false
disable-model-invocation: false
skills:
  - execute
---

# security-review

## Purpose

A read-only security review gate for changed source. Surfaces high-confidence
exploitable vulnerabilities and security-relevant logic errors in a diff or
change set, ignoring style and non-security noise.

## Use when

- `04-implementing` finishes a change touching auth, IO, parsing, or untrusted
  input paths and needs a security preflight before green-testing.
- `05-green-testing` needs to confirm no security regression was introduced.
