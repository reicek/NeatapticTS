---
name: dependency-audit
description: 'Use when: auditing dependency manifest, lockfile, or transitive packages for license/vulnerability/attribution drift.'
argument-hint: 'Describe the dependency manifest/lockfile, audit scope, and the validation gate to satisfy.'
user-invocable: false
disable-model-invocation: false
skills:
  - execute
---

# dependency-audit

## Purpose

Audit a dependency manifest and lockfile for license attribution, known
vulnerabilities, and unexpected transitive changes. Keeps `package.json` and
lockfile drift aligned with the repo license-attribution policy.

## Use when

- `04-implementing` introduces or upgrades a dependency and needs an audit
  before green-testing.
- `06-documenting` needs license/attribution confirmation for a changed
  dependency set.
