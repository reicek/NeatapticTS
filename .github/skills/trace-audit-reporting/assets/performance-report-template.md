# Performance Audit

## Scope

This report summarizes the performance audit of `<trace-file>` for
`<feature-area>`.

The audit focused on:

1. Where the system spends time in practice.
2. Which issues belong to the app layer, protocol layer, or core runtime.

The trace was analyzed with:

```bash
npm run trace:analyze -- <trace-path> --top=15
```

## Executive Summary

State the dominant bottleneck first, then the secondary bottleneck, then the
highest-leverage fix.

## Trace Summary

- Trace window: `<value>`
- Event count: `<value>`
- Frames observed: `<value>`
- Dropped frames: `<value>`

### Thread Summary

- `<thread>`: `<summary>`
- `<thread>`: `<summary>`

### Longest Events

- `<duration>` on `<thread>` for `<event>`
- `<duration>` on `<thread>` for `<event>`

### Event Rollups

- `<event>`: `<summary>`
- `<event>`: `<summary>`

## Detailed Findings

### 1. Primary bottleneck

Explain the strongest repeated signal and why it matters.

Evidence:

- `<evidence>`
- `<evidence>`

Interpretation:

- `<interpretation>`

### 2. Secondary bottleneck

Explain the next most important cost.

### 3. Root-cause detail

Tie the hotspot to specific source files and runtime behavior.

## Root Cause Summary

### App or demo layer

1. `<cause>`
2. `<cause>`

### Core runtime layer

1. `<cause>`
2. `<cause>`

## Prioritized Action Plan

### Priority 1

Goal:

- `<goal>`

Actions:

1. `<action>`
2. `<action>`

Expected impact:

- `<impact>`

Risk:

- `<risk>`

### Priority 2

Repeat the same structure for the next action group.

## Validation Plan

After each optimization pass, collect a fresh trace and compare:

- dropped frame count,
- renderer main-thread total,
- worker total,
- hottest `FunctionCall`,
- hottest transport or GPU events.

## Final Conclusions

Summarize:

1. what is bottlenecked now,
2. what matters most to users,
3. what matters most for scalability,
4. what should be implemented next.