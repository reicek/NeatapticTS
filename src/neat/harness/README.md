# neat/harness

Type helpers for test harnesses exercising Neat lineage behaviour.

## neat/harness/neat.harness.types.ts

### LineageTrackedNetwork

Network subtype that surfaces lineage metadata fields for assertions.

Example:

const lineageAware = child as LineageTrackedNetwork;
console.log(lineageAware._parents);

### NeatLineageHarness

Narrow Neat surface exposing lineage helper methods used in tests.

Example:

const helper: NeatLineageHarness = neat as NeatLineageHarness;
const child = helper.spawnFromParent(parent, 1);

### PhasedComplexityHarness

Minimal surface exposing phased complexity internals for testing.
