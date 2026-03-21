# neat/lineage

Lineage and ancestry analysis helpers for NEAT populations.

The lineage boundary answers a different question from the broader diversity
chapter: not just whether genomes look different now, but whether they come
from meaningfully different recent families. That ancestry view matters when
telemetry, diagnostics, or adaptive policy need to detect whether a run is
still exploring multiple lineages or quietly collapsing onto descendants of a
small ancestor set.

The root chapter keeps two public read models together:

- `buildAnc()` builds one genome's shallow ancestor set.
- `computeAncestorUniqueness()` turns many shallow ancestor sets into one
  sampled population signal.

Read the chapter in this order:

- `buildAnc()` when you want the raw ancestry evidence for one genome.
- `computeAncestorUniqueness()` when you want the controller-facing summary
  used by telemetry, diagnostics, or adaptive lineage pressure.
- `core/` when you need the breadth-first traversal, pair sampling, or
  Jaccard-distance mechanics behind the public helpers.

```mermaid
flowchart TD
  Genome[Genome with parent ids] --> Ancestors[Shallow ancestor set]
  Population[Population genomes] --> Pairs[Sample genome pairs]
  Ancestors --> Distance[Compare ancestor overlap]
  Pairs --> Distance
  Distance --> Uniqueness[Ancestor uniqueness summary]
  Uniqueness --> Consumers[Telemetry diagnostics and adaptive lineage policy]
```

The root chapter stays compact on purpose. `core/` owns the queue mechanics,
sampled pair generation, and distance aggregation so this file can stay
focused on what the ancestry reads mean at the controller surface.

## neat/lineage/lineage.ts

### buildAnc

```ts
buildAnc(
  genome: GenomeLike,
): Set<number>
```

Build the shallow ancestor ID set for a genome using breadth-first traversal.

"Shallow" means this helper intentionally stops after a small ancestry
window instead of walking the entire historical tree. That keeps the result
useful for runtime telemetry: it captures the recent family neighborhood that
most directly explains current convergence or branching without turning every
read into an unbounded genealogy crawl.

Use this when you need the raw ancestry evidence behind later population
summaries. The returned set is most helpful for pairwise overlap checks,
debugging parent tracking, or validating that speciation and reproduction are
still producing multiple recent family branches.

Parameters:
- `this` - - NEAT lineage context providing the current population.
- `genome` - - Genome whose shallow ancestor set should be computed.

Returns: Set of ancestor IDs within the configured depth window.

Example:

```ts
const ancestorIds = neat.buildAnc(neat.population[0]);

console.log(ancestorIds.has(42));
```

### computeAncestorUniqueness

```ts
computeAncestorUniqueness(): number
```

Compute the ancestor uniqueness metric for the current population.

This is the controller-facing lineage summary. It samples genome pairs,
builds a shallow ancestor set for each genome in the pair, then measures how
different those ancestor sets are using Jaccard distance.

Interpret the returned value as a bounded trend signal:

- lower values mean many genomes still share recent ancestors,
- higher values mean recent ancestry is spread across more distinct family
  branches.

The helper is intentionally sampled rather than exhaustive so telemetry and
adaptive controllers can reuse it during a run without paying the full cost
of comparing every genome pair. It complements the diversity chapter by
focusing on ancestry overlap rather than structural size or compatibility
distance.

Parameters:
- `this` - - NEAT lineage context exposing the population and RNG provider.

Returns: Mean sampled Jaccard distance across shallow ancestor sets.

Example:

```ts
const ancestorUniqueness = neat.computeAncestorUniqueness();

if (ancestorUniqueness < 0.2) {
  console.log('Recent ancestry is collapsing into a narrow family band.');
}
```

### GenomeLike

Minimal genome shape used by lineage helpers.

Lineage analysis only needs two structural facts from each genome: a stable
identifier and the identifiers of its recorded parents. Everything else is
intentionally left open-ended so ancestry helpers can run against richer
runtime objects without importing or depending on all of their fields.

In practice this interface is the bridge between reproduction-time lineage
bookkeeping and read-side lineage metrics. If those ids are present and
stable, the rest of the ancestry pipeline can stay decoupled from mutation,
evaluation, telemetry, and speciation internals.

### NeatLineageContext

Minimal NEAT context required by lineage helpers.

The lineage boundary only needs the current population and the RNG provider
used for sampled ancestor uniqueness. That small host contract makes the
ownership model explicit: lineage reporting is a read-side controller
concern, not a stateful subsystem with its own storage or mutation rules.

The population supplies the ancestry graph to inspect. The RNG provider keeps
sampled uniqueness deterministic so the same run can replay the same sampled
comparisons during tests or exported-state debugging.
