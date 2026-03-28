# neat/diversity/core

Contracts for the bounded diversity-reporting pipeline.

The diversity core chapter builds one cheap population snapshot from four
complementary signals:
structural size, structural shape, compatibility distance, and lineage
spread. These contracts stay intentionally small so telemetry and diagnostic
reads can project just the fields that report needs instead of importing the
entire `Neat` runtime surface.

Read the chapter in this order:
- `GenomeWithMetrics` describes the minimal per-genome projection.
- `CompatComputer` supplies the one expensive cross-genome measurement.
- `DiversityStats` names the aggregated output fields that consumers compare
  across generations.

## neat/diversity/core/diversity.types.ts

### NodeWithConnections

Minimal node interface used by diversity computations.

Diversity helpers only need each node's outgoing connection count to build a
structural-entropy fingerprint, so this type keeps the reporting boundary
narrower than the full runtime node model.

### GenomeWithMetrics

Minimal genome shape used by diversity computations.

This projection is the per-genome input to the diversity report. It exposes
exactly the data needed to answer four cheap read-side questions:

- how large genomes are right now,
- how uneven that structural size has become,
- how far sampled ancestry depth has spread,
- and how compatible each genome remains with sampled peers.

Keeping the contract this small lets diagnostics and telemetry reuse the
diversity helpers against genome-like snapshots rather than the full NEAT
controller state.

### CompatComputer

Minimal interface for computing compatibility distance between genomes.

Compatibility is the one report dimension that needs a cross-genome callback
instead of direct field reads. This seam keeps the diversity chapter focused
on sampling and aggregation while the broader NEAT controller owns the actual
compatibility policy.

### DiversityStats

Diversity statistics returned by sampled population analysis.

Treat this as a compact population-health report rather than as a single
scalar "diversity score." The fields are grouped deliberately:

- lineage fields show whether ancestry depth is spreading or collapsing,
- node and connection fields show average structural size and unevenness,
- compatibility sampling estimates how genetically separated sampled peers
  remain,
- entropy adds a shape signal that raw size counts cannot capture.

In practice, telemetry consumers compare this object across generations to
see whether mutation, speciation, and pruning are still producing meaningful
variation without paying for exhaustive all-pairs analysis.

## neat/diversity/core/diversity.core.ts

Diversity-statistics mechanics used by telemetry and diagnostics.

This core chapter assembles one bounded diversity report from four
complementary measurements:

- lineage depth spread explains how ancestry is fanning out or collapsing,
- node and connection counts describe topology size and uneven growth,
- sampled compatibility distance estimates genetic separation,
- structural entropy captures topology shape beyond raw size alone.

The helpers stay deliberately cheap. Instead of running exhaustive all-pairs
analysis, they sample the expensive comparisons and fold the results into a
report that is accurate enough for telemetry trends, diagnostics, and
generation-over-generation comparisons.

### calculateStructuralEntropy

```ts
calculateStructuralEntropy(
  graph: default,
): number
```

Compute the Shannon-style entropy of a network's out-degree distribution.

This is the chapter's shape metric. Two genomes can share similar node and
connection counts while still distributing edges very differently, so the
entropy read adds a lightweight topology fingerprint beside the raw size
aggregates.

Parameters:
- `graph` - - Network instance to evaluate.

Returns: Shannon-style entropy value.

### calculateDiversityStats

```ts
calculateDiversityStats(
  population: GenomeWithMetrics[],
  compatibilityComputer: CompatComputer,
): DiversityStats | undefined
```

Compute diversity statistics for a NEAT population.

This is the orchestration step for the bounded diversity report. It projects
lineage depths, structure counts, sampled compatibility distances, and per-
genome structural entropy into one object that callers can log, chart, or
compare across generations.

The returned metrics are intentionally read-oriented rather than prescriptive.
They help answer questions such as whether species are collapsing toward a
narrow family of similar genomes or whether structural experimentation is
still producing spread.

Parameters:
- `population` - - Population genomes exposing nodes, connections, and optional `_depth`.
- `compatibilityComputer` - - Object exposing `_compatibilityDistance(a, b)`.

Returns: Diversity report or `undefined` when the population is empty.

Example:

```ts
const diversity = calculateDiversityStats(neat.population, neat);

if (diversity) {
  console.log(diversity.lineageMeanPairDist, diversity.graphletEntropy);
}
```

### MAX_LINEAGE_PAIR_SAMPLE

Maximum lineage sample size for pairwise depth comparisons.

Lineage spread is useful for telemetry, but full all-pairs ancestry distance
becomes expensive quickly. This cap keeps the lineage side of the report
bounded while still surfacing whether ancestry depth is bunching up or
staying distributed.

### MAX_COMPATIBILITY_SAMPLE

Maximum population sample size for compatibility comparisons.

Compatibility distance is the most obviously quadratic part of the diversity
report. Sampling lets the controller estimate genetic separation cheaply
enough to keep diversity reporting on the hot path for telemetry.

### mean

```ts
mean(
  values: number[],
): number
```

Compute the arithmetic mean of a numeric array.

The diversity report uses this helper for the direct summary columns such as
average lineage depth, average node count, average connection count, and the
mean entropy across genomes.

Parameters:
- `values` - - Values to average.

Returns: Arithmetic mean, or `0` when the array is empty.

### variance

```ts
variance(
  values: number[],
): number
```

Compute the population variance of a numeric array.

Variance complements the raw averages by showing whether the population is
staying structurally tight or spreading into a wider range of topology sizes.

Parameters:
- `values` - - Values to evaluate.

Returns: Population variance, or `0` when the array is empty.

### computeMeanAbsolutePairDistance

```ts
computeMeanAbsolutePairDistance(
  values: number[],
  sampleLimit: number,
): number
```

Compute the mean absolute pairwise distance across a sampled value list.

The diversity report uses this for lineage depth because the question is not
just "how deep are genomes on average?" but also "how far apart are sampled
genomes along that depth axis?" Sampling keeps that pairwise comparison
cheap enough for repeated telemetry reads.

Parameters:
- `values` - - Values to compare.
- `sampleLimit` - - Maximum number of sampled values to include.

Returns: Mean absolute pair distance across the sampled values.

### computeMeanCompatibilityDistance

```ts
computeMeanCompatibilityDistance(
  genomes: GenomeWithMetrics[],
  compatibilityComputer: CompatComputer,
  sampleLimit: number,
): number
```

Compute the mean compatibility distance across sampled genome pairs.

Compatibility is the most expensive dimension in the report because it needs
host-provided pairwise comparisons. This helper bounds the work by sampling
a prefix of genomes, then averaging the pair distances so callers get a
stable separation trend instead of an exhaustive matrix.

Parameters:
- `genomes` - - Population genomes to compare.
- `compatibilityComputer` - - Compatibility-distance provider.
- `sampleLimit` - - Maximum number of genomes to include.

Returns: Mean compatibility distance across the sampled pairs.
