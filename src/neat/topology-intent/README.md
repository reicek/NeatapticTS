# neat/topology-intent

Shared feed-forward topology-intent policy for NEAT population entry points.

This chapter keeps the feed-forward promotion contract in one small shared
boundary because both pool bootstrapping and next-population provenance need
to answer the same question: when does a genome already satisfy the stricter
feed-forward runtime assumptions exposed by the public mutation policy?

Example:
```ts
const shouldPromote = usesFeedForwardMutationPolicy(neat.options.mutation);
promoteGenomeToFeedForwardIntentWhenEligible(genome, shouldPromote);
```

## neat/topology-intent/neat.topology-intent.ts

### isGenomeEligibleForFeedForwardIntentPromotion

```ts
isGenomeEligibleForFeedForwardIntentPromotion(
  genome: TopologyIntentGenome,
): boolean
```

Check whether a genome can safely adopt feed-forward topology intent.

Eligibility is intentionally conservative: the graph must already be free of
gates and self-connections, and every normal connection must follow the
current node ordering. This keeps the helper focused on preserving an
already-feed-forward structure instead of rewriting arbitrary graphs into a
different interpretation.

Parameters:
- `genome` - Genome candidate.

Returns: True when the genome can safely adopt feed-forward intent.

### matchesCanonicalFeedForwardPool

```ts
matchesCanonicalFeedForwardPool(
  configuredPool: TopologyIntentMutationMethod[],
  canonicalPool: TopologyIntentMutationMethod[],
): boolean
```

Check whether a configured mutation pool matches the canonical FFW pool.

Parameters:
- `configuredPool` - Mutation pool configured on the NEAT instance.
- `canonicalPool` - Canonical feed-forward mutation pool.

Returns: True when both pools align by operator name and order.

### promoteGenomeToFeedForwardIntentWhenEligible

```ts
promoteGenomeToFeedForwardIntentWhenEligible(
  genome: TopologyIntentGenome,
  shouldPromote: boolean,
): void
```

Promote a genome to feed-forward topology intent when the structure is eligible.

The promotion is intentionally conservative. The helper only switches the
runtime contract when the current graph is already ordered like a pure
feed-forward network, which avoids changing the meaning of recurrent or
gated seed genomes during bootstrapping and provenance insertion.

Parameters:
- `genome` - Genome candidate being inserted into a population.
- `shouldPromote` - Whether the active NEAT options request feed-forward semantics.

Returns: Nothing.

### TopologyIntentGenome

Minimal genome surface required to promote feed-forward intent safely.

### TopologyIntentMutationMethod

Minimal mutation descriptor used by topology-intent helpers.

### usesFeedForwardMutationPolicy

```ts
usesFeedForwardMutationPolicy(
  mutationConfig: unknown,
): boolean
```

Determine whether the configured mutation policy communicates feed-forward intent.

Accepts the canonical mutation pool reference, the legacy single-item array
wrapper, or a flattened pool that exactly matches the canonical feed-forward
operator order. The comparison stays strict on purpose so the controller does
not silently reinterpret custom mutation pools as feed-forward mode.

Parameters:
- `mutationConfig` - Configured mutation option.

Returns: True when the option expresses canonical feed-forward intent.
