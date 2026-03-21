# neat/topology-intent

Shared feed-forward topology-intent policy for NEAT population entry points.

This chapter keeps the feed-forward promotion contract in one small shared
boundary because both pool bootstrapping and next-population provenance need
to answer the same question: when does a genome already satisfy the stricter
feed-forward runtime assumptions exposed by the public mutation policy?

Read this as a policy bridge, not as a graph-rewrite chapter. Mutation
configuration can communicate feed-forward intent, but that intent should only
become a runtime topology contract when an actual genome already satisfies the
stricter structural rules. This boundary keeps those two questions together:
does the active mutation policy request feed-forward behavior, and is the
candidate genome already safe to promote without changing its meaning?

The helpers form one short decision flow:

1. `usesFeedForwardMutationPolicy()` recognizes the canonical public signal.
2. `isGenomeEligibleForFeedForwardIntentPromotion()` checks whether the
   current graph is already ordered like a feed-forward network.
3. `promoteGenomeToFeedForwardIntentWhenEligible()` applies the public runtime
   contract only when both earlier checks agree.

Move back to `mutation/` when you want the broader operator-policy story,
`init/` when you want constructor-time pool setup, and `helpers/` when you
want the provenance and population-entry flows that reuse this bridge.

Example:
```ts
const shouldPromote = usesFeedForwardMutationPolicy(neat.options.mutation);
promoteGenomeToFeedForwardIntentWhenEligible(genome, shouldPromote);
```

## neat/topology-intent/neat.topology-intent.ts

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

In practice this helper answers the policy question only. It does not inspect
a concrete genome, and it does not attempt any runtime promotion by itself.
That separation is important because a caller may request feed-forward
mutation semantics while still holding seed genomes whose current graphs are
recurrent, gated, or otherwise not yet eligible for the stricter runtime
contract.

Parameters:
- `mutationConfig` - Configured mutation option.

Returns: True when the option expresses canonical feed-forward intent.

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

Read this as the handoff point between policy and runtime contract. The
caller has already decided that feed-forward intent is desired; this helper
makes sure that intent is only written onto genomes that already behave like
feed-forward networks under the current node ordering.

Parameters:
- `genome` - Genome candidate being inserted into a population.
- `shouldPromote` - Whether the active NEAT options request feed-forward semantics.

Returns: Nothing.

### TopologyIntentMutationMethod

Minimal mutation descriptor used by topology-intent helpers.

The bridge only needs one stable piece of information from a mutation entry:
its public name. That keeps the topology-intent checks aligned with the
canonical feed-forward pool without importing the full mutation subsystem.

### TopologyIntentGenome

Minimal genome surface required to promote feed-forward intent safely.

This contract stays deliberately small because the bridge is not trying to
own general graph validation. It only needs the current node order, directed
connections, recurrent-only collections such as gates and self-connections,
and the public topology-intent setter exposed by `Network`.

### matchesCanonicalFeedForwardPool

```ts
matchesCanonicalFeedForwardPool(
  configuredPool: TopologyIntentMutationMethod[],
  canonicalPool: TopologyIntentMutationMethod[],
): boolean
```

Check whether a configured mutation pool matches the canonical FFW pool.

This helper exists so callers can recognize the feed-forward mutation policy
even after options have been flattened, copied, or wrapped by legacy code.
The comparison deliberately stays order-sensitive because the canonical pool
is treated as one explicit public signal rather than as a fuzzy set of
approximately similar operators.

Parameters:
- `configuredPool` - Mutation pool configured on the NEAT instance.
- `canonicalPool` - Canonical feed-forward mutation pool.

Returns: True when both pools align by operator name and order.

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

The helper therefore answers a structural-preservation question, not a graph
repair question. A `false` result does not mean the genome is invalid. It
means only that this bridge should not relabel it as feed-forward yet.

Parameters:
- `genome` - Genome candidate.

Returns: True when the genome can safely adopt feed-forward intent.
