# neat/selection/facade

Public population-summary facade helpers for the stable `Neat` entrypoint.

The broader selection chapter already owns ordering and parent-choice
behavior. This facade keeps only the stable class-friendly wrappers that the
top-level [src/neat.ts](src/neat.ts) entrypoint exposes: sorting the current
population, reading the fittest genome, and reading the average score.

That split is intentional. The root [selection](../selection.ts) chapter is
the controller-facing story about how selection works, when parent-choice
strategy matters, and which fallback score rules keep the population
inspectable. This facade is the smaller promise preserved by `Neat` itself:
callers can ask for a deterministic best-first ordering, inspect the current
champion, or read the population mean without importing the lower-level
selection module directly.

Read this as the stable public contract, not as a second mechanics chapter.
If a caller only needs class-friendly inspection helpers, this is the right
boundary. If the caller needs to understand parent-choice strategy or the
exact guard rails beneath those reads, the root selection chapter and
`core/` own that deeper explanation.

Read these helpers as one compact summary flow:

1. {@link sort} restores descending score order when an explicit best-first
   population view is useful,
2. {@link getFittest} returns the current champion using the same safety
   rules as the root selection helpers,
3. {@link getAverage} gives the whole-generation companion signal that says
   how the rest of the population is doing beside that champion.

Keeping that wrapper surface in `selection/facade/` makes the ownership story
match the generated docs and the direct-path chapter layout used by the newer
RNG, pruning, and telemetry facades.

Read the symbols in this order:

- `NeatPopulationSummaryFacadeHost` defines the narrow host seam.
- `FittestNetwork` names the champion-shaped return contract used by the
  public `getFittest()` wrapper.
- `sort`, `getFittest`, and `getAverage` form the stable summary flow that
  remains on the `Neat` surface.

Invariant: this boundary only summarizes or reorders the current population.
It does not change parent-selection strategy, crossover policy, or mutation
behavior.

## neat/selection/facade/selection.facade.ts

### FittestNetwork

Champion-shaped network returned by the public `getFittest()` facade.

The facade intentionally returns the real network instance from the current
population rather than a detached summary object. That keeps the stable
`Neat` contract aligned with long-standing usage patterns where callers read
the champion's score and then continue inspecting the network itself.

The extra `score` field is called out here so the generated README can name
the returned shape explicitly instead of falling back to a rough imported
default-type rendering.

### getAverage

```ts
getAverage(
  host: NeatPopulationSummaryFacadeHost,
): number
```

Compute the average score across the current population.

This is a compact inspection helper for telemetry, tests, and quick
debugging where the caller only needs the current mean fitness.

Read it as the companion to {@link getFittest}: the champion tells you how
high the best genome has climbed, while the average tells you whether that
progress is representative of the wider generation or still concentrated in
a small leading group. Because this wrapper stays at the summary layer, it
reports on population state without drifting into parent-choice or breeding
policy.

Together, `getFittest()` and `getAverage()` are the facade's public promise:
one read for the leader and one read for the generation as a whole.

Parameters:

- `host` - `Neat` instance exposing population and evaluation state.

Returns: Mean score across the current population.

Example:

```ts
const meanScore = neat.getAverage();
console.log(`Average score: ${meanScore}`);
```

### getFittest

```ts
getFittest(
  host: NeatPopulationSummaryFacadeHost,
): FittestNetwork
```

Return the fittest genome in the current population.

If scores are missing, the underlying selection helper will trigger the
existing evaluation path before resolving the best genome. If scores exist
but the population is not in descending order yet, the shared selection logic
restores that ordering before returning the leading genome.

That keeps the public `neat.getFittest()` contract pleasantly boring: call
sites can ask for the current champion without remembering whether
evaluation or sorting already happened earlier in the generation.

In the public `Neat` surface, this is the highest-value inspection read: it
answers "who is winning right now?" without exposing lower-level parent
strategy mechanics that belong to root selection and `core/`.

Parameters:

- `host` - `Neat` instance exposing population and evaluation state.

Returns: Champion network with the highest current score.

Example:

```ts
const champion = neat.getFittest();
console.log(champion.score);
```

### NeatPopulationSummaryFacadeHost

Narrow `Neat` host surface required by the public population-summary facade.

The host keeps the contract small on purpose: a population, selection-aware
options, the existing sort hook, and the evaluation entrypoint used when
callers ask for summary data before scores have been computed.

That narrowness preserves the stable `Neat` wrapper semantics without turning
this facade into a second controller chapter. Parent-choice math, RNG usage,
and lower-level selection policy stay in the root selection boundary and its
`core/` helpers. This host only exposes what the summary wrappers genuinely
need to preserve long-standing `neat.sort()`, `neat.getFittest()`, and
`neat.getAverage()` behavior.

### sort

```ts
sort(
  host: NeatPopulationSummaryFacadeHost,
): void
```

Sort the population in descending fitness order.

This preserves the historical `neat.sort()` behavior while keeping the
top-level class free of inline ordering details.

Use this wrapper when later reads should see the current population in an
explicit best-first order, such as before manual inspection, debugging, or a
deterministic test assertion. The wrapper deliberately stays narrow: it only
forwards to the shared selection ordering helper and keeps the familiar class
method available at the stable `Neat` surface.

This is the facade's one explicit reordering primitive. Everything else in
the chapter is read-only population inspection.

Parameters:

- `host` - `Neat` instance exposing population sorting state.

Returns: Nothing. The population array is reordered in place.

Example:

```ts
neat.sort();
console.log(neat.population[0].score);
```
