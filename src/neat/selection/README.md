# neat/selection

Parent-selection helpers for the NEAT controller.

The root selection chapter keeps the public controller-facing methods small
and readable, while `core/` holds the selection strategy mechanics,
constants, and narrow runtime contracts. The sibling `facade/` chapter keeps
the stable `Neat` class wrappers for callers that interact through the main
controller entrypoint instead of the lower-level selection module.

- `core/` explains score defaults, ordering checks, and parent-selection strategies.
- `facade/` keeps the stable population-summary wrappers used by `Neat`.

## neat/selection/selection.ts

### DEFAULT_POWER

Default power exponent for POWER selection when none is configured.

### DEFAULT_SCORE

Default score when a genome has no explicit score.

### DEFAULT_TOURNAMENT_PROBABILITY

Default tournament win probability when none is configured.

### DEFAULT_TOURNAMENT_SIZE

Default tournament size when none is configured.

### FIRST_INDEX

Index of the first element in an array.

### getAverage

```ts
getAverage(): number
```

Compute the average fitness across the population.

Returns: Mean fitness across the current population.

### getFittest

```ts
getFittest(): GenomeWithScore
```

Return the fittest genome in the population.

Returns: Genome with the highest current score.

### getParent

```ts
getParent(): GenomeWithScore
```

Select a parent genome according to the configured selection strategy.

Returns: Genome chosen according to the active selection strategy.

### INITIAL_CUMULATIVE_FITNESS

Initial cumulative fitness value for threshold scans.

### INITIAL_MOST_NEGATIVE_SCORE

Initial most-negative score sentinel for fitness scans.

### INITIAL_TOTAL_FITNESS

Initial total fitness accumulator value.

### LAST_ELEMENT_INDEX

Index used with `at()` to access the last element.

### LAST_INDEX_OFFSET

Offset for retrieving the last element via length arithmetic.

### LOOP_INDEX_INCREMENT

Step size for index-based loops.

### SECOND_INDEX

Index of the second element in an array.

### sort

```ts
sort(): void
```

Sort the internal population in place by descending fitness.

Returns: Nothing. The population array is reordered in place.
