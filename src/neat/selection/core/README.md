# neat/selection/core

Selection mechanics used by the NEAT controller.

This chapter collects the shared constants, ordering helpers, and the three
built-in parent-selection strategies.

## neat/selection/core/selection.types.ts

### GenomeWithScore

Genome with a fitness score and arbitrary additional metadata.

This stays intentionally loose because selection only cares about the score
and should not over-constrain other genome metadata carried through the run.

### NeatLikeWithSelection

NEAT-like instance extended with selection-specific state and helpers.

The selection boundary only needs population access, selection options, RNG,
optional tournament overflow suppression, and the in-place sort hook.

### SelectionContext

Shared state passed through the internal selection strategies.

### SelectionOptions

Selection strategy settings used by the NEAT controller.

## neat/selection/core/selection.core.ts

### calculateFitnessTotals

```ts
calculateFitnessTotals(
  population: GenomeWithScore[],
): { totalFitness: number; minFitnessShift: number; }
```

Compute the total fitness and minimal shift used by roulette selection.

Parameters:
- `population` - - Genomes in the current population.

Returns: Aggregate fitness totals with the negative-score shift.

### calculateTotalScore

```ts
calculateTotalScore(
  population: GenomeWithScore[],
): number
```

Calculate the total fitness across the population.

Parameters:
- `population` - - Genomes in the current population.

Returns: Sum of all scores with missing scores treated as zero.

### DEFAULT_POWER

Default power exponent for POWER selection when none is configured.

### DEFAULT_SCORE

Default score when a genome has no explicit score.

### DEFAULT_TOURNAMENT_PROBABILITY

Default tournament win probability when none is configured.

### DEFAULT_TOURNAMENT_SIZE

Default tournament size when none is configured.

### ensurePopulationEvaluated

```ts
ensurePopulationEvaluated(
  internal: NeatLikeWithSelection,
): void
```

Ensure population scores exist by running evaluation if needed.

Parameters:
- `internal` - - NEAT host containing population and evaluation support.

Returns: Nothing. Evaluation is triggered only when the population is unevaluated.

### ensurePopulationSortedDescending

```ts
ensurePopulationSortedDescending(
  internal: NeatLikeWithSelection,
): void
```

Ensure the population is sorted descending by score when out of order.

Parameters:
- `internal` - - NEAT host containing the current population.

Returns: Nothing. Sorting only runs when the first two scores are out of order.

### ensurePopulationSortedDescendingForPower

```ts
ensurePopulationSortedDescendingForPower(
  selectionContext: SelectionContext,
): void
```

Ensure the population is sorted descending by score for POWER selection.

Parameters:
- `selectionContext` - - Shared selection state.

Returns: Nothing. Sorting only runs when the first two entries are out of order.

### FIRST_INDEX

Index of the first element in an array.

### getRandomPopulationMember

```ts
getRandomPopulationMember(
  selectionContext: SelectionContext,
): GenomeWithScore
```

Select a random population member using the configured RNG.

Parameters:
- `selectionContext` - - Shared selection state.

Returns: Randomly chosen genome from the current population.

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

### pickByShiftedThreshold

```ts
pickByShiftedThreshold(
  population: GenomeWithScore[],
  selectionThreshold: number,
  minFitnessShift: number,
): GenomeWithScore | undefined
```

Pick the first genome whose shifted cumulative fitness exceeds the threshold.

Parameters:
- `population` - - Genomes in the current population.
- `selectionThreshold` - - Random threshold in shifted fitness space.
- `minFitnessShift` - - Amount added to each score to shift negatives.

Returns: The chosen genome when a threshold crossing occurs.

### pickTournamentWinner

```ts
pickTournamentWinner(
  selectionContext: SelectionContext,
  sortedParticipants: GenomeWithScore[],
): GenomeWithScore
```

Select a winner from sorted tournament participants.

Parameters:
- `selectionContext` - - Shared selection state.
- `sortedParticipants` - - Participants sorted by descending score.

Returns: The chosen tournament winner.

### resolveTournamentOverflow

```ts
resolveTournamentOverflow(
  selectionContext: SelectionContext,
): GenomeWithScore
```

Resolve what happens when tournament size exceeds population size.

Parameters:
- `selectionContext` - - Shared selection state.

Returns: A fallback parent genome.

### sampleTournamentParticipants

```ts
sampleTournamentParticipants(
  selectionContext: SelectionContext,
  tournamentSize: number,
): GenomeWithScore[]
```

Sample tournament participants with possible repeats.

Parameters:
- `selectionContext` - - Shared selection state.
- `tournamentSize` - - Number of competitors to sample.

Returns: Sampled participants.

### SECOND_INDEX

Index of the second element in an array.

### selectParentByFitnessProportionate

```ts
selectParentByFitnessProportionate(
  selectionContext: SelectionContext,
): GenomeWithScore
```

Select a parent using roulette-wheel fitness proportionate selection.

Parameters:
- `selectionContext` - - Shared selection state.

Returns: The chosen parent genome.

### selectParentByPower

```ts
selectParentByPower(
  selectionContext: SelectionContext,
): GenomeWithScore
```

Select a parent by power-law distribution on the sorted population.

Parameters:
- `selectionContext` - - Shared selection state.

Returns: The chosen parent genome.

### selectParentByStrategy

```ts
selectParentByStrategy(
  internal: NeatLikeWithSelection,
): GenomeWithScore
```

Select a parent genome according to the configured selection strategy.

Parameters:
- `internal` - - NEAT host containing population, options, and RNG access.

Returns: A genome chosen according to the active selection strategy.

### selectParentByTournament

```ts
selectParentByTournament(
  selectionContext: SelectionContext,
): GenomeWithScore
```

Select a parent by tournament selection.

Parameters:
- `selectionContext` - - Shared selection state.

Returns: The chosen parent genome.
