# neat/genome/heredity

## neat/genome/heredity/genome.heredity.types.ts

### GenomeHereditySelectionContext

Pure selection context for innovation-aligned genome heredity.

This contract keeps the heredity pass structural-first: two strict genomes,
the fitness or equality policy that decides disjoint inheritance, and the
deterministic RNG that resolves matching-gene choices plus disabled-gene
re-enable behavior.

### GenomeHereditySourceParent

Stable string literal labels identifying which parent contributed a given connection gene during the genome-owned heredity selection pass.

### SelectedGenomeConnectionGene

One inherited connection gene selected by the genome-owned heredity pass.

The source-parent label preserves parent provenance for runtime adapters and
future narrow seams without pushing runtime node indexing into the genome
surface.

## neat/genome/heredity/genome.heredity.ts

### selectGenomeHeredityConnectionGenes

```ts
selectGenomeHeredityConnectionGenes(
  context: GenomeHereditySelectionContext,
): SelectedGenomeConnectionGene[]
```

Select inherited connection genes using only the strict genome contract.

Step 7.2a moves innovation-aligned heredity selection behind the genome
boundary without widening the runtime crossover facade. The runtime shelf
still owns node scaffolding and phenotype materialization, while this helper
owns three structural decisions:

1. collect parent connection genes by preserved innovation number,
2. resolve matching, disjoint, and excess inheritance from scores plus
   equal-mode policy,
3. apply the explicit disabled-gene re-enable rule through the inherited RNG.

Parameters:

- `context` - Pure genome heredity context.

Returns: Ordered inherited connection genes plus their source-parent labels.

Example:

```ts
const selectedGenes = selectGenomeHeredityConnectionGenes({
  parent1Genome,
  parent2Genome,
  parent1Score: 2,
  parent2Score: 1,
  equal: false,
  randomGenerator: () => 0.25,
});
```
