# NGE Core Reference Notes

This file stores paraphrased reference notes for the `nge-core-algorithm`
skill.

These notes summarize upstream material instead of copying it verbatim. Use the
linked sources for canonical wording and details.

## Source Map

### 1. NGE plan in this repo

- Source: `plans/NEAT_Genesis_EvoDevo.md`
- Why it matters:
  - This is the primary architectural source for NGE.
  - It defines DNA as a program, the deterministic lifecycle, computation motifs,
    memory tiers, neuromodulation, reproduction modes, and the boundary between
    core algorithm and follow-on demo plans.

### 2. Wikipedia: evolutionary developmental biology

- URL: https://en.wikipedia.org/wiki/Evolutionary_developmental_biology
- Authors: Wikipedia contributors
- License note: Wikipedia text is available under CC BY-SA 4.0.
- Why it matters:
  - Evo-devo emphasizes that form emerges from regulated developmental programs,
    not from one explicit list of final structures.
  - Deep homology, developmental bias, and genetic assimilation provide useful
    language for NGE's deterministic development and slow structural write-back.
  - Regulatory deployment matters more than raw structural parts alone.

### 3. Wikipedia: stigmergy

- URL: https://en.wikipedia.org/wiki/Stigmergy
- Authors: Wikipedia contributors
- License note: Wikipedia text is available under CC BY-SA 4.0.
- Why it matters:
  - Stigmergy is indirect coordination through environmental traces.
  - The environment can act as an external memory that simple agents read and
    modify.
  - This maps well to NGE's shared-field and ant-hive collective-intelligence
    primitives.

### 4. Wikipedia: neuromodulation

- URL: https://en.wikipedia.org/wiki/Neuromodulation
- Authors: Wikipedia contributors
- License note: Wikipedia text is available under CC BY-SA 4.0.
- Why it matters:
  - Neuromodulators create broad, longer-lasting mode shifts across populations of
    neurons.
  - Tonic versus phasic effects are a useful analogy for steady context versus
    fast event-driven mode switching.
  - This supports the plan's separation between slow structural edits and fast
    behavioral bias signals.

## Practical Notes

### DNA is policy plus generator, not phenotype dump

- Core NGE design should prefer compact generator rules, module archetypes, and
  budget knobs over storing a fully realized graph in DNA.

### Development must stay deterministic

- If rule passes, module IDs, or edge realization are unstable, the whole
  evo-devo framing collapses into nondeterministic graph generation.

### Shared environment can be memory

- Stigmergic traces let a colony coordinate without direct pairwise messaging.
- In NGE terms, this means shared fields are not demo decoration; they are a core
  algorithmic primitive.

### Fast mode switching is not morphogenesis

- Neuromodulation changes how circuits behave quickly.
- Morphogenesis changes structure slowly.
- NGE should keep those mechanisms separate in both API and implementation.

## Working Heuristics For This Repo

- Stabilize core motifs and DNA contracts before building demo-local workarounds.
- Treat deterministic build hashes as a first-class acceptance signal.
- Keep reproduction semantics explicit and testable.
- Keep collective-intelligence primitives generic enough that multiple NGE demos
  can reuse them.
