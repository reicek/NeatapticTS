/**
 * Test-harness contracts for narrow NEAT controller seams.
 *
 * This chapter exists for tests that need direct access to a small slice of the
 * controller without pretending the whole `Neat` surface is stable, minimal, or
 * easy to mock. The goal is not to create a second public runtime API. The goal
 * is to make high-value assertions readable when tests need to inspect lineage
 * metadata, parent-derived spawning, external genome registration, or phased
 * complexity state.
 *
 * Read this chapter as a seam map between tests and the richer runtime
 * chapters:
 *
 * - `lineage/` explains what the parent and depth metadata mean,
 * - `speciation/` explains why lineage-aware assertions matter for diversity
 *   preservation,
 * - `adaptive/` explains what phased-complexity state is trying to control.
 *
 * The harness boundary stays intentionally narrow so tests can ask precise
 * questions without importing unrelated controller behavior.
 *
 * The important teaching move is to separate two kinds of truth:
 *
 * - harness types answer "what is the smallest surface a test needs to ask one
 *   precise question?"
 * - runtime chapters answer "what does that field or helper actually mean in
 *   the live controller?"
 *
 * Without that split, tests tend to widen the public controller surface just to
 * make assertions convenient. Over time that blurs the difference between a
 * stable runtime contract and a narrow testing seam. This chapter exists to
 * prevent that drift.
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   tests[Test assertions]:::accent --> harness[Harness types]:::base
 *   harness --> lineage[Lineage metadata and parent tracking]:::base
 *   harness --> spawn[Spawn and add helpers used in test setup]:::base
 *   harness --> phase[Phased complexity state checks]:::base
 *   lineage --> chapters[Read lineage, speciation, and adaptive chapters for runtime meaning]:::base
 *   spawn --> chapters
 *   phase --> chapters
 * ```
 *
 * A good way to read the harness root is as a translation layer from test
 * intent to runtime meaning:
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   tests[Test question]:::accent --> tracked[Did lineage metadata survive?]:::base
 *   tests --> spawnQuestion[Can I spawn or register a genome?]:::base
 *   tests --> phaseQuestion[Did adaptive phase change?]:::base
 *   tracked --> lineageMeaning[lineage chapter explains ancestry meaning]:::base
 *   spawnQuestion --> helpersMeaning[helpers chapter explains population-entry meaning]:::base
 *   phaseQuestion --> adaptiveMeaning[adaptive chapter explains phase-policy meaning]:::base
 * ```
 *
 * Practical reading order:
 * 1. Start with `LineageTrackedNetwork` when a test needs to inspect the
 *    metadata written onto a child genome.
 * 2. Continue to `NeatLineageHarness` when a test needs the minimal controller
 *    surface for spawning or registering genomes.
 * 3. Finish with `PhasedComplexityHarness` when a test needs to observe the
 *    adaptive phase flag without depending on the entire adaptive controller.
 *
 * Historically, this kind of boundary becomes necessary once a controller grows
 * enough internal structure that test clarity and runtime API clarity start to
 * pull in different directions. The harness chapter is the compromise: keep
 * tests honest and precise without pretending those seams are the main user
 * story of the library.
 */
import type Neat from '../../neat';
import type Network from '../../architecture/network/network';
import type { GenomeDetailed } from '../shared/neat.shared.types';

/**
 * Network subtype that surfaces lineage metadata fields for assertions.
 *
 * Runtime code usually treats these fields as controller-owned bookkeeping, not
 * as part of the everyday network programming surface. Tests, however, often
 * need to ask whether a spawned or imported genome preserved provenance
 * correctly: which genome id it received, which parents were recorded, and what
 * derived generation depth it now carries.
 *
 * This alias keeps those assertions explicit without forcing tests to widen the
 * main `Network` type everywhere they touch lineage-aware genomes.
 *
 * @example
 * ```ts
 * const lineageAware = child as LineageTrackedNetwork;
 * console.log(lineageAware._parents);
 * ```
 */
export type LineageTrackedNetwork = Network &
  Pick<GenomeDetailed, '_id' | '_parents' | '_depth'>;

/**
 * Narrow Neat surface exposing lineage helper methods used in tests.
 *
 * This is the harness contract for tests that need to exercise one small part
 * of controller behavior without depending on every public `Neat` method. The
 * two exposed helpers cover the common lineage-oriented test workflow:
 * produce a child from an existing parent, then optionally register a genome
 * into the live population with explicit recorded ancestry.
 *
 * Read this as a testing seam, not as a second facade. The full runtime meaning
 * of these helpers still lives in the stronger `helpers/` and `lineage/`
 * chapters; this alias simply makes those behaviors easier to call from focused
 * tests.
 *
 * @example
 * ```ts
 * const helper: NeatLineageHarness = neat as NeatLineageHarness;
 * const child = helper.spawnFromParent(parent, 1);
 * helper.addGenome(child, [parent._id]);
 * ```
 */
export type NeatLineageHarness = Neat & {
  spawnFromParent(parent: Network, mutateCount: number): Network;
  addGenome(genome: Network, parentIds?: number[]): void;
};

/**
 * Minimal surface exposing phased complexity state for testing.
 *
 * Some adaptive-behavior tests do not need the whole complexity-budget system;
 * they only need to observe which phase the controller believes it is in after
 * a transition. This narrow harness keeps those assertions honest by exposing
 * the one internal flag the tests are actually checking instead of widening the
 * full adaptive controller contract.
 *
 * Move back to `adaptive/` when you want the runtime meaning of the phase
 * itself. This harness type only answers the test-oriented question of whether
 * the phase marker changed when expected.
 */
export type PhasedComplexityHarness = Neat & {
  _phase?: string;
};
