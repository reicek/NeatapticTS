import type { NeatLike } from '../../shared/neat.shared.types';

/**
 * Contracts for the bounded parent-selection mechanics chapter.
 *
 * The selection core only needs a small runtime seam to answer one practical
 * question repeatedly during evolution: given an evaluated population, which
 * genome should parent the next child? These contracts keep that seam narrow so
 * the mechanics chapter can focus on guard rails, strategy dispatch, and
 * strategy-specific choice flow rather than broader controller policy.
 *
 * Read the chapter in this order:
 * - `GenomeWithScore` narrows the per-genome view to the score-centered fields
 *   selection actually consumes.
 * - `SelectionOptions` describes the small set of knobs that alter parent
 *   choice behavior.
 * - `NeatLikeWithSelection` captures the host hooks for evaluation, sorting,
 *   and RNG access.
 * - `SelectionContext` is the per-call packet shared after dispatch begins.
 */

/**
 * Genome with a fitness score and arbitrary additional metadata.
 *
 * The selection core deliberately narrows its view of a genome to one decisive
 * field: the score that earlier evaluation work produced. That keeps this
 * boundary reusable across parent selection, ordering, and summary reads
 * without forcing the selection chapter to understand topology, telemetry,
 * lineage, or other controller-owned metadata.
 *
 * Keep the contract loose when you add new genome-side fields elsewhere in the
 * controller. If a selection helper can still do its job from score plus opaque
 * metadata, the extra fields belong outside this boundary.
 */
export interface GenomeWithScore {
  /** Fitness score used by selection strategies. */
  score?: number;
  /** Additional genome metadata carried by the runtime. */
  [key: string]: unknown;
}

/**
 * Selection strategy settings used by the NEAT controller.
 *
 * These options are intentionally small because the public selection story is
 * already taught at the root selection chapter and exposed through the stable
 * `Neat` wrappers. The core layer only needs the knobs that alter actual parent
 * choice mechanics: which strategy runs, how strongly POWER biases toward the
 * front, and how TOURNAMENT sizes and probabilistic winner walks behave.
 *
 * Keeping this contract narrow prevents the core chapter from turning into a
 * second facade for broader controller policy.
 */
export interface SelectionOptions {
  /** Strategy name, such as `POWER`, `FITNESS_PROPORTIONATE`, or `TOURNAMENT`. */
  name?: string;
  /** Power exponent that controls how strongly POWER favors the sorted front. */
  power?: number;
  /** Participant count for the temporary TOURNAMENT bracket. */
  size?: number;
  /** Win probability used when walking the sorted TOURNAMENT bracket. */
  probability?: number;
}

/**
 * NEAT-like instance extended with selection-specific state and helpers.
 *
 * This host contract captures the minimum runtime surface the mechanics need in
 * order to preserve stable controller assumptions.
 *
 * Those assumptions are:
 * - selection reads the current in-memory population rather than rebuilding it,
 * - evaluation can be triggered before score-dependent reads when needed,
 * - sorting remains centralized through the host's in-place population order,
 * - randomness always comes from the controller's configured RNG stream,
 * - tournament overflow behavior can be relaxed only through an explicit host
 *   escape hatch.
 *
 * The result is a small, practical seam for selection behavior and tests rather
 * than another public API surface.
 */
export interface NeatLikeWithSelection extends NeatLike {
  /** Current population used by selection strategies. */
  population: GenomeWithScore[];
  /** Configuration options for selection logic. */
  options: {
    /** Selection strategy settings. */
    selection?: SelectionOptions;
    /** Additional options carried by the runtime. */
    [key: string]: unknown;
  };
  /** Factory that returns the RNG function used for selection. */
  _getRNG: () => () => number;
  /** When true, tournament size overflow falls back to a random member. */
  _suppressTournamentError?: boolean;
  /** Sorts the population by descending fitness. */
  sort: () => void;
}

/**
 * Shared state passed through the internal selection strategies.
 *
 * This bundles the stable inputs that every strategy needs once dispatch has
 * already decided which algorithm to run. It lets POWER, FITNESS_PROPORTIONATE,
 * and TOURNAMENT share one context shape without each helper repeatedly pulling
 * data off the host.
 *
 * Read this as the per-selection call frame: one host, one population view,
 * one resolved selection-options object, and one RNG source.
 */
export type SelectionContext = {
  /** NEAT host that owns evaluation, sorting, and overflow policy hooks. */
  internal: NeatLikeWithSelection;
  /** Current population view used by the active selection strategy. */
  population: GenomeWithScore[];
  /** Resolved strategy options for the current parent-selection call. */
  selectionOptions: SelectionOptions | undefined;
  /** RNG accessor preserved from the controller so all strategies stay deterministic. */
  getRngFactory: () => () => number;
};
