import type { NeatLike } from '../../shared/neat.shared.types';

/**
 * Genome with a fitness score and arbitrary additional metadata.
 *
 * This stays intentionally loose because selection only cares about the score
 * and should not over-constrain other genome metadata carried through the run.
 */
export interface GenomeWithScore {
  /** Fitness score used by selection strategies. */
  score?: number;
  /** Additional genome metadata carried by the runtime. */
  [key: string]: unknown;
}

/**
 * Selection strategy settings used by the NEAT controller.
 */
export interface SelectionOptions {
  /** Strategy name, such as `POWER`, `FITNESS_PROPORTIONATE`, or `TOURNAMENT`. */
  name?: string;
  /** Power exponent for POWER selection. */
  power?: number;
  /** Participant count for TOURNAMENT selection. */
  size?: number;
  /** Win probability for TOURNAMENT selection. */
  probability?: number;
}

/**
 * NEAT-like instance extended with selection-specific state and helpers.
 *
 * The selection boundary only needs population access, selection options, RNG,
 * optional tournament overflow suppression, and the in-place sort hook.
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
 */
export type SelectionContext = {
  internal: NeatLikeWithSelection;
  population: GenomeWithScore[];
  selectionOptions: SelectionOptions | undefined;
  getRngFactory: () => () => number;
};
