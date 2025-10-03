/**
 * Type helpers for test harnesses exercising Neat lineage behaviour.
 */
import type Neat from '../neat';
import type Network from '../architecture/network';
import type { GenomeDetailed } from './neat.types';

/**
 * Network subtype that surfaces lineage metadata fields for assertions.
 *
 * @example
 * const lineageAware = child as LineageTrackedNetwork;
 * console.log(lineageAware._parents);
 */
export type LineageTrackedNetwork = Network & Pick<GenomeDetailed, '_id' | '_parents' | '_depth'>;

/**
 * Narrow Neat surface exposing lineage helper methods used in tests.
 *
 * @example
 * const helper: NeatLineageHarness = neat as NeatLineageHarness;
 * const child = helper.spawnFromParent(parent, 1);
 */
export type NeatLineageHarness = Neat & {
  spawnFromParent(parent: Network, mutateCount: number): Network;
  addGenome(genome: Network, parentIds?: number[]): void;
};

/**
 * Minimal surface exposing phased complexity internals for testing.
 */
export type PhasedComplexityHarness = Neat & {
  _phase?: string;
};
