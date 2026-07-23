/**
 * Common enemy population abstraction for the Neatenstein co-evolution harness.
 *
 * This module re-exports the {@link EnemyPopulation} interface so backend-
 * specific implementations (MLP, SWARM) can declare they satisfy the same
 * contract. The harness swaps backends without changing selection, barrier,
 * or main-runner logic.
 *
 * @module
 */

export type { EnemyPopulation } from './types';
