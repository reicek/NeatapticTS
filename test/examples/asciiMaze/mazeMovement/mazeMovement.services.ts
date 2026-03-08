/**
 * Shared mutable services for the dedicated mazeMovement module.
 *
 * This module owns the pooled buffers, PRNG state, output-history plumbing,
 * and shared run-scoped counters used by the legacy MazeMovement facade while
 * Step 2 incrementally moves helper categories into the dedicated boundary.
 */

import type { INetwork } from '../interfaces';
import type {
	MazeMovementBufferPools,
	MazeMovementRunServiceState,
} from './mazeMovement.types';
import {
	isFiniteNumberArray,
	materializePath,
	nextPowerOfTwo,
} from './mazeMovement.utils';

interface InternalMazeMovementBufferPools {
	visitedFlags: Uint8Array | null;
	visitCounts: Uint16Array | null;
	pathX: Int32Array | null;
	pathY: Int32Array | null;
	gridCapacity: number;
	pathCapacity: number;
	cachedWidth: number;
	cachedHeight: number;
}

const BUFFER_POOLS: InternalMazeMovementBufferPools = {
	visitedFlags: null,
	visitCounts: null,
	pathX: null,
	pathY: null,
	gridCapacity: 0,
	pathCapacity: 0,
	cachedWidth: 0,
	cachedHeight: 0,
};

const RUN_SERVICE_STATE: MazeMovementRunServiceState = {
	saturations: 0,
	noMoveStreak: 0,
	prevDistanceStep: undefined,
};

const PRNG_STATE = {
	value: null as Uint32Array | null,
};

/**
 * Expose the shared mutable run-scoped state used across helper categories.
 *
 * @returns The singleton mutable run-state object for the current process.
 */
export function getMazeMovementRunServiceState(): MazeMovementRunServiceState {
	return RUN_SERVICE_STATE;
}

/**
 * Reset the shared mutable run-scoped state before a new simulation begins.
 *
 * @returns The reused singleton state after reset.
 */
export function resetMazeMovementRunServiceState(): MazeMovementRunServiceState {
	RUN_SERVICE_STATE.saturations = 0;
	RUN_SERVICE_STATE.noMoveStreak = 0;
	RUN_SERVICE_STATE.prevDistanceStep = undefined;
	return RUN_SERVICE_STATE;
}

/**
 * Ensure the pooled grid and path buffers are initialized for a run.
 *
 * @param width - Maze width in cells.
 * @param height - Maze height in cells.
 * @param maxSteps - Maximum path length expected for the run.
 * @returns The initialized pooled buffer surface.
 */
export function initializeMazeMovementBufferPools(
	width: number,
	height: number,
	maxSteps: number,
): MazeMovementBufferPools {
	const requiredCellCount = width * height;

	if (
		BUFFER_POOLS.visitedFlags == null ||
		requiredCellCount > BUFFER_POOLS.gridCapacity
	) {
		const newCellCapacity = nextPowerOfTwo(requiredCellCount);
		BUFFER_POOLS.visitedFlags = new Uint8Array(newCellCapacity);
		BUFFER_POOLS.visitCounts = new Uint16Array(newCellCapacity);
		BUFFER_POOLS.gridCapacity = newCellCapacity;
	} else {
		BUFFER_POOLS.visitedFlags.fill(0, 0, requiredCellCount);
		BUFFER_POOLS.visitCounts!.fill(0, 0, requiredCellCount);
	}

	const requiredPathEntries = maxSteps + 1;
	if (
		BUFFER_POOLS.pathX == null ||
		requiredPathEntries > BUFFER_POOLS.pathCapacity
	) {
		const newPathCapacity = nextPowerOfTwo(requiredPathEntries);
		BUFFER_POOLS.pathX = new Int32Array(newPathCapacity);
		BUFFER_POOLS.pathY = new Int32Array(newPathCapacity);
		BUFFER_POOLS.pathCapacity = newPathCapacity;
	}

	BUFFER_POOLS.cachedWidth = width;
	BUFFER_POOLS.cachedHeight = height;

	return requireMazeMovementBufferPools();
}

/**
 * Return the initialized pooled buffer surface for the current run.
 *
 * @returns The shared buffer pools.
 * @throws Error when a caller reaches the pools before initialization.
 */
export function requireMazeMovementBufferPools(): MazeMovementBufferPools {
	if (
		BUFFER_POOLS.visitedFlags == null ||
		BUFFER_POOLS.visitCounts == null ||
		BUFFER_POOLS.pathX == null ||
		BUFFER_POOLS.pathY == null
	) {
		throw new Error('Maze movement buffer pools were used before initialization.');
	}

	return BUFFER_POOLS as MazeMovementBufferPools;
}

/**
 * Read the currently cached maze dimensions for bounds and index helpers.
 *
 * @returns Cached width and height for the active pooled buffers.
 */
export function getMazeMovementBufferMetadata(): {
	cachedWidth: number;
	cachedHeight: number;
} {
	return {
		cachedWidth: BUFFER_POOLS.cachedWidth,
		cachedHeight: BUFFER_POOLS.cachedHeight,
	};
}

/**
 * Convert a cell coordinate into the pooled linear grid index.
 *
 * @param x - Zero-based maze column.
 * @param y - Zero-based maze row.
 * @returns Linearized index used by pooled grid buffers.
 */
export function indexMazeMovementCell(x: number, y: number): number {
	return Math.imul(y, BUFFER_POOLS.cachedWidth) + x;
}

/**
 * Generate a pseudo-random number in the range `[0, 1)`.
 *
 * @returns A deterministic or host-random unit float for exploration logic.
 */
export function randomMazeMovementUnit(): number {
	const pooledState = PRNG_STATE.value;
	if (pooledState == null || pooledState.length === 0) {
		return Math.random();
	}

	const current = (pooledState[0] + 0x6d2b79f5) >>> 0;
	pooledState[0] = current;

	let mixed = current;
	mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1) >>> 0;
	mixed =
		(mixed ^ (mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61))) >>> 0;

	const final32 = (mixed ^ (mixed >>> 14)) >>> 0;
	return final32 / 4_294_967_296;
}

/**
 * Read the reflected `_lastStepOutputs` network history when present.
 *
 * @param network - Network that may carry the reflected output history.
 * @returns Sanitized output history or `undefined` when absent or invalid.
 */
export function readMazeMovementOutputHistory(
	network: INetwork,
): number[][] | undefined {
	const historyCandidate = Reflect.get(network as object, '_lastStepOutputs');
	if (!Array.isArray(historyCandidate)) return undefined;
	return historyCandidate.every(isFiniteNumberArray)
		? (historyCandidate as number[][])
		: undefined;
}

/**
 * Persist the reflected `_lastStepOutputs` network history.
 *
 * @param network - Network receiving the reflected output history.
 * @param history - Bounded output-history payload to persist.
 */
export function writeMazeMovementOutputHistory(
	network: INetwork,
	history: number[][],
): void {
	Reflect.set(network as object, '_lastStepOutputs', history);
}

/**
 * Materialize the active pooled path buffers into a fresh tuple array.
 *
 * @param length - Number of active path entries to copy.
 * @returns A newly allocated materialized path snapshot.
 */
export function materializeMazeMovementPath(
	length: number,
): [number, number][] {
	const bufferPools = requireMazeMovementBufferPools();
	return materializePath(length, bufferPools.pathX, bufferPools.pathY);
}

export {};