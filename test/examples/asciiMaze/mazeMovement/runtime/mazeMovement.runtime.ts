/**
 * Runtime/environment helpers for the dedicated mazeMovement module.
 *
 * This file owns the low-level simulation primitives that do not define maze
 * policy: cell-open checks, distance lookup, run-state creation, visit-ring
 * bookkeeping, and perception-state updates.
 */

import { MAZE_MOVEMENT_CONSTANTS } from '../mazeMovement.constants';
import {
	getMazeMovementBufferMetadata,
	getMazeMovementRunServiceState,
	indexMazeMovementCell,
	initializeMazeMovementBufferPools,
	requireMazeMovementBufferPools,
	resetMazeMovementRunServiceState,
} from '../mazeMovement.services';
import type { SimulationState } from '../mazeMovement.types';
import { MazeVision } from '../../mazeVision';

const C = MAZE_MOVEMENT_CONSTANTS;

/**
 * Determine whether a maze cell is inside bounds and not a wall.
 *
 * @param encodedMaze - Maze grid to inspect.
 * @param x - Zero-based maze column.
 * @param y - Zero-based maze row.
 * @param coordinateScratch - Reused integer scratch buffer for coordinate coercion.
 * @returns True when the target cell is within bounds and open.
 */
export function isMazeMovementCellOpen(
	encodedMaze: ReadonlyArray<ReadonlyArray<number>>,
	x: number,
	y: number,
	coordinateScratch: Int32Array,
): boolean {
	const providedRowCount = encodedMaze?.length ?? 0;
	const firstRow = encodedMaze?.[0];
	const providedColumnCount = firstRow?.length ?? 0;
	const { cachedWidth: cachedColumnCount, cachedHeight: cachedRowCount } =
		getMazeMovementBufferMetadata();

	const mazeColumnCount =
		cachedColumnCount > 0 &&
		cachedRowCount === providedRowCount &&
		cachedColumnCount === providedColumnCount
			? cachedColumnCount
			: providedColumnCount;
	const mazeRowCount =
		cachedRowCount > 0 &&
		cachedColumnCount === providedColumnCount &&
		cachedRowCount === providedRowCount
			? cachedRowCount
			: providedRowCount;

	coordinateScratch[0] = x | 0;
	coordinateScratch[1] = y | 0;
	const columnIndex = coordinateScratch[0];
	const rowIndex = coordinateScratch[1];

	if (rowIndex < 0 || rowIndex >= mazeRowCount) return false;
	if (columnIndex < 0 || columnIndex >= mazeColumnCount) return false;

	const targetRow = encodedMaze[rowIndex];
	if (!targetRow) return false;
	const cellValue = targetRow[columnIndex];
	return cellValue !== -1;
}

/**
 * Resolve the current distance value for a maze coordinate.
 *
 * @param encodedMaze - Maze grid aligned with the optional distance map.
 * @param coordinates - Zero-based `[x, y]` coordinate tuple.
 * @param distanceMap - Optional precomputed distance map.
 * @returns Finite distance when present, otherwise `Infinity`.
 */
export function getMazeMovementDistance(
	encodedMaze: ReadonlyArray<ReadonlyArray<number>>,
	[x, y]: readonly [number, number],
	distanceMap?: number[][],
): number {
	const xCoordinate = x | 0;
	const yCoordinate = y | 0;

	if (
		distanceMap &&
		distanceMap[yCoordinate] !== undefined &&
		Number.isFinite(distanceMap[yCoordinate][xCoordinate])
	) {
		return distanceMap[yCoordinate][xCoordinate];
	}

	const providedHeight = encodedMaze?.length ?? 0;
	const firstRow = encodedMaze?.[0];
	const providedWidth = firstRow?.length ?? 0;
	const { cachedWidth, cachedHeight } = getMazeMovementBufferMetadata();

	const mazeWidth =
		cachedWidth > 0 &&
		cachedHeight === providedHeight &&
		cachedWidth === providedWidth
			? cachedWidth
			: providedWidth;
	const mazeHeight =
		cachedHeight > 0 &&
		cachedWidth === providedWidth &&
		cachedHeight === providedHeight
			? cachedHeight
			: providedHeight;

	if (xCoordinate < 0 || xCoordinate >= mazeWidth) return Infinity;
	if (yCoordinate < 0 || yCoordinate >= mazeHeight) return Infinity;

	return Infinity;
}

/**
 * Create the initial run-state object for one simulation episode.
 *
 * @param encodedMaze - Maze grid used by the run.
 * @param startPos - Starting coordinate.
 * @param distanceMap - Optional precomputed distance map.
 * @param maxSteps - Maximum number of allowed simulation steps.
 * @returns Fresh simulation state backed by the shared buffer pools.
 */
export function createMazeMovementRunState(
	encodedMaze: number[][],
	startPos: readonly [number, number],
	distanceMap: number[][] | undefined,
	maxSteps: number,
): SimulationState {
	resetMazeMovementRunServiceState();
	const mazeHeight = encodedMaze.length;
	const mazeWidth = encodedMaze[0].length;
	const hasDistanceMap =
		Array.isArray(distanceMap) && distanceMap.length === mazeHeight;
	const bufferPools = initializeMazeMovementBufferPools(
		mazeWidth,
		mazeHeight,
		maxSteps,
	);

	const position: [number, number] = [startPos[0], startPos[1]];
	bufferPools.pathX[0] = position[0];
	bufferPools.pathY[0] = position[1];
	const historyCapacity = C.MOVE_HISTORY_LENGTH;

	return {
		position,
		steps: 0,
		pathLength: 1,
		visitedUniqueCount: 0,
		hasDistanceMap,
		distanceMap,
		minDistanceToExit: hasDistanceMap
			? (distanceMap![position[1]]?.[position[0]] ?? Infinity)
			: getMazeMovementDistance(encodedMaze, position, distanceMap),
		progressReward: 0,
		newCellExplorationBonus: 0,
		invalidMovePenalty: 0,
		prevAction: C.NO_MOVE,
		stepsSinceImprovement: 0,
		lastDistanceGlobal: getMazeMovementDistance(
			encodedMaze,
			position,
			distanceMap,
		),
		saturatedSteps: 0,
		recentPositions: [],
		localAreaPenalty: 0,
		directionCounts: [0, 0, 0, 0],
		moveHistoryRing: new Int32Array(historyCapacity),
		moveHistoryLength: 0,
		moveHistoryHead: 0,
		currentCellIndex: 0,
		loopPenalty: 0,
		memoryPenalty: 0,
		revisitPenalty: 0,
		visitsAtCurrent: 0,
		distHere: Infinity,
		vision: [],
		actionStats: null,
		direction: C.NO_MOVE,
		moved: false,
		prevDistance: Infinity,
		earlyTerminate: false,
	};
}

/**
 * Push a cell index into the circular visit-history ring.
 *
 * @param state - Mutable simulation state containing the ring buffer.
 * @param cellIndex - Linearized cell index to append.
 */
export function pushMazeMovementHistory(
	state: SimulationState,
	cellIndex: number,
): void {
	const ring = state.moveHistoryRing;
	let headIndex = state.moveHistoryHead | 0;
	const currentLength = state.moveHistoryLength;
	const ringCapacity = ring.length;
	if (ringCapacity === 0) return;

	ring[headIndex] = cellIndex;
	headIndex = (headIndex + 1) % ringCapacity;
	state.moveHistoryHead = headIndex;
	if (currentLength < ringCapacity) state.moveHistoryLength = currentLength + 1;
}

/**
 * Return the `nth` most recent cell index from the visit-history ring.
 *
 * @param state - Mutable simulation state containing the ring buffer.
 * @param nth - One-based index from the history tail.
 * @returns The requested cell index or `undefined` when out of range.
 */
export function getMazeMovementHistoryFromEnd(
	state: SimulationState,
	nth: number,
): number | undefined {
	const requested = nth | 0;
	const historyLength = state.moveHistoryLength | 0;
	if (requested <= 0 || requested > historyLength) return undefined;

	const ring = state.moveHistoryRing;
	const ringCapacity = ring.length;
	if (ringCapacity === 0) return undefined;
	const headIndex = state.moveHistoryHead | 0;

	let rawIndex = headIndex - requested;
	rawIndex = ((rawIndex % ringCapacity) + ringCapacity) % ringCapacity;
	return ring[rawIndex];
}

/**
 * Record the current cell visit and update visit-driven penalties.
 *
 * @param state - Mutable simulation state for the active run.
 */
export function recordMazeMovementVisitAndPenalties(
	state: SimulationState,
): void {
	const bufferPools = requireMazeMovementBufferPools();
	const visitedFlags = bufferPools.visitedFlags;
	const visitCounts = bufferPools.visitCounts;
	const rewardScale = C.REWARD_SCALE;

	const cellIndex = indexMazeMovementCell(
		state.position[0],
		state.position[1],
	);
	state.currentCellIndex = cellIndex;
	if (!visitedFlags[cellIndex]) {
		visitedFlags[cellIndex] = 1;
		state.visitedUniqueCount++;
	}

	visitCounts[cellIndex] = (visitCounts[cellIndex] + 1) as number;
	pushMazeMovementHistory(state, cellIndex);
	const visitsAtCell = (state.visitsAtCurrent = visitCounts[cellIndex]);

	state.loopPenalty = 0;
	if (state.moveHistoryLength >= C.OSCILLATION_DETECT_LENGTH) {
		const last = getMazeMovementHistoryFromEnd(state, 1)!;
		const secondLast = getMazeMovementHistoryFromEnd(state, 2);
		const thirdLast = getMazeMovementHistoryFromEnd(state, 3);
		const fourthLast = getMazeMovementHistoryFromEnd(state, 4);
		if (
			last === thirdLast &&
			secondLast !== undefined &&
			fourthLast !== undefined &&
			secondLast === fourthLast
		) {
			state.loopPenalty = -C.LOOP_PENALTY * rewardScale;
		}
	}

	state.memoryPenalty = 0;
	if (state.moveHistoryLength > 1) {
		for (
			let historyOffset = 2;
			historyOffset <= state.moveHistoryLength;
			historyOffset++
		) {
			const recentIndex = getMazeMovementHistoryFromEnd(state, historyOffset);
			if (recentIndex === cellIndex) {
				state.memoryPenalty = -C.MEMORY_RETURN_PENALTY * rewardScale;
				break;
			}
		}
	}

	state.revisitPenalty = 0;
	if (visitsAtCell > 1) {
		state.revisitPenalty =
			-C.REVISIT_PENALTY_PER_VISIT * (visitsAtCell - 1) * rewardScale;
	}

	if (visitsAtCell > C.VISIT_TERMINATION_THRESHOLD) {
		state.invalidMovePenalty -= C.INVALID_MOVE_PENALTY_HARSH * rewardScale;
		state.earlyTerminate = true;
	}
}

/**
 * Build the current perception vector and update distance-tracking state.
 *
 * @param state - Mutable simulation state for the active run.
 * @param encodedMaze - Maze grid used for perception and distance lookup.
 * @param exitPos - Goal coordinate for the current run.
 * @param distanceMap - Optional precomputed distance map.
 */
export function buildMazeMovementVisionAndDistance(
	state: SimulationState,
	encodedMaze: number[][],
	exitPos: readonly [number, number],
	distanceMap?: number[][],
): void {
	if (state.earlyTerminate) return;

	const currentPosition = state.position;
	const positionX = currentPosition[0] | 0;
	const positionY = currentPosition[1] | 0;
	const hasPrecomputedDistances = state.hasDistanceMap;
	const runServices = getMazeMovementRunServiceState();

	const preMoveDistance = hasPrecomputedDistances
		? (distanceMap![positionY]?.[positionX] ?? undefined)
		: getMazeMovementDistance(encodedMaze, currentPosition, distanceMap);

	const visionInputs = MazeVision.buildInputs6(
		encodedMaze,
		currentPosition,
		exitPos,
		distanceMap,
		runServices.prevDistanceStep,
		preMoveDistance,
		state.prevAction,
	);

	state.vision = (
		Array.isArray(visionInputs)
			? visionInputs
			: Array.from(visionInputs as Iterable<number>)
	) as number[];

	runServices.prevDistanceStep = preMoveDistance;
	state.distHere = hasPrecomputedDistances
		? (distanceMap![positionY]?.[positionX] ?? Infinity)
		: getMazeMovementDistance(encodedMaze, currentPosition, distanceMap);
}

export {};