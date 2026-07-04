import type { SharedField } from './neat.nge-collective.types';

export type { SharedField };

/**
 * @module neat.nge-collective.shared-field
 *
 * Float32Array-backed 2D pheromone/signal field for multi-agent collective evaluation.
 *
 * The `SharedField` abstraction is the stigmergy substrate that connects agents within one
 * evaluation tick. Because the backing `Float32Array` is passed by reference through the
 * `CollectiveEvaluationContext`, writes committed by an earlier evaluator are immediately
 * visible to every later evaluator in the same tick — exactly the indirect coordination
 * contract required by collective signal fields and multi-agent coordination channels.
 *
 * ## Cell layout
 *
 * ```
 * col → 0   1   2   ...  (width - 1)
 * row 0: [0,  1,  2,  ..., w-1        ]
 * row 1: [w,  w+1,w+2,..., 2w-1       ]
 * ...
 * ```
 *
 * Cell `(x, y)` maps to flat index `y × width + x`.
 *
 * ## Field lifecycle (one generation)
 *
 * ```mermaid
 * stateDiagram-v2
 *   [*] --> Created: createSharedField
 *   Created --> Active: bound into CollectiveEvaluationContext
 *   Active --> Active: writeCell / readCell (per evaluator, same tick)
 *   Active --> Decayed: applyDecay (returns new field)
 *   Decayed --> Diffused: applyDiffusion (returns new field)
 *   Diffused --> Cleared: clearField / resetCollectiveEvaluationState
 *   Cleared --> Active: next generation tick begins
 * ```
 *
 * ## Immutability contract
 *
 * `applyDecay`, `applyDiffusion`, and `clearField` all return **new** `SharedField` instances
 * and never mutate the source field. Only `writeCell` mutates in-place (deliberately, to
 * allow sequential evaluators within the same tick to observe each other's writes).
 */

/**
 * Creates a new zeroed shared field with the given dimensions.
 *
 * The backing store is a `Float32Array` of `width × height` elements, all initialized to `0`.
 * Cell `(x, y)` maps to flat index `y * width + x` (row-major order).
 *
 * @param width - Number of columns in the 2D grid.
 * @param height - Number of rows in the 2D grid.
 * @returns A new `SharedField` with all cells initialized to `0`.
 *
 * @example
 * ```ts
 * const field = createSharedField(10, 10); // 100-element Float32Array
 * ```
 */
export function createSharedField(width: number, height: number): SharedField {
  return { width, height, cells: new Float32Array(width * height) };
}

/**
 * Writes a value to the cell at `(x, y)` in the shared field.
 *
 * Mutates the backing `Float32Array` **in-place** so that sequential evaluators
 * within the same collective tick observe each other's writes via the shared reference.
 * Returns the same `field` reference for fluent chaining.
 *
 * @param field - The shared field to write into.
 * @param x - Column index (0-based).
 * @param y - Row index (0-based).
 * @param value - Value to store at the target cell.
 * @returns The same `field` reference (mutation is in-place).
 *
 * @example
 * ```ts
 * const field = createSharedField(3, 3);
 * writeCell(field, 1, 1, 0.75); // field.cells[4] === 0.75
 * ```
 */
export function writeCell(
  field: SharedField,
  x: number,
  y: number,
  value: number,
): SharedField {
  // Step 1: Convert 2D coordinate to row-major flat index and write in-place.
  field.cells[y * field.width + x] = value;
  return field;
}

/**
 * Reads the value stored at cell `(x, y)` in the shared field.
 *
 * @param field - The shared field to read from.
 * @param x - Column index (0-based).
 * @param y - Row index (0-based).
 * @returns The stored cell value, or `0` if the index is out of bounds.
 *
 * @example
 * ```ts
 * const value = readCell(field, 1, 1); // 0.75
 * ```
 */
export function readCell(field: SharedField, x: number, y: number): number {
  return field.cells[y * field.width + x] ?? 0;
}

/**
 * Applies exponential pheromone decay to every cell in the field.
 * Returns a **new** `SharedField`; the original is **not** mutated.
 *
 * Each cell's new value is: `oldValue × factor`.
 *
 * @param field - Source field to decay.
 * @param factor - Decay multiplier in `[0, 1]`. A value of `0.95` retains 95% per tick.
 * @returns A new `SharedField` with decayed cell values.
 *
 * @example
 * ```ts
 * const decayed = applyDecay(field, 0.95); // all cells × 0.95
 * ```
 */
export function applyDecay(field: SharedField, factor: number): SharedField {
  // Step 1: Allocate a new backing array.
  const newCells = new Float32Array(field.cells.length);

  // Step 2: Multiply every cell by the decay factor.
  for (let cellIndex = 0; cellIndex < field.cells.length; cellIndex++) {
    newCells[cellIndex] = field.cells[cellIndex] * factor;
  }

  return { width: field.width, height: field.height, cells: newCells };
}

/**
 * Applies one lateral diffusion step to the shared field.
 * Returns a **new** `SharedField`; the original is **not** mutated.
 *
 * Each cell donates `rate × cellValue / neighborCount` to each of its 4-connected
 * grid neighbors and retains `cellValue × (1 − rate)`. Deterministic: identical
 * inputs always produce byte-identical outputs.
 *
 * @param field - Source field to diffuse.
 * @param rate - Diffusion rate in `[0, 1]`. A value of `0.5` spreads half the cell's value.
 * @returns A new `SharedField` with diffused cell values.
 *
 * @example
 * ```ts
 * const diffused = applyDiffusion(field, 0.1); // 10% of each cell spreads to neighbors
 * ```
 */
export function applyDiffusion(field: SharedField, rate: number): SharedField {
  // Step 1: Allocate a zeroed output array.
  const newCells = new Float32Array(field.cells.length);

  // Step 2: For each source cell, distribute its value to self and 4-connected neighbors.
  for (let row = 0; row < field.height; row++) {
    for (let col = 0; col < field.width; col++) {
      const cellIndex = row * field.width + col;
      const cellValue = field.cells[cellIndex];
      const neighborCoords = collectNeighborCoordinates(
        col,
        row,
        field.width,
        field.height,
      );
      const neighborCount = neighborCoords.length;

      if (neighborCount === 0) {
        newCells[cellIndex] = newCells[cellIndex] + cellValue;
        continue;
      }

      const outflowFraction = rate * cellValue;
      const outflowPerNeighbor = outflowFraction / neighborCount;

      // Step 3: Cell retains its value minus the total outflow.
      newCells[cellIndex] = newCells[cellIndex] + cellValue - outflowFraction;

      // Step 4: Distribute outflow evenly across each valid 4-connected neighbor.
      for (const [neighborCol, neighborRow] of neighborCoords) {
        const neighborIndex = neighborRow * field.width + neighborCol;
        newCells[neighborIndex] = newCells[neighborIndex] + outflowPerNeighbor;
      }
    }
  }

  return { width: field.width, height: field.height, cells: newCells };
}

/**
 * Zeros every cell in the field and returns a **new** `SharedField`.
 * The original field is **not** mutated.
 *
 * @param field - Source field to clear.
 * @returns A new `SharedField` with all cells set to `0`.
 *
 * @example
 * ```ts
 * const clean = clearField(field); // all cells === 0
 * ```
 */
export function clearField(field: SharedField): SharedField {
  return {
    width: field.width,
    height: field.height,
    cells: new Float32Array(field.cells.length),
  };
}

/**
 * Collects the 4-connected neighbor coordinates for a given grid cell,
 * filtering out coordinates that fall outside the grid boundaries.
 *
 * @param col - Column index of the source cell.
 * @param row - Row index of the source cell.
 * @param width - Grid width used as the column boundary.
 * @param height - Grid height used as the row boundary.
 * @returns Array of valid `[col, row]` neighbor coordinate pairs.
 */
function collectNeighborCoordinates(
  col: number,
  row: number,
  width: number,
  height: number,
): Array<[number, number]> {
  const candidates: Array<[number, number]> = [
    [col, row - 1],
    [col, row + 1],
    [col - 1, row],
    [col + 1, row],
  ];
  return candidates.filter(
    ([neighborCol, neighborRow]) =>
      neighborCol >= 0 &&
      neighborCol < width &&
      neighborRow >= 0 &&
      neighborRow < height,
  );
}
