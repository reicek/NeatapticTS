/**
 * Red-phase test contract for the NGE collective shared-field primitive.
 *
 * Target production module: `./neat.nge-collective.shared-field`
 * Phase G Step 03 — intended failures before any implementation exists.
 *
 * Covered behaviors:
 * 1. Typed-array-backed field construction with correct dimension metadata.
 * 2. Deterministic write/read round-trip at an arbitrary cell coordinate.
 * 3. Decay step scales every cell by the decay factor exactly.
 * 4. Diffusion step spreads value from a seeded center cell to adjacent cells.
 * 5. Clear step zeros every cell in the field.
 */

import {
  applyDecay,
  applyDiffusion,
  clearField,
  createSharedField,
  readCell,
  writeCell,
} from './neat.nge-collective.shared-field';

describe('neat.nge-collective shared-field primitive', () => {
  describe('createSharedField', () => {
    describe('given width=5 and height=4', () => {
      it('returns a field with the declared width', () => {
        // Arrange / Act
        const field = createSharedField(5, 4);

        // Assert
        expect(field.width).toBe(5);
      });

      it('returns a field with the declared height', () => {
        // Arrange / Act
        const field = createSharedField(5, 4);

        // Assert
        expect(field.height).toBe(4);
      });

      it('returns a field whose backing Float32Array has width × height elements', () => {
        // Arrange / Act
        const field = createSharedField(5, 4);

        // Assert
        expect(field.cells.length).toBe(20);
      });

      it('initializes every cell to zero', () => {
        // Arrange / Act
        const field = createSharedField(5, 4);

        // Assert — every cell starts at 0
        expect(Array.from(field.cells).every((value) => value === 0)).toBe(
          true,
        );
      });
    });
  });

  describe('writeCell / readCell', () => {
    describe('given a 3×3 field', () => {
      it('round-trips a positive value written to an interior cell', () => {
        // Arrange
        const field = createSharedField(3, 3);
        const writtenValue = 0.75;

        // Act
        const updatedField = writeCell(field, 1, 1, writtenValue);

        // Assert
        expect(readCell(updatedField, 1, 1)).toBeCloseTo(writtenValue, 6);
      });

      it('does not mutate cells outside the written coordinate', () => {
        // Arrange
        const field = createSharedField(3, 3);

        // Act
        const updatedField = writeCell(field, 1, 1, 1.0);

        // Assert — corner cell at (0,0) remains zero
        expect(readCell(updatedField, 0, 0)).toBe(0);
      });

      it('overwrites an existing value at the same coordinate', () => {
        // Arrange
        const field = createSharedField(3, 3);
        const firstWrite = writeCell(field, 2, 2, 0.5);

        // Act
        const secondWrite = writeCell(firstWrite, 2, 2, 0.3);

        // Assert
        expect(readCell(secondWrite, 2, 2)).toBeCloseTo(0.3, 6);
      });
    });

    describe('given an out-of-bounds coordinate', () => {
      it('returns 0 when the requested coordinate is outside the field dimensions', () => {
        // Arrange — 2×2 field; coordinate (10, 10) is out of bounds
        const field = createSharedField(2, 2);

        // Act
        const value = readCell(field, 10, 10);

        // Assert — out-of-bounds access returns the default 0
        expect(value).toBe(0);
      });
    });
  });

  describe('applyDecay', () => {
    describe('given a 3×3 field with all cells set to 1.0 and decay factor 0.9', () => {
      it('scales every cell to 0.9 after one decay step', () => {
        // Arrange
        const rawField = createSharedField(3, 3);
        const initialField = Array.from({ length: 9 }, (_, cellIndex) =>
          writeCell(rawField, cellIndex % 3, Math.floor(cellIndex / 3), 1.0),
        ).at(-1)!;

        // Act
        const decayedField = applyDecay(initialField, 0.9);

        // Assert — all cells are 0.9 (1.0 × 0.9)
        expect(
          (Array.from(decayedField.cells) as number[]).every(
            (value) => Math.abs(value - 0.9) < 1e-5,
          ),
        ).toBe(true);
      });

      it('produces a new field object rather than mutating the original', () => {
        // Arrange
        const rawField = createSharedField(2, 2);
        const filledField = writeCell(
          writeCell(
            writeCell(writeCell(rawField, 0, 0, 1.0), 1, 0, 1.0),
            0,
            1,
            1.0,
          ),
          1,
          1,
          1.0,
        );

        // Act
        applyDecay(filledField, 0.5);

        // Assert — original cell is unchanged after decay is applied to a copy
        expect(readCell(filledField, 0, 0)).toBeCloseTo(1.0, 6);
      });

      it('returns the decayed cell value in the new field', () => {
        // Arrange
        const rawField = createSharedField(2, 2);
        const filledField = writeCell(
          writeCell(
            writeCell(writeCell(rawField, 0, 0, 1.0), 1, 0, 1.0),
            0,
            1,
            1.0,
          ),
          1,
          1,
          1.0,
        );

        // Act
        const decayedField = applyDecay(filledField, 0.5);

        // Assert — new field carries the decayed value
        expect(readCell(decayedField, 0, 0)).toBeCloseTo(0.5, 6);
      });
    });
  });

  describe('applyDiffusion', () => {
    describe('given a 3×3 field with center cell = 1.0 and all others = 0.0, diffusion rate = 0.5', () => {
      it('reduces the center cell value below 1.0 after one diffusion step', () => {
        // Arrange
        const field = writeCell(createSharedField(3, 3), 1, 1, 1.0);

        // Act
        const diffusedField = applyDiffusion(field, 0.5);

        // Assert — center gave some value to neighbors
        expect(readCell(diffusedField, 1, 1)).toBeLessThan(1.0);
      });

      it('raises at least one adjacent cell above 0.0 after one diffusion step', () => {
        // Arrange
        const field = writeCell(createSharedField(3, 3), 1, 1, 1.0);

        // Act
        const diffusedField = applyDiffusion(field, 0.5);

        // Assert — top neighbor received some value
        expect(readCell(diffusedField, 1, 0)).toBeGreaterThan(0);
      });

      it('produces identical output for two calls with the same inputs (determinism)', () => {
        // Arrange
        const field = writeCell(createSharedField(3, 3), 1, 1, 1.0);

        // Act
        const firstRun = applyDiffusion(field, 0.5);
        const secondRun = applyDiffusion(field, 0.5);

        // Assert — same inputs → byte-identical Float32Array
        expect(Array.from(firstRun.cells)).toEqual(Array.from(secondRun.cells));
      });
    });

    describe('given a 1×1 field (no 4-connected neighbors)', () => {
      it('preserves the cell value unchanged because there are no neighbors to diffuse to', () => {
        // Arrange — a 1×1 grid has neighborCount === 0 for its only cell
        const field = writeCell(createSharedField(1, 1), 0, 0, 1.0);

        // Act
        const diffusedField = applyDiffusion(field, 0.5);

        // Assert — value retained intact; no neighbors to receive outflow
        expect(readCell(diffusedField, 0, 0)).toBeCloseTo(1.0, 6);
      });
    });
  });

  describe('clearField', () => {
    describe('given a 3×3 field with several nonzero cells', () => {
      it('returns a field with all cells set to zero', () => {
        // Arrange
        const field = writeCell(
          writeCell(createSharedField(3, 3), 0, 0, 0.8),
          2,
          2,
          0.5,
        );

        // Act
        const clearedField = clearField(field);

        // Assert
        expect(
          Array.from(clearedField.cells).every((value) => value === 0),
        ).toBe(true);
      });

      it('does not mutate the original field', () => {
        // Arrange
        const field = writeCell(createSharedField(3, 3), 1, 1, 0.9);

        // Act
        clearField(field);

        // Assert — original cell unchanged
        expect(readCell(field, 1, 1)).toBeCloseTo(0.9, 6);
      });
    });
  });
});
