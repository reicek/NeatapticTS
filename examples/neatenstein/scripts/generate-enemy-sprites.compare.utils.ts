/**
 * @module generate-enemy-sprites.compare.utils
 *
 * Snapshot comparison executors for the enemy sprite generator.
 *
 * Compares a generated snapshot PNG against an approved reference PNG using
 * silhouette IoU and color-class overlap metrics.
 */

import { decodePng } from './generate-enemy-sprites.png.utils';
import type { SnapshotComparison } from './generate-enemy-sprites.types';
import { RGBA_CHANNELS } from './generate-enemy-sprites.constants';

/**
 * Compare a generated snapshot PNG against an approved reference PNG.
 *
 * The metric is intentionally loose because the reference images are stylized
 * 192×192 art targets while the runtime pipeline renders a detailed voxel
 * model with shading. Two scores are returned:
 * - **Silhouette IoU**: overlap of opaque pixels.
 * - **Color-class overlap**: each pixel is classified to the nearest dominant
 *   color from the reference, then the per-class counts are compared with a
 *   Jaccard-like `sum(min) / sum(max)` ratio.
 *
 * The current implementation documents the thresholds in the test suite, not
 * in this helper, so callers can choose appropriate passing criteria.
 *
 * @param generatedPng - Generated PNG file contents.
 * @param referencePng - Approved reference PNG file contents.
 * @returns Comparison scores and opaque-pixel counts.
 */
export function compareSnapshotBuffers(
  generatedPng: Buffer,
  referencePng: Buffer,
): SnapshotComparison {
  const generated = decodePng(generatedPng);
  const reference = decodePng(referencePng);

  if (
    generated.width !== reference.width ||
    generated.height !== reference.height
  ) {
    throw new Error(
      `Size mismatch: generated ${generated.width}x${generated.height} vs reference ${reference.width}x${reference.height}`,
    );
  }

  const palette = extractReferencePalette(reference.data);
  const genClasses = classifyImage(generated.data, palette);
  const refClasses = classifyImage(reference.data, palette);

  let bothOpaque = 0;
  let genOpaque = 0;
  let refOpaque = 0;
  let sumMin = 0;
  let sumMax = 0;

  for (let i = 0; i < palette.length; i++) {
    const genCount = genClasses[i];
    const refCount = refClasses[i];
    sumMin += Math.min(genCount, refCount);
    sumMax += Math.max(genCount, refCount);
  }

  for (let i = 0; i < generated.data.length; i += RGBA_CHANNELS) {
    const genO = generated.data[i + 3] > 0;
    const refO = reference.data[i + 3] > 0;
    if (genO) genOpaque++;
    if (refO) refOpaque++;
    if (genO && refO) bothOpaque++;
  }

  const union = genOpaque + refOpaque - bothOpaque;
  const iou = union > 0 ? bothOpaque / union : 0;
  const colorSimilarity = sumMax > 0 ? sumMin / sumMax : 0;

  return {
    iou,
    colorSimilarity,
    generatedOpaque: genOpaque,
    referenceOpaque: refOpaque,
  };
}

/**
 * Extract the dominant opaque RGB colors from a reference image.
 *
 * @param data - Flat RGBA buffer.
 * @returns Up to 6 colors sorted by descending frequency.
 */
function extractReferencePalette(
  data: Buffer,
): Array<[number, number, number]> {
  const counts = new Map<string, number>();
  for (let i = 0; i < data.length; i += RGBA_CHANNELS) {
    if (data[i + 3] === 0) continue;
    const key = `${data[i]},${data[i + 1]},${data[i + 2]}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }

  return Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([key]) => {
      const [r, g, b] = key.split(',').map(Number);
      return [r, g, b] as [number, number, number];
    });
}

/**
 * Classify every opaque pixel of an image to the nearest reference-palette
 * color and return per-class counts.
 *
 * @param data - Flat RGBA buffer.
 * @param palette - Reference color palette.
 * @returns Per-class pixel counts.
 */
function classifyImage(
  data: Buffer,
  palette: Array<[number, number, number]>,
): number[] {
  const counts = new Array(palette.length).fill(0);
  if (palette.length === 0) return counts;

  for (let i = 0; i < data.length; i += RGBA_CHANNELS) {
    if (data[i + 3] === 0) continue;
    let best = 0;
    let bestDist = Infinity;
    for (let c = 0; c < palette.length; c++) {
      const [pr, pg, pb] = palette[c];
      const dist =
        (data[i] - pr) ** 2 + (data[i + 1] - pg) ** 2 + (data[i + 2] - pb) ** 2;
      if (dist < bestDist) {
        bestDist = dist;
        best = c;
      }
    }
    counts[best]++;
  }

  return counts;
}
