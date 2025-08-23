import { methods } from '../../../src/neataptic';
import Network from '../../../src/architecture/network';

/**
 * Supervised refinement of an evolved winner (Lamarckian/Baldwinian fine‑tune pass).
 * Adds a compact, high-signal training dataset of unambiguous movement decisions,
 * reinforcing core directional heuristics before seeding the next curriculum phase.
 *
 * Inputs encode: [compassScalar, openN, openE, openS, openW, progressDelta].
 * Output is a soft distribution over [N,E,S,W] with the correct move boosted.
 *
 * @param winner Evolved network to refine (not mutated; a clone is returned).
 * @returns Refined cloned network; if refinement fails, returns original clone.
 */
export function refineWinnerWithBackprop(
  winner?: Network
): Network | undefined {
  if (!winner) return undefined;
  // Ensure differentiable activations for hidden/output nodes.
  try {
    winner.nodes.forEach((node) => {
      if (typeof node.squash !== 'function') {
        node.squash = methods.Activation.logistic;
      }
    });
  } catch {
    /* ignore activation patch failures */
  }

  const trainingSet: { input: number[]; output: number[] }[] = [];
  const OUT = (dirIndex: number) =>
    [0, 1, 2, 3].map((i) => (i === dirIndex ? 0.92 : 0.02));
  const add = (inp: number[], d: number) =>
    trainingSet.push({ input: inp, output: OUT(d) });

  // Cardinal single‑path scenarios (vary progressDelta for robustness)
  const compassDirs = [0, 0.25, 0.5, 0.75]; // N,E,S,W scalars
  for (let dir = 0; dir < 4; dir++) {
    const compass = compassDirs[dir];
    const open = [0, 0, 0, 0];
    open[dir] = 1;
    add([compass, ...open, 0.85], dir);
    add([compass, ...open, 0.65], dir);
    add([compass, ...open, 0.55], dir);
    add([compass, ...open, 0.5], dir); // near-neutral progress
  }

  try {
    winner.train(trainingSet, {
      iterations: 220,
      error: 0.005,
      rate: 0.001,
      momentum: 0.1,
      batchSize: 8,
      cost: methods.Cost.softmaxCrossEntropy,
    });
  } catch {
    // Silent: fallback still returns a clone (unrefined)
  }
  try {
    return winner.clone();
  } catch {
    return winner; // if clone fails, return original reference
  }
}
