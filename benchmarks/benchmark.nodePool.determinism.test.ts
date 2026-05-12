import fs from 'fs';
import path from 'path';
import Network from '../src/architecture/network';
import { config } from '../src/config';

/**
 * Runtime interface for mutable config properties in tests.
 */
interface MutableConfig {
  enableNodePooling: boolean;
}

interface DeterminismReplayArtifact {
  determinismReplay?: {
    checks: Array<{
      passed: boolean;
      scenario: string;
    }>;
    generatedAt: string;
    passed: boolean;
  };
}

/**
 * Persist the current determinism replay result into the shared benchmark artifact.
 *
 * @param passed - Whether the determinism replay check succeeded.
 * @returns Parsed artifact after the determinism section is written.
 */
function persistDeterminismReplayResult(
  passed: boolean,
): DeterminismReplayArtifact {
  const resultsPath = path.resolve(__dirname, 'benchmark.results.json');
  const currentArtifact = readDeterminismReplayArtifact(resultsPath);
  const updatedArtifact: DeterminismReplayArtifact = {
    ...currentArtifact,
    determinismReplay: {
      checks: [
        {
          passed,
          scenario: 'nodePool-forward-parity',
        },
      ],
      generatedAt: new Date().toISOString(),
      passed,
    },
  };

  fs.writeFileSync(
    resultsPath,
    JSON.stringify(updatedArtifact, null, 2),
    'utf8',
  );

  return updatedArtifact;
}

/**
 * Read the persisted benchmark artifact when present.
 *
 * @param resultsPath - Absolute benchmark artifact path.
 * @returns Parsed artifact or an empty object when the file cannot be read.
 */
function readDeterminismReplayArtifact(
  resultsPath: string,
): DeterminismReplayArtifact {
  if (!fs.existsSync(resultsPath)) {
    return {};
  }

  try {
    return JSON.parse(
      fs.readFileSync(resultsPath, 'utf8'),
    ) as DeterminismReplayArtifact;
  } catch {
    return {};
  }
}

describe('benchmark.nodePool.determinism', () => {
  describe('forward parity pooling off vs on', () => {
    const seed = 1337;
    const inputValues = [0.25, -0.1, 0.9];
    let determinismReplayArtifact: DeterminismReplayArtifact;

    beforeAll(() => {
      // Step 1: Build matching seeded networks with pooling disabled and enabled.
      (config as unknown as MutableConfig).enableNodePooling = false;
      const networkWithoutPooling = new Network(3, 2, { seed });
      (config as unknown as MutableConfig).enableNodePooling = true;
      const networkWithPooling = new Network(3, 2, { seed });

      // Step 2: Replay the same input vector through both networks.
      const outputWithoutPooling = networkWithoutPooling.activate(
        inputValues.slice(),
      );
      const outputWithPooling = networkWithPooling.activate(
        inputValues.slice(),
      );

      // Step 3: Persist the replay outcome for the Phase 10 release gate.
      determinismReplayArtifact = persistDeterminismReplayResult(
        JSON.stringify(outputWithoutPooling) ===
          JSON.stringify(outputWithPooling),
      );
    });

    it('should produce identical outputs with same seed (pool off vs on)', () => {
      // Assert
      expect(determinismReplayArtifact.determinismReplay?.passed).toBe(true);
    });

    it('persists a determinism replay section in the benchmark artifact', () => {
      // Assert
      expect(
        determinismReplayArtifact.determinismReplay?.checks[0]?.scenario,
      ).toBe('nodePool-forward-parity');
    });
  });
});
