import fs from 'fs';
import path from 'path';
import { evaluateBenchmarkReleaseGates } from './benchmark.release-gates.utils';

describe('benchmark release gates', () => {
  describe('given the artifact is stable and regression-free', () => {
    it('returns a passing gate result', () => {
      // Arrange
      const gateResult = evaluateBenchmarkReleaseGates({
        aggregated: [
          {
            bytesPerConnMean: 64,
            mode: 'dist',
            scenario: 'buildForward',
            size: 100000,
          },
        ],
        determinismReplay: {
          checks: [
            {
              passed: true,
              scenario: 'nodePool-forward-parity',
            },
          ],
          generatedAt: '2026-05-10T00:00:00.000Z',
          passed: true,
        },
        generatedAt: '2026-05-10T00:00:00.000Z',
        history: [
          {
            commit: 'abc1234',
            distBundle: {
              hash: 'deadbeef1234',
            },
            generatedAt: '2026-05-09T00:00:00.000Z',
            summary: [
              {
                bytesPerConnMean: 64,
                size: 100000,
              },
              {
                bytesPerConnMean: 64,
                size: 200000,
              },
            ],
          },
          {
            commit: 'def5678',
            distBundle: {
              hash: 'feedface5678',
            },
            generatedAt: '2026-05-10T00:00:00.000Z',
            summary: [
              {
                bytesPerConnMean: 64,
                size: 100000,
              },
            ],
          },
        ],
        meta: {
          distBundle: {
            bytes: 2048,
            exists: true,
            hash: 'feedface5678',
          },
          varianceRepeatsLarge: 3,
        },
        variance: [
          {
            buildMsCvPct: 4,
            fwdAvgMsCvPct: 5,
            mode: 'dist',
            samples: 3,
            size: 100000,
          },
          {
            buildMsCvPct: 6,
            fwdAvgMsCvPct: 6,
            mode: 'dist',
            samples: 3,
            size: 200000,
          },
        ],
      });

      // Assert
      expect(gateResult.passed).toBe(true);
    });
  });

  describe('given the artifact contains a memory regression', () => {
    it('fails the gate result', () => {
      // Arrange
      const gateResult = evaluateBenchmarkReleaseGates({
        aggregated: [
          {
            bytesPerConnMean: 70,
            mode: 'dist',
            scenario: 'buildForward',
            size: 100000,
          },
        ],
        determinismReplay: {
          checks: [
            {
              passed: true,
              scenario: 'nodePool-forward-parity',
            },
          ],
          generatedAt: '2026-05-10T00:00:00.000Z',
          passed: true,
        },
        generatedAt: '2026-05-10T00:00:00.000Z',
        history: [
          {
            commit: 'abc1234',
            distBundle: {
              hash: 'deadbeef1234',
            },
            generatedAt: '2026-05-09T00:00:00.000Z',
            summary: [
              {
                bytesPerConnMean: 64,
                size: 100000,
              },
            ],
          },
          {
            commit: 'def5678',
            distBundle: {
              hash: 'feedface5678',
            },
            generatedAt: '2026-05-10T00:00:00.000Z',
            summary: [
              {
                bytesPerConnMean: 70,
                size: 100000,
              },
            ],
          },
        ],
        meta: {
          distBundle: {
            bytes: 2048,
            exists: true,
            hash: 'feedface5678',
          },
          varianceRepeatsLarge: 3,
        },
        variance: [
          {
            buildMsCvPct: 4,
            fwdAvgMsCvPct: 5,
            mode: 'dist',
            samples: 3,
            size: 100000,
          },
          {
            buildMsCvPct: 6,
            fwdAvgMsCvPct: 6,
            mode: 'dist',
            samples: 3,
            size: 200000,
          },
        ],
      });

      // Assert
      expect(
        gateResult.failures.some((failure) => failure.name === 'memory'),
      ).toBe(true);
    });
  });

  describe('given determinism replay evidence is missing', () => {
    it('fails the determinism gate', () => {
      // Arrange
      const gateResult = evaluateBenchmarkReleaseGates({
        aggregated: [
          {
            bytesPerConnMean: 64,
            mode: 'dist',
            scenario: 'buildForward',
            size: 100000,
          },
        ],
        generatedAt: '2026-05-10T00:00:00.000Z',
        history: [
          {
            commit: 'abc1234',
            distBundle: {
              hash: 'deadbeef1234',
            },
            generatedAt: '2026-05-09T00:00:00.000Z',
            summary: [
              {
                bytesPerConnMean: 64,
                size: 100000,
              },
            ],
          },
          {
            commit: 'def5678',
            distBundle: {
              hash: 'feedface5678',
            },
            generatedAt: '2026-05-10T00:00:00.000Z',
            summary: [
              {
                bytesPerConnMean: 64,
                size: 100000,
              },
            ],
          },
        ],
        meta: {
          distBundle: {
            bytes: 2048,
            exists: true,
            hash: 'feedface5678',
          },
          varianceRepeatsLarge: 3,
        },
        variance: [
          {
            buildMsCvPct: 4,
            fwdAvgMsCvPct: 5,
            mode: 'dist',
            samples: 3,
            size: 100000,
          },
          {
            buildMsCvPct: 6,
            fwdAvgMsCvPct: 6,
            mode: 'dist',
            samples: 3,
            size: 200000,
          },
        ],
      });

      // Assert
      expect(
        gateResult.failures.some((failure) => failure.name === 'determinism'),
      ).toBe(true);
    });
  });

  describe('given the latest snapshot lacks a stable dist bundle hash', () => {
    it('fails the audit gate', () => {
      // Arrange
      const gateResult = evaluateBenchmarkReleaseGates({
        aggregated: [
          {
            bytesPerConnMean: 64,
            mode: 'dist',
            scenario: 'buildForward',
            size: 100000,
          },
        ],
        determinismReplay: {
          checks: [
            {
              passed: true,
              scenario: 'nodePool-forward-parity',
            },
          ],
          generatedAt: '2026-05-10T00:00:00.000Z',
          passed: true,
        },
        generatedAt: '2026-05-10T00:00:00.000Z',
        history: [
          {
            commit: 'abc1234',
            generatedAt: '2026-05-09T00:00:00.000Z',
            summary: [
              {
                bytesPerConnMean: 64,
                size: 100000,
              },
            ],
          },
          {
            commit: 'def5678',
            generatedAt: '2026-05-10T00:00:00.000Z',
            summary: [
              {
                bytesPerConnMean: 64,
                size: 100000,
              },
            ],
          },
        ],
        meta: {
          distBundle: {
            bytes: 2048,
            exists: true,
            hash: 'feedface5678',
          },
          varianceRepeatsLarge: 3,
        },
        variance: [
          {
            buildMsCvPct: 4,
            fwdAvgMsCvPct: 5,
            mode: 'dist',
            samples: 3,
            size: 100000,
          },
          {
            buildMsCvPct: 6,
            fwdAvgMsCvPct: 6,
            mode: 'dist',
            samples: 3,
            size: 200000,
          },
        ],
      });

      // Assert
      expect(
        gateResult.failures.some((failure) => failure.name === 'audit'),
      ).toBe(true);
    });
  });

  describe('given the checked-in benchmark artifact is current', () => {
    it('passes the real artifact release gate', () => {
      // Arrange
      const artifact = JSON.parse(
        fs.readFileSync(
          path.resolve(__dirname, 'benchmark.results.json'),
          'utf8',
        ),
      ) as Parameters<typeof evaluateBenchmarkReleaseGates>[0];

      // Act
      const gateResult = evaluateBenchmarkReleaseGates(artifact);

      // Assert
      expect(gateResult).toStrictEqual({
        failures: [],
        passed: true,
      });
    });
  });
});
