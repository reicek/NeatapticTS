import {
  createRuntimeParityTensor,
  findPhase9RuntimeParityFixture,
  getPhase9RuntimeParityInventory,
  resolveRuntimeTensorDimensions,
  runSeededOnnxRuntimeParitySamples,
  runOnnxRuntimeParityFixture,
} from './network.onnx.parity';

describe('network onnx runtime parity chapter', () => {
  describe('getPhase9RuntimeParityInventory', () => {
    it('freezes the narrow Phase 9A lane inventory with explicit execution and skip status', () => {
      // Arrange
      const parityInventory = getPhase9RuntimeParityInventory();

      // Act
      const summarizedInventory = parityInventory.map((fixtureDescriptor) => ({
        id: fixtureDescriptor.id,
        lane: fixtureDescriptor.lane,
        executionMode: fixtureDescriptor.executionMode,
        skipReason: fixtureDescriptor.skipReason ?? null,
      }));

      // Assert
      expect(summarizedInventory).toEqual([
        {
          id: 'baseline-float32-dense',
          lane: 'baseline-float32-dense',
          executionMode: 'execute',
          skipReason: null,
        },
        {
          id: 'storage-fp16-dense',
          lane: 'storage-fp16-dense',
          executionMode: 'execute',
          skipReason: null,
        },
        {
          id: 'static-8bit-dense-qlinear',
          lane: 'static-8bit-dense-qlinear',
          executionMode: 'execute',
          skipReason: null,
        },
        {
          id: 'static-8bit-conv-qlinear',
          lane: 'static-8bit-conv-qlinear',
          executionMode: 'execute',
          skipReason: null,
        },
        {
          id: 'dynamic-uint8-dense-guidance',
          lane: 'dynamic-uint8-dense-guidance',
          executionMode: 'execute',
          skipReason: null,
        },
      ]);
    });
  });

  describe('findPhase9RuntimeParityFixture', () => {
    it('rejects unknown fixture identifiers with a named error', () => {
      // Assert
      expect(() =>
        findPhase9RuntimeParityFixture('missing-phase-9-fixture' as never),
      ).toThrow('Unknown ONNX runtime parity fixture: missing-phase-9-fixture');
    });
  });

  describe('createRuntimeParityTensor', () => {
    it('normalizes float32 ArrayBuffer payloads into a standard tensor', () => {
      // Arrange
      const sourceBuffer = Float32Array.from([0.5, -1.25]).buffer;

      // Act
      const runtimeTensor = createRuntimeParityTensor(
        'float32',
        sourceBuffer,
        [1, 2],
      );

      // Assert
      expect({
        type: runtimeTensor.type,
        constructorName: runtimeTensor.data.constructor.name,
        values: Array.from(runtimeTensor.data as ArrayLike<number>),
      }).toEqual({
        type: 'float32',
        constructorName: 'Float32Array',
        values: [0.5, -1.25],
      });
    });

    it('falls back to the standard tensor constructor for non-float32 payloads', () => {
      // Act
      const runtimeTensor = createRuntimeParityTensor(
        'int32',
        new Int32Array([3, 4]),
        [1, 2],
      );

      // Assert
      expect({
        type: runtimeTensor.type,
        constructorName: runtimeTensor.data.constructor.name,
        values: Array.from(runtimeTensor.data as ArrayLike<number>),
      }).toEqual({
        type: 'int32',
        constructorName: 'Int32Array',
        values: [3, 4],
      });
    });

    it('passes through float32 typed-array payloads without re-wrapping them', () => {
      // Arrange
      const sourceValues = new Float32Array([1.5, -2.5]);

      // Act
      const runtimeTensor = createRuntimeParityTensor(
        'float32',
        sourceValues,
        [1, 2],
      );

      // Assert
      expect({
        type: runtimeTensor.type,
        constructorName: runtimeTensor.data.constructor.name,
        values: Array.from(runtimeTensor.data as ArrayLike<number>),
      }).toEqual({
        type: 'float32',
        constructorName: 'Float32Array',
        values: [1.5, -2.5],
      });
    });

    it('normalizes float32 non-Float32Array views into a standard tensor', () => {
      // Arrange
      const sourceValues = new Float32Array([2.25, -3.75]);
      const sourceView = new Uint8Array(sourceValues.buffer);

      // Act
      const runtimeTensor = createRuntimeParityTensor(
        'float32',
        sourceView,
        [1, 2],
      );

      // Assert
      expect({
        type: runtimeTensor.type,
        constructorName: runtimeTensor.data.constructor.name,
        values: Array.from(runtimeTensor.data as ArrayLike<number>),
      }).toEqual({
        type: 'float32',
        constructorName: 'Float32Array',
        values: [2.25, -3.75],
      });
    });
  });

  describe('resolveRuntimeTensorDimensions', () => {
    it('maps numeric and symbolic runtime dimensions into numeric feed dimensions', () => {
      // Assert
      expect(resolveRuntimeTensorDimensions([1, 'N', 3])).toEqual([1, 1, 3]);
    });
  });

  describe('fixture factories', () => {
    it('builds the deferred fixture networks with the expected IO widths', () => {
      // Arrange
      const fixtureIds = [
        'static-8bit-dense-qlinear',
        'static-8bit-conv-qlinear',
        'dynamic-uint8-dense-guidance',
      ] as const;

      // Act
      const fixtureSummaries = fixtureIds.map((fixtureId) => {
        const fixtureDescriptor = findPhase9RuntimeParityFixture(fixtureId);
        const network = fixtureDescriptor.createNetwork();
        return {
          id: fixtureDescriptor.id,
          inputSize: network.input,
          outputSize: network.output,
          hiddenNodeCount: network.nodes.filter(
            (nodeEntry) => nodeEntry.type === 'hidden',
          ).length,
        };
      });

      // Assert
      expect(fixtureSummaries).toEqual([
        {
          id: 'static-8bit-dense-qlinear',
          inputSize: 2,
          outputSize: 1,
          hiddenNodeCount: 2,
        },
        {
          id: 'static-8bit-conv-qlinear',
          inputSize: 9,
          outputSize: 3,
          hiddenNodeCount: 8,
        },
        {
          id: 'dynamic-uint8-dense-guidance',
          inputSize: 2,
          outputSize: 1,
          hiddenNodeCount: 2,
        },
      ]);
    });
  });

  describe('runOnnxRuntimeParityFixture', () => {
    it('returns the named skipped result for deferred fixtures', async () => {
      // Arrange
      const skippedFixtureDescriptor = {
        ...findPhase9RuntimeParityFixture('baseline-float32-dense'),
        executionMode: 'skip' as const,
        skipReason: 'Deferred golden parity fixture.',
      };

      // Act
      const parityResult = await runOnnxRuntimeParityFixture(
        skippedFixtureDescriptor,
      );

      // Assert
      expect(parityResult).toEqual({
        fixture: skippedFixtureDescriptor,
        skipped: true,
        skipReason: 'Deferred golden parity fixture.',
      });
    });

    it('executes the baseline float32 dense fixture within the declared tolerance packet', async () => {
      // Arrange
      const fixtureDescriptor = findPhase9RuntimeParityFixture(
        'baseline-float32-dense',
      );

      // Act
      const parityResult = await runOnnxRuntimeParityFixture(fixtureDescriptor);

      if (parityResult.skipped) {
        throw new Error(parityResult.skipReason);
      }

      // Assert
      expect({
        fixtureId: parityResult.fixture.id,
        lane: parityResult.fixture.lane,
        isWithinTolerance: parityResult.isWithinTolerance,
        meanSquaredErrorWithinTolerance:
          parityResult.meanSquaredError <=
          parityResult.fixture.tolerance.maximumMeanSquaredError,
        maxAbsoluteDifferenceWithinTolerance:
          parityResult.maxAbsoluteDifference <=
          parityResult.fixture.tolerance.maximumAbsoluteDifference,
        inputNameCount: parityResult.inputNames.length,
        outputNameCount: parityResult.outputNames.length,
        outputValueCount: parityResult.runtimeOutput.length,
      }).toEqual({
        fixtureId: 'baseline-float32-dense',
        lane: 'baseline-float32-dense',
        isWithinTolerance: true,
        meanSquaredErrorWithinTolerance: true,
        maxAbsoluteDifferenceWithinTolerance: true,
        inputNameCount: 1,
        outputNameCount: 1,
        outputValueCount: 1,
      });
    });

    it('executes the dynamic guidance fixture within the declared tolerance packet', async () => {
      // Arrange
      const fixtureDescriptor = findPhase9RuntimeParityFixture(
        'dynamic-uint8-dense-guidance',
      );

      // Act
      const parityResult = await runOnnxRuntimeParityFixture(fixtureDescriptor);

      if (parityResult.skipped) {
        throw new Error(parityResult.skipReason);
      }

      // Assert
      expect({
        fixtureId: parityResult.fixture.id,
        lane: parityResult.fixture.lane,
        isWithinTolerance: parityResult.isWithinTolerance,
        meanSquaredErrorWithinTolerance:
          parityResult.meanSquaredError <=
          parityResult.fixture.tolerance.maximumMeanSquaredError,
        maxAbsoluteDifferenceWithinTolerance:
          parityResult.maxAbsoluteDifference <=
          parityResult.fixture.tolerance.maximumAbsoluteDifference,
      }).toEqual({
        fixtureId: 'dynamic-uint8-dense-guidance',
        lane: 'dynamic-uint8-dense-guidance',
        isWithinTolerance: true,
        meanSquaredErrorWithinTolerance: true,
        maxAbsoluteDifferenceWithinTolerance: true,
      });
    });

    it('executes the storage-fp16 dense fixture after the raw binding is already initialized', async () => {
      // Arrange
      const fixtureDescriptor = findPhase9RuntimeParityFixture(
        'storage-fp16-dense',
      );

      // Act
      const parityResult = await runOnnxRuntimeParityFixture(fixtureDescriptor);

      if (parityResult.skipped) {
        throw new Error(parityResult.skipReason);
      }

      // Assert
      expect({
        fixtureId: parityResult.fixture.id,
        isWithinTolerance: parityResult.isWithinTolerance,
        meanSquaredErrorWithinTolerance:
          parityResult.meanSquaredError <=
          parityResult.fixture.tolerance.maximumMeanSquaredError,
      }).toEqual({
        fixtureId: 'storage-fp16-dense',
        isWithinTolerance: true,
        meanSquaredErrorWithinTolerance: true,
      });
    });

    it('freezes the approved execute lanes as named golden parity fixtures', async () => {
      // Arrange
      const executeFixtureIds = [
        'baseline-float32-dense',
        'storage-fp16-dense',
        'static-8bit-dense-qlinear',
        'static-8bit-conv-qlinear',
        'dynamic-uint8-dense-guidance',
      ] as const;

      // Act
      const goldenFixtureSummaries = [];
      for (const fixtureId of executeFixtureIds) {
        const fixtureDescriptor = findPhase9RuntimeParityFixture(fixtureId);
        const parityResult = await runOnnxRuntimeParityFixture(fixtureDescriptor);

        if (parityResult.skipped) {
          throw new Error(parityResult.skipReason);
        }

        goldenFixtureSummaries.push({
          fixtureId: parityResult.fixture.id,
          nativeOutput: parityResult.nativeOutput.map((outputValue) =>
            Number(outputValue.toFixed(6)),
          ),
          runtimeOutput: parityResult.runtimeOutput.map((outputValue) =>
            Number(outputValue.toFixed(6)),
          ),
          meanSquaredError: Number(parityResult.meanSquaredError.toExponential(6)),
          maxAbsoluteDifference: Number(
            parityResult.maxAbsoluteDifference.toExponential(6),
          ),
        });
      }

      // Assert
      expect(goldenFixtureSummaries).toEqual([
        {
          fixtureId: 'baseline-float32-dense',
          nativeOutput: [0.667613],
          runtimeOutput: [0.667613],
          meanSquaredError: 1.738861e-15,
          maxAbsoluteDifference: 4.169966e-8,
        },
        {
          fixtureId: 'storage-fp16-dense',
          nativeOutput: [0.50534],
          runtimeOutput: [0.505341],
          meanSquaredError: 2.290053e-12,
          maxAbsoluteDifference: 0.000001513292,
        },
        {
          fixtureId: 'static-8bit-dense-qlinear',
          nativeOutput: [0.515376],
          runtimeOutput: [0.515436],
          meanSquaredError: 3.631531e-9,
          maxAbsoluteDifference: 0.00006026219,
        },
        {
          fixtureId: 'static-8bit-conv-qlinear',
          nativeOutput: [0.979586, 0.980875, 0.982084],
          runtimeOutput: [0.940304, 0.942982, 0.945548],
          meanSquaredError: 0.001437951,
          maxAbsoluteDifference: 0.03928246,
        },
        {
          fixtureId: 'dynamic-uint8-dense-guidance',
          nativeOutput: [0.507751],
          runtimeOutput: [0.50775],
          meanSquaredError: 2.053784e-13,
          maxAbsoluteDifference: 4.531869e-7,
        },
      ]);
    });

    it('freezes the Phase 9C seeded randomized parity summary for the approved runtime subset', async () => {
      // Arrange
      const executeFixtureIds = [
        'baseline-float32-dense',
        'storage-fp16-dense',
        'static-8bit-dense-qlinear',
        'static-8bit-conv-qlinear',
        'dynamic-uint8-dense-guidance',
      ] as const;

      // Act
      const randomizedParitySummaries = [];
      for (const fixtureId of executeFixtureIds) {
        const fixtureDescriptor = findPhase9RuntimeParityFixture(fixtureId);
        const randomizedResults = await runSeededOnnxRuntimeParitySamples(
          fixtureDescriptor,
          {
            seed: 20260517,
            sampleCount: 2,
          },
        );

        randomizedParitySummaries.push({
          fixtureId,
          sampleCount: randomizedResults.length,
          inputValueCounts: randomizedResults.map(
            (randomizedResult) => randomizedResult.nativeInputValues.length,
          ),
          runtimeInputDimensions: randomizedResults.map(
            (randomizedResult) => randomizedResult.runtimeInputDimensions ?? null,
          ),
          allWithinTolerance: randomizedResults.every(
            (randomizedResult) => randomizedResult.isWithinTolerance,
          ),
          maximumMeanSquaredError: Number(
            Math.max(
              ...randomizedResults.map(
                (randomizedResult) => randomizedResult.meanSquaredError,
              ),
            ).toExponential(6),
          ),
          maximumAbsoluteDifference: Number(
            Math.max(
              ...randomizedResults.map(
                (randomizedResult) => randomizedResult.maxAbsoluteDifference,
              ),
            ).toExponential(6),
          ),
        });
      }

      // Assert
      expect(randomizedParitySummaries).toEqual([
        {
          fixtureId: 'baseline-float32-dense',
          sampleCount: 2,
          inputValueCounts: [2, 3],
          runtimeInputDimensions: [null, null],
          allWithinTolerance: true,
          maximumMeanSquaredError: 7.568595e-16,
          maximumAbsoluteDifference: 2.751108e-8,
        },
        {
          fixtureId: 'storage-fp16-dense',
          sampleCount: 2,
          inputValueCounts: [2, 3],
          runtimeInputDimensions: [null, null],
          allWithinTolerance: true,
          maximumMeanSquaredError: 3.03877e-10,
          maximumAbsoluteDifference: 0.00001743207,
        },
        {
          fixtureId: 'static-8bit-dense-qlinear',
          sampleCount: 2,
          inputValueCounts: [2, 3],
          runtimeInputDimensions: [null, null],
          allWithinTolerance: true,
          maximumMeanSquaredError: 5.745503e-8,
          maximumAbsoluteDifference: 0.0002396978,
        },
        {
          fixtureId: 'static-8bit-conv-qlinear',
          sampleCount: 2,
          inputValueCounts: [12, 12],
          runtimeInputDimensions: [
            [1, 1, 3, 4],
            [1, 1, 4, 3],
          ],
          allWithinTolerance: true,
          maximumMeanSquaredError: 0.008907063,
          maximumAbsoluteDifference: 0.09518136,
        },
        {
          fixtureId: 'dynamic-uint8-dense-guidance',
          sampleCount: 2,
          inputValueCounts: [2, 3],
          runtimeInputDimensions: [null, null],
          allWithinTolerance: true,
          maximumMeanSquaredError: 2.534419e-11,
          maximumAbsoluteDifference: 0.000005034301,
        },
      ]);
    });

    it('rejects randomized parity sample counts outside the supported Phase 9C bounds', async () => {
      // Arrange
      const fixtureDescriptor = findPhase9RuntimeParityFixture(
        'baseline-float32-dense',
      );

      // Assert
      await expect(
        runSeededOnnxRuntimeParitySamples(fixtureDescriptor, {
          seed: 20260517,
          sampleCount: 0,
        }),
      ).rejects.toThrow(
        'Phase 9 randomized parity sampleCount must be an integer between 1 and 4.',
      );
    });

    it('rejects randomized parity requests for skipped fixtures', async () => {
      // Arrange
      const skippedFixtureDescriptor = {
        ...findPhase9RuntimeParityFixture('baseline-float32-dense'),
        executionMode: 'skip' as const,
        skipReason: 'Deferred randomized parity lane.',
      };

      // Assert
      await expect(
        runSeededOnnxRuntimeParitySamples(skippedFixtureDescriptor, {
          seed: 20260517,
          sampleCount: 1,
        }),
      ).rejects.toThrow(
        'Randomized parity requires an executed fixture: baseline-float32-dense',
      );
    });

    afterEach(() => {
      jest.resetModules();
      jest.unmock('node:child_process');
    });

    async function loadParityModuleWithMockedSpawnSync(
      mockSpawnSyncResult: {
        error?: Error;
        status: number | null;
        stdout: string;
        stderr: string;
      },
    ): Promise<typeof import('./network.onnx.parity')> {
      jest.doMock('node:child_process', () => ({
        spawnSync: jest.fn(() => mockSpawnSyncResult),
      }));

      return import('./network.onnx.parity');
    }

    it('propagates child-process execution errors directly', async () => {
      // Arrange
      const parityModule = await loadParityModuleWithMockedSpawnSync({
        error: new Error('subprocess launch failed'),
        status: null,
        stdout: '',
        stderr: '',
      });
      const baselineFixture = parityModule.findPhase9RuntimeParityFixture(
        'baseline-float32-dense',
      );

      // Assert
      await expect(
        parityModule.runOnnxRuntimeParityFixture(baselineFixture),
      ).rejects.toThrow('subprocess launch failed');
    });

    it('prefers stderr text for nonzero child-process exits', async () => {
      // Arrange
      const parityModule = await loadParityModuleWithMockedSpawnSync({
        status: 1,
        stdout: 'stdout fallback',
        stderr: 'stderr failure',
      });
      const baselineFixture = parityModule.findPhase9RuntimeParityFixture(
        'baseline-float32-dense',
      );

      // Assert
      await expect(
        parityModule.runOnnxRuntimeParityFixture(baselineFixture),
      ).rejects.toThrow('stderr failure');
    });

    it('falls back to stdout text for nonzero child-process exits without stderr', async () => {
      // Arrange
      const parityModule = await loadParityModuleWithMockedSpawnSync({
        status: 1,
        stdout: 'stdout failure',
        stderr: '   ',
      });
      const baselineFixture = parityModule.findPhase9RuntimeParityFixture(
        'baseline-float32-dense',
      );

      // Assert
      await expect(
        parityModule.runOnnxRuntimeParityFixture(baselineFixture),
      ).rejects.toThrow('stdout failure');
    });

    it('uses the default message for nonzero child-process exits without output', async () => {
      // Arrange
      const parityModule = await loadParityModuleWithMockedSpawnSync({
        status: 1,
        stdout: '   ',
        stderr: '   ',
      });
      const baselineFixture = parityModule.findPhase9RuntimeParityFixture(
        'baseline-float32-dense',
      );

      // Assert
      await expect(
        parityModule.runOnnxRuntimeParityFixture(baselineFixture),
      ).rejects.toThrow(
        'ONNX runtime parity subprocess failed without error output.',
      );
    });
  });
});