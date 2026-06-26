import type {
  CarControlOutput,
  EnvironmentState,
} from '../environment/environment.types';
import { generateTrack } from '../track/track.generator';
import type { TrackSpec } from '../track/track.generator.types';

/**
 * Red-phase contracts for the owner-local NGE controller seam.
 *
 * These tests pin the replacement boundary for the scripted controller before
 * Step 04 introduces any production implementation. The module is loaded via a
 * runtime string import so the suite can stay red while the implementation file
 * does not exist yet.
 */
describe('nge.controller', () => {
  describe('createNgeController', () => {
    it('maps the network forward pass to throttle and steer while providing a non-empty observation vector', async () => {
      const envState = createExpandedEnvironmentState();
      const trackSpec = createTrackSpec();
      const activationInputs: number[][] = [];
      const network: ControllerNetwork = {
        activate(inputVector) {
          activationInputs.push(Array.from(inputVector));
          return [0.25, -0.75];
        },
      };

      await expect(
        loadNgeControllerModule().then(({ createNgeController }) => {
          const controller = createNgeController(network);
          return {
            control: controller.computeControl(envState, trackSpec),
            didProvideObservationVector: (activationInputs[0]?.length ?? 0) > 0,
          };
        }),
      ).resolves.toEqual({
        control: { throttle: 0.25, steer: -0.75 },
        didProvideObservationVector: true,
      });
    });

    it('writes the last seven network outputs into the Tier 2 self-radio channel', async () => {
      const envState = createExpandedEnvironmentState();
      const trackSpec = createTrackSpec();
      const radioOutputs = [0.11, -0.22, 0.33, -0.44, 0.55, -0.66, 0.77];
      const networkOutputs = [0.25, -0.75, ...radioOutputs];
      const network: ControllerNetwork = {
        activate() {
          return networkOutputs;
        },
      };

      await expect(
        loadNgeControllerModule().then(
          ({ createNgeController, createSingleCarRadioChannel }) => {
            const radioChannel = createSingleCarRadioChannel(7);
            const controller = createNgeController(network, {
              tier: 2,
              radioChannel,
            });
            controller.computeControl(envState, trackSpec);
            return Array.from(radioChannel.readSelf());
          },
        ),
      ).resolves.toEqual(radioOutputs);
    });
  });

  describe('resolveGuidanceAlphaForTier', () => {
    it('keeps full guidance in Tier 0, fades in Tier 1, and disables the overlay in Tier 2', async () => {
      await expect(
        loadNgeControllerModule().then(({ resolveGuidanceAlphaForTier }) => {
          const tier0 = resolveGuidanceAlphaForTier(0);
          const tier1 = resolveGuidanceAlphaForTier(1);
          const tier2 = resolveGuidanceAlphaForTier(2);

          return {
            tier0,
            tier1,
            tier2,
            hasOrderedFade: tier0 > tier1 && tier1 > tier2 && tier2 >= 0,
          };
        }),
      ).resolves.toEqual({
        tier0: 1,
        tier1: expect.any(Number),
        tier2: 0,
        hasOrderedFade: true,
      });
    });
  });

  describe('createSingleCarRadioChannel', () => {
    it('round-trips a seven-channel Tier 2 self-monitoring payload without reordering it', async () => {
      const payload = [0.11, -0.22, 0.33, -0.44, 0.55, -0.66, 0.77];

      await expect(
        loadNgeControllerModule().then(({ createSingleCarRadioChannel }) => {
          const channel = createSingleCarRadioChannel(payload.length);
          channel.writeSelf(payload);
          return Array.from(channel.readSelf());
        }),
      ).resolves.toEqual(payload);
    });
  });
});

type ExpandedEnvironmentState = EnvironmentState & {
  forwardSpeedWorld: number;
  lateralSpeedWorld: number;
  speedWorld: number;
  yawRateRadiansPerSecond: number;
  slipAngleRadians: number;
  progress01: number;
  lapProgress01: number;
  boundaryDistanceLeftWorld: number;
  boundaryDistanceRightWorld: number;
  hazardDistanceWorld: number;
  waypointDistanceWorld: number;
  optimalLineLateralOffsetWorld: number;
  optimalLineHeadingErrorRadians: number;
  targetSpeedWorld: number;
  memoryTrace: readonly number[];
  radioField: Float32Array;
};

interface ControllerNetwork {
  activate(inputVector: readonly number[] | Float32Array): readonly number[];
}

interface NgeControllerOptions {
  tier?: number;
  radioChannel?: SingleCarRadioChannel;
  radioDim?: number;
}

interface NgeController {
  computeControl(
    envState: ExpandedEnvironmentState,
    trackSpec: TrackSpec,
  ): CarControlOutput;
}

interface SingleCarRadioChannel {
  writeSelf(radioValues: readonly number[] | Float32Array): void;
  readSelf(): readonly number[] | Float32Array;
}

interface NgeControllerModule {
  createNgeController(
    network: ControllerNetwork,
    options?: NgeControllerOptions,
  ): NgeController;
  resolveGuidanceAlphaForTier(tier: 0 | 1 | 2): number;
  createSingleCarRadioChannel(radioDim: number): SingleCarRadioChannel;
}

function createExpandedEnvironmentState(
  radioField: Float32Array = new Float32Array(0),
): ExpandedEnvironmentState {
  return {
    tick: 12,
    carX: 18,
    carY: -24,
    carHeading: Math.PI / 4,
    forwardSpeedWorld: 18,
    lateralSpeedWorld: -4,
    speedWorld: 18.4390889146,
    yawRateRadiansPerSecond: -0.25,
    slipAngleRadians: 0.12,
    progress01: 0.35,
    lapProgress01: 0.6,
    boundaryDistanceLeftWorld: 14,
    boundaryDistanceRightWorld: 9,
    hazardDistanceWorld: 32,
    waypointDistanceWorld: 21,
    optimalLineLateralOffsetWorld: -3.5,
    optimalLineHeadingErrorRadians: 0.2,
    targetSpeedWorld: 24,
    memoryTrace: [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5],
    radioField,
  };
}

function createTrackSpec(): TrackSpec {
  return generateTrack({ seed: 42, layoutVersion: 1, sizeBucket: 'medium' });
}

async function loadNgeControllerModule(): Promise<NgeControllerModule> {
  const modulePath = './nge.controller';
  return (await import(modulePath)) as NgeControllerModule;
}
