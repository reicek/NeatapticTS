import fs from 'fs';
import path from 'path';

import {
  createInitialState,
  decayTireState,
  stepEnvironment,
  stepEnvironmentBatch,
} from './environment.step.service';
import type { EnvironmentState, RacingCarState } from './environment.types';
import { generateTrack } from '../track/track.generator';
import type { TrackSpec } from '../track/track.generator.types';
import {
  resolveInnerLaneCenterlinePoint,
  resolveSplineSampleFrame,
} from '../track/track.spline.utils';

function computeSignedLateralOffset(
  x: number,
  y: number,
  trackSpec: TrackSpec,
): number {
  let nearestSample = trackSpec.splineSamples[0]!;
  let nearestDistance = Infinity;
  for (const sample of trackSpec.splineSamples) {
    const dx = x - sample.x;
    const dy = y - sample.y;
    const distance = dx * dx + dy * dy;
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearestSample = sample;
    }
  }
  const frame = resolveSplineSampleFrame(
    trackSpec.splineSamples,
    nearestSample.globalIndex,
  );
  const dx = x - nearestSample.x;
  const dy = y - nearestSample.y;
  return dx * frame.normalX + dy * frame.normalY;
}

describe('environment step service sibling seam', () => {
  describe('stepEnvironmentBatch', () => {
    it('advances the tick count by the requested batch length', () => {
      const nextState = stepEnvironmentBatch(
        createInitialState(),
        { throttle: 1, steer: 0 },
        3,
      );

      expect(nextState.tick).toBe(3);
    });
  });
});

describe('Tier 1/Tier 2 track boundary walls', () => {
  it('keeps a car from crossing the inner track boundary when driving inward', () => {
    const trackSpec = generateTrack({
      seed: 42,
      layoutVersion: 1,
      sizeBucket: 'medium',
    });
    const sample = trackSpec.splineSamples[0]!;
    const halfWidth = sample.width / 2;
    const frame = resolveSplineSampleFrame(
      trackSpec.splineSamples,
      sample.globalIndex,
    );
    const heading = Math.atan2(frame.normalY, frame.normalX);
    const carX = sample.x + frame.normalX * halfWidth;
    const carY = sample.y + frame.normalY * halfWidth;
    const car: RacingCarState = {
      carX,
      carY,
      carHeading: heading,
      teamIndex: 0,
      tireState: [1, 1, 1, 1],
    };
    const state: EnvironmentState = {
      tick: 0,
      carX,
      carY,
      carHeading: heading,
      teamIndex: 0,
      tireState: [1, 1, 1, 1],
      cars: [car],
      trackSpec,
    };

    const nextState = stepEnvironment(state, { throttle: 1, steer: 0 });
    const nextCar = nextState.cars![0]!;
    const offset = computeSignedLateralOffset(
      nextCar.carX,
      nextCar.carY,
      trackSpec,
    );

    expect(offset).toBeLessThanOrEqual(halfWidth + 1e-6);
  });

  it('keeps a car from crossing the outer track boundary when driving outward', () => {
    const trackSpec = generateTrack({
      seed: 42,
      layoutVersion: 1,
      sizeBucket: 'medium',
    });
    const sample = trackSpec.splineSamples[0]!;
    const halfWidth = sample.width / 2;
    const frame = resolveSplineSampleFrame(
      trackSpec.splineSamples,
      sample.globalIndex,
    );
    const heading = Math.atan2(-frame.normalY, -frame.normalX);
    const carX = sample.x - frame.normalX * halfWidth;
    const carY = sample.y - frame.normalY * halfWidth;
    const car: RacingCarState = {
      carX,
      carY,
      carHeading: heading,
      teamIndex: 0,
      tireState: [1, 1, 1, 1],
    };
    const state: EnvironmentState = {
      tick: 0,
      carX,
      carY,
      carHeading: heading,
      teamIndex: 0,
      tireState: [1, 1, 1, 1],
      cars: [car],
      trackSpec,
    };

    const nextState = stepEnvironment(state, { throttle: 1, steer: 0 });
    const nextCar = nextState.cars![0]!;
    const offset = computeSignedLateralOffset(
      nextCar.carX,
      nextCar.carY,
      trackSpec,
    );

    expect(offset).toBeGreaterThanOrEqual(-halfWidth - 1e-6);
  });
});

describe('Tier 1 physics hardening — penalties and car separation', () => {
  it('applies a negative reward when a car is clamped back from outside the track', () => {
    const trackSpec = generateTrack({
      seed: 42,
      layoutVersion: 1,
      sizeBucket: 'medium',
    });
    const sample = trackSpec.splineSamples[0]!;
    const halfWidth = sample.width / 2;
    const frame = resolveSplineSampleFrame(
      trackSpec.splineSamples,
      sample.globalIndex,
    );
    const carX = sample.x + frame.normalX * (halfWidth + 5);
    const carY = sample.y + frame.normalY * (halfWidth + 5);
    const car: RacingCarState = {
      carX,
      carY,
      carHeading: frame.tangentHeadingRadians,
      teamIndex: 0,
      tireState: [1, 1, 1, 1],
    };
    const state: EnvironmentState = {
      tick: 0,
      carX,
      carY,
      carHeading: frame.tangentHeadingRadians,
      teamIndex: 0,
      tireState: [1, 1, 1, 1],
      cars: [car],
      trackSpec,
    };

    const nextState = stepEnvironment(state, { throttle: 0, steer: 0 });
    const nextCar = nextState.cars![0]!;
    const reward = (nextCar as { reward?: number }).reward;

    expect(reward).toBeLessThan(0);
  });

  it('applies a wrong-direction penalty when the car moves opposite to the track tangent', () => {
    const trackSpec = generateTrack({
      seed: 42,
      layoutVersion: 1,
      sizeBucket: 'medium',
    });
    const sample = trackSpec.splineSamples[0]!;
    const frame = resolveSplineSampleFrame(
      trackSpec.splineSamples,
      sample.globalIndex,
    );
    const startPoint = resolveInnerLaneCenterlinePoint(sample, frame);
    const wrongHeading = frame.tangentHeadingRadians + Math.PI;
    const car: RacingCarState = {
      carX: startPoint.x,
      carY: startPoint.y,
      carHeading: wrongHeading,
      teamIndex: 0,
      tireState: [1, 1, 1, 1],
    };
    const state: EnvironmentState = {
      tick: 0,
      carX: startPoint.x,
      carY: startPoint.y,
      carHeading: wrongHeading,
      teamIndex: 0,
      tireState: [1, 1, 1, 1],
      cars: [car],
      trackSpec,
    };

    const nextState = stepEnvironment(state, { throttle: 1, steer: 0 });
    const nextCar = nextState.cars![0]!;
    const reward = (nextCar as { reward?: number }).reward;

    expect(reward).toBeLessThan(0);
  });

  it('pushes overlapping cars apart so their centers are no longer coincident', () => {
    const trackSpec = generateTrack({
      seed: 42,
      layoutVersion: 1,
      sizeBucket: 'medium',
    });
    const sample = trackSpec.splineSamples[0]!;
    const frame = resolveSplineSampleFrame(
      trackSpec.splineSamples,
      sample.globalIndex,
    );
    const startPoint = resolveInnerLaneCenterlinePoint(sample, frame);
    const carA: RacingCarState = {
      carX: startPoint.x,
      carY: startPoint.y,
      carHeading: frame.tangentHeadingRadians,
      teamIndex: 0,
      tireState: [1, 1, 1, 1],
    };
    const carB: RacingCarState = {
      carX: startPoint.x,
      carY: startPoint.y,
      carHeading: frame.tangentHeadingRadians,
      teamIndex: 1,
      tireState: [1, 1, 1, 1],
    };
    const state: EnvironmentState = {
      tick: 0,
      carX: startPoint.x,
      carY: startPoint.y,
      carHeading: frame.tangentHeadingRadians,
      teamIndex: 0,
      tireState: [1, 1, 1, 1],
      cars: [carA, carB],
      trackSpec,
    };

    const nextState = stepEnvironment(state, { throttle: 0, steer: 0 });
    const separation = Math.hypot(
      nextState.cars![0]!.carX - nextState.cars![1]!.carX,
      nextState.cars![0]!.carY - nextState.cars![1]!.carY,
    );

    expect(separation).toBeGreaterThan(1e-6);
  });
});

describe('pit entry teleport and hold', () => {
  it('teleports an entering car to the team pit-box center', () => {
    const trackSpec = generateTrack({
      seed: 42,
      layoutVersion: 1,
      sizeBucket: 'medium',
    });
    const teamZeroPitBox = trackSpec.pitBoxes!.find(
      (pitBox) => pitBox.teamIndex === 0,
    )!;
    const corridor = teamZeroPitBox.entranceCorridor;
    const car: RacingCarState = {
      carX: corridor.x + corridor.width / 2,
      carY: corridor.y + corridor.height / 2,
      carHeading: 0,
      teamIndex: 0,
      tireState: [0.8, 0.8, 0.8, 0.8],
    };
    const emptyPitOccupancy = createEmptyPitOccupancy();
    const state: EnvironmentState = {
      tick: 0,
      carX: car.carX,
      carY: car.carY,
      carHeading: car.carHeading,
      teamIndex: 0,
      tireState: [0.8, 0.8, 0.8, 0.8],
      cars: [car],
      trackSpec,
      pitOccupancy: emptyPitOccupancy,
      pitStatus: emptyPitOccupancy,
    };

    const nextState = stepEnvironment(state, { throttle: 0, steer: 0 });
    const nextCar = nextState.cars![0]!;

    expect({
      carX: nextCar.carX,
      carY: nextCar.carY,
    }).toEqual({
      carX: teamZeroPitBox.boxCenter!.x,
      carY: teamZeroPitBox.boxCenter!.y,
    });
  });

  it('holds the car at the pit-box center on the following step', () => {
    const trackSpec = generateTrack({
      seed: 42,
      layoutVersion: 1,
      sizeBucket: 'medium',
    });
    const teamZeroPitBox = trackSpec.pitBoxes!.find(
      (pitBox) => pitBox.teamIndex === 0,
    )!;
    const corridor = teamZeroPitBox.entranceCorridor;
    const car: RacingCarState = {
      carX: corridor.x + corridor.width / 2,
      carY: corridor.y + corridor.height / 2,
      carHeading: 0,
      teamIndex: 0,
      tireState: [0.8, 0.8, 0.8, 0.8],
    };
    const emptyPitOccupancy = createEmptyPitOccupancy();
    const state: EnvironmentState = {
      tick: 0,
      carX: car.carX,
      carY: car.carY,
      carHeading: car.carHeading,
      teamIndex: 0,
      tireState: [0.8, 0.8, 0.8, 0.8],
      cars: [car],
      trackSpec,
      pitOccupancy: emptyPitOccupancy,
      pitStatus: emptyPitOccupancy,
    };

    const steppedOnce = stepEnvironment(state, { throttle: 0, steer: 0 });
    const steppedTwice = stepEnvironment(steppedOnce, {
      throttle: 0,
      steer: 0,
    });
    const nextCar = steppedTwice.cars![0]!;

    expect({
      carX: nextCar.carX,
      carY: nextCar.carY,
    }).toEqual({
      carX: teamZeroPitBox.boxCenter!.x,
      carY: teamZeroPitBox.boxCenter!.y,
    });
  });
});

describe('Phase 8 Step 18 pit release re-entry red contract', () => {
  it('does not re-trap a car released with mean tire health at or above the service threshold', () => {
    const trackSpec = generateTrack({
      seed: 42,
      layoutVersion: 1,
      sizeBucket: 'medium',
    });
    const teamZeroPitBox = trackSpec.pitBoxes!.find(
      (pitBox) => pitBox.teamIndex === 0,
    )!;
    const boxCenter = teamZeroPitBox.boxCenter!;
    const serviceThreshold = 0.85;
    const tireState: [number, number, number, number] = [
      serviceThreshold,
      serviceThreshold,
      serviceThreshold,
      serviceThreshold,
    ];
    const car: RacingCarState = {
      carX: boxCenter.x,
      carY: boxCenter.y,
      carHeading: 0,
      teamIndex: 0,
      tireState,
    };
    const emptyPitOccupancy = createEmptyPitOccupancy();
    const state: EnvironmentState = {
      tick: 0,
      carX: car.carX,
      carY: car.carY,
      carHeading: car.carHeading,
      teamIndex: 0,
      tireState,
      cars: [car],
      trackSpec,
      pitOccupancy: emptyPitOccupancy,
      pitStatus: emptyPitOccupancy,
    };

    const nextState = stepEnvironment(state, { throttle: 0, steer: 0 });
    const isCarReTrapped = nextState.pitStatus!.some(
      (record) => record.occupyingCarIndex === 0,
    );

    expect(isCarReTrapped).toBe(false);
  });
});

describe('tire decay rate calibration', () => {
  it('retains at least 80% mean tire health after 1000 repeated load steps', () => {
    let tireState: [number, number, number, number] = [1, 1, 1, 1];

    for (let step = 0; step < 1000; step++) {
      tireState = decayTireState(tireState, 0.5, 0.5, 50);
    }

    const meanTireHealth =
      tireState.reduce((sum, health) => sum + health, 0) / tireState.length;

    expect(meanTireHealth).toBeGreaterThanOrEqual(0.8);
  });
});

function createEmptyPitOccupancy() {
  return Array.from({ length: 6 }, () => ({
    occupyingCarIndex: 255,
    remainingStopTicks: 0,
  }));
}

describe('P8S22 — AC-RC-22-006: OFF_TRACK_CLAMP_REWARD increased from -1 to <= -5', () => {
  it('(P8S22) OFF_TRACK_CLAMP_REWARD constant is at most -5 (not -1)', () => {
    const sourcePath = path.join(__dirname, 'environment.step.service.ts');
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    const match = sourceText.match(
      /const\s+OFF_TRACK_CLAMP_REWARD\s*=\s*(-?[0-9.]+)/,
    );
    const rewardValue = match ? Number(match[1]) : 0;

    expect(rewardValue).toBeLessThanOrEqual(-5);
  });
});

describe('P8S22 — AC-RC-22-007: WRONG_DIRECTION_REWARD increased from -1 to <= -5', () => {
  it('(P8S22) WRONG_DIRECTION_REWARD constant is at most -5 (not -1)', () => {
    const sourcePath = path.join(__dirname, 'environment.step.service.ts');
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    const match = sourceText.match(
      /const\s+WRONG_DIRECTION_REWARD\s*=\s*(-?[0-9.]+)/,
    );
    const rewardValue = match ? Number(match[1]) : 0;

    expect(rewardValue).toBeLessThanOrEqual(-5);
  });
});

describe('P8S22 — AC-RC-22-009: guide-following positive reward', () => {
  it('(P8S22) source contains guide-following reward logic', () => {
    const sourcePath = path.join(__dirname, 'environment.step.service.ts');
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    // After the fix, a guide-following positive reward should exist that
    // applies +2 to +5 when the car is near the guide line.
    const hasGuideFollow =
      sourceText.includes('guideFollow') ||
      sourceText.includes('guideReward') ||
      sourceText.includes('GUIDE_FOLLOW') ||
      sourceText.includes('GUIDE_REWARD') ||
      sourceText.includes('guideLineReward');

    expect(hasGuideFollow).toBe(true);
  });
});

describe('P8S22 — AC-RC-22-010: diverging-from-guide penalty', () => {
  it('(P8S22) source contains diverging-from-guide penalty logic', () => {
    const sourcePath = path.join(__dirname, 'environment.step.service.ts');
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    // After the fix, a diverging-from-guide penalty should exist that
    // penalizes lateral divergence from the guide line more strongly
    // than the generic offTrackPenalty.
    const hasDivergencePenalty =
      sourceText.includes('diverg') ||
      sourceText.includes('Diverg') ||
      sourceText.includes('guideDivergence') ||
      sourceText.includes('GUIDE_DIVERGENCE') ||
      sourceText.includes('lateralDivergence');

    expect(hasDivergencePenalty).toBe(true);
  });
});

describe('P8S22 — AC-RC-22-011: escalating penalties for consecutive border contact', () => {
  it('(P8S22) source contains escalating penalty logic with consecutive border ticks', () => {
    const sourcePath = path.join(__dirname, 'environment.step.service.ts');
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    // After the fix, escalating penalties should exist that scale with
    // consecutive border contact ticks (-1 × consecutiveTicks, capped).
    const hasEscalatingPenalty =
      sourceText.includes('consecutiveBorder') ||
      sourceText.includes('escalat') ||
      sourceText.includes('Escalat') ||
      sourceText.includes('consecutiveTicks') ||
      sourceText.includes('borderTickCount') ||
      sourceText.includes('consecutiveOffTrack');

    expect(hasEscalatingPenalty).toBe(true);
  });
});
