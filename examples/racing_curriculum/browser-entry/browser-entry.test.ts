/** @jest-environment jsdom */

import type { EnvironmentState } from '../environment/environment.types';
import {
  stabilizeCurriculumTierTireGrip,
  start,
} from './browser-entry';

describe('racing curriculum browser entry start()', () => {
  beforeEach(() => {
    document.body.innerHTML = '';
  });

  it('rejects when the requested host container is missing', async () => {
    await expect(start('missing-racing-curriculum-host')).rejects.toThrow(
      'Racing curriculum container "missing-racing-curriculum-host" was not found.',
    );
  });

  it('keeps active panel readiness copy free of future-facing and deferred placeholders', async () => {
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);

    runHandle.stop();

    expect(
      hostElement.textContent?.includes('Future-facing') ||
        hostElement.textContent?.includes('Deferred'),
    ).toBe(false);
  });

  it('renders a live network canvas in the focused network panel', async () => {
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);

    runHandle.stop();

    expect(
      hostElement.querySelector(
        '.racing-host__region--network canvas.racing-network-canvas',
      ),
    ).not.toBeNull();
  });

  it('keeps tire wear disabled before the wear tier and preserves it at the wear tier', () => {
    const wornEnvironmentState = {
      tick: 1,
      carX: 0,
      carY: 0,
      carHeading: 0,
      tireState: [0.4, 0.3, 0.2, 0.1] as const,
      cars: [
        {
          carX: 0,
          carY: 0,
          carHeading: 0,
          teamIndex: 0,
          tireState: [0.4, 0.3, 0.2, 0.1] as const,
        },
      ],
    } satisfies EnvironmentState;

    expect({
      tier1: stabilizeCurriculumTierTireGrip(wornEnvironmentState, 1),
      tier4: stabilizeCurriculumTierTireGrip(wornEnvironmentState, 4),
    }).toMatchObject({
      tier1: {
        tireState: [1, 1, 1, 1],
        cars: [{ tireState: [1, 1, 1, 1] }],
      },
      tier4: {
        tireState: [0.4, 0.3, 0.2, 0.1],
        cars: [{ tireState: [0.4, 0.3, 0.2, 0.1] }],
      },
    });
  });
});
