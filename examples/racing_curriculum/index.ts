import {
  start,
  type RacingCurriculumRunHandle,
} from './browser-entry/browser-entry';

declare global {
  interface Window {
    racingCurriculum?: {
      start: (
        container?: HTMLElement | string,
      ) => Promise<RacingCurriculumRunHandle>;
    };
    racingCurriculumStart?: (
      container?: HTMLElement | string,
    ) => Promise<RacingCurriculumRunHandle>;
  }
}

if (typeof window !== 'undefined') {
  window.racingCurriculum = window.racingCurriculum ?? { start };
  window.racingCurriculum.start = start;
  window.racingCurriculumStart = start;
}

export { start };
export type { RacingCurriculumRunHandle };
