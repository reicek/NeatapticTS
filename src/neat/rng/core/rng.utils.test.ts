import {
  getOrCreateRng,
  restoreRngState,
} from './rng.utils';
import {
  RNG_DEFAULT_SEED_FALLBACK,
  RNG_TIME_SCRAMBLE_CONSTANT,
} from './rng.constants';
import type { RngHost } from './rng.types';

describe('rng core utilities', () => {
  describe('getOrCreateRng', () => {
    describe('given a host with an injected rng function', () => {
      it('caches and returns the injected rng function', () => {
        const injectedRng = () => 0.125;
        const host: RngHost = {
          options: { rng: injectedRng },
        };

        const resolvedRng = getOrCreateRng(host);

        expect([resolvedRng, host._rng]).toEqual([injectedRng, injectedRng]);
      });
    });

    describe('given the host resolves a zero seed string and the live state is then cleared', () => {
      it('normalizes the string seed and still advances from the guarded fallback state', () => {
        const host: RngHost = {
          options: { seed: '0' },
        };

        const resolvedRng = getOrCreateRng(host);
        host._rngState = undefined;
        const randomValue = resolvedRng();

        expect([typeof randomValue, typeof host._rngState]).toEqual([
          'number',
          'number',
        ]);
      });
    });

    describe('given the host population is not an array and the fallback seed scramble resolves to zero', () => {
      it('derives the default guarded seed from the time scramble fallback', () => {
        const dateNowSpy = jest
          .spyOn(Date, 'now')
          .mockReturnValue(RNG_TIME_SCRAMBLE_CONSTANT);
        const host: RngHost = {
          population: undefined,
        };

        try {
          getOrCreateRng(host);

          expect(host._rngState).toBe(RNG_DEFAULT_SEED_FALLBACK);
        } finally {
          dateNowSpy.mockRestore();
        }
      });
    });
  });

  describe('restoreRngState', () => {
    describe('given the restored seed string is not finite', () => {
      it('clears the cached RNG and leaves the restored state undefined', () => {
        const host: RngHost = {
          options: {},
        };

        restoreRngState(host, 'not-a-number');

        expect([host._rngState, host._rng]).toEqual([undefined, undefined]);
      });
    });
  });
});