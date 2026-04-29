import { crossover, selection } from './methods';

describe('methods barrel chapter', () => {
  describe('barrel re-export surface', () => {
    describe('given selection is imported through the barrel', () => {
      describe('when accessed', () => {
        it('exposes the POWER strategy', () => {
          expect(typeof selection.POWER).toBe('object');
        });
      });
    });

    describe('given crossover is imported through the barrel', () => {
      describe('when accessed', () => {
        it('exposes the UNIFORM strategy', () => {
          expect(typeof crossover.UNIFORM).toBe('object');
        });
      });
    });
  });
});
