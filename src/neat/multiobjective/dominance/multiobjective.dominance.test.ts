import { vectorDominates } from './multiobjective.dominance';

describe('vectorDominates()', () => {
  describe('given a descriptor without an explicit direction', () => {
    describe('when the candidate is strictly higher on that objective', () => {
      it('treats the missing direction as max and determines the candidate dominates', () => {
        // Arrange – no direction property → ?? 'max' fallback (line 130)
        const descriptors = [{ accessor: () => 0 }];

        // Act
        const dominates = vectorDominates([1], [0], descriptors);

        // Assert – 1 > 0 in 'max' direction → dominates
        expect(dominates).toBe(true);
      });
    });
  });
});
