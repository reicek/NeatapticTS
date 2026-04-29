import type {
  ConnectionLike,
  GenomeDetailed,
} from '../../../shared/neat.shared.types';
import { summarizeSpeciesConnections } from './species.core.shared';

function createGenomeMember(input: {
  compatibilityMode?: 'allow-fallback';
  connections: ConnectionLike[];
  genomeId: number;
}): GenomeDetailed {
  return {
    _id: input.genomeId,
    nodes: [],
    connections: input.connections,
    ...(input.compatibilityMode
      ? { _compatInnovationMode: input.compatibilityMode }
      : {}),
  };
}

describe('species core shared chapter', () => {
  describe('summarizeSpeciesConnections', () => {
    describe('given the species has no member genomes', () => {
      describe('when the species summary is computed', () => {
        it('returns the zero-style summary defaults', () => {
          // Arrange
          const members: GenomeDetailed[] = [];

          // Act
          const summary = summarizeSpeciesConnections(members, undefined);

          // Assert
          expect(summary).toEqual({
            enabledRatio: 0,
            innovationRange: 0,
          });
        });
      });
    });

    describe('given one native member omits both innovation and endpoint gene ids', () => {
      describe('when the species summary is computed', () => {
        it('throws the native missing-innovation error', () => {
          // Arrange
          const members = [
            createGenomeMember({
              connections: [
                {
                  enabled: true,
                  from: {},
                  to: {},
                },
              ],
              genomeId: 1,
            }),
          ];
          Reflect.deleteProperty(members[0] as object, '_id');
          const summarizeWithoutExplicitInnovation = () =>
            summarizeSpeciesConnections(members, undefined);

          // Assert
          expect(summarizeWithoutExplicitInnovation).toThrow(
            /Species history backfill requires explicit connection innovations/,
          );
        });
      });
    });

    describe('given one legacy member opts into fallback innovations without a resolver', () => {
      describe('when the species summary is computed', () => {
        it('throws the missing fallback resolver error', () => {
          // Arrange
          const members = [
            createGenomeMember({
              compatibilityMode: 'allow-fallback',
              connections: [
                {
                  enabled: true,
                  from: { geneId: 1 },
                  to: { geneId: 2 },
                },
              ],
              genomeId: 1,
            }),
          ];
          const summarizeWithoutFallbackResolver = () =>
            summarizeSpeciesConnections(members, undefined);

          // Assert
          expect(summarizeWithoutFallbackResolver).toThrow(
            /Species history backfill requires `_fallbackInnov`/,
          );
        });
      });
    });

    describe('given one legacy member omits both innovation and endpoint gene ids without a resolver', () => {
      describe('when the species summary is computed', () => {
        it('throws the missing fallback resolver error', () => {
          // Arrange
          const members = [
            createGenomeMember({
              compatibilityMode: 'allow-fallback',
              connections: [
                {
                  enabled: true,
                  from: {},
                  to: {},
                },
              ],
              genomeId: 1,
            }),
          ];
          Reflect.deleteProperty(members[0] as object, '_id');
          const summarizeWithoutFallbackResolver = () =>
            summarizeSpeciesConnections(members, undefined);

          // Assert
          expect(summarizeWithoutFallbackResolver).toThrow(
            /Species history backfill requires `_fallbackInnov`/,
          );
        });
      });
    });
  });
});
