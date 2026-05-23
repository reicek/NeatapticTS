interface SnapshotSearchResult {
  score: number;
  document: { file_path: string };
  chunk: { heading_path: string; body_text: string };
}

interface SearchSnapshotModule {
  searchSnapshot: (
    snapshot: SemanticSnapshotFixture,
    query: string,
  ) => SnapshotSearchResult[];
}

interface SemanticSnapshotFixture {
  generated_at: string;
  schema_version: '1';
  families: string[];
  documents: Array<{
    doc_id: number;
    file_path: string;
    family: string;
    chunks: Array<{
      chunk_id: number;
      heading_path: string;
      body_text: string;
      char_start: number;
      char_end: number;
    }>;
  }>;
}

describe('semantic browser snapshot', () => {
  describe('semantic-snapshot-search.ts', () => {
    it('returns at least one result for a known query', async () => {
      const { searchSnapshot } = await importSearchModule();
      const results = searchSnapshot(createFixtureSnapshot(), 'NEAT');

      expect(results.length).toBeGreaterThanOrEqual(1);
    });

    it('sorts results by descending heading-weighted score', async () => {
      const { searchSnapshot } = await importSearchModule();
      const results = searchSnapshot(createFixtureSnapshot(), 'NEAT');

      expect(results.map(({ score }) => score)).toEqual(
        results
          .map(({ score }) => score)
          .toSorted((leftScore, rightScore) => rightScore - leftScore),
      );
    });
  });
});

async function importSearchModule(): Promise<SearchSnapshotModule> {
  const modulePath = './semantic-snapshot-search';
  return import(modulePath) as Promise<SearchSnapshotModule>;
}

function createFixtureSnapshot(): SemanticSnapshotFixture {
  return {
    generated_at: '2026-05-23T00:00:00.000Z',
    schema_version: '1',
    families: ['readme'],
    documents: [
      {
        doc_id: 1,
        file_path: 'README.md',
        family: 'readme',
        chunks: [
          {
            chunk_id: 1,
            heading_path: '# NEAT Overview',
            body_text:
              'NeuroEvolution of Augmenting Topologies introduces NEAT.',
            char_start: 0,
            char_end: 60,
          },
          {
            chunk_id: 2,
            heading_path: '# Glossary',
            body_text: 'NEAT appears once in body text.',
            char_start: 61,
            char_end: 91,
          },
        ],
      },
    ],
  };
}

export {};
