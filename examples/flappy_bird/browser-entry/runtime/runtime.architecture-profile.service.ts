import {
  DEFAULT_FLAPPY_ARCHITECTURE_PROFILE_ID,
  getApprovedExampleArchitectureProfiles,
  resolveExampleArchitectureProfile,
  type ExampleArchitectureProfile,
  type ExampleArchitectureProfileId,
} from '../../../architectureProfiles';
import type { SerializedNetwork } from '../browser-entry.worker.types';
import type { HostArchitectureSelectorItem } from '../host/host.types';

const FLAPPY_ARCHITECTURE_HISTORY_STORAGE_KEY =
  'neataptic:flappy-bird:architecture-history:v1';
const FLAPPY_ARCHITECTURE_CHAMPION_STORAGE_KEY =
  'neataptic:flappy-bird:architecture-champions:v1';

type RuntimeArchitectureHistoryStorage = Pick<Storage, 'getItem' | 'setItem'>;

/** Best-known local browser record for one Flappy architecture profile. */
export interface RuntimeArchitectureBestScore {
  framesSurvived: number;
  pipesPassed: number;
}

/** Local-browser record table keyed by the shared Flappy architecture profile id. */
export type RuntimeArchitectureHistoryByProfileId = Partial<
  Record<ExampleArchitectureProfileId, RuntimeArchitectureBestScore>
>;

/** Browser-local champion table keyed by the shared Flappy architecture profile id. */
export type RuntimeArchitectureChampionByProfileId = Partial<
  Record<ExampleArchitectureProfileId, SerializedNetwork>
>;

/**
 * Resolves the selected shared Flappy profile for the next browser session.
 *
 * @param profileId - Optional requested profile id.
 * @returns Resolved Flappy-ready shared profile.
 */
export function resolveSelectedRuntimeArchitectureProfile(
  profileId?: ExampleArchitectureProfileId,
): ExampleArchitectureProfile {
  return resolveExampleArchitectureProfile(
    'flappy-bird',
    profileId ?? DEFAULT_FLAPPY_ARCHITECTURE_PROFILE_ID,
  );
}

/**
 * Resolves the currently approved shared Flappy architecture profiles.
 *
 * @returns Approved shared profiles in the curated Flappy selector order.
 */
export function resolveAvailableRuntimeArchitectureProfiles(): ExampleArchitectureProfile[] {
  return getApprovedExampleArchitectureProfiles('flappy-bird');
}

/**
 * Reads persisted local browser architecture records when storage is available.
 *
 * @param storage - Optional storage override for tests.
 * @returns Previously stored local records or an empty table.
 */
export function resolveRuntimeArchitectureHistory(
  storage:
    | RuntimeArchitectureHistoryStorage
    | undefined = resolveRuntimeArchitectureHistoryStorage(),
): RuntimeArchitectureHistoryByProfileId {
  if (!storage) {
    return {};
  }

  try {
    const serializedHistory = storage.getItem(
      FLAPPY_ARCHITECTURE_HISTORY_STORAGE_KEY,
    );
    if (!serializedHistory) {
      return {};
    }

    const parsedHistory = JSON.parse(serializedHistory) as
      | RuntimeArchitectureHistoryByProfileId
      | undefined;
    return parsedHistory ?? {};
  } catch {
    return {};
  }
}

/**
 * Reads persisted browser-local champion networks when storage is available.
 *
 * @param storage - Optional storage override for tests.
 * @returns Previously stored champion table or an empty table.
 */
export function resolveRuntimeArchitectureChampions(
  storage:
    | RuntimeArchitectureHistoryStorage
    | undefined = resolveRuntimeArchitectureHistoryStorage(),
): RuntimeArchitectureChampionByProfileId {
  if (!storage) {
    return {};
  }

  try {
    const serializedChampions = storage.getItem(
      FLAPPY_ARCHITECTURE_CHAMPION_STORAGE_KEY,
    );
    if (!serializedChampions) {
      return {};
    }

    const parsedChampions = JSON.parse(serializedChampions) as
      | RuntimeArchitectureChampionByProfileId
      | undefined;
    return parsedChampions ?? {};
  } catch {
    return {};
  }
}

/**
 * Persists the current browser-local architecture record table when storage exists.
 *
 * @param historyByProfileId - Local history table to persist.
 * @param storage - Optional storage override for tests.
 * @returns Nothing.
 */
export function persistRuntimeArchitectureHistory(
  historyByProfileId: RuntimeArchitectureHistoryByProfileId,
  storage:
    | RuntimeArchitectureHistoryStorage
    | undefined = resolveRuntimeArchitectureHistoryStorage(),
): void {
  if (!storage) {
    return;
  }

  try {
    storage.setItem(
      FLAPPY_ARCHITECTURE_HISTORY_STORAGE_KEY,
      JSON.stringify(historyByProfileId),
    );
  } catch {
    // Ignore storage failures in restrictive browser contexts.
  }
}

/**
 * Persists the current browser-local champion table when storage exists.
 *
 * @param championByProfileId - Champion table to persist.
 * @param storage - Optional storage override for tests.
 * @returns Nothing.
 */
export function persistRuntimeArchitectureChampions(
  championByProfileId: RuntimeArchitectureChampionByProfileId,
  storage:
    | RuntimeArchitectureHistoryStorage
    | undefined = resolveRuntimeArchitectureHistoryStorage(),
): void {
  if (!storage) {
    return;
  }

  try {
    storage.setItem(
      FLAPPY_ARCHITECTURE_CHAMPION_STORAGE_KEY,
      JSON.stringify(championByProfileId),
    );
  } catch {
    // Ignore storage failures in restrictive browser contexts.
  }
}

/**
 * Clears all persisted browser-local Flappy score history and champion state.
 *
 * Resetting the selector should remove both the visible best-score captions and
 * the stored champion seeds they were derived from, so the next session starts
 * from the shared architecture template instead of reusing a saved winner.
 *
 * @param storage - Optional storage override for tests.
 * @returns Nothing.
 */
export function resetRuntimeArchitectureProgress(
  storage:
    | RuntimeArchitectureHistoryStorage
    | undefined = resolveRuntimeArchitectureHistoryStorage(),
): void {
  persistRuntimeArchitectureHistory({}, storage);
  persistRuntimeArchitectureChampions({}, storage);
}

/**
 * Folds one session-best Flappy record into the persisted architecture history table.
 *
 * Records compare by pipes passed first and frames survived as a stable
 * tiebreaker so the selector caption reflects the most meaningful browser score.
 *
 * @param historyByProfileId - Existing browser-local history table.
 * @param profileId - Shared architecture profile receiving the score update.
 * @param candidateBestScore - Session-best browser score for the profile.
 * @returns Updated history table when the candidate improves the stored record.
 */
export function updateRuntimeArchitectureHistory(
  historyByProfileId: RuntimeArchitectureHistoryByProfileId,
  profileId: ExampleArchitectureProfileId,
  candidateBestScore: RuntimeArchitectureBestScore,
): RuntimeArchitectureHistoryByProfileId {
  const currentBestScore = historyByProfileId[profileId];
  if (!isRuntimeArchitectureScoreBetter(candidateBestScore, currentBestScore)) {
    return historyByProfileId;
  }

  return {
    ...historyByProfileId,
    [profileId]: candidateBestScore,
  };
}

/**
 * Resolves render-ready selector items from approved profiles and local history.
 *
 * @param options - Approved profile set, selected profile id, and local history table.
 * @returns Render-ready selector items for the Flappy host UI.
 */
export function resolveRuntimeArchitectureSelectorItems(options: {
  availableProfiles: ExampleArchitectureProfile[];
  selectedProfileId: ExampleArchitectureProfileId;
  historyByProfileId: RuntimeArchitectureHistoryByProfileId;
}): HostArchitectureSelectorItem[] {
  const leaderProfileId = resolveRuntimeArchitectureHistoryLeaderProfileId(
    options.historyByProfileId,
  );

  return options.availableProfiles.map((profile) => {
    const bestScore = options.historyByProfileId[profile.id];
    const isLeader = profile.id === leaderProfileId;

    return {
      id: profile.id,
      label: `${profile.label}${isLeader ? ' *' : ''}`,
      caption: bestScore ? `Best ${bestScore.pipesPassed} pipes` : undefined,
      selected: profile.id === options.selectedProfileId,
      tooltipHeading: resolveRuntimeArchitectureTooltipHeading(profile),
      tooltipBodyLines: resolveRuntimeArchitectureTooltipBodyLines(profile),
    };
  });
}

/**
 * Resolves the punchy tooltip heading used by the Flappy architecture selector.
 *
 * @param profile - Shared Flappy architecture profile.
 * @returns Tooltip heading shown above the selector button.
 */
function resolveRuntimeArchitectureTooltipHeading(
  profile: ExampleArchitectureProfile,
): string {
  switch (profile.id) {
    case 'mlp':
      return 'MLP · Multi-Layer Perceptron';

    case 'random-sparse':
      return 'Sparse · Sparse Feed-Forward Graph';

    case 'narx':
      return 'NARX · Explicit Delay-Line Memory';

    case 'gru':
      return 'GRU · Gated Recurrent Unit';

    case 'lstm':
      return 'LSTM · Long Short-Term Memory';
  }
}

/**
 * Resolves the educational tooltip copy shown for one Flappy architecture profile.
 *
 * @param profile - Shared Flappy architecture profile.
 * @returns Short, punchy tooltip lines for the selector button.
 */
function resolveRuntimeArchitectureTooltipBodyLines(
  profile: ExampleArchitectureProfile,
): string[] {
  switch (profile.id) {
    case 'mlp':
      return [
        'MLP is the plain feed-forward baseline: the current numbers go in, action scores come out, and nothing is remembered between frames.',
        'In simple words, it reacts only to what it sees right now. There is no built-in memory cell or delay shelf inside the network.',
        'That makes MLP the easiest model to read and a clean reference point for comparing the memory-based families.',
      ];

    case 'random-sparse':
      return [
        'Sparse starts from the same feed-forward idea as MLP, but it begins with many fewer wires already drawn.',
        'Think of it as a rough sketch instead of a finished diagram: evolution has to discover which connections deserve to exist.',
        'That often makes the first generations messier, but it is great for watching useful structure emerge over time.',
      ];

    case 'narx':
      return [
        'NARX is a recurrent network with an explicit short memory shelf for recent inputs and recent outputs.',
        'Instead of hiding memory behind gates, it keeps a visible rolling window of the recent past. You can picture it as a tiny notepad of what just happened.',
        'That makes NARX one of the easiest memory-based models to explain when timing and rhythm matter.',
      ];

    case 'gru':
      return [
        'GRU is a gated recurrent network that learns what recent information to keep, refresh, or forget as the bird flies.',
        'In simple words, it builds its own short-term memory instead of relying on a fixed delay shelf. The gates act like small traffic lights for remembered state.',
        'GRU is usually lighter than LSTM while still giving the policy real temporal memory.',
      ];

    case 'lstm':
      return [
        'LSTM is a gated recurrent network designed to carry information forward over longer stretches of time.',
        'It uses a dedicated cell state plus gates that decide what to keep, write, and reveal. A simple picture is a memory lane with controlled entry and exit points.',
        'That extra control can capture longer rhythms, but it also makes LSTM the heaviest and most complex option in this selector.',
      ];
  }
}

/**
 * Resolves the highest-scoring architecture profile from the local history table.
 *
 * @param historyByProfileId - Browser-local history table.
 * @returns Leading profile id when at least one stored record exists.
 */
function resolveRuntimeArchitectureHistoryLeaderProfileId(
  historyByProfileId: RuntimeArchitectureHistoryByProfileId,
): ExampleArchitectureProfileId | undefined {
  let leaderProfileId: ExampleArchitectureProfileId | undefined;
  let leaderBestScore: RuntimeArchitectureBestScore | undefined;

  (Object.keys(historyByProfileId) as ExampleArchitectureProfileId[]).forEach(
    (profileId) => {
      const candidateBestScore = historyByProfileId[profileId];
      if (!candidateBestScore) {
        return;
      }

      if (
        isRuntimeArchitectureScoreBetter(candidateBestScore, leaderBestScore)
      ) {
        leaderProfileId = profileId;
        leaderBestScore = candidateBestScore;
      }
    },
  );

  return leaderProfileId;
}

/**
 * Resolves whether a candidate browser score should replace the current record.
 *
 * @param candidateBestScore - Candidate score being considered.
 * @param currentBestScore - Current stored record for the profile.
 * @returns True when the candidate is strictly better.
 */
function isRuntimeArchitectureScoreBetter(
  candidateBestScore: RuntimeArchitectureBestScore,
  currentBestScore: RuntimeArchitectureBestScore | undefined,
): boolean {
  if (!currentBestScore) {
    return true;
  }

  if (candidateBestScore.pipesPassed !== currentBestScore.pipesPassed) {
    return candidateBestScore.pipesPassed > currentBestScore.pipesPassed;
  }

  return candidateBestScore.framesSurvived > currentBestScore.framesSurvived;
}

/**
 * Resolves browser storage for Flappy local architecture history when available.
 *
 * @returns Browser storage implementation or `undefined` outside the browser.
 */
function resolveRuntimeArchitectureHistoryStorage():
  | RuntimeArchitectureHistoryStorage
  | undefined {
  try {
    return window.localStorage;
  } catch {
    return undefined;
  }
}
