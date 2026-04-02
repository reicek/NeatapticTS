/**
 * Shared simulation vocabulary reused across environment, evaluation, worker,
 * and browser-adjacent helpers.
 *
 * This boundary exists so the example can share observation semantics and
 * deterministic spawn logic without letting every runtime invent its own near-
 * duplicate types. The payoff is consistency: when a policy sees a gap, or when
 * a helper estimates urgency, those meanings stay aligned across the whole
 * example.
 *
 * This is the common language layer for the control problem. It does not own
 * full simulation stepping, scoring, or rendering. It owns the smaller
 * contracts those larger boundaries must agree on: difficulty profiles, pipe
 * geometry, observation features, temporal memory, and action decoding.
 *
 * That shared vocabulary is what keeps the demo honest across runtimes. The
 * environment can advance the world, the evaluation layer can score policies,
 * the worker can stream playback, and browser helpers can inspect decisions
 * without silently redefining what "next gap" or "urgent correction" means.
 *
 * ## What This Folder Is Trying To Teach
 *
 * Read this chapter if you want to answer three practical questions:
 *
 * 1. Which geometric signals does the policy actually see?
 * 2. How does the example add short-horizon memory without requiring recurrent
 *    networks?
 * 3. How do difficulty, spawn, observation, and control helpers stay reusable
 *    across Node training and browser playback?
 *
 * ## Shared Vocabulary Map
 *
 * ```mermaid
 * flowchart LR
 *     Difficulty["difficulty utils\ncurriculum profile"] --> Spawn["spawn utils\nnext pipe cadence and gap"]
 *     Spawn --> World["environment + worker playback\nconcrete world state"]
 *     World --> Features["observation/\nfeature synthesis"]
 *     Features --> Memory["memory utils\nstack recent frames and actions"]
 *     Features --> Vector["observation/\ncanonical policy vectors"]
 *     Vector --> Control["control utils\nresolve flap decision"]
 *     Memory --> Control
 *
 *     Browser["browser inspection helpers"] -.-> Features
 *     Evaluation["evaluation/"] -.-> Features
 *     Worker["flappy-evolution-worker/"] -.-> Memory
 *
 *     classDef boundary fill:#001522,stroke:#0fb5ff,color:#9fdcff,stroke-width:2px;
 *     classDef runtime fill:#03111f,stroke:#00e5ff,color:#d8f6ff,stroke-width:2px;
 *     classDef highlight fill:#2a1029,stroke:#ff4a8d,color:#ffd7e8,stroke-width:3px;
 *
 *     class Difficulty,Spawn,Features,Memory,Vector,Control boundary;
 *     class World,Browser,Evaluation,Worker runtime;
 *     class Features highlight;
 * ```
 *
 * The key teaching point is that the policy does not read pixels. It reads a
 * curated state representation: gap geometry, velocity, urgency, and a short
 * action-conditioned memory trail. That makes the control problem easier to
 * inspect and keeps training, evaluation, and playback aligned around the same
 * semantics.
 *
 * ## Choose Your Route
 *
 * - Start with `simulation-shared.types.ts` if you want the stable nouns of the
 *   subsystem.
 * - Read `simulation-shared.difficulty.utils.ts` and
 *   `simulation-shared.spawn.utils.ts` if you want the curriculum and course
 *   generation rules.
 * - Read [simulation-shared/observation/README.md](./observation/README.md) if
 *   you want the feature-engineering story in more detail.
 * - Read `simulation-shared.memory.utils.ts` if you want the frame-stacking and
 *   recent-action channels.
 * - Read `simulation-shared.control.utils.ts` if you want the final step from
 *   network outputs to `flap` versus `no flap`.
 *
 * Example sketch:
 *
 * ```ts
 * const difficultyProfile = resolveAdaptiveDifficultyProfile(pipesPassed, 1);
 * const features = resolveObservationFeatures({
 *   birdYPx,
 *   velocityYPxPerFrame,
 *   pipes,
 *   visibleWorldWidthPx,
 *   difficultyProfile,
 *   activeSpawnIntervalFrames: difficultyProfile.pipeSpawnIntervalFrames,
 * });
 * const networkInput = resolveTemporalObservationVector(
 *   features,
 *   observationMemoryState,
 * );
 * const didFlap = resolveFlapDecision(network.activate(networkInput));
 * ```
 *
 * If you want background reading, Wikipedia contributors on feature
 * engineering, curriculum learning, and frame stacking are useful bridges, but
 * this folder is where those ideas become concrete contracts for the Flappy
 * Bird example.
 */
/**
 * Minimal deterministic random contract used by shared spawn helpers.
 *
 * The shared layer keeps its RNG contract intentionally small so the same spawn
 * helpers can work with both Node-side and browser-side deterministic sources.
 */
export interface SharedRngLike {
  /**
   * Returns integer in `[min, max)`.
   *
   * @param min - Inclusive lower bound.
   * @param max - Exclusive upper bound.
   * @returns Pseudo-random integer.
   */
  nextInt(min: number, max: number): number;
}

/**
 * Common pipe shape consumed by observation helpers.
 *
 * This is the narrowest useful pipe contract for feature synthesis: horizontal
 * position plus the vertical gap geometry seen by the bird.
 */
export interface SharedPipeLike {
  /** Horizontal left position in world pixels. */
  xPx: number;

  /** Gap center y position in world pixels. */
  gapCenterYPx: number;

  /** Gap size in world pixels. */
  gapSizePx: number;
}

/**
 * Shared runtime difficulty profile used by browser and environment simulators.
 *
 * By projecting difficulty into one plain object, the example can keep
 * curriculum logic independent from rendering, evaluation, and worker runtime
 * concerns.
 */
export interface SharedDifficultyProfile {
  /** Active target gap size. */
  pipeGapPx: number;

  /** Active pipe horizontal speed. */
  pipeSpeedPxPerFrame: number;

  /** Active spawn interval in simulation frames. */
  pipeSpawnIntervalFrames: number;
}

/**
 * Structured observation features for network input.
 *
 * Educational note:
 * These features make the policy input interpretable. The example does not feed
 * raw pixels into NEAT; it feeds geometric signals such as distance to the next
 * pipe, corridor clearance, and urgency of recovering to the gap center.
 */
export interface SharedObservationFeatures {
  /** Bird y position normalized to [0, 1]. */
  normalizedBirdY: number;

  /** Bird vertical velocity normalized to [-1, 1]. */
  normalizedVelocity: number;

  /** Distance to next pipe normalized to [0, 1]. */
  normalizedDistanceToNextPipe: number;

  /** Delta to next gap center normalized to [-1, 1]. */
  normalizedDeltaToNextGap: number;

  /** Next gap top normalized to [0, 1]. */
  normalizedNextGapTop: number;

  /** Next gap bottom normalized to [0, 1]. */
  normalizedNextGapBottom: number;

  /** Distance to second pipe normalized to [0, 1]. */
  normalizedDistanceToSecondPipe: number;

  /** Delta to second gap center normalized to [-1, 1]. */
  normalizedDeltaToSecondGap: number;

  /** Time-to-next-pipe closeness normalized to [0, 1]. */
  normalizedTimeToNextPipe: number;

  /** Signed next-gap clearance in [-1, 1]. */
  normalizedNextGapClearance: number;

  /** Required vertical velocity to center next gap normalized to [-1, 1]. */
  normalizedRequiredVerticalVelocityToNextGap: number;

  /** Transition between next and second gap centers normalized to [-1, 1]. */
  normalizedNextToSecondGapTransition: number;

  /** Frames-to-gap-entry estimate normalized to [0, 1]. */
  normalizedFramesToGapEntry: number;

  /** Frames-to-gap-exit estimate normalized to [0, 1]. */
  normalizedFramesToGapExit: number;

  /** Required vertical velocity at gap entry normalized to [-1, 1]. */
  normalizedRequiredVerticalVelocityAtGapEntry: number;

  /** Required vertical velocity at gap exit normalized to [-1, 1]. */
  normalizedRequiredVerticalVelocityAtGapExit: number;

  /** Urgency signal for recovering to center before entry normalized to [0, 1]. */
  normalizedEntryUrgency: number;

  /** Reachability signal indicating whether entry can be recovered in <=1 flap. */
  normalizedOneFlapReachabilityAtGapEntry: number;
}

/**
 * Mutable temporal memory attached to one policy-controlled bird.
 *
 * The memory stores recent core observation frames and recent action history,
 * allowing feedforward policies to consume short-term context without adding
 * recurrent connections.
 *
 * If you want background reading, the Wikipedia article on "frame stacking"
 * captures the basic idea of giving a feed-forward policy a short motion trail
 * instead of full recurrent state.
 */
export interface SharedObservationMemoryState {
  /** Previous core frames, newest-first, excluding the current frame. */
  previousCoreObservationFrames: number[][];

  /** Recent flap actions, newest-first, encoded as `1` (flap) or `0` (no flap). */
  recentFlapActions: number[];
}

/**
 * Input shape for observation-feature synthesis.
 *
 * This object is the raw world snapshot from which normalized features are
 * derived. It intentionally separates world geometry from the later feature
 * projection step.
 */
export interface SharedObservationInput {
  /** Bird vertical position in world pixels. */
  birdYPx: number;

  /** Bird vertical velocity in world pixels/frame. */
  velocityYPxPerFrame: number;

  /** Current pipe list. */
  pipes: SharedPipeLike[];

  /** Active visible width used for distance normalization. */
  visibleWorldWidthPx: number;

  /** Active difficulty profile. */
  difficultyProfile: SharedDifficultyProfile;

  /** Active spawn interval. */
  activeSpawnIntervalFrames: number;

  /** Default gap size used when no next pipe exists. */
  defaultGapSizePx?: number;

  /** Bird center x-position used for upcoming-pipe resolution. */
  birdCenterXPx?: number;

  /** Bird collision radius used for upcoming-pipe resolution. */
  birdRadiusPx?: number;

  /** Pipe width used for distance and upcoming-pipe calculations. */
  pipeWidthPx?: number;

  /** World height used for normalization. */
  worldHeightPx?: number;

  /** Maximum fall speed used for velocity normalization. */
  maxFallSpeedPxPerFrame?: number;

  /** Division guard used for time-to-next-pipe estimate. */
  normalizationEpsilon?: number;
}
