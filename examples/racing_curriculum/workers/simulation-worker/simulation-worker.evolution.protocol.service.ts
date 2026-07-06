/**
 * Racing worker protocol FSM router.
 *
 * This module owns the `routeRacingWorkerProtocolMessage` function, which maps
 * one host-to-worker message onto the next FSM state and an optional response.
 * It enforces the forward-only lifecycle:
 *
 *   idle → initialised → generation-ready → racing → (stopped)
 *
 * The lifecycle is a straightforward finite-state machine; see
 * [Finite-state machine (Wikipedia)](https://en.wikipedia.org/wiki/Finite-state_machine)
 * for background on why deterministic state transitions help avoid authority
 * confusion between a host and a worker.
 *
 * ## Protocol FSM
 *
 * The diagram below shows the host-to-worker message contract as a state
 * machine.  Each transition is triggered by a single inbound message.  The
 * worker never advances simulation time unless the host asks for a race step,
 * and the host never mutates population state.
 *
 * ```mermaid
 * stateDiagram-v2
 *     [*] --> idle
 *     idle --> initialised : init
 *     initialised --> generation_ready : request-generation
 *     generation_ready --> racing : start-race
 *     racing --> racing : request-race-step
 *     racing --> generation_ready : race finished
 *     idle --> stopped : stop
 *     initialised --> stopped : stop
 *     generation_ready --> stopped : stop
 *     racing --> stopped : stop
 * ```
 *
 * ## Host/worker authority boundary
 *
 * This router runs **inside the worker**.  The host calls `worker.postMessage`
 * with an `RacingWorkerInboundMessage`; the worker calls this router to
 * determine the next FSM state and the outbound payload to send back.
 * The host never advances simulation ticks directly — it only requests steps
 * and renders the snapshots it receives.
 *
 * ## Fallback transport
 *
 * When SharedArrayBuffer or nested worker pools are unavailable (e.g., the
 * host document lacks the required COOP/COEP headers), the same FSM and packed
 * snapshot types work over a single `Worker` with `postMessage`.  No
 * SharedArrayBuffer dependency exists in this module.  The transport layer can
 * be upgraded to a SharedArrayBuffer ring-buffer by wrapping the `postMessage`
 * call site without changing this router.  See
 * [Transferable objects (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferring_objects)
 * for the zero-copy transfer semantics used by the generation-ready response.
 *
 * ## Extension points
 *
 * | Hook | Where to extend | Current seam |
 * | --- | --- | --- |
 * | Frozen opponent selection | `simulation-worker.opponent-snapshot.service.ts` | Swap or weight the snapshot sampler |
 * | Generation lifecycle | `simulation-worker.coevolution.service.ts` | Add elitism, diversity pressure, or Lamarckian updates |
 * | Race pack layout | `simulation-worker.race-pack.service.ts` | Add track geometry, curricula, or sensor channels |
 * | Transfer transport | Wrap the `postMessage` caller | Replace structured clone with a ring buffer or batched frames |
 * | Neuromodulation / plasticity | `ModulatorBroadcaster` (not available in Tier 0) | Extend the protocol when dynamic activation or synaptic change primitives are available |
 * | Strategy-divergence analytics | `simulation-worker.strategy-divergence.service.ts` | Adjust classifier thresholds or add new divergence metrics |
 *
 * ## Multi-generation evaluation loop
 *
 * The diagram below shows how stateful resources persist across generations.
 * The coevolution container, adaptation engines, opponent snapshot store,
 * hall-of-fame snapshot pool, and strategy-divergence tracker are all created
 * once on the first `request-generation` and reused on every subsequent
 * generation. This avoids discarding accumulated population state between
 * races.
 *
 * ```mermaid
 * flowchart TD
 *     INIT["First request-generation"] --> CREATE["Create once:<br/>coevolution container<br/>adaptation engines<br/>snapshot store<br/>hall-of-fame pool<br/>strategy-divergence tracker"]
 *     CREATE --> GEN_READY["generation-ready"]
 *     GEN_READY --> START["start-race"]
 *     START --> RACE["racing<br/>(request-race-step*)"]
 *     RACE --> DONE{"race finished?"}
 *     DONE -- "no" --> RACE
 *     DONE -- "yes" --> TRANS["transitionToGenerationReady"]
 *     TRANS --> ADVANCE["Increment generation counter<br/>Advance team generation counters"]
 *     ADVANCE --> SNAPSHOT["Store opponent snapshot<br/>Accumulate hall-of-fame pool"]
 *     SNAPSHOT --> DIVERGE["Record strategy-divergence snapshot<br/>Classify trajectory"]
 *     DIVERGE --> FITNESS["Extract per-car fitness<br/>Compute team fitness"]
 *     FITNESS --> REPRODUCE["Polyandric reproduction per team<br/>Queen template + drone patches<br/>Offspring seed assignment"]
 *     REPRODUCE --> GEN_READY
 *     GEN_READY --> NEXT["Next request-generation<br/>(reuses all stateful resources)"]
 * ```
 *
 * The reproduction step follows the queen-plus-drones pattern described by the
 * racing reference design. The biological analogy is polyandry; see
 * [Polyandry (Wikipedia)](https://en.wikipedia.org/wiki/Polyandry) for
 * background on why one primary genome can benefit from multiple donor
 * contributions.
 *
 * ### Hall-of-fame opponent snapshot pool
 *
 * The `OpponentSnapshotPool` from `src/neat/nge-collective/` provides a
 * fixed-capacity rolling buffer with FIFO eviction. Each completed generation
 * adds per-car serialized genome snapshots to the pool so that historical
 * opponents persist across generations. The pool capacity is
 * `OPPONENT_SNAPSHOT_POOL_CAPACITY` (10). See
 * [Coevolution (Wikipedia)](https://en.wikipedia.org/wiki/Coevolution)
 * for why evaluating against historical opponents stabilises competitive
 * coevolution.
 *
 * ### Strategy-divergence analytics
 *
 * The `StrategyDivergenceTracker` accumulates per-generation team-level
 * observables (aggregate fitness, pit-lap distributions, reproduction-mode
 * mix) and classifies the resulting time series to detect whether the two
 * teams' strategies are diverging in an alternating arms-race pattern or one
 * team is consistently dominant. These metrics are observability-only — they
 * do NOT change fitness or reproduction.
 */
import type {
  EvolutionProtocolRouteResult,
  EvolutionProtocolState,
  GenerationReadyResponse,
  RacingWorkerInboundMessage,
  RacingWorkerPhase,
  StrategyDivergenceClassifierResult,
  PitLapDistribution,
} from './simulation-worker.evolution.types';
import {
  createCoevolutionContainer,
  createCarGenome,
  selectQueenPerTeam,
  type CoevolutionContainer,
  type CarGenome,
} from './simulation-worker.coevolution.service';
import {
  createOpponentSnapshotStore,
  type OpponentSnapshotStore,
} from './simulation-worker.opponent-snapshot.service';
import {
  addOpponentSnapshot,
  createOpponentSnapshotPool,
} from '../../../../src/neat/nge-collective/neat.nge-collective.metrics';
import {
  createRaceEpisodeRunner,
  extractPitLapDistribution,
  type RaceControllerNetwork,
  type RaceAdaptationContext,
  type OpponentSnapshot,
} from './simulation-worker.race-pack.service';
import {
  createPerCarAdaptationEngines,
  type RuntimeAdaptationEngine,
} from '../../controller/runtime.adaptation';
import type { Network } from '../../../../src/browser-entry.ts';
import { createStrategyDivergenceTracker } from './simulation-worker.strategy-divergence.service';
import { reproducePolyandric } from '../../../../src/neat/nge-evolution/neat.nge-evolution';
import type {
  NgeDnaCanonicalEnvelope,
  NgeReproductionPolicy,
} from '../../../../src/neat/nge-dna/neat.nge-dna.types';

/**
 * Computes shared-equal team fitness as the average of all team members' fitness.
 *
 * Each car on a team shares equal fitness — the team's fitness is the arithmetic
 * mean of all member fitness scores.  This cooperative pressure ensures a team
 * is only as strong as its average member, not its single best performer.
 *
 * See [Coevolution (Wikipedia)](https://en.wikipedia.org/wiki/Coevolution) for
 * background on why shared-equal fitness promotes cooperative team strategies
 * over free-rider exploitation.
 *
 * @param carFitnessScores - Per-car fitness scores (one entry per car).
 * @param teamLayout - Team id per car: 0 for blue (Team A), 1 for red (Team B).
 * @param teamId - 0 for Team A (blue), 1 for Team B (red).
 * @returns Average fitness of all team members, or 0 when the team has no members.
 *
 * @example
 * ```ts
 * const teamFitness = computeSharedEqualTeamFitness([10, 20, 30, 40], [0, 0, 1, 1], 0);
 * // → 15 (average of 10 and 20 — blue team members)
 * ```
 */
export function computeSharedEqualTeamFitness(
  carFitnessScores: readonly number[],
  teamLayout: readonly (0 | 1)[],
  teamId: 0 | 1,
): number {
  const teamMemberScores = carFitnessScores.filter(
    (_, carIndex) => teamLayout[carIndex] === teamId,
  );

  if (teamMemberScores.length === 0) {
    return 0;
  }

  const totalFitness = teamMemberScores.reduce((sum, score) => sum + score, 0);

  return totalFitness / teamMemberScores.length;
}

/**
 * Extracts per-car fitness scores from a race episode runner.
 *
 * Priority:
 * 1. When the runner exposes a `computeFitness` method (the real race episode
 *    runner or a mock that provides it), per-car fitness is read directly.
 * 2. When the runner exposes lap-completion data (`lapCompleted`,
 *    `lapTimeTicks`, `frame.progress01`) but no `computeFitness`, fitness is
 *    derived from real finish positions: lap finishers ranked by lap time,
 *    non-finishers ranked by progress, with a lap-completion bonus.
 * 3. Otherwise (e.g., a minimal mock in tests without lap data), a
 *    deterministic non-zero fallback based on car index is used so fitness
 *    feedback is never absent after a completed race.
 *
 * @param runner - Race episode runner (may be a mock without computeFitness).
 * @param carCount - Number of cars in the race pack.
 * @returns Per-car fitness scores (one entry per car).
 */
function extractCarFitnessScores(runner: unknown, carCount: number): number[] {
  interface RunnerWithFitness {
    computeFitness(carIndex: number): number;
  }
  const maybeRunner = runner as Partial<RunnerWithFitness>;
  if (typeof maybeRunner.computeFitness === 'function') {
    return Array.from({ length: carCount }, (_, carIndex) =>
      maybeRunner.computeFitness!(carIndex),
    );
  }
  // Finish-position extraction from real runner lap data (when available).
  const finishPositions = tryExtractFinishPositions(runner, carCount);
  if (finishPositions !== null) {
    return finishPositions;
  }
  // Deterministic non-zero fallback: finish position by car index (1-based).
  return Array.from({ length: carCount }, (_, carIndex) => carIndex + 1);
}

/** Fitness scale factor applied to finish-position rank (higher rank = lower fitness). */
const FINISH_POSITION_FITNESS_SCALE = 100 as const;

/** Fitness bonus awarded to cars that completed at least one lap. */
const LAP_COMPLETION_FITNESS_BONUS = 1000 as const;

/** Maximum number of opponent snapshots retained in the hall-of-fame pool. */
const OPPONENT_SNAPSHOT_POOL_CAPACITY = 10 as const;

/** Minimum generations before strategy-divergence classification produces non-zero output. */
const STRATEGY_DIVERGENCE_MIN_GENERATIONS = 2 as const;

/** Maximum number of same-team drone donors passed to `reproducePolyandric`. */
const MAX_POLYANDRIC_DRONES = 2 as const;

/**
 * Polyandric reproduction policy used by the racing FSM.
 *
 * Mirrors the reference spec: non-overlapping region assignment, queen-weighted
 * seed governance, a strong queen bias, and an evolvable mode flag. The seed
 * policy is passed as the racing shorthand `'queen-weighted'` and expanded by
 * the NGE_DNA constructor; the operator path only spreads the provided policy
 * object, so the shorthand is safe at runtime.
 */
const RACING_POLYANDRIC_POLICY = {
  mode: 'polyandric',
  polyandricDroneCount: 2,
  polyandricDroneContributionFraction: 0.25,
  queenBias: 0.85,
  assignedRegionStrategy: 'non-overlapping',
  modeIsEvolvable: true,
  seedPolicy: 'queen-weighted',
  parthenogenesisMutationRate: 0,
} as const;

/**
 * Extracts 1-based finish positions from real lap-completion data.
 *
 * Cars that completed at least one lap are ranked by lap time ascending;
 * non-finishers are ranked by track progress descending. The returned array
 * uses 1-based positions (1 = first place) indexed by carIndex.
 *
 * Returns `null` when the runner does not expose the required lap-data fields,
 * so callers can fall back to raw fitness scores.
 *
 * @param runner - Race episode runner (may be a mock without lap data).
 * @param carCount - Number of cars in the race pack.
 * @returns Per-car finish positions, or `null` when lap data is unavailable.
 */
function tryExtractFinishPositionRanks(
  runner: unknown,
  carCount: number,
): number[] | null {
  interface RunnerWithLapData {
    readonly lapCompleted: Uint8Array;
    readonly lapTimeTicks: Uint32Array;
    readonly frame: { readonly progress01: Float32Array };
  }
  const maybeRunner = runner as Partial<RunnerWithLapData>;
  if (
    !(maybeRunner.lapCompleted instanceof Uint8Array) ||
    !(maybeRunner.lapTimeTicks instanceof Uint32Array) ||
    !(maybeRunner.frame?.progress01 instanceof Float32Array)
  ) {
    return null;
  }

  const lapCompleted = maybeRunner.lapCompleted;
  const lapTimeTicks = maybeRunner.lapTimeTicks;
  const progress01 = maybeRunner.frame.progress01;

  const entries = Array.from({ length: carCount }, (_, carIndex) => ({
    carIndex,
    completedLap: lapCompleted[carIndex],
    lapTicks: lapTimeTicks[carIndex],
    progress: progress01[carIndex],
  }));

  const ranked = entries.toSorted((a, b) => {
    if (a.completedLap && b.completedLap) {
      return a.lapTicks - b.lapTicks;
    }
    if (a.completedLap) return -1;
    if (b.completedLap) return 1;
    return b.progress - a.progress;
  });

  const finishPositions = new Array<number>(carCount);
  for (let rank = 0; rank < ranked.length; rank++) {
    finishPositions[ranked[rank].carIndex] = rank + 1;
  }
  return finishPositions;
}

/**
 * Normalizes the return of `reproducePolyandric` so the FSM can use both the
 * real operator (which returns `{ offspring }`) and test mocks that return the
 * envelope directly.
 *
 * @param result - Raw operator return value.
 * @returns The offspring canonical envelope.
 */
function extractOffspringEnvelope(
  result:
    NgeDnaCanonicalEnvelope | { readonly offspring?: NgeDnaCanonicalEnvelope },
): NgeDnaCanonicalEnvelope {
  if (
    result &&
    typeof result === 'object' &&
    'offspring' in result &&
    result.offspring !== undefined
  ) {
    return result.offspring;
  }
  return result as NgeDnaCanonicalEnvelope;
}

/**
 * Derives per-car fitness from real finish positions when lap data is available.
 *
 * Cars that completed at least one lap (`lapCompleted[car] === 1`) are ranked
 * by lap time ascending (fewer ticks = better finish).  Cars that did not
 * complete a lap are ranked by track progress descending (further along =
 * better finish).  Each car receives a base fitness of
 * `(carCount - rank) * FINISH_POSITION_FITNESS_SCALE` plus a
 * `LAP_COMPLETION_FITNESS_BONUS` when the lap was completed.
 *
 * Returns `null` when the runner does not expose the required lap-data fields,
 * signalling the caller to use a fallback strategy.
 *
 * @param runner - Race episode runner (may be a mock without lap data).
 * @param carCount - Number of cars in the race pack.
 * @returns Per-car fitness scores, or `null` when lap data is unavailable.
 */
function tryExtractFinishPositions(
  runner: unknown,
  carCount: number,
): number[] | null {
  interface RunnerWithLapData {
    readonly lapCompleted: Uint8Array;
    readonly lapTimeTicks: Uint32Array;
    readonly frame: { readonly progress01: Float32Array };
  }
  const maybeRunner = runner as Partial<RunnerWithLapData>;
  if (
    !(maybeRunner.lapCompleted instanceof Uint8Array) ||
    !(maybeRunner.lapTimeTicks instanceof Uint32Array) ||
    !(maybeRunner.frame?.progress01 instanceof Float32Array)
  ) {
    return null;
  }

  const lapCompleted = maybeRunner.lapCompleted;
  const lapTimeTicks = maybeRunner.lapTimeTicks;
  const progress01 = maybeRunner.frame.progress01;

  // Step 1: Collect per-car lap data tuples.
  const entries = Array.from({ length: carCount }, (_, carIndex) => ({
    carIndex,
    completedLap: lapCompleted[carIndex],
    lapTicks: lapTimeTicks[carIndex],
    progress: progress01[carIndex],
  }));

  // Step 2: Rank — lap finishers first (by lap time ascending), then non-finishers (by progress descending).
  const ranked = entries.toSorted((a, b) => {
    if (a.completedLap && b.completedLap) {
      return a.lapTicks - b.lapTicks;
    }
    if (a.completedLap) return -1;
    if (b.completedLap) return 1;
    return b.progress - a.progress;
  });

  // Step 3: Assign fitness — higher for better finish position, with lap bonus.
  const fitnessScores = new Array<number>(carCount);
  for (let rank = 0; rank < ranked.length; rank++) {
    const { carIndex, completedLap } = ranked[rank];
    const baseFitness = (carCount - rank) * FINISH_POSITION_FITNESS_SCALE;
    const lapBonus = completedLap ? LAP_COMPLETION_FITNESS_BONUS : 0;
    fitnessScores[carIndex] = baseFitness + lapBonus;
  }
  return fitnessScores;
}

/** Allowed inbound message types per worker phase. */
const PHASE_ALLOWED_MESSAGES: Record<RacingWorkerPhase, readonly string[]> = {
  idle: ['init', 'stop'],
  initialised: ['request-generation', 'stop'],
  'generation-ready': ['start-race', 'request-generation', 'stop'],
  racing: ['request-race-step', 'stop'],
  stopped: ['stop'],
};

/**
 * Returns the canonical starting state for the racing worker protocol FSM.
 *
 * The FSM lifecycle is: idle → initialised → generation-ready → racing → stopped.
 *
 * @returns Idle protocol state.
 *
 * @example
 * ```ts
 * const state = createInitialProtocolState();
 * // state.phase === 'idle'
 * ```
 */
export function createInitialProtocolState(): EvolutionProtocolState {
  return { phase: 'idle', generation: 0 };
}

/**
 * Routes one inbound host-to-worker message through the evolution protocol FSM.
 *
 * Messages that arrive in a phase where they are not allowed return an error
 * string and leave `nextState` unchanged.  The `stop` message is always
 * accepted from any phase and unconditionally transitions to `stopped`.
 *
 * Lifecycle transitions:
 * - idle        + init             → initialised
 * - initialised + request-generation → generation-ready
 * - generation-ready + start-race  → racing
 * - racing      + request-race-step → racing (or generation-ready when done)
 * - any phase   + stop             → stopped
 *
 * @param message - Inbound host-to-worker protocol message.
 * @param state - Current FSM state.
 * @returns Next state plus optional response or rejection error.
 */
export function routeRacingWorkerProtocolMessage(
  message: RacingWorkerInboundMessage,
  state: EvolutionProtocolState,
): EvolutionProtocolRouteResult {
  // Step 1: stop is always accepted — transition unconditionally to stopped.
  if (message.type === 'stop') {
    return { nextState: { phase: 'stopped' } };
  }

  // Step 2: Reject messages not permitted in the current phase.
  const allowedTypes = PHASE_ALLOWED_MESSAGES[state.phase];
  if (!allowedTypes.includes(message.type)) {
    return {
      nextState: state,
      error: `Message type '${message.type}' is not allowed in phase '${state.phase}'.`,
    };
  }

  // Step 3: Apply permitted transitions.
  return applyTransition(message, state);

  /** Applies the permitted FSM transition for a validated message. */
  function applyTransition(
    msg: Exclude<RacingWorkerInboundMessage, { type: 'stop' }>,
    currentState: EvolutionProtocolState,
  ): EvolutionProtocolRouteResult {
    switch (msg.type) {
      case 'init':
        // Store the init configuration so it is no longer silently dropped.
        return {
          nextState: {
            phase: 'initialised',
            initConfig: {
              populationSize: msg.populationSize,
              rngSeed: msg.rngSeed,
              tier: msg.tier,
            },
          },
        };
      case 'request-generation': {
        // Step 1: Reuse the existing coevolution container when one is already
        // present in protocol state.  Recreating the container each generation
        // discards accumulated population state and team generation counters.
        const populationSize = currentState.initConfig?.populationSize ?? 10;
        const rngSeed = currentState.initConfig?.rngSeed ?? 1;
        const tier = currentState.initConfig?.tier ?? 1;
        const existingContainer = currentState.coevolutionContainer as
          CoevolutionContainer | undefined;
        const container =
          existingContainer ??
          createCoevolutionContainer({ populationSize, rngSeed, tier });
        const carGenomes = container.getCarGenomes();

        // Step 2: Reuse adaptation engines when present; create once on the
        // first request-generation.
        const existingEngines = currentState.adaptationEngines as
          ReadonlyMap<number, unknown> | undefined;
        const adaptationEngines =
          existingEngines ?? createPerCarAdaptationEngines(carGenomes.length);

        // Step 3: Create the rolling opponent snapshot store once so it
        // persists across generations.
        const opponentSnapshotStore =
          (currentState.opponentSnapshotStore as
            OpponentSnapshotStore | undefined) ??
          createOpponentSnapshotStore({ updateEveryNGenerations: 1 });

        // Step 3b: Create the opponent snapshot pool (hall-of-fame) once so
        // snapshots accumulate across generations via addOpponentSnapshot.
        const existingPool = currentState.opponentSnapshotPool;
        const opponentSnapshotPool =
          existingPool ??
          createOpponentSnapshotPool(OPPONENT_SNAPSHOT_POOL_CAPACITY);

        // Step 3c: Create the strategy-divergence tracker once so per-generation
        // team advantage snapshots accumulate across the racing coevolution loop.
        const existingTracker = currentState.strategyDivergenceTracker;
        const strategyDivergenceTracker =
          existingTracker ??
          createStrategyDivergenceTracker({
            teamSize: Math.max(1, Math.floor(carGenomes.length / 2)),
            minGenerations: STRATEGY_DIVERGENCE_MIN_GENERATIONS,
          });

        // Step 4: Carry the generation counter forward (incremented at the
        // racing→generation-ready transition, not here).
        const generation = currentState.generation ?? 0;

        // Step 5: Initial generation has no race results yet — fitness is zero.
        const initialFitnessScores = carGenomes.map(() => 0);

        return {
          nextState: {
            ...currentState,
            phase: 'generation-ready',
            coevolutionContainer: container,
            adaptationEngines,
            opponentSnapshotStore,
            opponentSnapshotPool,
            strategyDivergenceTracker,
            generation,
          },
          response: buildGenerationReadyResponse(
            carGenomes,
            generation,
            initialFitnessScores,
          ),
        };
      }
      case 'start-race':
        return {
          nextState: {
            ...currentState,
            phase: 'racing',
            raceRunner: createRaceRunnerForState(currentState),
          },
        };
      case 'request-race-step':
        return handleRaceStep(msg, currentState);
    }
  }

  /**
   * Creates a race episode runner with per-car adaptation engines.
   *
   * Uses the coevolution container and adaptation engines stored in the
   * protocol state from the `request-generation` transition. The runner
   * owns all per-car networks and runs continuous adaptation per tick.
   */
  function createRaceRunnerForState(
    currentState: EvolutionProtocolState,
  ): ReturnType<typeof createRaceEpisodeRunner> | undefined {
    const container = currentState.coevolutionContainer as
      ReturnType<typeof createCoevolutionContainer> | undefined;
    if (!container) {
      return undefined;
    }

    const carGenomes = container.getCarGenomes();
    const controllerNetworks: RaceControllerNetwork[] = carGenomes.map(
      (genome) => ({
        activate: (inputs: number[]) => genome.activate(inputs),
      }),
    );

    // Build per-car adaptation context: engines + live Network instances.
    const engines = currentState.adaptationEngines as Map<
      number,
      RuntimeAdaptationEngine
    >;
    const adaptationNetworks = new Map<number, Network>();
    for (const genome of carGenomes) {
      adaptationNetworks.set(genome.carIndex, genome.getNetwork());
    }

    const adaptationContext: RaceAdaptationContext = {
      engines,
      networks: adaptationNetworks,
    };

    const seed = currentState.initConfig!.rngSeed;
    const opponentSnapshot: OpponentSnapshot = {
      snapshotId: 'race-start',
      generation: 0,
      networkPayloads: [],
    };

    return createRaceEpisodeRunner(
      seed,
      opponentSnapshot,
      controllerNetworks,
      adaptationContext,
    );
  }

  /**
   * Handles a request-race-step message by ticking the race episode runner.
   *
   * The runner advances physics + inference + adaptation for all cars. Only
   * the packed render frame is sent back to the host; per-car networks stay
   * worker-side except for car 0's visualization payload.
   */
  function handleRaceStep(
    msg: { requestId: string; stepsToAdvance: number },
    currentState: EvolutionProtocolState,
  ): EvolutionProtocolRouteResult {
    const runner = currentState.raceRunner as
      ReturnType<typeof createRaceEpisodeRunner> | undefined;
    if (!runner) {
      return {
        nextState: currentState,
        response: {
          type: 'error' as const,
          message: 'No race episode runner available.',
        },
      };
    }

    for (let step = 0; step < msg.stepsToAdvance; step++) {
      runner.tick();
      if (runner.frame.done) {
        break;
      }
    }

    // When the race episode is finished, transition back to generation-ready
    // so the host can request the next generation or start a new race.
    if (runner.frame.done) {
      return transitionToGenerationReady(currentState, runner);
    }

    return {
      nextState: currentState,
      response: {
        type: 'race-step' as const,
        requestId: msg.requestId,
        done: false,
        visualizationPayload: runner.serializeVisualizationNetwork(),
      },
    };
  }

  /**
   * Transitions the FSM from racing to generation-ready after a race completes.
   *
   * Performs the full generation-boundary bookkeeping:
   * - Increments the generation counter.
   * - Advances both teams' isolated population generation counters via
   *   `advanceTeamGeneration`.
   * - Stores an opponent snapshot via `tryUpdateSnapshot` so the rolling
   *   snapshot accumulates across generations.
   * - Accumulates serialized car genomes into the hall-of-fame opponent
   *   snapshot pool with FIFO eviction.
   * - Records a strategy-divergence snapshot and classifies the coevolutionary
   *   trajectory.
   * - Extracts per-car fitness scores and pit-lap distributions from the race.
   * - Runs polyandric reproduction once per team to build the next generation's
   *   genomes (see below).
   * - Clears the race runner from state (recreated on the next `start-race`).
   * - Builds a generation-ready response with the real generation index and
   *   fitness scores.
   *
   * ## Polyandric reproduction sub-step
   *
   * For each team the queen is the car with the best individual finishing
   * position (selected by `selectQueenPerTeam`). The remaining team cars are
   * drone donors. Up to `MAX_POLYANDRIC_DRONES` drones are passed to
   * `reproducePolyandric`; the cap keeps the merge input stable when a team
   * has more cars than the operator expects. The operator uses
   * `RACING_POLYANDRIC_POLICY`: non-overlapping region assignment, strong
   * queen bias, and an evolvable mode flag. The returned offspring envelope
   * replaces every car genome on that team, so the next race starts from a
   * fresh, shared team genome.
   *
   * This pattern — one queen template combined with contributions from
   * multiple drone donors — mirrors the biological analogy of polyandry;
   * see [Polyandry (Wikipedia)](https://en.wikipedia.org/wiki/Polyandry).
   *
   * ## Offspring seed policy
   *
   * Each new car genome gets a deterministic seed derived from the original
   * `rngSeed`, the next generation index, the team id, and the car index:
   *
   * ```
   * offspringSeed = baseSeed + nextGeneration * 10000 + teamId * 1000 + carIndex
   * ```
   *
   * The formula guarantees that siblings on the same team differ by
   * `carIndex`, and the combination of generation, team, and car produces a
   * unique offset, so no two offspring share the same seed. This makes
   * repeated runs with the same initial seed reproducible at the genome level.
   *
   * @param currentState - Current FSM state (phase: racing).
   * @param runner - The completed race episode runner.
   * @returns Next state (phase: generation-ready) plus generation-ready response.
   */
  function transitionToGenerationReady(
    currentState: EvolutionProtocolState,
    runner: ReturnType<typeof createRaceEpisodeRunner>,
  ): EvolutionProtocolRouteResult {
    const container = currentState.coevolutionContainer as CoevolutionContainer;
    const carGenomes = container.getCarGenomes();

    // Step 1: Increment the generation counter.
    const nextGeneration = currentState.generation! + 1;

    // Step 2: Advance both teams' isolated generation counters.
    container.advanceTeamGeneration('team-a');
    container.advanceTeamGeneration('team-b');

    // Step 3: Store an opponent snapshot for the completed generation so the
    // rolling snapshot store accumulates opponents across generations.
    const store = currentState.opponentSnapshotStore as OpponentSnapshotStore;
    const snapshotPayload = carGenomes.map((genome) => genome.serialize());
    store.tryUpdateSnapshot(nextGeneration, snapshotPayload);

    // Step 3b: Accumulate opponent snapshots into the hall-of-fame pool so
    // snapshots persist across generations with FIFO eviction.
    let updatedPool = currentState.opponentSnapshotPool!;
    for (let carIndex = 0; carIndex < carGenomes.length; carIndex++) {
      const agentId = `car-${carIndex}-gen-${nextGeneration}`;
      const payload: Readonly<Record<string, unknown>> = {
        networkPayloads: [carGenomes[carIndex].serialize()],
      };
      updatedPool = addOpponentSnapshot(
        updatedPool,
        agentId,
        payload,
        nextGeneration,
      );
    }

    // Step 4: Extract per-car fitness scores from the completed race.
    const carFitnessScores = extractCarFitnessScores(runner, carGenomes.length);

    // Step 4b: Compute shared-equal team fitness for strategy-divergence snapshot.
    const teamLayout = carGenomes.map((genome) => genome.teamId);
    const teamAFitness = computeSharedEqualTeamFitness(
      carFitnessScores,
      teamLayout,
      0,
    );
    const teamBFitness = computeSharedEqualTeamFitness(
      carFitnessScores,
      teamLayout,
      1,
    );

    // Step 4c: Extract per-team pit-lap distributions from the completed race.
    const teamAPitLapDistribution = extractPitLapDistribution(runner, 0);
    const teamBPitLapDistribution = extractPitLapDistribution(runner, 1);

    // Step 4d: Record strategy-divergence snapshot and classify the trajectory.
    const tracker = currentState.strategyDivergenceTracker!;
    tracker.recordSnapshot({
      generation: nextGeneration,
      teamAFitness,
      teamBFitness,
      teamAPitLapDistribution,
      teamBPitLapDistribution,
      reproductionModeMix: {},
    });
    const strategyDivergenceClassifier = tracker.classify();

    // Step 5: Polyandric reproduction for each team.
    const finishPositionRanks = tryExtractFinishPositionRanks(
      runner,
      carGenomes.length,
    );
    const queenSelections = selectQueenPerTeam(
      finishPositionRanks ?? carFitnessScores,
      teamLayout,
    );

    for (const selection of queenSelections) {
      const queenGenome = carGenomes[selection.queenCarIndex];
      const selectedDrones = selection.droneCarIndices.slice(
        0,
        MAX_POLYANDRIC_DRONES,
      );

      const drones = selectedDrones.map((droneIndex) => ({
        dna: carGenomes[droneIndex].envelope,
        parentId: `car-${droneIndex}`,
      }));

      const reproductionResult = reproducePolyandric({
        ngeEnabled: true,
        queen: queenGenome.envelope,
        queenId: `car-${selection.queenCarIndex}`,
        drones,
        policy: RACING_POLYANDRIC_POLICY as unknown as NgeReproductionPolicy,
      });

      const offspringEnvelope = extractOffspringEnvelope(reproductionResult);
      const baseSeed = currentState.initConfig!.rngSeed;

      const teamCarIndices = carGenomes
        .map((genome, index) => ({ index, teamId: genome.teamId }))
        .filter(({ teamId }) => teamId === selection.teamId)
        .map(({ index }) => index);

      for (const carIndex of teamCarIndices) {
        const offspringSeed =
          baseSeed +
          nextGeneration * 10000 +
          selection.teamId * 1000 +
          carIndex;
        container.replaceCarGenome(
          carIndex,
          createCarGenome({
            carIndex,
            seed: offspringSeed,
            teamId: selection.teamId,
            populationId: carGenomes[carIndex].populationId,
            inputSize: carGenomes[carIndex].inputSize,
            outputSize: carGenomes[carIndex].outputSize,
            envelope: offspringEnvelope,
          }),
        );
      }
    }

    // Step 6: Re-read post-reproduction car genomes for the response.
    const postReproductionCarGenomes = container!.getCarGenomes();

    return {
      nextState: {
        ...currentState,
        phase: 'generation-ready',
        raceRunner: undefined,
        generation: nextGeneration,
        opponentSnapshotPool: updatedPool,
      },
      response: buildGenerationReadyResponse(
        postReproductionCarGenomes,
        nextGeneration,
        carFitnessScores,
        strategyDivergenceClassifier,
        teamAPitLapDistribution,
        teamBPitLapDistribution,
      ),
    };
  }

  /**
   * Builds a generation-ready response with per-car network payloads.
   *
   * Each car gets its own serialized genome payload.  Only car 0 (blue team #1)
   * is copied back to the browser for visualization; other cars' networks stay
   * in the worker.  The transfer list includes all payload ArrayBuffers for
   * zero-copy `postMessage` transfer.
   *
   * Team fitness uses shared-equal semantics: each team's fitness is the average
   * of all its members' fitness scores.  Individual per-car fitness is tracked
   * separately for NGE growth and adaptation.
   *
   * @param carGenomes - Per-car genomes from the coevolution container.
   * @returns Generation-ready response with per-car payloads and transfer list.
   */
  function buildGenerationReadyResponse(
    carGenomes: readonly CarGenome[],
    generation: number,
    carFitnessScores: readonly number[],
    strategyDivergenceClassifier?: StrategyDivergenceClassifierResult,
    teamAPitLapDistribution?: PitLapDistribution,
    teamBPitLapDistribution?: PitLapDistribution,
  ): GenerationReadyResponse {
    // Serialize each car's genome to a Float32Array payload.
    const carNetworkPayloads = carGenomes.map((genome) => genome.serialize());

    // Team layout derived from car genomes: 0 for blue (Team A), 1 for red (Team B).
    const teamLayout = carGenomes.map((genome) => genome.teamId);

    // Shared-equal team fitness: average of all team members' fitness scores.
    const teamABestFitness = computeSharedEqualTeamFitness(
      carFitnessScores,
      teamLayout,
      0,
    );
    const teamBBestFitness = computeSharedEqualTeamFitness(
      carFitnessScores,
      teamLayout,
      1,
    );

    // Car 0 (blue team #1) is the visualization car.
    const visualizationCarIndex = 0;
    const visualizationPayload = carNetworkPayloads[visualizationCarIndex];
    const bestNetworkPayload = visualizationPayload as Float32Array;

    // Collect all ArrayBuffer backs for zero-copy transfer.
    const transferList = carNetworkPayloads.map(
      (payload) => (payload as Float32Array).buffer as ArrayBuffer,
    );

    return {
      type: 'generation-ready',
      generation,
      teamABestFitness,
      teamBBestFitness,
      bestNetworkPayload,
      transferList,
      carNetworkPayloads,
      carFitnessScores,
      visualizationCarIndex,
      visualizationPayload,
      strategyDivergenceClassifier,
      teamAPitLapDistribution,
      teamBPitLapDistribution,
    };
  }
}
