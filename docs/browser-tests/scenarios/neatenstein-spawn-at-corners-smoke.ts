import { createGameState, gameTick } from '../../../examples/neatenstein/browser-entry/host/game/tick.ts';
import { spawnWaveTick, allEnemiesCleared } from '../../../examples/neatenstein/browser-entry/host/game/waves.ts';
import {
  NEATENSTEIN_MAP_SIZE,
  NEATENSTEIN_SPAWN_CENTER_X,
  NEATENSTEIN_SPAWN_CENTER_Y,
  NEATENSTEIN_PLAYER_MAX_HEALTH,
  NEATENSTEIN_PLAYER_MAX_AMMO,
  NEATENSTEIN_ENEMY_MAX_CONCURRENT,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
} from '../../../examples/neatenstein/browser-entry/host/game/constants.ts';

interface SpawnAtCornersSmokeResult {
  passed: boolean;
  durationMs: number;
  browserVisibility: string;
  scenario: string;
  url: string;
  consoleErrors: string[];
  notes: string;
  checks: Record<string, boolean>;
  metrics: Record<string, number>;
}

const SCENARIO = 'neatenstein-spawn-at-corners-smoke';
const startMs = performance.now();

const result: SpawnAtCornersSmokeResult = {
  passed: true,
  browserVisibility: document.visibilityState === 'visible' ? 'visible-foreground' : document.visibilityState,
  scenario: SCENARIO,
  url: location.href,
  consoleErrors: [],
  notes: '',
  checks: {},
  metrics: {},
};

const originalError = console.error;
console.error = (...args: unknown[]) => {
  result.consoleErrors.push(args.map(String).join(' '));
  originalError.apply(console, args);
};
window.addEventListener('error', (event: ErrorEvent) => {
  result.consoleErrors.push(event.message ?? String(event.error));
});
window.addEventListener('unhandledrejection', (event: PromiseRejectionEvent) => {
  result.consoleErrors.push(String(event.reason));
});

function fail(check: string, message: string) {
  result.passed = false;
  result.checks[check] = false;
  result.notes += `FAIL ${check}: ${message}\n`;
}

function pass(check: string) {
  result.checks[check] = true;
}

function isEdgePosition(p: { x: number; y: number }): boolean {
  const edgeMax = NEATENSTEIN_MAP_SIZE - 0.75;
  return p.x <= 0.75 || p.x >= edgeMax || p.y <= 0.75 || p.y >= edgeMax;
}

function isCenterPosition(p: { x: number; y: number }): boolean {
  return p.x === NEATENSTEIN_SPAWN_CENTER_X && p.y === NEATENSTEIN_SPAWN_CENTER_Y;
}

function killEnemy(state: ReturnType<typeof createGameState>, index: number) {
  const enemies = [...state.enemies];
  if (enemies[index]) {
    enemies[index] = { ...enemies[index], health: 0, active: false };
  }
  return { ...state, enemies, kills: (state.kills ?? 0) + 1 };
}

async function runChecks() {
  let state = createGameState({ seed: 42 });
  const spawnPositions: { x: number; y: number }[] = [];
  for (let i = 0; i < NEATENSTEIN_ENEMY_MAX_CONCURRENT * 3; i += 1) {
    const tickResult = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS);
    state = tickResult.state;
    const enemy = state.enemies[state.enemies.length - 1];
    if (tickResult.spawnedThisTick === 1 && enemy) {
      spawnPositions.push(enemy.position);
    }
  }

  const nonEdgePositions = spawnPositions.filter((p) => !isEdgePosition(p));
  if (nonEdgePositions.length > 0) {
    fail('spawn-on-edge', `found ${nonEdgePositions.length} non-edge spawns: ${JSON.stringify(nonEdgePositions)}`);
  } else {
    pass('spawn-on-edge');
  }

  const centerSpawns = spawnPositions.filter((p) => isCenterPosition(p));
  if (centerSpawns.length > 0) {
    fail('spawn-no-center', `spawned at center ${centerSpawns.length} times: ${JSON.stringify(centerSpawns)}`);
  } else {
    pass('spawn-no-center');
  }

  state = createGameState({ seed: 42 });
  for (let i = 0; i < NEATENSTEIN_ENEMY_MAX_CONCURRENT; i += 1) {
    const tickResult = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS);
    state = tickResult.state;
  }

  let waitTicks = 0;
  while (waitTicks < 50) {
    const tickResult = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS);
    state = tickResult.state;
    if (tickResult.spawnedThisTick !== 0) {
      fail('batch-wait-alive', `enemy spawned while batch still alive at wait tick ${waitTicks}`);
      break;
    }
    waitTicks += 1;
  }
  if (result.checks['batch-wait-alive'] === undefined) {
    pass('batch-wait-alive');
  }

  for (let i = 0; i < state.enemies.length; i += 1) {
    state = killEnemy(state, i);
  }
  const afterKillTick = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS);
  if (afterKillTick.spawnedThisTick !== 1) {
    fail('batch-resume-after-clear', `expected 1 spawn after clearing, got ${afterKillTick.spawnedThisTick}`);
  } else {
    pass('batch-resume-after-clear');
  }

  state = createGameState({ seed: 7 });
  state = {
    ...state,
    player: {
      ...state.player,
      position: { x: 1, y: 1 },
      previousPosition: { x: 1, y: 1 },
      health: 0,
      ammo: 0,
    },
  };
  state = gameTick(state, { move: { x: 0, y: 0 }, lookDelta: 0, fire: false, dash: false });

  if (state.player.health !== NEATENSTEIN_PLAYER_MAX_HEALTH) {
    fail('hero-health-restored', `health=${state.player.health}`);
  } else {
    pass('hero-health-restored');
  }
  if (state.player.ammo !== NEATENSTEIN_PLAYER_MAX_AMMO) {
    fail('hero-ammo-restored', `ammo=${state.player.ammo}`);
  } else {
    pass('hero-ammo-restored');
  }
  if (!isCenterPosition(state.player.position)) {
    fail('hero-spawn-center', `position=${JSON.stringify(state.player.position)}`);
  } else {
    pass('hero-spawn-center');
  }
  if ((state.deaths ?? 0) !== 1) {
    fail('hero-deaths-increment', `deaths=${state.deaths}`);
  } else {
    pass('hero-deaths-increment');
  }

  state = createGameState({ seed: 99 });
  let killCount = 0;
  for (let wave = 0; wave < 12; wave += 1) {
    for (let s = 0; s < NEATENSTEIN_ENEMY_MAX_CONCURRENT; s += 1) {
      const spawn = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS);
      state = spawn.state;
    }
    const start = state.enemies.length - NEATENSTEIN_ENEMY_MAX_CONCURRENT;
    for (let i = start; i < state.enemies.length; i += 1) {
      state = killEnemy(state, i);
      killCount += 1;
    }
  }

  result.metrics.killCount = killCount;
  result.metrics.enemyRosterSizeAfterKills = state.enemies.length;

  if (killCount < 86) {
    fail('kills-above-86', `killCount=${killCount}`);
  } else {
    pass('kills-above-86');
  }

  const aliveAfterKillLoop = state.enemies.filter((e) => (e.health ?? 0) > 0 && e.active !== false).length;
  if (aliveAfterKillLoop !== 0) {
    fail('all-killed-before-extended-ticks', `${aliveAfterKillLoop} enemies still alive after kill loop`);
  } else {
    pass('all-killed-before-extended-ticks');
  }

  if (!allEnemiesCleared(state.enemies)) {
    fail('all-enemies-cleared-after-kills', 'roster not fully cleared after all kills');
  } else {
    pass('all-enemies-cleared-after-kills');
  }

  const spawnCountBeforeExtendedTicks = state.spawnCount;
  for (let i = 0; i < 200; i += 1) {
    state = gameTick(state, { move: { x: 0, y: 0 }, lookDelta: 0, fire: false, dash: false });
  }

  result.durationMs = Math.round(performance.now() - startMs);
  result.metrics.spawnCount = state.spawnCount;
  result.metrics.extendedTickCount = 200;
  result.metrics.deaths = state.deaths ?? 0;

  if (state.spawnCount <= spawnCountBeforeExtendedTicks) {
    fail('extended-ticks-spawned', `spawnCount did not increase during extended ticks: ${state.spawnCount}`);
  } else {
    pass('extended-ticks-spawned');
  }

  result.notes += `killCount=${killCount} spawnCount=${state.spawnCount} deaths=${state.deaths}`;

  const status = document.getElementById('status');
  if (status) {
    status.textContent = JSON.stringify(result, null, 2);
  }
  (window as unknown as Record<string, unknown>)['neatensteinSpawnAtCornersSmokeResult'] = result;
}

runChecks().catch((err) => {
  result.passed = false;
  result.consoleErrors.push(String(err));
  result.durationMs = Math.round(performance.now() - startMs);
  result.notes += `UNCAUGHT: ${err instanceof Error ? err.message : String(err)}\n`;
  const status = document.getElementById('status');
  if (status) {
    status.textContent = JSON.stringify(result, null, 2);
  }
  (window as unknown as Record<string, unknown>)['neatensteinSpawnAtCornersSmokeResult'] = result;
});
