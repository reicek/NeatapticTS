/**
 * Red contract tests for the neatRacing right-world render-pass scene assembly
 * (slices P1-sprite-spacing + P1-bleacher-shape + P1-crowd2-red-contracts +
 * P1-sprite-scale-red-contracts, Step 14J + Step 14K + Step 14L,
 * AC-065 + AC-065b + AC-073 + AC-074 + AC-075 + AC-076 + AC-077).
 *
 * GREEN TARGET (04-implementing must make exactly these pass):
 * - {@link buildRightWorldSceneData} returns a scene with non-empty props,
 *   atlas frames, and billboards.
 * - Every assembled robot billboard keeps a personal-space radius >= 1 m
 *   equivalent in the XZ plane.
 * - Supports host only as many robots as their horizontal top area affords at
 *   >= 1 m spacing.
 * - RED-C (AC-073 + AC-074): every bleacher of the assembled scene hosts at
 *   least one SEATED billboard at every fixture seed, the scene carries at
 *   least one WALKER and at least one STANDING off-bleacher billboard, and
 *   the scene props carry the mechanic/crew/team-banner kinds on the INNER
 *   ring.
 * - RED-D (AC-074): the mechanic billboards use a TEAM accent, they carry the
 *   look-at-car STAND frame, the walker billboards carry the slow walk-cycle
 *   frames, and the off-bleacher agents keep the 1 m personal space.
 * - MAJOR-1 (AC-072): every bleacher of every fixture seed carries EXACTLY
 *   twelve seated spectators - four on each of its three levels - measured on
 *   the assembled scene, so the personal-space filter can never thin a
 *   bleacher behind a silently lowered bound.
 * - AC-070 (retained clauses): the atlas is PUBLIC-ONLY (`neatenstein-public`,
 *   144 frames, no mechanic frame id) and a seed rebuilds an identical scene.
 * - AC-071: the packed atlas fits within the device-reported
 *   `gl.getParameter(gl.MAX_TEXTURE_SIZE)` bound, and an over-limit frame
 *   count raises the documented atlas error instead of uploading a texture.
 * - RED-E (AC-075): the recorded atlas WIDTH AND HEIGHT each stay within the
 *   device-reported `gl.getParameter(gl.MAX_TEXTURE_SIZE)` bound FOR THE
 *   RE-POPULATED WORLD (20 bleachers, at least 240 seated billboards), the
 *   atlas stays the public-only 144-frame pack of 192 px cells, and the
 *   over-limit path raises the documented atlas error instead of uploading a
 *   texture.
 * - RED-F (AC-076): the per-frame billboard selection keeps EVERY culled-in
 *   seated billboard of the re-populated 20 x 12 crowd, and the per-frame
 *   billboard upload buffer is sized for that crowd, so the retired 128
 *   billboard cap cannot silently truncate the visible crowd.
 * - RED-C (AC-077): the RENDERED crowd billboard quad of the right-world pass
 *   is HALF the pre-14L world size. The uploaded vertex quad measures a full
 *   height of `frame.height * 0.5 * 0.05` (4.8 m) and a half width of
 *   `frame.width * 0.5 * 0.05 / 2` (2.4 m), and the quad still expands UPWARD
 *   from the foot anchor, so the measured bottom vertex sits at the billboard
 *   base Y. The halving is proven on the RENDERED geometry, not only on the
 *   module constant.
 * - RED-D (AC-077 + AC-072 + AC-074): the HALVED sprites change NO crowd
 *   COUNT. Every fixture seed still seats twelve spectators on every bleacher
 *   (240 seated billboards over the 20 bleachers), still carries 20 walkers,
 *   10 standing spectators and 4 mechanics, and every seated billboard still
 *   stands ON its bleacher level instead of dropping to the ground.
 * - RED-G (AC-077): the right-world pass keeps NO module-local copy of the
 *   meters-per-pixel constant. The pass IMPORTS the shared export
 *   `NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M` from `public.sprite.ts` and
 *   sizes the rendered quad from the shared constants, so the duplicated
 *   twin can never drift apart from the placement footprint again.
 *
 * GREEN STATUS (measured on the current tree after
 * P1-crowd2-off-bleacher-people): the RED-F CAPACITY contracts are GREEN -
 * the per-frame billboard upload buffer is sized from the named
 * `NEATRACING_BILLBOARD_FRAME_BUDGET` (274 quads = 32880 bytes), which is
 * INDEPENDENT of the shared prop LOD budget (128), and the per-frame
 * selection budget no longer shrinks with the visible prop count. The RED-E
 * re-populated-world atlas bound is GREEN as well: the bound is asserted
 * against the re-populated world itself - 20 bleachers with 12 seated
 * billboards each (240 seated) - so the world under the bound carries that
 * crowd. The RETAINED RED-E guards (the width bound, the height bound, the
 * public-only pack and the over-limit throw path) stay GREEN: the plan keeps
 * the AC-071 `gl.MAX_TEXTURE_SIZE` BOUND under AC-075 and the public-only
 * 144-frame pack does not grow with the re-populated crowd. The two RED-F
 * selection contracts stay GREEN: they pin that the per-frame selection
 * keeps every billboard the cull distance admits.
 */

import { describe, expect, it } from '@jest/globals';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  NEATRACING_LIGHTING_UNIFORM_NAMES,
  NEATRACING_SUN_COLOR,
  NEATRACING_SUN_DIRECTION,
} from '../engine/lighting';
import {
  getBleacherAxisAlignedHalfSize,
  getBleacherSeatAnchors,
  getBleacherSeatTopY,
  NEATRACING_BLEACHER_LEVEL_COUNT,
  NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M,
  NEATRACING_ROBOT_PERSONAL_SPACE_M,
  type RightWorldProp,
} from './right-world.builder';
import {
  buildRightWorldSceneData,
  createRightWorldPass,
  type RightWorldPassDrawState,
  type RightWorldSceneData,
} from './right-world.pass';
import {
  computePublicRuntimeFrameIndex,
  DIRECTION_ORDER,
  NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M,
  NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE,
  NEATRACING_PUBLIC_OFF_BLEACHER_AGENT_ALLOWANCE,
  PUBLIC_POSE_COUNT,
  PUBLIC_POSE_INDEX_STAND,
  PUBLIC_POSE_INDEX_WALK1,
  PUBLIC_POSE_INDEX_WALK2,
  PUBLIC_RUNTIME_ROLES,
  resolvePublicRuntimeRole,
  type PublicBillboard,
  type PublicRuntimeRole,
} from './public.sprite';
import { NEATRACING_TEAM_COLORS } from '../voxel/team-remap';
import { resolvePropHalfSize } from './right-world.prop-sizes';
import {
  NEATRACING_DECORATION_CULL_DISTANCE_M,
  NEATRACING_DECORATION_LOD_BUDGET,
} from './decoration.lod';

/**
 * Tolerance for matching a billboard foot anchor to a prop top surface.
 * Low pedestals are only ~0.5m tall, so a small vertical window is enough to
 * detect them while ignoring tall structures such as bleachers.
 */
const NEATRACING_PROP_HOST_TOLERANCE_M = 0.2;

/**
 * The people prop kinds the re-populated INNER ring must carry (AC-074).
 *
 * The `RightWorldProp['kind']` union carries the three pit people kinds, so
 * the roster below is typed with that union: a kind that the prop type does
 * not carry cannot slip into the contract.
 */
const REQUIRED_PEOPLE_PROP_KINDS: readonly RightWorldProp['kind'][] = [
  'mechanic',
  'crew',
  'team-banner',
];

/**
 * Runtime role of the bleacher SEATED spectators.
 *
 * The role names are typed with the {@link PublicRuntimeRole} union, which
 * carries all four roles of the re-populated crowd - seated, walker,
 * standing and mechanic - so a contract can never name a role the runtime
 * does not know.
 */
const CROWD_ROLE_SEATED: PublicRuntimeRole = 'seated';

/** Runtime role of the off-bleacher WALKERS. */
const CROWD_ROLE_WALKER: PublicRuntimeRole = 'walker';

/** Runtime role of the off-bleacher STANDING spectators. */
const CROWD_ROLE_STANDING: PublicRuntimeRole = 'standing';

/** Runtime role of the inner-ring MECHANICS. */
const CROWD_ROLE_MECHANIC: PublicRuntimeRole = 'mechanic';

/**
 * Number of bleacher props the re-populated right world carries (AC-073).
 *
 * The re-populated world reserves its bleacher slots BY CONSTRUCTION, so the
 * count is a named target of the crowd repopulation and not a measurement of
 * the current scene.
 */
const REPOPULATED_BLEACHER_COUNT = 20;

/**
 * Seated billboards every bleacher of the re-populated crowd carries
 * (AC-072: 4 evenly spread sprites per bleacher level x 3 levels).
 */
const REPOPULATED_SEATED_BILLBOARDS_PER_BLEACHER = 12;

/**
 * Seated billboards every bleacher LEVEL of the re-populated crowd carries
 * (AC-072: 4 evenly spread sprites per level, on 3 levels per bleacher).
 *
 * The count is a named target of the crowd repopulation, not a measurement
 * of the current scene, so the exact-count guard below can never be
 * satisfied by lowering the bound.
 */
const REPOPULATED_SEATED_BILLBOARDS_PER_LEVEL = 4;

/**
 * Seated billboards of the full re-populated crowd (AC-072 + AC-073).
 *
 * The per-frame billboard budget and the per-frame billboard upload buffer
 * must be sized for this crowd, because the retired shared cap
 * (`NEATRACING_DECORATION_LOD_BUDGET` = 128) was sized for the retired
 * single-host crowd and would silently truncate the visible crowd.
 */
const REPOPULATED_SEATED_CROWD =
  REPOPULATED_BLEACHER_COUNT * REPOPULATED_SEATED_BILLBOARDS_PER_BLEACHER;

/** Vertices emitted for one camera-facing billboard quad. */
const BILLBOARD_UPLOAD_VERTICES_PER_QUAD = 6;

/** Floats per billboard vertex in the upload buffer (position 3 + uv 2). */
const BILLBOARD_UPLOAD_FLOATS_PER_VERTEX = 5;

/**
 * Byte size of one billboard quad inside the per-frame upload buffer.
 *
 * The pass writes six vertices of five floats per billboard quad, so one
 * quad occupies 120 bytes of the GPU buffer.
 */
const BILLBOARD_UPLOAD_BYTES =
  BILLBOARD_UPLOAD_VERTICES_PER_QUAD * BILLBOARD_UPLOAD_FLOATS_PER_VERTEX * 4;

/**
 * Number of frames of the public-only sprite atlas (AC-075).
 *
 * The atlas serves the seated spectators, the mechanics, the walkers and the
 * standing spectators, and those roles REUSE the public frames instead of
 * adding a mechanic frame block: 6 accents x 8 directions x 3 poses.
 */
const PUBLIC_ONLY_ATLAS_FRAME_COUNT = 144;

/** Pixel size of one public-only atlas cell. */
const PUBLIC_ONLY_ATLAS_CELL_PX = 192;

/**
 * Side length of the public-only atlas pack (6 accents x 8 directions x
 * 3 poses is a 12 x 12 grid of 192 px cells).
 */
const PUBLIC_ONLY_ATLAS_GRID = 12;

/** Side length of the packed public-only atlas texture in pixels. */
const PUBLIC_ONLY_ATLAS_SIDE_PX =
  PUBLIC_ONLY_ATLAS_GRID * PUBLIC_ONLY_ATLAS_CELL_PX;

/**
 * World-space scale the crowd billboards must render at (SPRITE-1).
 *
 * The named production lever `NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE` is
 * introduced by the implementation slice, so the red contract pins the
 * FACTOR itself: the crowd billboards must render at EXACTLY HALF of the
 * 1:1 pixel-to-meter world size they carried before the halving.
 */
const EXPECTED_BILLBOARD_WORLD_SCALE = 0.5;

/**
 * Meters per sprite pixel the crowd billboards are sized with.
 *
 * The meters-per-pixel factor is NOT part of the halving: it stays at
 * 0.05 m/px, so a 192 px frame goes from 9.6 m tall to 4.8 m tall.
 */
const EXPECTED_METERS_PER_SPRITE_PIXEL = 0.05;

/**
 * Path of the right-world pass source read by the RED-G single-source guard.
 *
 * The guard asserts a STRUCTURAL property of the module (it must not keep a
 * second copy of the meters-per-pixel constant), so it reads the source file
 * itself instead of a generated bundle copy.
 */
const RIGHT_WORLD_PASS_SOURCE_PATH = resolve(
  process.cwd(),
  'examples',
  'neatRacing',
  'track',
  'right-world.pass.ts',
);

/**
 * Matches a module-local DECLARATION of a meters-per-pixel constant.
 *
 * The pattern requires a `const` declaration, so a plain reference to the
 * shared export `NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M` is never a match:
 * the trailing `_M` of the shared name can never be followed by `=` or by a
 * type annotation.
 */
const LOCAL_METERS_PER_PIXEL_DECLARATION_PATTERN =
  /const\s+NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE\s*(?::[^=]+)?=/;

/**
 * Matches the VALUE import block of the public sprite module.
 *
 * The type-only import (`import type { ... } from './public.sprite'`) never
 * matches, because the pattern requires `import` to be followed directly by
 * the opening brace.
 */
const PUBLIC_SPRITE_VALUE_IMPORT_PATTERN =
  /import\s*\{([\s\S]*?)\}\s*from\s*'\.\/public\.sprite'/;

/**
 * Tolerance of the rendered-quad measurements, in meters.
 *
 * The uploaded vertices are single-precision floats, so a world-space
 * measurement carries a float32 rounding residue far below this bound.
 */
const RENDERED_QUAD_TOLERANCE_M = 0.001;

/**
 * Tolerance of the seated billboard base Y against its bleacher level, in
 * meters.
 *
 * The seated spectators are placed on the seat surface plus the named
 * surface epsilon, so their base Y sits a millimeter above the bleacher
 * level. The bound stays far below the gap between two bleacher levels, so a
 * billboard that dropped to the ground - or onto another level - fails.
 */
const SEATED_BASE_Y_TOLERANCE_M = 0.05;

/**
 * Camera basis used to resolve the runtime atlas frame of a crowd billboard.
 *
 * The default camera of the right-world pass is module-private, so the
 * crowd contract re-declares the same draw-state basis the lighting red
 * tests use: the camera sits at [0, 2, 20] and looks down -Z (the forward
 * axis the pass derives as `cameraUp x cameraRight`), and the car sits at
 * the origin.
 */
const CROWD_CAMERA_WORLD: readonly [number, number, number] = [0, 2, 20];

/** Camera forward axis of {@link CROWD_CAMERA_WORLD}. */
const CROWD_CAMERA_FORWARD: readonly [number, number, number] = [0, 0, -1];

/** Car position the crowd agents look at. */
const CROWD_CAR_POSITION: readonly [number, number, number] = [0, 0, 0];

/**
 * Clock time used to resolve the runtime atlas frame of a crowd billboard.
 *
 * Six seconds sit in the middle of the 16 s look-around cycle, so the
 * deterministic look-around pause can never swap a walk pose for the stand
 * pose while the walk-cycle contract is measured.
 */
const CROWD_RUNTIME_TIME_SECONDS = 6;

/** Clock time one second after {@link CROWD_RUNTIME_TIME_SECONDS}. */
const CROWD_RUNTIME_TIME_LATER_SECONDS = 7;

/**
 * Deterministic seed sweep for the AC-070 crowd-host contract.
 *
 * Hand-rolled on purpose (`fast-check` is not a dependency of this repo): a
 * fixed seed set that covers every observed failure shape of the hosting
 * bleacher (wrong host and two hosting bleachers).
 */
const NEATRACING_CROWD_FIXTURE_SEEDS: readonly number[] = [
  0, 7, 42, 61, 123, 1337, 2777, 4242,
];

/**
 * Determinism tuple fields shared by every seed of the crowd sweep. The
 * production constant `NEATRACING_RIGHT_WORLD_BUILD_INPUT` is module-private,
 * so the sweep re-declares the same layout version and size bucket.
 */
const NEATRACING_CROWD_BUILD_INPUT = {
  layoutVersion: 1,
  sizeBucket: 'large',
} as const;

/**
 * Serializes a scene into a stable string so that two builds of the SAME
 * input can be compared field by field.
 *
 * Coordinates are rounded to 1e-6 m so the comparison is free of float-print
 * noise while still catching any real placement drift.
 *
 * @param scene - Scene data to serialize.
 * @returns A stable string digest of the scene props and billboards.
 */
function summarizeScene(scene: RightWorldSceneData): string {
  const round = (value: number): number => Number(value.toFixed(6));

  return JSON.stringify({
    props: scene.props.map((prop) => [
      prop.kind,
      prop.position.map(round),
      prop.ringSide,
    ]),
    billboards: scene.billboards.map((billboard) => [
      billboard.frameId,
      billboard.position.map(round),
      billboard.runtimeRole ?? 'seated',
    ]),
  });
}

/** Computes the smallest XZ distance between any two billboard positions. */
function minRobotXZDistance(
  positions: readonly (readonly [number, number, number])[],
): number {
  let minDistance = Number.POSITIVE_INFINITY;

  for (let i = 0; i < positions.length; i += 1) {
    const a = positions[i]!;
    for (let j = i + 1; j < positions.length; j += 1) {
      const b = positions[j]!;
      const dx = a[0] - b[0];
      const dz = a[2] - b[2];
      const distance = Math.hypot(dx, dz);
      if (distance < minDistance) {
        minDistance = distance;
      }
    }
  }

  return minDistance;
}

/**
 * Finds the prop whose top surface hosts a billboard foot anchor.
 * Returns `undefined` when the billboard stands on the ground plane.
 *
 * A billboard is considered hosted by a prop when its horizontal footprint
 * center lies inside the prop's XZ bounds and its foot Y is within a small
 * tolerance of the prop's top surface.  The closest matching prop top wins
 * so that low pedestals are not mistaken for the ground plane.
 */
function findHostProp(
  billboard: { readonly position: readonly [number, number, number] },
  props: readonly RightWorldProp[],
): RightWorldProp | undefined {
  const [x, y, z] = billboard.position;
  let best: RightWorldProp | undefined;
  let bestDeltaY = Number.POSITIVE_INFINITY;

  for (const prop of props) {
    const [px, py, pz] = prop.position;
    const [hx, hy, hz] = resolvePropHalfSize(prop.kind);
    const topY = py + 2 * hy;

    if (Math.abs(x - px) > hx || Math.abs(z - pz) > hz) {
      continue;
    }

    const deltaY = Math.abs(y - topY);
    if (deltaY <= NEATRACING_PROP_HOST_TOLERANCE_M && deltaY < bestDeltaY) {
      best = prop;
      bestDeltaY = deltaY;
    }
  }

  return best;
}

/** Computes the horizontal top surface area of a prop in square meters. */
function propTopArea(prop: RightWorldProp): number {
  const [hx, , hz] = resolvePropHalfSize(prop.kind);
  return 2 * hx * (2 * hz);
}

/**
 * Counts how many billboards are hosted by each prop, keyed by prop identity.
 * Billboards standing on the ground are not counted.
 */
function countHostedBillboardsPerProp(
  billboards: readonly {
    readonly position: readonly [number, number, number];
  }[],
  props: readonly RightWorldProp[],
): Map<RightWorldProp, number> {
  const counts = new Map<RightWorldProp, number>();

  for (const billboard of billboards) {
    const host = findHostProp(billboard, props);
    if (host) {
      counts.set(host, (counts.get(host) ?? 0) + 1);
    }
  }

  return counts;
}

/**
 * Finds the bleacher prop that hosts a billboard using real voxel geometry:
 * the billboard foot must be inside the bleacher XZ footprint and within
 * tolerance of the actual seat-top surface at that XZ location.
 *
 * @param billboard - Billboard to place.
 * @param props - Scene props to search.
 * @returns The hosting bleacher prop, or `undefined` when the billboard does
 *   not sit on a bleacher.
 */
function findBleacherHostProp(
  billboard: { readonly position: readonly [number, number, number] },
  props: readonly RightWorldProp[],
): RightWorldProp | undefined {
  const [x, y, z] = billboard.position;

  for (const prop of props) {
    if (prop.kind !== 'bleacher' || !prop.bleacherGeometry) {
      continue;
    }

    const [px, , pz] = prop.position;
    const [hx, hz] = getBleacherAxisAlignedHalfSize(prop);

    if (Math.abs(x - px) > hx || Math.abs(z - pz) > hz) {
      continue;
    }

    const topY = getBleacherSeatTopY(prop, x, z);
    if (Math.abs(y - topY) <= NEATRACING_PROP_HOST_TOLERANCE_M) {
      return prop;
    }
  }

  return undefined;
}

/**
 * Returns the public billboards that are really hosted on a bleacher.
 *
 * The re-populated crowd carries FOUR roles, and only the SEATED spectators
 * stand on a bleacher: the mechanics stand on their own inner-ring pit
 * props, the standing spectators wait outside the ring and the walkers
 * stroll the trackside, so those three roles are NOT bleacher-hosted.
 *
 * @param scene - Assembled right-world scene.
 * @returns The bleacher-hosted public billboards.
 */
function getBleacherHostedPublicBillboards(scene: {
  readonly props: readonly RightWorldProp[];
  readonly billboards: readonly {
    readonly position: readonly [number, number, number];
    readonly runtimeRole?: string;
  }[];
}): {
  readonly position: readonly [number, number, number];
  readonly runtimeRole?: string;
}[] {
  return scene.billboards.filter(
    (billboard) => findBleacherHostProp(billboard, scene.props) !== undefined,
  );
}

/**
 * Cache of the assembled fixture scenes, keyed by seed.
 *
 * The crowd contracts sweep the same eight fixture seeds from several
 * tests, and one scene build costs about a second, so every scene is built
 * once and shared. The contracts only READ the scene, so the shared
 * instance stays valid.
 */
const fixtureSceneCache = new Map<number, RightWorldSceneData>();

/**
 * Builds the assembled scene of one fixture seed.
 *
 * @param seed - Fixture seed of the crowd sweep.
 * @returns The assembled right-world scene of that seed.
 */
function buildFixtureScene(seed: number): RightWorldSceneData {
  const cached = fixtureSceneCache.get(seed);
  if (cached) {
    return cached;
  }

  const scene = buildRightWorldSceneData({
    seed,
    ...NEATRACING_CROWD_BUILD_INPUT,
  });
  fixtureSceneCache.set(seed, scene);

  return scene;
}

/**
 * Returns the runtime role of a billboard.
 *
 * The role is read through the shared {@link resolvePublicRuntimeRole}
 * resolver, so a role-less billboard counts as a walker exactly like the
 * atlas frame-index helper reads it. The runtime role union carries all
 * four roles of the re-populated crowd - seated, walker, standing and
 * mechanic - so the result keeps that union and needs no widening.
 *
 * @param billboard - Billboard to classify.
 * @returns The resolved runtime role.
 */
function runtimeRoleOf(billboard: PublicBillboard): PublicRuntimeRole {
  return resolvePublicRuntimeRole(billboard);
}

/**
 * Returns the billboards of a scene that do NOT stand on a bleacher.
 *
 * The off-bleacher agents of the re-populated crowd are the walkers, the
 * standing spectators outside the ring, and the inner-ring mechanics.
 *
 * @param scene - Assembled right-world scene.
 * @returns The off-bleacher billboards.
 */
function getOffBleacherBillboards(
  scene: RightWorldSceneData,
): readonly PublicBillboard[] {
  return scene.billboards.filter(
    (billboard) => findBleacherHostProp(billboard, scene.props) === undefined,
  );
}

/**
 * Counts the billboards of a scene that survive the decoration cull
 * distance for a camera.
 *
 * The per-frame billboard selection is bounded by the cull distance, so a
 * billboard beyond `NEATRACING_DECORATION_CULL_DISTANCE_M` is never selected
 * and must not be counted as a truncation.
 *
 * @param scene - Assembled right-world scene.
 * @param cameraWorld - Camera position in world space.
 * @returns The number of billboards inside the cull distance.
 */
function countCulledInBillboards(
  scene: RightWorldSceneData,
  cameraWorld: readonly [number, number, number],
): number {
  return scene.billboards.filter((billboard) => {
    const distance = Math.hypot(
      billboard.position[0] - cameraWorld[0],
      billboard.position[1] - cameraWorld[1],
      billboard.position[2] - cameraWorld[2],
    );
    return distance <= NEATRACING_DECORATION_CULL_DISTANCE_M;
  }).length;
}

/**
 * Counts the SEATED billboards of a scene that survive the decoration cull
 * distance for a camera.
 *
 * The seated billboards are the ones the bleachers host, so the count is
 * the visible part of the re-populated 20 x 12 crowd.
 *
 * @param scene - Assembled right-world scene.
 * @param cameraWorld - Camera position in world space.
 * @returns The number of culled-in seated billboards.
 */
function countCulledInSeatedBillboards(
  scene: RightWorldSceneData,
  cameraWorld: readonly [number, number, number],
): number {
  return scene.billboards.filter((billboard) => {
    if (runtimeRoleOf(billboard) !== CROWD_ROLE_SEATED) {
      return false;
    }
    if (findBleacherHostProp(billboard, scene.props) === undefined) {
      return false;
    }
    const distance = Math.hypot(
      billboard.position[0] - cameraWorld[0],
      billboard.position[1] - cameraWorld[1],
      billboard.position[2] - cameraWorld[2],
    );
    return distance <= NEATRACING_DECORATION_CULL_DISTANCE_M;
  }).length;
}

/**
 * Counts the billboards the pass actually submits in one frame.
 *
 * The billboard quad is the LAST `drawArrays` of a frame, and every
 * billboard contributes six vertices, so the selected count is recovered
 * from the vertex count of that last draw call.
 *
 * @param drawCalls - Draw calls recorded by the mock GL harness.
 * @returns The number of billboards selected for the frame.
 */
function countSelectedBillboards(
  drawCalls: ReadonlyArray<{ readonly count: number }>,
): number {
  const lastDrawCall = drawCalls[drawCalls.length - 1];
  if (lastDrawCall === undefined) {
    return 0;
  }
  return lastDrawCall.count / BILLBOARD_UPLOAD_VERTICES_PER_QUAD;
}

/**
 * Builds the direction×pose atlas index table the runtime frame-index helper
 * expects.
 *
 * The table mirrors the production index-table builder: the frame ids are
 * not unique across the colour blocks, so the LAST colour block wins, which
 * is the block the helper offsets the per-robot colour from.
 *
 * @param scene - Assembled right-world scene.
 * @returns The direction×pose index table.
 */
function buildRuntimeIndexTable(scene: RightWorldSceneData): Int16Array {
  const table = new Int16Array(DIRECTION_ORDER.length * PUBLIC_POSE_COUNT).fill(
    -1,
  );

  for (
    let frameIndex = 0;
    frameIndex < scene.atlas.frames.length;
    frameIndex += 1
  ) {
    const frame = scene.atlas.frames[frameIndex]!;
    const [direction, pose] = frame.id.split('-');
    const directionIndex = (DIRECTION_ORDER as readonly string[]).indexOf(
      direction,
    );
    let poseIndex = -1;
    if (pose === 'stand') {
      poseIndex = PUBLIC_POSE_INDEX_STAND;
    } else if (pose === 'walk1') {
      poseIndex = PUBLIC_POSE_INDEX_WALK1;
    } else if (pose === 'walk2') {
      poseIndex = PUBLIC_POSE_INDEX_WALK2;
    }
    if (directionIndex >= 0 && poseIndex >= 0) {
      table[directionIndex * PUBLIC_POSE_COUNT + poseIndex] = frameIndex;
    }
  }

  return table;
}

/**
 * Returns the direction×pose slot a runtime atlas frame index renders.
 *
 * The public atlas is colour-major (6 accents x 8 directions x 3 poses), so
 * the slot survives the per-robot colour offset in the low bits of the
 * frame index.
 *
 * @param frameIndex - Runtime atlas frame index.
 * @returns The direction×pose slot.
 */
function directionPoseSlotOfRuntimeFrameIndex(frameIndex: number): number {
  const blockSize = DIRECTION_ORDER.length * PUBLIC_POSE_COUNT;
  return ((frameIndex % blockSize) + blockSize) % blockSize;
}

/**
 * Returns the pose a runtime atlas frame index renders.
 *
 * @param frameIndex - Runtime atlas frame index.
 * @returns The pose index inside the direction×pose slot.
 */
function poseIndexOfRuntimeFrameIndex(frameIndex: number): number {
  return directionPoseSlotOfRuntimeFrameIndex(frameIndex) % PUBLIC_POSE_COUNT;
}

/**
 * Returns the direction a runtime atlas frame index renders.
 *
 * @param frameIndex - Runtime atlas frame index.
 * @returns The direction index inside the direction×pose slot.
 */
function directionIndexOfRuntimeFrameIndex(frameIndex: number): number {
  return Math.floor(
    directionPoseSlotOfRuntimeFrameIndex(frameIndex) / PUBLIC_POSE_COUNT,
  );
}

/**
 * Builds a controlled SEATED billboard at a given world position.
 *
 * The reference billboard resolves the camera-relative look-at-car direction
 * of that exact spot, so the mechanic contract can compare the mechanic
 * frame against the look-at-car frame without duplicating the bearing math.
 *
 * @param position - World position of the reference billboard.
 * @returns A seated billboard at that position.
 */
function seatedReferenceBillboard(
  position: readonly [number, number, number],
): PublicBillboard {
  return {
    position,
    runtimeRole: 'seated',
    seed: 0,
    index: 0,
    walkDirectionIndex: 0,
    frameId: 'front-stand',
    scale: 1,
    distance: 10,
    quad: [0, 0, 1, 1],
  };
}

/**
 * Computes a 0..360 hue angle, returning NaN for grayscale pixels.
 *
 * @param red - Red channel byte.
 * @param green - Green channel byte.
 * @param blue - Blue channel byte.
 * @returns The hue angle in degrees, or NaN for a grayscale colour.
 */
function rgbToHue(red: number, green: number, blue: number): number {
  const max = Math.max(red, green, blue);
  const min = Math.min(red, green, blue);

  if (max === min) {
    return Number.NaN;
  }

  const delta = max - min;
  let hue: number;

  if (max === red) {
    hue = ((green - blue) / delta) % 6;
  } else if (max === green) {
    hue = (blue - red) / delta + 2;
  } else {
    hue = (red - green) / delta + 4;
  }

  hue *= 60;
  if (hue < 0) {
    hue += 360;
  }

  return hue;
}

/**
 * Returns the dominant accent hue bucket of a sprite frame.
 *
 * The accent pixels are the recoloured ones, so the dominant saturated hue
 * of a frame is the accent colour the robot is painted in. Grayscale pixels
 * (the body greys and the white trim) and low-saturation pixels are skipped.
 *
 * @param rgba - Raw RGBA pixel data of the frame.
 * @returns The dominant 30° hue bucket, or `null` when the frame carries no
 *   saturated accent pixel.
 */
function dominantAccentHue(rgba: Uint8ClampedArray): number | null {
  const counts = new Map<number, number>();

  for (let index = 0; index < rgba.length; index += 4) {
    const red = rgba[index]!;
    const green = rgba[index + 1]!;
    const blue = rgba[index + 2]!;
    const alpha = rgba[index + 3]!;

    if (alpha < 32) {
      continue;
    }

    if (red === green && green === blue) {
      continue;
    }

    const max = Math.max(red, green, blue);
    const min = Math.min(red, green, blue);
    if (max > 0 && (max - min) / max < 0.5) {
      continue;
    }

    const hue = rgbToHue(red, green, blue);
    if (Number.isNaN(hue)) {
      continue;
    }

    const bucket = Math.round(hue / 30) * 30;
    counts.set(bucket, (counts.get(bucket) ?? 0) + 1);
  }

  let bestBucket: number | null = null;
  let bestCount = 0;
  for (const [bucket, count] of counts.entries()) {
    if (count > bestCount) {
      bestBucket = bucket;
      bestCount = count;
    }
  }

  return bestBucket;
}

/**
 * The hue buckets of the four NAMED team accents.
 *
 * The buckets are DERIVED from {@link NEATRACING_TEAM_COLORS} - the neon
 * cyan / red / green / blue liveries of the four Neat Racing teams - so the
 * team-accent contract reads the SAME named table the team remap uses and
 * can never drift from it.
 */
const TEAM_ACCENT_HUE_BUCKETS: readonly number[] = NEATRACING_TEAM_COLORS.map(
  (team) =>
    Math.round(rgbToHue(team.rgb[0], team.rgb[1], team.rgb[2]) / 30) * 30,
);

/**
 * Angular distance between two hue buckets, wrapped around the colour wheel.
 *
 * @param left - First hue bucket.
 * @param right - Second hue bucket.
 * @returns The distance in degrees.
 */
function hueBucketDistance(left: number, right: number): number {
  const raw = Math.abs(left - right) % 360;
  return Math.min(raw, 360 - raw);
}

/**
 * Tells whether a hue bucket belongs to the named team-accent family.
 *
 * A bucket counts as a team accent when it is the hue of one of the four
 * NAMED team colours, or when it sits in the SAME 30° hue band next to it:
 * the LD-24 team liveries and the neon atlas accents are two shades of the
 * same colour family (the team green and the neon green, the team red and
 * the neon pink), so a mechanic painted in the family of a team colour is a
 * team-accented mechanic.
 *
 * @param hue - Hue bucket of the accent colour.
 * @returns `true` when the hue belongs to the team-accent family.
 */
function isTeamAccentHue(hue: number): boolean {
  return TEAM_ACCENT_HUE_BUCKETS.some(
    (bucket) => hueBucketDistance(hue, bucket) <= 30,
  );
}

/** `gl.DEPTH_TEST` enum value (0x0B71). */
const MOCK_GL_DEPTH_TEST = 0x0b71;

/** `gl.CULL_FACE` enum value (0x0B44). */
const MOCK_GL_CULL_FACE = 0x0b44;

/** `gl.BLEND` enum value (0x0BE2). */
const MOCK_GL_BLEND = 0x0be2;

/** `gl.TRIANGLES` enum value (0x0004). */
const MOCK_GL_TRIANGLES = 0x0004;

/** `gl.ARRAY_BUFFER` enum value (0x8892). */
const MOCK_GL_ARRAY_BUFFER = 0x8892;

/** `gl.TEXTURE0` enum value. */
const MOCK_GL_TEXTURE0 = 0x84c0;

/** Mutable call recorder mirroring the right-world pass mock-GL shape. */
interface RightWorldMockGlHarness {
  readonly gl: WebGL2RenderingContext;
  readonly uniformLocationRequests: readonly string[];
  readonly uniform3fFeeds: ReadonlyArray<{
    readonly name: string;
    readonly channels: readonly [number, number, number];
  }>;
  readonly uniform1fFeeds: ReadonlyArray<{
    readonly name: string;
    readonly value: number;
  }>;
  readonly uniform1iFeeds: ReadonlyArray<{
    readonly name: string;
    readonly value: number;
  }>;
  readonly uniformMatrixFeeds: ReadonlyArray<{
    readonly name: string;
    readonly matrix: readonly number[];
  }>;
  readonly texImage2DCalls: ReadonlyArray<{
    readonly width: number;
    readonly height: number;
    readonly format: number;
    readonly type: number;
  }>;
  readonly drawCalls: ReadonlyArray<{
    readonly mode: number;
    readonly first: number;
    readonly count: number;
  }>;
  readonly bufferDataCalls: ReadonlyArray<{
    readonly size: number;
  }>;
  readonly bufferSubDataCalls: ReadonlyArray<{
    readonly data: Float32Array;
  }>;
}

/** Filters recorded uniform3f feeds down to one uniform name. */
function uniform3fFeedsNamed(
  harness: RightWorldMockGlHarness,
  name: string,
): ReadonlyArray<{
  readonly name: string;
  readonly channels: readonly [number, number, number];
}> {
  return harness.uniform3fFeeds.filter((feed) => feed.name === name);
}

/** Filters recorded uniform1f feeds down to one uniform name. */
function uniform1fFeedsNamed(
  harness: RightWorldMockGlHarness,
  name: string,
): ReadonlyArray<{ readonly name: string; readonly value: number }> {
  return harness.uniform1fFeeds.filter((feed) => feed.name === name);
}

/**
 * Measures the byte size a `bufferData` call requests.
 *
 * The right-world pass sizes its vertex buffers with the numeric overload,
 * but the typed-array overload is measured the same way, so the harness
 * reports the requested byte size of either form.
 *
 * @param sizeOrData - Numeric size or the uploaded data.
 * @returns The requested byte size in bytes.
 */
function measureBufferDataBytes(
  sizeOrData: number | ArrayBufferView | ArrayBuffer | null,
): number {
  if (typeof sizeOrData === 'number') {
    return sizeOrData;
  }
  if (sizeOrData === null) {
    return 0;
  }
  return sizeOrData.byteLength;
}

/**
 * Reads the byte size the pass reserved for the per-frame billboard upload
 * buffer.
 *
 * The pass creates the prop vertex buffer first and the billboard vertex
 * buffer second, so the LAST `bufferData` call of the create step sizes the
 * per-frame billboard upload buffer.
 *
 * @param harness - Mock GL harness of a created right-world pass.
 * @returns The reserved byte size of the billboard upload buffer.
 */
function billboardUploadBufferBytes(harness: RightWorldMockGlHarness): number {
  const lastBufferDataCall =
    harness.bufferDataCalls[harness.bufferDataCalls.length - 1];
  return lastBufferDataCall?.size ?? 0;
}

/**
 * One rendered billboard quad measured from the uploaded vertex data.
 *
 * The right-world pass expands every billboard quad UPWARD from the foot
 * anchor along the camera up axis and sideways along the camera right axis,
 * so the measured extents of a quad are its world height, its world half
 * width and the world Y of its foot anchor.
 */
interface RenderedBillboardQuad {
  /** World-space Y of the quad's BOTTOM vertices - the foot anchor. */
  readonly baseY: number;
  /** World-space Y of the quad's TOP vertices. */
  readonly topY: number;
  /** World-space X of the quad's center. */
  readonly centerX: number;
  /** World-space Z of the quad's center. */
  readonly centerZ: number;
  /** World-space half width of the quad along the camera right axis. */
  readonly halfWidth: number;
  /** World-space full height of the quad along the camera up axis. */
  readonly height: number;
}

/**
 * Reads the rendered billboard quads out of the recorded vertex upload.
 *
 * The billboard upload is the LAST `bufferSubData` of a frame (the prop
 * upload comes first) and the billboard draw is the LAST `drawArrays`, so
 * the drawn vertex count bounds the quads that are read. The recorded
 * source array holds the whole scratch buffer, so only the drawn prefix is
 * measured.
 *
 * @param harness - Mock GL harness of a drawn right-world pass.
 * @returns The rendered crowd billboard quads of the frame.
 */
function readRenderedBillboardQuads(
  harness: RightWorldMockGlHarness,
): readonly RenderedBillboardQuad[] {
  const upload =
    harness.bufferSubDataCalls[harness.bufferSubDataCalls.length - 1];
  const lastDrawCall = harness.drawCalls[harness.drawCalls.length - 1];

  if (upload === undefined || lastDrawCall === undefined) {
    return [];
  }

  const quadCount = Math.floor(
    lastDrawCall.count / BILLBOARD_UPLOAD_VERTICES_PER_QUAD,
  );
  const requiredFloats =
    quadCount *
    BILLBOARD_UPLOAD_VERTICES_PER_QUAD *
    BILLBOARD_UPLOAD_FLOATS_PER_VERTEX;

  if (upload.data.length < requiredFloats) {
    return [];
  }

  const quads: RenderedBillboardQuad[] = [];

  for (let quadIndex = 0; quadIndex < quadCount; quadIndex += 1) {
    const quadOffset =
      quadIndex *
      BILLBOARD_UPLOAD_VERTICES_PER_QUAD *
      BILLBOARD_UPLOAD_FLOATS_PER_VERTEX;
    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    let minZ = Number.POSITIVE_INFINITY;
    let maxZ = Number.NEGATIVE_INFINITY;

    for (
      let vertexIndex = 0;
      vertexIndex < BILLBOARD_UPLOAD_VERTICES_PER_QUAD;
      vertexIndex += 1
    ) {
      const offset =
        quadOffset + vertexIndex * BILLBOARD_UPLOAD_FLOATS_PER_VERTEX;
      const x = upload.data[offset]!;
      const y = upload.data[offset + 1]!;
      const z = upload.data[offset + 2]!;

      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      minZ = Math.min(minZ, z);
      maxZ = Math.max(maxZ, z);
    }

    quads.push({
      baseY: minY,
      topY: maxY,
      centerX: (minX + maxX) / 2,
      centerZ: (minZ + maxZ) / 2,
      halfWidth: (maxX - minX) / 2,
      height: maxY - minY,
    });
  }

  return quads;
}

/** Outcome of the rendered-quad probe of the right-world pass. */
interface RenderedQuadProbe {
  /** The assembled scene the pass renders. */
  readonly scene: RightWorldSceneData;
  /** The rendered crowd billboard quads of the drawn frame. */
  readonly quads: readonly RenderedBillboardQuad[];
}

/**
 * Cache of the rendered-quad probe.
 *
 * The probe builds the whole right-world scene and draws a frame, so it is
 * built once and shared by the rendered-quad contracts.
 */
let renderedQuadProbe: RenderedQuadProbe | undefined;

/**
 * Reads the right-world pass source (not a generated bundle copy).
 *
 * @returns The TypeScript source of the right-world render pass.
 */
function readRightWorldPassSource(): string {
  return readFileSync(RIGHT_WORLD_PASS_SOURCE_PATH, 'utf-8');
}

/**
 * Draws one frame of the right-world pass and measures the crowd quads the
 * pass actually uploaded.
 *
 * @returns The assembled scene and the rendered crowd billboard quads.
 */
function probeRenderedCrowdQuads(): RenderedQuadProbe {
  if (renderedQuadProbe !== undefined) {
    return renderedQuadProbe;
  }

  const harness = createRightWorldMockGlHarness();
  const pass = createRightWorldPass(harness.gl);
  pass.draw(createRightWorldDrawState());

  const probe: RenderedQuadProbe = {
    scene: buildRightWorldSceneData(),
    quads: readRenderedBillboardQuads(harness),
  };
  renderedQuadProbe = probe;

  return probe;
}

/** Outcome of one atlas probe against the mock GL harness. */
interface RightWorldAtlasProbe {
  readonly harness: RightWorldMockGlHarness;
  readonly thrown: unknown;
  readonly atlasCalls: ReadonlyArray<{
    readonly width: number;
    readonly height: number;
    readonly format: number;
    readonly type: number;
  }>;
}

/**
 * Creates the right-world pass against a mock GL harness with a mocked
 * device texture limit and records the outcome plus the atlas uploads.
 *
 * @param maxTextureSize - Mocked `gl.getParameter(gl.MAX_TEXTURE_SIZE)`
 *   value of the probe.
 * @returns The harness, the thrown error (if any), and the recorded atlas
 *   texture uploads.
 */
function probeRightWorldAtlas(maxTextureSize: number): RightWorldAtlasProbe {
  const harness = createRightWorldMockGlHarness({ maxTextureSize });
  let thrown: unknown = null;

  try {
    createRightWorldPass(harness.gl);
  } catch (error) {
    thrown = error;
  }

  const atlasCalls = harness.texImage2DCalls.filter(
    (call) => call.format === 0x1908 && call.type === 0x1401,
  );

  return { harness, thrown, atlasCalls };
}

/** Default right-world draw state used by the lighting red tests. */
function createRightWorldDrawState(): RightWorldPassDrawState {
  return {
    viewProjection: new Float32Array([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    ]),
    cameraWorld: [0, 2, 20],
    cameraRight: [1, 0, 0],
    cameraUp: [0, 1, 0],
    timeSeconds: 0,
    carPosition: [0, 0, 0],
  };
}

/**
 * Builds a recording mock WebGL2 context covering every GL entry point the
 * right-world pass touches during create and draw, including the state
 * save/restore queries around the prop and billboard scopes.
 *
 * @param options - Optional device overrides. `maxTextureSize` replaces the
 *   mocked `gl.getParameter(gl.MAX_TEXTURE_SIZE)` value so the AC-071
 *   device-sourced atlas bound and its over-limit throw path can be exercised
 *   without touching production code. Defaults to 16384, which keeps every
 *   existing call site behaving exactly as before.
 */
function createRightWorldMockGlHarness(
  options: { readonly maxTextureSize?: number } = {},
): RightWorldMockGlHarness {
  const maxTextureSize = options.maxTextureSize ?? 16384;
  const uniformLocationRequests: string[] = [];
  const uniform3fFeeds: Array<{
    readonly name: string;
    readonly channels: readonly [number, number, number];
  }> = [];
  const uniform1fFeeds: Array<{
    readonly name: string;
    readonly value: number;
  }> = [];
  const uniform1iFeeds: Array<{
    readonly name: string;
    readonly value: number;
  }> = [];
  const uniformMatrixFeeds: Array<{
    readonly name: string;
    readonly matrix: readonly number[];
  }> = [];
  const texImage2DCalls: Array<{
    readonly width: number;
    readonly height: number;
    readonly format: number;
    readonly type: number;
  }> = [];
  const drawCalls: Array<{
    readonly mode: number;
    readonly first: number;
    readonly count: number;
  }> = [];
  const bufferDataCalls: Array<{ readonly size: number }> = [];
  const bufferSubDataCalls: Array<{ readonly data: Float32Array }> = [];

  const parameterDefaults = new Map<number, unknown>([
    [0x0b71, false], // DEPTH_TEST
    [0x0be2, false], // BLEND
    [0x0b44, false], // CULL_FACE
    [0x0b71, false], // DEPTH_TEST (duplicate, ignored)
    [0x0b90, true], // DEPTH_WRITEMASK
    [0x0302, 1], // BLEND_SRC_RGB = ONE
    [0x0303, 0], // BLEND_DST_RGB = ZERO
    [0x0306, 1], // BLEND_SRC_ALPHA = ONE
    [0x0307, 0], // BLEND_DST_ALPHA = ZERO
    [0x84e0, 0], // ACTIVE_TEXTURE = TEXTURE0
    [0x8b8d, null], // CURRENT_PROGRAM
    [0x8894, null], // ARRAY_BUFFER_BINDING
    [0x8b8d, null], // CURRENT_PROGRAM (duplicate)
    [0x806a, null], // TEXTURE_BINDING_2D
    [0x0d33, maxTextureSize], // MAX_TEXTURE_SIZE
  ]);

  const gl = {
    VERTEX_SHADER: 0x8b31,
    FRAGMENT_SHADER: 0x8b20,
    COMPILE_STATUS: 0x8b81,
    LINK_STATUS: 0x8b82,
    ARRAY_BUFFER: MOCK_GL_ARRAY_BUFFER,
    STATIC_DRAW: 0x88e4,
    DYNAMIC_DRAW: 0x88e8,
    FLOAT: 0x1406,
    UNSIGNED_BYTE: 0x1401,
    RGBA: 0x1908,
    TEXTURE_2D: 0x0de1,
    TEXTURE0: MOCK_GL_TEXTURE0,
    DEPTH_TEST: MOCK_GL_DEPTH_TEST,
    CULL_FACE: MOCK_GL_CULL_FACE,
    BLEND: MOCK_GL_BLEND,
    TRIANGLES: MOCK_GL_TRIANGLES,
    CLAMP_TO_EDGE: 0x812f,
    LINEAR: 0x2601,
    SRC_ALPHA: 0x0302,
    ONE_MINUS_SRC_ALPHA: 0x0303,
    MAX_TEXTURE_SIZE: 0x0d33,
    createShader: (): object => ({}),
    shaderSource: (): void => undefined,
    compileShader: (): void => undefined,
    getShaderParameter: (): boolean => true,
    getShaderInfoLog: (): string => '',
    createProgram: (): object => ({}),
    attachShader: (): void => undefined,
    linkProgram: (): void => undefined,
    getProgramParameter: (): boolean => true,
    getProgramInfoLog: (): string => '',
    getAttribLocation: (_program: unknown, _name: string): number => {
      void _program;
      void _name;
      return 0;
    },
    getUniformLocation: (_program: unknown, name: string): object => {
      uniformLocationRequests.push(name);
      return { name };
    },
    createBuffer: (): object => ({}),
    bindBuffer: (): void => undefined,
    bufferData: (
      _target: number,
      sizeOrData: number | ArrayBufferView | ArrayBuffer | null,
      _usage: number,
    ): void => {
      void _target;
      void _usage;
      bufferDataCalls.push({ size: measureBufferDataBytes(sizeOrData) });
    },
    bufferSubData: (
      _target: number,
      _dstByteOffset: number,
      srcData?: ArrayBufferView | ArrayBuffer | null,
      _srcOffset?: number,
      _length?: number,
    ): void => {
      void _target;
      void _dstByteOffset;
      void _srcOffset;
      void _length;

      if (srcData === undefined || srcData === null) {
        return;
      }

      // The uploaded source is COPIED, so the recorded vertex data keeps the
      // values of the frame it was uploaded in even when the pass reuses the
      // same scratch array in a later frame.
      const source =
        srcData instanceof ArrayBuffer
          ? new Float32Array(srcData)
          : new Float32Array(
              srcData.buffer,
              srcData.byteOffset,
              srcData.byteLength / Float32Array.BYTES_PER_ELEMENT,
            );

      bufferSubDataCalls.push({ data: new Float32Array(source) });
    },
    enableVertexAttribArray: (): void => undefined,
    disableVertexAttribArray: (): void => undefined,
    vertexAttribPointer: (): void => undefined,
    enable: (): void => undefined,
    disable: (): void => undefined,
    depthMask: (): void => undefined,
    blendFunc: (): void => undefined,
    blendFuncSeparate: (): void => undefined,
    useProgram: (): void => undefined,
    activeTexture: (): void => undefined,
    bindTexture: (): void => undefined,
    createTexture: (): object => ({}),
    texImage2D: (
      _target: number,
      _level: number,
      _internalformat: number,
      width: number,
      height: number,
      _border: number,
      format: number,
      type: number,
      _data: unknown,
    ): void => {
      void _target;
      void _level;
      void _internalformat;
      void _border;
      void _data;
      texImage2DCalls.push({ width, height, format, type });
    },
    texParameteri: (): void => undefined,
    uniformMatrix4fv: (
      location: { name: string },
      _transpose: boolean,
      matrix: Float32Array,
    ): void => {
      uniformMatrixFeeds.push({
        name: location.name,
        matrix: Array.from(matrix),
      });
    },
    uniform3f: (
      location: { name: string },
      red: number,
      green: number,
      blue: number,
    ): void => {
      uniform3fFeeds.push({
        name: location.name,
        channels: [red, green, blue],
      });
    },
    uniform3fv: (
      location: { name: string },
      channels: readonly [number, number, number],
    ): void => {
      uniform3fFeeds.push({ name: location.name, channels });
    },
    uniform1f: (location: { name: string }, value: number): void => {
      uniform1fFeeds.push({ name: location.name, value });
    },
    uniform1fv: (
      location: { name: string },
      values: readonly number[],
    ): void => {
      for (const value of values) {
        uniform1fFeeds.push({ name: location.name, value });
      }
    },
    uniform1i: (location: { name: string }, value: number): void => {
      uniform1iFeeds.push({ name: location.name, value });
    },
    drawArrays: (mode: number, first: number, count: number): void => {
      drawCalls.push({ mode, first, count });
    },
    isEnabled: (): boolean => false,
    getParameter: (pname: number): unknown =>
      parameterDefaults.get(pname) ?? null,
    getVertexAttrib: (): boolean => false,
    deleteProgram: (): void => undefined,
    deleteShader: (): void => undefined,
    deleteBuffer: (): void => undefined,
    deleteTexture: (): void => undefined,
  };

  return {
    gl: gl as unknown as WebGL2RenderingContext,
    uniformLocationRequests,
    uniform3fFeeds,
    uniform1fFeeds,
    uniform1iFeeds,
    uniformMatrixFeeds,
    texImage2DCalls,
    drawCalls,
    bufferDataCalls,
    bufferSubDataCalls,
  };
}

describe('right-world.pass scene assembly (P1-sprite-spacing, AC-065)', () => {
  it('returns a scene with non-empty props, atlas, and billboards', () => {
    const scene = buildRightWorldSceneData();

    expect(scene.props.length).toBeGreaterThan(0);
    expect(scene.atlas.frames.length).toBeGreaterThan(0);
    expect(scene.billboards.length).toBeGreaterThan(1);
  });

  it('assembles robot billboards with >= 1m XZ personal-space separation', () => {
    const scene = buildRightWorldSceneData();
    const positions = scene.billboards.map((billboard) => billboard.position);

    expect(positions.length).toBeGreaterThan(1);

    const minDistance = minRobotXZDistance(positions);

    // B4: the retained AC-065 contract is RE-EXPRESSED against the named
    // margin of AC-069. The effective clearance is still 1 m - the named
    // epsilon only covers the double-precision residue of the rotated seat
    // lattice, so the contract is never relaxed.
    expect(minDistance).toBeGreaterThanOrEqual(
      NEATRACING_ROBOT_PERSONAL_SPACE_M -
        NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M,
    );
  });
});

describe('right-world.pass scene anchoring (P1-bleacher-shape, AC-065b)', () => {
  it('hosts only as many robots per support as its top area affords at 1m spacing', () => {
    const scene = buildRightWorldSceneData();
    const counts = countHostedBillboardsPerProp(scene.billboards, scene.props);

    for (const [prop, count] of counts.entries()) {
      const maxHosted = Math.floor(propTopArea(prop));

      expect(count).toBeLessThanOrEqual(maxHosted);
    }
  });
});

describe('right-world.pass prop shared sun lighting (P1-world-lighting-coherence)', () => {
  it('resolves the shared sun direction uniform in the prop program', () => {
    const harness = createRightWorldMockGlHarness();
    createRightWorldPass(harness.gl);
    expect(harness.uniformLocationRequests).toContain('uSunDirection');
  });

  it('resolves all shared lighting uniforms in the prop program', () => {
    const harness = createRightWorldMockGlHarness();
    createRightWorldPass(harness.gl);
    for (const name of NEATRACING_LIGHTING_UNIFORM_NAMES) {
      expect(harness.uniformLocationRequests).toContain(name);
    }
  });

  it('feeds the shared sun direction and color during prop draw', () => {
    const harness = createRightWorldMockGlHarness();
    const pass = createRightWorldPass(harness.gl);
    pass.draw(createRightWorldDrawState());
    const sunDirectionFeeds = uniform3fFeedsNamed(harness, 'uSunDirection');
    const sunColorFeeds = uniform3fFeedsNamed(harness, 'uSunColor');
    expect(sunDirectionFeeds.length).toBeGreaterThan(0);
    expect(sunColorFeeds.length).toBeGreaterThan(0);
    expect(sunDirectionFeeds[0].channels).toEqual(NEATRACING_SUN_DIRECTION);
    expect(sunColorFeeds[0].channels).toEqual(NEATRACING_SUN_COLOR);
  });

  it('feeds the camera world position and fog density during prop draw', () => {
    const harness = createRightWorldMockGlHarness();
    const pass = createRightWorldPass(harness.gl);
    const state = createRightWorldDrawState();
    pass.draw(state);
    const cameraFeeds = uniform3fFeedsNamed(harness, 'uCameraWorld');
    const fogDensityFeeds = uniform1fFeedsNamed(harness, 'uFogDensity');
    expect(cameraFeeds.length).toBeGreaterThan(0);
    expect(cameraFeeds[0].channels).toEqual(state.cameraWorld);
    expect(fogDensityFeeds.length).toBeGreaterThan(0);
  });
});

describe('bleacher crowd hosting and gaze (fix-packet-P1-bleacher-crowd-iteration-1, OBS-B + OBS-C)', () => {
  it('hosts at least one public spectator on a real bleacher stand cell', () => {
    const scene = buildRightWorldSceneData();
    const hosted = getBleacherHostedPublicBillboards(scene);

    expect(hosted.length).toBeGreaterThan(0);
  });

  it('keeps the majority of bleacher-hosted spectators in look-at-car mode', () => {
    const scene = buildRightWorldSceneData();
    const hosted = getBleacherHostedPublicBillboards(scene);

    expect(hosted.length).toBeGreaterThan(0);

    const lookingAtCar = hosted.filter(
      (billboard) => billboard.runtimeRole === 'seated',
    );
    const ratio = lookingAtCar.length / hosted.length;

    expect(ratio).toBeGreaterThan(0.5);
  });
});

describe('right-world.pass public-only atlas and seed-invariant scene (AC-070 retained clauses)', () => {
  it('packs a public-only sprite atlas of exactly 144 frames', () => {
    // AC-070: the atlas is PUBLIC-ONLY - the mechanic atlas must no longer
    // be merged into it. 144 frames = 6 accents x 8 directions x 3 poses.
    const scene = buildRightWorldSceneData();
    const frameIds = scene.atlas.frames.map((frame) => String(frame.id));

    expect(scene.atlas.source).toBe('neatenstein-public');
    expect(scene.atlas.frames.length).toBe(144);
    expect(frameIds.filter((id) => id.includes('mechanic'))).toEqual([]);
  });

  it('rebuilds an identical scene when the same seed is built twice', () => {
    // F3 (determinism reviewer): the crowd-host rule has to be STABLE for
    // a fixed seed. Two builds of the same input must yield the same
    // props, the same hosting bleacher and the same billboard anchors -
    // the reserved bleacher slot, the host selection and the seat anchors
    // must not drift between two builds of one seed. The sweep runs over
    // BOTH size buckets, because the reserved slot, the ring slot counts,
    // and therefore the host selection all depend on the track geometry
    // the bucket selects.
    const driftedSeeds: string[] = [];

    for (const sizeBucket of ['small', 'large'] as const) {
      for (const seed of NEATRACING_CROWD_FIXTURE_SEEDS) {
        const first = buildRightWorldSceneData({
          seed,
          layoutVersion: NEATRACING_CROWD_BUILD_INPUT.layoutVersion,
          sizeBucket,
        });
        const second = buildRightWorldSceneData({
          seed,
          layoutVersion: NEATRACING_CROWD_BUILD_INPUT.layoutVersion,
          sizeBucket,
        });

        if (summarizeScene(first) !== summarizeScene(second)) {
          driftedSeeds.push(`seed=${seed} bucket=${sizeBucket}`);
        }
      }
    }

    expect(driftedSeeds).toEqual([]);
  });
});

describe('right-world.pass re-populated crowd (P1-crowd2-red-contracts, RED-C, AC-073 + AC-074)', () => {
  it('populates every bleacher of the scene with seated billboards at every fixture seed', () => {
    // RED-C (AC-073): the retired single-host crowd filter left ONE
    // bleacher populated and every other bleacher empty. The re-populated
    // world keeps EVERY bleacher, so each one of them has to host at least
    // one SEATED billboard at every fixture seed.
    const violations: string[] = [];
    const bleacherCounts: number[] = [];

    for (const seed of NEATRACING_CROWD_FIXTURE_SEEDS) {
      const scene = buildFixtureScene(seed);
      const bleachers = scene.props.filter((prop) => prop.kind === 'bleacher');
      bleacherCounts.push(bleachers.length);

      if (bleachers.length === 0) {
        violations.push(`seed=${seed} the scene carries no bleacher at all`);
        continue;
      }

      for (const bleacher of bleachers) {
        const hosted = scene.billboards.filter(
          (billboard) =>
            runtimeRoleOf(billboard) === CROWD_ROLE_SEATED &&
            findBleacherHostProp(billboard, scene.props) === bleacher,
        );

        if (hosted.length === 0) {
          violations.push(
            `seed=${seed} bleacherProp=${scene.props.indexOf(bleacher)} ` +
              'hosts no seated billboard',
          );
        }
      }
    }

    // More than one bleacher per scene, so "every bleacher" is a real
    // constraint and not a vacuous one.
    expect(Math.min(...bleacherCounts)).toBeGreaterThan(1);
    expect(violations).toEqual([]);
  });

  it('seats exactly twelve spectators on every bleacher and four on every level at every fixture seed', () => {
    // MAJOR-1 (AC-072): the seated count is an EXACT contract, not a lower
    // bound. Every bleacher of every fixture seed carries exactly twelve
    // seated spectators - four on each of its three levels - and the
    // count is measured on the ASSEMBLED scene, so the personal-space
    // filter is part of the measurement. A filter that thins a bleacher
    // shows up as a violation instead of a silently lower bound, the
    // violation names the seed, the bleacher and the level, and the
    // expected counts are constants that are never lowered to match a
    // drifting scene.
    const violations: string[] = [];

    for (const seed of NEATRACING_CROWD_FIXTURE_SEEDS) {
      const scene = buildFixtureScene(seed);
      const bleachers = scene.props.filter((prop) => prop.kind === 'bleacher');

      for (const bleacher of bleachers) {
        const bleacherIndex = scene.props.indexOf(bleacher);
        const hosted = scene.billboards.filter(
          (billboard) =>
            runtimeRoleOf(billboard) === CROWD_ROLE_SEATED &&
            findBleacherHostProp(billboard, scene.props) === bleacher,
        );

        if (hosted.length !== REPOPULATED_SEATED_BILLBOARDS_PER_BLEACHER) {
          violations.push(
            `seed=${seed} bleacherProp=${bleacherIndex} seats ` +
              `${hosted.length} spectators, expected ` +
              `${REPOPULATED_SEATED_BILLBOARDS_PER_BLEACHER}`,
          );
          continue;
        }

        // The seat anchors are LEVEL-MAJOR - four anchors per level, the
        // lowest platform first - so the level of a seated billboard is
        // read from the real seat geometry instead of a guessed height
        // window.
        const anchors = getBleacherSeatAnchors(bleacher);
        const levelCounts = new Map<number, number>();

        for (const billboard of hosted) {
          const [x, , z] = billboard.position;
          let nearestAnchor = -1;
          let nearestDistance = Number.POSITIVE_INFINITY;

          for (let index = 0; index < anchors.length; index += 1) {
            const anchor = anchors[index]!;
            const distance = Math.hypot(x - anchor[0], z - anchor[2]);

            if (distance < nearestDistance) {
              nearestDistance = distance;
              nearestAnchor = index;
            }
          }

          const level = Math.floor(
            nearestAnchor / REPOPULATED_SEATED_BILLBOARDS_PER_LEVEL,
          );
          levelCounts.set(level, (levelCounts.get(level) ?? 0) + 1);
        }

        for (
          let level = 0;
          level < NEATRACING_BLEACHER_LEVEL_COUNT;
          level += 1
        ) {
          const levelCount = levelCounts.get(level) ?? 0;

          if (levelCount !== REPOPULATED_SEATED_BILLBOARDS_PER_LEVEL) {
            violations.push(
              `seed=${seed} bleacherProp=${bleacherIndex} level=${level} seats ` +
                `${levelCount} spectators, expected ` +
                `${REPOPULATED_SEATED_BILLBOARDS_PER_LEVEL}`,
            );
          }
        }
      }
    }

    expect(violations).toEqual([]);
  });

  it('keeps the off-bleacher crowd inside its allowance and every role inside the runtime role set', () => {
    // The off-bleacher crowd is a CAP, never a quota: the allowance is the
    // ceiling of the walkers, standing spectators and mechanics, and the
    // clearance search SKIPS an agent that has no clear spot. The contract
    // therefore pins the ceiling and the role set, not an exact
    // off-bleacher count.
    const violations: string[] = [];

    for (const seed of NEATRACING_CROWD_FIXTURE_SEEDS) {
      const scene = buildFixtureScene(seed);
      const offBleacher = getOffBleacherBillboards(scene);

      if (offBleacher.length > NEATRACING_PUBLIC_OFF_BLEACHER_AGENT_ALLOWANCE) {
        violations.push(
          `seed=${seed} carries ${offBleacher.length} off-bleacher agents, ` +
            `above the allowance of ` +
            `${NEATRACING_PUBLIC_OFF_BLEACHER_AGENT_ALLOWANCE}`,
        );
      }

      for (const billboard of scene.billboards) {
        const role = runtimeRoleOf(billboard);

        if (!PUBLIC_RUNTIME_ROLES.includes(role)) {
          violations.push(`seed=${seed} carries the unknown role ${role}`);
        }
      }
    }

    expect(violations).toEqual([]);
  });

  it('carries at least one walker and at least one standing off-bleacher billboard at every fixture seed', () => {
    // RED-C (AC-074): the off-bleacher agents are back - the WALKERS on the
    // seeded slow walk cycle and the STANDING spectators outside the ring.
    const violations: string[] = [];

    for (const seed of NEATRACING_CROWD_FIXTURE_SEEDS) {
      const scene = buildFixtureScene(seed);
      const walkers = scene.billboards.filter(
        (billboard) => runtimeRoleOf(billboard) === CROWD_ROLE_WALKER,
      );
      const standingOffBleacher = scene.billboards.filter(
        (billboard) =>
          runtimeRoleOf(billboard) === CROWD_ROLE_STANDING &&
          findBleacherHostProp(billboard, scene.props) === undefined,
      );

      if (walkers.length === 0) {
        violations.push(`seed=${seed} the scene carries no walker billboard`);
      }
      if (standingOffBleacher.length === 0) {
        violations.push(
          `seed=${seed} the scene carries no standing off-bleacher billboard`,
        );
      }
    }

    expect(violations).toEqual([]);
  });

  it('carries the mechanic, crew and team-banner prop kinds on the inner ring', () => {
    // RED-C (AC-074): the mechanic pit props are RE-AUTHORED on the INNER
    // ring - closer to the road than the bleachers - so the assembled scene
    // carries the three people kinds and every one of them sits on the
    // inner ring.
    const violations: string[] = [];

    for (const seed of NEATRACING_CROWD_FIXTURE_SEEDS) {
      const scene = buildFixtureScene(seed);

      for (const kind of REQUIRED_PEOPLE_PROP_KINDS) {
        const ofKind = scene.props.filter((prop) => prop.kind === kind);

        if (ofKind.length === 0) {
          violations.push(`seed=${seed} the scene carries no ${kind} prop`);
          continue;
        }

        for (const prop of ofKind) {
          if (prop.ringSide !== 'inner') {
            violations.push(
              `seed=${seed} the ${kind} prop sits on the ` +
                `${prop.ringSide} ring`,
            );
          }
        }
      }
    }

    expect(violations).toEqual([]);
  });
});

describe('right-world.pass off-bleacher agents (P1-crowd2-red-contracts, RED-D, AC-074)', () => {
  it('paints the inner-ring mechanics with a team accent', () => {
    // RED-D (AC-074): the mechanics keep TEAM-color accents instead of the
    // colour-diverse spectator rule. The accent of the frame a mechanic
    // actually renders has to belong to the family of one of the four
    // NAMED team colours.
    const violations: string[] = [];
    let mechanicCount = 0;

    for (const seed of NEATRACING_CROWD_FIXTURE_SEEDS) {
      const scene = buildFixtureScene(seed);
      const table = buildRuntimeIndexTable(scene);
      const mechanics = scene.billboards.filter(
        (billboard) => runtimeRoleOf(billboard) === CROWD_ROLE_MECHANIC,
      );
      mechanicCount += mechanics.length;

      for (const mechanic of mechanics) {
        const frameIndex = computePublicRuntimeFrameIndex(
          mechanic,
          CROWD_CAMERA_WORLD,
          CROWD_CAR_POSITION,
          CROWD_RUNTIME_TIME_SECONDS,
          table,
          CROWD_CAMERA_FORWARD,
        );

        if (frameIndex < 0) {
          violations.push(`seed=${seed} a mechanic resolves no atlas frame`);
          continue;
        }

        const frame = scene.atlas.frames[frameIndex];
        if (!frame) {
          violations.push(`seed=${seed} a mechanic resolves a missing frame`);
          continue;
        }

        const hue = dominantAccentHue(frame.rgba);
        if (hue === null) {
          violations.push(
            `seed=${seed} a mechanic frame carries no accent hue`,
          );
          continue;
        }

        if (!isTeamAccentHue(hue)) {
          violations.push(
            `seed=${seed} a mechanic renders accent hue ${hue}, ` +
              'which is no team accent',
          );
        }
      }
    }

    expect(mechanicCount).toBeGreaterThan(0);
    expect(violations).toEqual([]);
  });

  it('renders the inner-ring mechanics on the look-at-car stand frame', () => {
    // RED-D (AC-074): a mechanic always looks at the car, so its frame is
    // the STAND pose of the camera-relative look-at-car direction. The
    // reference billboard resolves that direction for the very same spot,
    // so the contract compares frames and never duplicates the bearing
    // math.
    const violations: string[] = [];
    let mechanicCount = 0;

    for (const seed of NEATRACING_CROWD_FIXTURE_SEEDS) {
      const scene = buildFixtureScene(seed);
      const table = buildRuntimeIndexTable(scene);
      const mechanics = scene.billboards.filter(
        (billboard) => runtimeRoleOf(billboard) === CROWD_ROLE_MECHANIC,
      );
      mechanicCount += mechanics.length;

      for (const mechanic of mechanics) {
        const frameIndex = computePublicRuntimeFrameIndex(
          mechanic,
          CROWD_CAMERA_WORLD,
          CROWD_CAR_POSITION,
          CROWD_RUNTIME_TIME_SECONDS,
          table,
          CROWD_CAMERA_FORWARD,
        );

        if (frameIndex < 0) {
          violations.push(`seed=${seed} a mechanic resolves no atlas frame`);
          continue;
        }

        const pose = poseIndexOfRuntimeFrameIndex(frameIndex);
        if (pose !== PUBLIC_POSE_INDEX_STAND) {
          violations.push(
            `seed=${seed} a mechanic renders pose ${pose} ` +
              `instead of the stand pose ${PUBLIC_POSE_INDEX_STAND}`,
          );
          continue;
        }

        const reference = seatedReferenceBillboard(mechanic.position);
        const referenceIndex = computePublicRuntimeFrameIndex(
          reference,
          CROWD_CAMERA_WORLD,
          CROWD_CAR_POSITION,
          CROWD_RUNTIME_TIME_SECONDS,
          table,
          CROWD_CAMERA_FORWARD,
        );

        if (referenceIndex < 0) {
          violations.push(
            `seed=${seed} the look-at-car reference resolves no atlas frame`,
          );
          continue;
        }

        const mechanicDirection = directionIndexOfRuntimeFrameIndex(frameIndex);
        const referenceDirection =
          directionIndexOfRuntimeFrameIndex(referenceIndex);

        if (mechanicDirection !== referenceDirection) {
          violations.push(
            `seed=${seed} a mechanic faces direction ${mechanicDirection}, ` +
              `the look-at-car direction is ${referenceDirection}`,
          );
        }
      }
    }

    expect(mechanicCount).toBeGreaterThan(0);
    expect(violations).toEqual([]);
  });

  it('gives every walker billboard a slow walk-cycle frame', () => {
    // RED-D (AC-074): the walkers walk again, on the seeded SLOW walk
    // cycle. The build-time frame id is a walk frame, the resolved pose is
    // a walk pose, and the pose does not flip inside a one-second window,
    // which is what makes the cycle slow.
    const violations: string[] = [];
    let walkerCount = 0;

    for (const seed of NEATRACING_CROWD_FIXTURE_SEEDS) {
      const scene = buildFixtureScene(seed);
      const table = buildRuntimeIndexTable(scene);
      const walkers = scene.billboards.filter(
        (billboard) => runtimeRoleOf(billboard) === CROWD_ROLE_WALKER,
      );
      walkerCount += walkers.length;

      for (const walker of walkers) {
        if (!/-walk[12]$/.test(walker.frameId)) {
          violations.push(
            `seed=${seed} a walker carries the frame id ` +
              `"${walker.frameId}" instead of a walk frame`,
          );
        }

        const frameIndex = computePublicRuntimeFrameIndex(
          walker,
          CROWD_CAMERA_WORLD,
          CROWD_CAR_POSITION,
          CROWD_RUNTIME_TIME_SECONDS,
          table,
          CROWD_CAMERA_FORWARD,
        );

        if (frameIndex < 0) {
          violations.push(`seed=${seed} a walker resolves no atlas frame`);
          continue;
        }

        const pose = poseIndexOfRuntimeFrameIndex(frameIndex);
        if (
          pose !== PUBLIC_POSE_INDEX_WALK1 &&
          pose !== PUBLIC_POSE_INDEX_WALK2
        ) {
          violations.push(
            `seed=${seed} a walker renders pose ${pose} ` +
              'instead of a walk pose',
          );
          continue;
        }

        const laterFrameIndex = computePublicRuntimeFrameIndex(
          walker,
          CROWD_CAMERA_WORLD,
          CROWD_CAR_POSITION,
          CROWD_RUNTIME_TIME_LATER_SECONDS,
          table,
          CROWD_CAMERA_FORWARD,
        );
        const laterPose = poseIndexOfRuntimeFrameIndex(laterFrameIndex);

        if (laterPose !== pose) {
          violations.push(
            `seed=${seed} a walker flips its walk pose inside one second ` +
              `(${pose} -> ${laterPose})`,
          );
        }
      }
    }

    expect(walkerCount).toBeGreaterThan(0);
    expect(violations).toEqual([]);
  });

  it('keeps the off-bleacher agents at 1 m personal space', () => {
    // RED-D (AC-074): the off-bleacher agents - the walkers, the standing
    // spectators, and the inner-ring mechanics - keep the 1 m personal
    // space of the re-populated world.
    const violations: string[] = [];
    let offBleacherCount = 0;

    for (const seed of NEATRACING_CROWD_FIXTURE_SEEDS) {
      const scene = buildFixtureScene(seed);
      const offBleacher = getOffBleacherBillboards(scene);
      offBleacherCount += offBleacher.length;

      if (offBleacher.length < 2) {
        violations.push(
          `seed=${seed} the scene carries ${offBleacher.length} ` +
            'off-bleacher agents, so no personal space is measured',
        );
        continue;
      }

      const minDistance = minRobotXZDistance(
        offBleacher.map((billboard) => billboard.position),
      );

      if (
        minDistance <
        NEATRACING_ROBOT_PERSONAL_SPACE_M -
          NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M
      ) {
        violations.push(
          `seed=${seed} two off-bleacher agents stand ` +
            `${minDistance.toFixed(4)} m apart`,
        );
      }
    }

    expect(offBleacherCount).toBeGreaterThan(1);
    expect(violations).toEqual([]);
  });
});

describe('right-world.pass device-sourced atlas texture bound (AC-071, RED-D)', () => {
  it('packs the sprite atlas into the device-reported GPU max texture size', () => {
    // AC-071 (d1): the atlas bound must come from the DEVICE, never from a
    // hard-coded 16384. The mocked limit below is 2400 px, which is TIGHTER
    // than 16384 and is only satisfiable by a public-only atlas:
    //   - before the AC-070 retirement (192 merged frames): 12 columns x
    //     16 rows of 192 px = 2304 x 3072 px, so the 3072 px height blows
    //     the 2400 px limit and the pass throws instead of packing;
    //   - after the AC-070 retirement (144 public frames): 12 x 12 cells of
    //     192 px = 2304 x 2304 px, which fits into 2400 px.
    // The limit is read back from the mock's getParameter so the assertion
    // can never drift into a hard-coded constant.
    const harness = createRightWorldMockGlHarness({ maxTextureSize: 2400 });
    const deviceLimit = harness.gl.getParameter(
      harness.gl.MAX_TEXTURE_SIZE,
    ) as number;
    let thrown: unknown = null;

    try {
      createRightWorldPass(harness.gl);
    } catch (error) {
      thrown = error;
    }

    const atlasCalls = harness.texImage2DCalls.filter(
      (call) => call.format === 0x1908 && call.type === 0x1401,
    );

    expect(thrown).toBeNull();
    expect(atlasCalls.length).toBe(1);
    // The larger of the two recorded dimensions is what has to respect the
    // device bound, so one comparison covers width AND height.
    expect(
      Math.max(atlasCalls[0]!.width, atlasCalls[0]!.height),
    ).toBeLessThanOrEqual(deviceLimit);
  });

  it('keeps the packed atlas within a device limit that is only just wide enough', () => {
    // AC-071 (d1): the bound must be honoured at the TIGHTEST limit that
    // can still hold the public-only atlas. 2304 px is exactly the 12 x 12
    // grid of 192 px cells the 144-frame public atlas needs, so a packer
    // that reads the device limit must produce a 2304 x 2304 texture and
    // still succeed. A packer that ignores the queried limit (or that only
    // bounds the column count and lets the HEIGHT overflow) fails here.
    const harness = createRightWorldMockGlHarness({ maxTextureSize: 2304 });
    const deviceLimit = harness.gl.getParameter(
      harness.gl.MAX_TEXTURE_SIZE,
    ) as number;
    let thrown: unknown = null;

    try {
      createRightWorldPass(harness.gl);
    } catch (error) {
      thrown = error;
    }

    const atlasCalls = harness.texImage2DCalls.filter(
      (call) => call.format === 0x1908 && call.type === 0x1401,
    );

    expect(thrown).toBeNull();
    expect(atlasCalls.length).toBe(1);
    expect(
      Math.max(atlasCalls[0]!.width, atlasCalls[0]!.height),
    ).toBeLessThanOrEqual(deviceLimit);
  });

  it('raises the documented atlas error instead of uploading when the device limit is too small', () => {
    // AC-071 (d2): the over-limit path must RAISE the documented atlas
    // error instead of uploading an oversized texture. The documented error
    // names the frame count of the atlas that is actually packed, so it can
    // only report the PUBLIC-ONLY frame count once the mechanic frames are
    // retired (AC-070). 1536 px is far below the 2304 px the public-only
    // atlas needs, so the throw is guaranteed.
    const harness = createRightWorldMockGlHarness({ maxTextureSize: 1536 });
    let thrown: unknown = null;

    try {
      createRightWorldPass(harness.gl);
    } catch (error) {
      thrown = error;
    }

    const message = thrown instanceof Error ? thrown.message : '';

    expect(thrown).toBeInstanceOf(Error);
    expect(message).toMatch(
      /cannot pack 144 sprite frames \(\d+x\d+\) within GPU max texture size 1536/,
    );
    // The atlas texture upload is the only texImage2D call of the pass, so
    // an over-limit pack must fail BEFORE any texture reaches the GPU.
    expect(harness.texImage2DCalls.length).toBe(0);
  });
});

describe('right-world.pass public-only atlas texture bound (AC-075, RED-E)', () => {
  it('keeps the recorded atlas WIDTH within the device-reported GPU max texture size', () => {
    // RED-E (AC-075): the atlas that now serves the seated spectators, the
    // mechanics, the walkers and the standing spectators must never exceed
    // the device bound in WIDTH. The limit is read back from the mock's own
    // getParameter, so the assertion can never drift into a hard-coded
    // constant. 2304 px is the tightest limit that still holds the
    // public-only 12 x 12 pack of 192 px cells.
    const probe = probeRightWorldAtlas(PUBLIC_ONLY_ATLAS_SIDE_PX);
    const deviceLimit = probe.harness.gl.getParameter(
      probe.harness.gl.MAX_TEXTURE_SIZE,
    ) as number;

    expect(probe.thrown).toBeNull();
    expect(probe.atlasCalls.length).toBe(1);
    expect(probe.atlasCalls[0]!.width).toBeLessThanOrEqual(deviceLimit);
  });

  it('keeps the recorded atlas HEIGHT within the device-reported GPU max texture size', () => {
    // RED-E (AC-075): the HEIGHT is a SEPARATE bound. A packer that only
    // bounds the column count and lets the row count overflow the device
    // limit would keep the width inside the bound and blow the height, so
    // the two dimensions are asserted independently.
    const probe = probeRightWorldAtlas(PUBLIC_ONLY_ATLAS_SIDE_PX);
    const deviceLimit = probe.harness.gl.getParameter(
      probe.harness.gl.MAX_TEXTURE_SIZE,
    ) as number;

    expect(probe.thrown).toBeNull();
    expect(probe.atlasCalls.length).toBe(1);
    expect(probe.atlasCalls[0]!.height).toBeLessThanOrEqual(deviceLimit);
  });

  it('packs the public-only atlas into the 12 x 12 grid of 192 px cells', () => {
    // AC-075: the atlas stays the PUBLIC-ONLY 144-frame atlas (6 accents x
    // 8 directions x 3 poses), so the pack is a 12 x 12 grid of 192 px
    // cells = 2304 x 2304 px. The off-bleacher roles REUSE those frames
    // instead of adding a mechanic frame block, so the pack geometry must
    // not grow with the re-populated crowd.
    const probe = probeRightWorldAtlas(16384);

    expect(probe.atlasCalls.length).toBe(1);
    expect(probe.atlasCalls[0]!.width).toBe(PUBLIC_ONLY_ATLAS_SIDE_PX);
    expect(probe.atlasCalls[0]!.height).toBe(PUBLIC_ONLY_ATLAS_SIDE_PX);
  });

  it('raises the documented atlas error when only the atlas HEIGHT exceeds the device limit', () => {
    // RED-E (AC-075): a 192 px limit forces the packer into a single column
    // of 144 cells, so the packed WIDTH (192 px) still fits the device
    // bound while the packed HEIGHT (144 x 192 = 27648 px) does not. The
    // over-limit path must therefore RAISE the documented atlas error
    // instead of uploading a texture whose height overflows the bound.
    const probe = probeRightWorldAtlas(PUBLIC_ONLY_ATLAS_CELL_PX);
    const message = probe.thrown instanceof Error ? probe.thrown.message : '';
    // The documented error names the PUBLIC-ONLY frame count of the atlas
    // that is actually packed, and the device limit that rejected it.
    const documentedError = new RegExp(
      `cannot pack ${PUBLIC_ONLY_ATLAS_FRAME_COUNT} sprite frames ` +
        `\\(\\d+x\\d+\\) within GPU max texture size ${PUBLIC_ONLY_ATLAS_CELL_PX}`,
    );

    expect(probe.thrown).toBeInstanceOf(Error);
    expect(message).toMatch(documentedError);
    // The atlas texture upload is the only texImage2D call of the pass, so
    // an over-limit pack must fail BEFORE any texture reaches the GPU.
    expect(probe.harness.texImage2DCalls.length).toBe(0);
  });

  it('keeps the recorded atlas WIDTH and HEIGHT within the device bound for the re-populated crowd', () => {
    // RED-E (AC-075): the atlas bound is asserted FOR THE RE-POPULATED
    // WORLD, not for the retired single-host crowd. The world under the
    // bound is the scene the pass itself packs (the default right-world
    // build tuple), and that world has to BE the re-populated one: 20
    // bleachers and at least 240 seated billboards (20 x 12).
    //
    // The device limit is the TIGHTEST one that still holds the public-only
    // 12 x 12 pack of 192 px cells (2304 px), and it is read back from the
    // mock's own getParameter, so the assertion can never drift into a
    // hard-coded constant.
    const harness = createRightWorldMockGlHarness({
      maxTextureSize: PUBLIC_ONLY_ATLAS_SIDE_PX,
    });
    const deviceLimit = harness.gl.getParameter(
      harness.gl.MAX_TEXTURE_SIZE,
    ) as number;
    let thrown: unknown = null;

    try {
      createRightWorldPass(harness.gl);
    } catch (error) {
      thrown = error;
    }

    const atlasCalls = harness.texImage2DCalls.filter(
      (call) => call.format === 0x1908 && call.type === 0x1401,
    );
    // The pass packs the default right-world scene, so the world under the
    // atlas bound is that very scene.
    const scene = buildRightWorldSceneData();
    const bleacherCount = scene.props.filter(
      (prop) => prop.kind === 'bleacher',
    ).length;
    const seatedCount = scene.billboards.filter(
      (billboard) => runtimeRoleOf(billboard) === CROWD_ROLE_SEATED,
    ).length;

    const violations: string[] = [];

    if (thrown !== null) {
      violations.push(
        'the pass threw while packing the atlas of the re-populated world: ' +
          (thrown instanceof Error ? thrown.message : String(thrown)),
      );
    }
    if (atlasCalls.length !== 1) {
      violations.push(
        `the atlas texture was uploaded ${atlasCalls.length} times, expected 1`,
      );
    }
    if (bleacherCount !== REPOPULATED_BLEACHER_COUNT) {
      violations.push(
        `the world under the atlas bound carries ${bleacherCount} bleachers, ` +
          `expected ${REPOPULATED_BLEACHER_COUNT}`,
      );
    }
    if (seatedCount < REPOPULATED_SEATED_CROWD) {
      violations.push(
        `the world under the atlas bound carries ${seatedCount} seated ` +
          `billboards, expected at least ${REPOPULATED_SEATED_CROWD}`,
      );
    }

    const atlasWidth = atlasCalls[0]?.width ?? 0;
    const atlasHeight = atlasCalls[0]?.height ?? 0;

    if (atlasWidth > deviceLimit) {
      violations.push(
        `the recorded atlas WIDTH ${atlasWidth} px exceeds the device limit ` +
          `${deviceLimit} px`,
      );
    }
    if (atlasHeight > deviceLimit) {
      violations.push(
        `the recorded atlas HEIGHT ${atlasHeight} px exceeds the device limit ` +
          `${deviceLimit} px`,
      );
    }

    expect(violations).toEqual([]);
  });
});

describe('right-world.pass per-frame billboard capacity (AC-076, RED-F)', () => {
  it('keeps every culled-in seated billboard of the re-populated crowd in the per-frame selection', () => {
    // RED-F (AC-076): the per-frame billboard selection must keep EVERY
    // seated billboard of EVERY bleacher that is inside the decoration
    // cull distance, so the retired 128-billboard cap cannot silently
    // truncate the visible crowd. The comparison is the plan's named one:
    // the selected billboard count against the assembled seated-billboard
    // count of the culled-in bleachers.
    const harness = createRightWorldMockGlHarness();
    const pass = createRightWorldPass(harness.gl);
    const state = createRightWorldDrawState();
    pass.draw(state);

    const scene = buildRightWorldSceneData();
    const selectedCount = countSelectedBillboards(harness.drawCalls);
    const culledInSeatedCount = countCulledInSeatedBillboards(
      scene,
      state.cameraWorld,
    );

    expect(selectedCount).toBeGreaterThanOrEqual(culledInSeatedCount);
  });

  it('keeps every billboard inside the decoration cull distance in the per-frame selection', () => {
    // RED-F (AC-076): the truncation of the retired cap happened in SCENE
    // ORDER: the collection loop stopped at the budget, so the billboards
    // beyond the cap were dropped even though they were inside the cull
    // distance. The per-frame selection must therefore keep EVERY
    // billboard the cull distance admits.
    const harness = createRightWorldMockGlHarness();
    const pass = createRightWorldPass(harness.gl);
    const state = createRightWorldDrawState();
    pass.draw(state);

    const scene = buildRightWorldSceneData();
    const selectedCount = countSelectedBillboards(harness.drawCalls);
    const culledInCount = countCulledInBillboards(scene, state.cameraWorld);

    expect(selectedCount).toBe(culledInCount);
  });

  it('sizes the per-frame billboard upload buffer for the re-populated crowd', () => {
    // RED-F (AC-076): the per-frame billboard upload buffer is sized from
    // the per-frame billboard budget, so it must hold the re-populated
    // crowd of 20 bleachers x 12 seated billboards = 240 quads of 120 bytes.
    // BEFORE AC-076 the buffer was sized from the RETIRED shared cap
    // (128 quads = 15360 bytes), which is what truncated the visible
    // crowd.
    const harness = createRightWorldMockGlHarness();
    createRightWorldPass(harness.gl);

    const uploadBytes = billboardUploadBufferBytes(harness);

    expect(uploadBytes).toBeGreaterThanOrEqual(
      REPOPULATED_SEATED_CROWD * BILLBOARD_UPLOAD_BYTES,
    );
    expect(uploadBytes / BILLBOARD_UPLOAD_BYTES).toBeGreaterThanOrEqual(
      REPOPULATED_SEATED_CROWD,
    );
  });

  it('sizes the per-frame billboard budget independently of the retired shared prop LOD budget', () => {
    // RED-F (AC-076): the billboard budget becomes its own NAMED constant
    // sized for the re-populated world, while the prop LOD budget keeps
    // governing the voxel props. BEFORE AC-076 the per-frame upload buffer
    // held exactly the retired shared cap of 128 quads - the coupling
    // AC-076 removes.
    const harness = createRightWorldMockGlHarness();
    createRightWorldPass(harness.gl);

    const billboardCapacity =
      billboardUploadBufferBytes(harness) / BILLBOARD_UPLOAD_BYTES;

    expect(billboardCapacity).not.toBe(NEATRACING_DECORATION_LOD_BUDGET);
  });
});

/**
 * Walkers of the off-bleacher crowd of the re-populated world (AC-074).
 *
 * The off-bleacher agents are a CAP, not a quota, so the counts below are
 * the named target the halving must not change.
 */
const EXPECTED_OFF_BLEACHER_WALKERS = 20;

/** Standing spectators of the off-bleacher crowd (AC-074). */
const EXPECTED_OFF_BLEACHER_STANDING = 10;

/** Inner-ring mechanics of the re-populated world (AC-074). */
const EXPECTED_OFF_BLEACHER_MECHANICS = 4;

describe('right-world.pass rendered crowd billboard quad (P1-sprite-scale-red-contracts, RED-C, AC-077)', () => {
  it('renders every crowd billboard quad at the halved world height', () => {
    // RED-C (AC-077): the halving has to be proven on the RENDERED
    // geometry, not only on the module constant. The pass uploads one
    // camera-facing quad per crowd billboard, and the quad is as tall as
    // the frame is high: `frame.height * world scale * meters per pixel`.
    // BEFORE the halving the quad measured 9.6 m (the 1:1 world scale),
    // so a 4.8 m quad can only come from the halved world scale.
    const { scene, quads } = probeRenderedCrowdQuads();
    const frame = scene.atlas.frames[0]!;
    const expectedHeight =
      frame.height *
      EXPECTED_BILLBOARD_WORLD_SCALE *
      EXPECTED_METERS_PER_SPRITE_PIXEL;
    const violations: string[] = [];

    if (quads.length === 0) {
      violations.push('the frame uploaded no crowd billboard quad');
    }

    for (let quadIndex = 0; quadIndex < quads.length; quadIndex += 1) {
      const quad = quads[quadIndex]!;

      if (Math.abs(quad.height - expectedHeight) > RENDERED_QUAD_TOLERANCE_M) {
        violations.push(
          `quad=${quadIndex} is ${quad.height.toFixed(3)} m tall, ` +
            `expected ${expectedHeight.toFixed(3)} m`,
        );
      }
    }

    expect(violations).toEqual([]);
  });

  it('renders every crowd billboard quad at the halved half width', () => {
    // RED-C (AC-077): the world WIDTH follows the SAME factor, so the
    // square 192 x 192 px atlas frames keep their aspect ratio. The half
    // width of the rendered quad is `frame.width * world scale * meters per
    // pixel / 2`, which is 2.4 m after the halving and 4.8 m before it.
    const { scene, quads } = probeRenderedCrowdQuads();
    const frame = scene.atlas.frames[0]!;
    const expectedHalfWidth =
      (frame.width *
        EXPECTED_BILLBOARD_WORLD_SCALE *
        EXPECTED_METERS_PER_SPRITE_PIXEL) /
      2;
    const violations: string[] = [];

    if (quads.length === 0) {
      violations.push('the frame uploaded no crowd billboard quad');
    }

    for (let quadIndex = 0; quadIndex < quads.length; quadIndex += 1) {
      const quad = quads[quadIndex]!;

      if (
        Math.abs(quad.halfWidth - expectedHalfWidth) > RENDERED_QUAD_TOLERANCE_M
      ) {
        violations.push(
          `quad=${quadIndex} is ${quad.halfWidth.toFixed(3)} m wide, ` +
            `expected ${expectedHalfWidth.toFixed(3)} m`,
        );
      }
    }

    expect(violations).toEqual([]);
  });

  it('expands every crowd billboard quad upward from the billboard base Y', () => {
    // RED-C (AC-077): the halving changes the SIZE of the quad, never its
    // ANCHOR. The bottom vertices of the uploaded quad have to sit exactly
    // on the base Y of the billboard the quad draws, and the top vertices
    // have to be ABOVE it, so the sprite keeps growing UPWARD out of its
    // foot anchor instead of sinking into the surface.
    const { scene, quads } = probeRenderedCrowdQuads();
    const violations: string[] = [];

    if (quads.length === 0) {
      violations.push('the frame uploaded no crowd billboard quad');
    }

    for (let quadIndex = 0; quadIndex < quads.length; quadIndex += 1) {
      const quad = quads[quadIndex]!;
      let nearest = scene.billboards[0];
      let nearestDistance = Number.POSITIVE_INFINITY;

      for (const billboard of scene.billboards) {
        const distance = Math.hypot(
          billboard.position[0] - quad.centerX,
          billboard.position[2] - quad.centerZ,
        );

        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = billboard;
        }
      }

      if (
        nearest === undefined ||
        nearestDistance > RENDERED_QUAD_TOLERANCE_M
      ) {
        violations.push(
          `quad=${quadIndex} is centered at ${quad.centerX.toFixed(3)}, ` +
            `${quad.centerZ.toFixed(3)}, which is no billboard position`,
        );
        continue;
      }

      if (
        Math.abs(quad.baseY - nearest.position[1]) > RENDERED_QUAD_TOLERANCE_M
      ) {
        violations.push(
          `quad=${quadIndex} starts at ${quad.baseY.toFixed(3)} m, the ` +
            `billboard base is ${nearest.position[1].toFixed(3)} m`,
        );
      }
      if (quad.topY <= quad.baseY) {
        violations.push(
          `quad=${quadIndex} does not grow upward ` +
            `(base ${quad.baseY.toFixed(3)} m, top ${quad.topY.toFixed(3)} m)`,
        );
      }
    }

    expect(violations).toEqual([]);
  });
});

describe('right-world.pass billboard meter scale is single-sourced (P1-sprite-scale-render-quad, RED-G, AC-077)', () => {
  it('keeps no module-local copy of the meters-per-pixel constant', () => {
    // RED-G (AC-077): the duplicated twin is a DRIFT hazard. A second copy
    // of the meters-per-pixel value can silently diverge from the shared
    // export, and the placement footprint would then disagree with the
    // rendered quad. The pass must declare NO local constant of its own.
    const source = readRightWorldPassSource();

    expect(source).not.toMatch(LOCAL_METERS_PER_PIXEL_DECLARATION_PATTERN);
  });

  it('imports the shared meters-per-pixel constant from the public sprite module', () => {
    // RED-G (AC-077): the single source of truth is the EXPORT
    // `NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M` of `public.sprite.ts`, so
    // the pass has to import it from that module.
    const source = readRightWorldPassSource();
    const importMatch = PUBLIC_SPRITE_VALUE_IMPORT_PATTERN.exec(source);
    const importedNames = (importMatch?.[1] ?? '')
      .split(',')
      .map((name) => name.trim())
      .filter((name) => name.length > 0);

    expect(importedNames).toContain('NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M');
  });

  it('sizes the rendered quad from the shared meters-per-pixel export', () => {
    // RED-G (AC-077): the two checks above prove the pass READS the shared
    // export; this one proves it USES it. The rendered quad height is
    // measured against the SHARED constants, so a twin that drifted to
    // another value fails here even when the import stayed in place.
    const { scene, quads } = probeRenderedCrowdQuads();
    const frame = scene.atlas.frames[0]!;
    const expectedHeight =
      frame.height *
      NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE *
      NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M;
    const violations: string[] = [];

    if (quads.length === 0) {
      violations.push('the frame uploaded no crowd billboard quad');
    }

    for (let quadIndex = 0; quadIndex < quads.length; quadIndex += 1) {
      const quad = quads[quadIndex]!;

      if (Math.abs(quad.height - expectedHeight) > RENDERED_QUAD_TOLERANCE_M) {
        violations.push(
          `quad=${quadIndex} is ${quad.height.toFixed(3)} m tall, ` +
            `expected ${expectedHeight.toFixed(3)} m`,
        );
      }
    }

    expect(violations).toEqual([]);
  });
});

describe('right-world.pass crowd counts under the halved sprite (P1-sprite-scale-red-contracts, RED-D, AC-077 + AC-072 + AC-074)', () => {
  it('seats twelve spectators on every bleacher and 240 seated billboards in total at every fixture seed', () => {
    // RED-D (AC-072 + AC-077): the halving changes the GEOMETRY of the
    // crowd, not its SIZE. Every bleacher of every fixture seed still
    // seats exactly twelve spectators - four on each of its three levels -
    // and the assembled scene still carries the full 240 seated billboards
    // of the 20 populated bleachers.
    const violations: string[] = [];

    for (const seed of NEATRACING_CROWD_FIXTURE_SEEDS) {
      const scene = buildFixtureScene(seed);
      const bleachers = scene.props.filter((prop) => prop.kind === 'bleacher');
      let seatedTotal = 0;

      if (bleachers.length !== REPOPULATED_BLEACHER_COUNT) {
        violations.push(
          `seed=${seed} carries ${bleachers.length} bleachers, ` +
            `expected ${REPOPULATED_BLEACHER_COUNT}`,
        );
      }

      for (const bleacher of bleachers) {
        const hosted = scene.billboards.filter(
          (billboard) =>
            runtimeRoleOf(billboard) === CROWD_ROLE_SEATED &&
            findBleacherHostProp(billboard, scene.props) === bleacher,
        );
        seatedTotal += hosted.length;

        if (hosted.length !== REPOPULATED_SEATED_BILLBOARDS_PER_BLEACHER) {
          violations.push(
            `seed=${seed} bleacher=${scene.props.indexOf(bleacher)} seats ` +
              `${hosted.length} spectators, expected ` +
              `${REPOPULATED_SEATED_BILLBOARDS_PER_BLEACHER}`,
          );
        }
      }

      if (seatedTotal !== REPOPULATED_SEATED_CROWD) {
        violations.push(
          `seed=${seed} carries ${seatedTotal} seated billboards, ` +
            `expected ${REPOPULATED_SEATED_CROWD}`,
        );
      }
    }

    expect(violations).toEqual([]);
  });

  it('keeps 20 walkers, 10 standing spectators and 4 mechanics off the bleachers at every fixture seed', () => {
    // RED-D (AC-074 + AC-077): the off-bleacher crowd of the re-populated
    // world - the slow walkers on the trackside, the standing spectators
    // outside the ring and the team-colored inner-ring mechanics - keeps
    // its named counts. The halving shrinks the sprite footprint of every
    // agent, so a count that silently grew or shrank with the sprite size
    // would break the crowd budget.
    const violations: string[] = [];

    for (const seed of NEATRACING_CROWD_FIXTURE_SEEDS) {
      const scene = buildFixtureScene(seed);
      const offBleacher = getOffBleacherBillboards(scene);
      const walkers = offBleacher.filter(
        (billboard) => runtimeRoleOf(billboard) === CROWD_ROLE_WALKER,
      );
      const standing = offBleacher.filter(
        (billboard) => runtimeRoleOf(billboard) === CROWD_ROLE_STANDING,
      );
      const mechanics = offBleacher.filter(
        (billboard) => runtimeRoleOf(billboard) === CROWD_ROLE_MECHANIC,
      );

      if (walkers.length !== EXPECTED_OFF_BLEACHER_WALKERS) {
        violations.push(
          `seed=${seed} carries ${walkers.length} walkers, ` +
            `expected ${EXPECTED_OFF_BLEACHER_WALKERS}`,
        );
      }
      if (standing.length !== EXPECTED_OFF_BLEACHER_STANDING) {
        violations.push(
          `seed=${seed} carries ${standing.length} standing spectators, ` +
            `expected ${EXPECTED_OFF_BLEACHER_STANDING}`,
        );
      }
      if (mechanics.length !== EXPECTED_OFF_BLEACHER_MECHANICS) {
        violations.push(
          `seed=${seed} carries ${mechanics.length} mechanics, ` +
            `expected ${EXPECTED_OFF_BLEACHER_MECHANICS}`,
        );
      }
      if (
        offBleacher.length !== NEATRACING_PUBLIC_OFF_BLEACHER_AGENT_ALLOWANCE
      ) {
        violations.push(
          `seed=${seed} carries ${offBleacher.length} off-bleacher agents, ` +
            `expected ${NEATRACING_PUBLIC_OFF_BLEACHER_AGENT_ALLOWANCE}`,
        );
      }
    }

    expect(violations).toEqual([]);
  });

  it('keeps every seated billboard on its bleacher level at every fixture seed', () => {
    // RED-D (AC-077 + AC-072): the halving may not drop the seated
    // spectators to the ground. Every seated billboard still stands on the
    // seat surface of its hosting bleacher, so the base Y of the billboard
    // matches the bleacher level it sits on.
    const violations: string[] = [];

    for (const seed of NEATRACING_CROWD_FIXTURE_SEEDS) {
      const scene = buildFixtureScene(seed);
      const seated = scene.billboards.filter(
        (billboard) => runtimeRoleOf(billboard) === CROWD_ROLE_SEATED,
      );

      for (const billboard of seated) {
        const host = findBleacherHostProp(billboard, scene.props);

        if (host === undefined) {
          violations.push(
            `seed=${seed} seated billboard at ` +
              `${billboard.position.join(',')} has no bleacher host`,
          );
          continue;
        }

        const levelY = getBleacherSeatTopY(
          host,
          billboard.position[0],
          billboard.position[2],
        );

        if (
          Math.abs(billboard.position[1] - levelY) > SEATED_BASE_Y_TOLERANCE_M
        ) {
          violations.push(
            `seed=${seed} seated billboard at ` +
              `${billboard.position.join(',')} stands at ` +
              `${billboard.position[1].toFixed(3)} m, the bleacher level ` +
              `is ${levelY.toFixed(3)} m`,
          );
        }
      }
    }

    expect(violations).toEqual([]);
  });
});
