/**
 * Arcade drift model for the neatRacing example (Step 14N; slice
 * `P1-drift-feel-tune-model`; AC-047 / AC-079 / AC-087 / AC-089; Step 14S
 * slice `P1-momentum-law`; AC-091).
 *
 * THE MODEL IS THE D1 COMMITTED-DRIFT MODEL. The drift is a MOMENTUM-DERIVED,
 * CLAMPED state; the body chases it; the countersteer gets its own
 * authority applied AGAINST the slide. The stage order is FIXED and is
 * the contract (integration artifact section 4.3):
 *
 *   stage 0  the SOFTENED MIN_SPEED gate             (stepDriftCommitment)
 *   stage 1  the MOMENTUM commitment (DEMAND / EXIT) (stepDriftCommitment)
 *   stage 2  the grip budget factor                 (computeGripBudgetFactor)
 *   stage 3  the committed chase                    (computeNextLateralVelocity)
 *   stage 4  the countersteer authority             (computeCountersteerAuthority)
 *   stage 5  the snap-back at the zero crossing     (computeNextLateralVelocity)
 *   stage 6  the longitudinal speed scrub           (computeLongitudinalSpeedScrub)
 *   stage 7  the merged pose spring                 (stepDriftPoseSpring)
 *
 * SIGN CONVENTION (production-pinned): `lateralVelocity > 0` means the
 * velocity vector lies to the LEFT of the heading, so
 * `slipAngle = atan2(lateralVelocity, speed)` is NEGATIVE in a LEFT turn
 * (the nose leads, the velocity lags). The visible pose therefore
 * shares the sign of the committed slip (`+committedSlip`), and a LEFT
 * turn yaws the body LEFT.
 *
 * THE MOMENTUM LAW (Step 14S, AC-091): the drift is CAUSED by the MOMENTUM,
 * not COMMANDED by the wheel. The committed slide is the MOMENTUM EXCESS -
 * the lateral acceleration the commanded turn DEMANDS minus the lateral
 * acceleration the tires can HOLD at the current speed - scaled by the
 * declared depth budget. The SAME wheel command therefore slides DEEPER the
 * faster the car goes, the slide is EXACTLY 0 at a standstill, and the
 * demand never exceeds the declared depth budget. The OLD speed-proportional
 * ENTRY-RATE COMMAND (its named constant and its named function) is DELETED
 * in this change and REPLACED by the momentum law - there is NO deferred
 * cleanup.
 *
 * THE ZERO-RATE CLAUSE IS SCOPED TO ENGAGEMENT ON PURPOSE (FP-14S-01): a
 * ZERO demanded turn rate produces an EXACTLY 0 committed slide AT
 * ENGAGEMENT - an UNCOMMITTED latch - because the ENGAGE branch of
 * `stepDriftCommitment` needs a POSITIVE momentum demand, so the latch is
 * never spent on a zero slide. UNDER A LIVE LATCH a ZERO momentum demand
 * HOLDS the committed depth (the HOLD branch of `stepDriftCommitment`):
 * the slide never SNAPS to zero under a live latch, and only the RELEASE -
 * the COUNTERSTEER or the NEUTRAL wheel - unwinds it. The UNSCOPED form
 * ("the slide is EXACTLY 0 when no turn rate is demanded") is TRUE ONLY AT
 * ENGAGEMENT and is FALSE under a live latch, so it is NOT the contract.
 *
 * THE LATCH (`turnSign`) IS THE TURN DIRECTION (+1 = LEFT), NEVER
 * `sign(slip)`. It is a three-state machine:
 *
 *   - ENGAGE (uncommitted -> committed) needs ALL of: `|steer| >=
 *     STEER_ENGAGE_THRESHOLD`, `committed.slip` exactly 0, the
 *     RE-ARM TOKEN present and a POSITIVE momentum demand (a wheel
 *     command the momentum does NOT carry opens NO slide). The MIN_SPEED
 *     gate is SOFTENED (Step 14Q, AC-089): it is no longer a hard cliff
 *     on the COMMITMENT - the committed depth is the MOMENTUM DEMAND of
 *     the tick, so the slide grows with the weight at speed instead of
 *     freezing - while the VISIBLE slide (the lateral velocity) stays
 *     EXACTLY 0 there;
 *   - HOLD: while the latch is live the committed depth FOLLOWS the
 *     momentum demand; when the demand is EXACTLY 0 (the momentum no
 *     longer carries the commanded turn) the latch HOLDS the committed
 *     depth and only the RELEASE unwinds it, so a live slide never
 *     SNAPS to zero;
 *   - the RE-ARM TOKEN (`reArmed`) starts TRUE and is latched again
 *     when the wheel is seen NEUTRAL (`|steer| < STEER_NEUTRAL_THRESHOLD`)
 *     for at least one tick; a RELEASE clears it;
 *   - RELEASE (committed -> uncommitted) happens at the hard zero, and
 *     the slide UNWINDS when the driver releases the wheel: the
 *     COUNTERSTEER or the NEUTRAL wheel (`|steer| <
 *     STEER_NEUTRAL_THRESHOLD`) unwinds the committed slide, and the
 *     release CLEARS the token, so an opposite steer ALONE can never
 *     re-latch and no phantom mirrored drift can appear;
 *   - CONTRACTION: an opposite-sign steer applies the countersteer
 *     authority and NEVER flips `turnSign`.
 *
 * The model stays PURE: no `Math.random`, no side effects, no hidden
 * globals. The same `(state, input)` always yields the same output ON THE
 * SAME RUNTIME (`Math.tan`, `Math.exp` and `Math.atan2` are
 * implementation-approximated, so the identity is same-engine, not
 * cross-engine), and the serialized zero channel is byte identical to
 * `createZeroDriftChannelState()` (AC-047D).
 */

/**
 * The COMMITTED-SLIDE budget (rad). The 14N slide is a BUDGET of 0.45 rad
 * (25.8 deg), not a quarter turn (it was `Math.PI / 2`).
 *
 * SPLIT CONTRACT (the GAP-N4 D1.3 escalation decision): this constant bounds
 * the COMMITTED slide only - `stepDriftCommitment` hard-clamps `committed.slip`
 * to +/- this value. The PUBLISHED slip (the slip the service publishes to
 * the renderer, the browser fixture and the published-slip pins) is a
 * DIFFERENT observable and MAY overshoot this budget by up to
 * `NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD`. That overshoot is
 * INTRINSIC for every scrub coefficient c > 4/9: no coefficient that clears
 * the D1.3 speed-deficit floor keeps the published slip <= 0.45.
 *
 * This constant is NEVER re-pinned - the 0.45 slide budget SURVIVES on the
 * committed slide.
 */
export const NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD = 0.45;

/**
 * The PUBLISHED-SLIP observation tolerance (rad).
 *
 * This export documents the PUBLISHED-slip OBSERVATION tolerance - it is NOT
 * the slide budget. The slide budget is
 * `NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD`, and it still hard-clamps the
 * COMMITTED slide.
 *
 * MECHANISM (MEASURED, not inferred): the published slip is the committed
 * slide carried through the first-order chase (tau 45 ms). The chase LAGS
 * ABOVE a target that SHRINKS as the speed scrub bleeds speed, so the
 * published slip peaks slightly above the committed budget - the measured
 * `vLat / target` ratio (the target taken from the TICK-START speed) is
 * 1.002317 at the peak, plus the post-scrub denominator on top of it.
 *
 * CALIBRATION: the measured peak is 0.4511617369362305 rad at t42 with
 * `NEATRACING_DRIFT_SPEED_SCRUB_COEFF_PER_S` = 0.55. The calibration envelope
 * is EXHAUSTED at c ~= 0.61 (measured: the peak is 0.45182287 rad at
 * c = 0.61, so only 0.00018 rad of headroom is left under the 0.452 ceiling;
 * the exact crossing is at c ~= 0.626), so ANY coefficient change FORCES a
 * re-derivation of this tolerance.
 *
 * The 0.45 slide budget SURVIVES on the COMMITTED slide: the hard clamp in
 * `stepDriftCommitment` (drift.model.ts:479-483) stays and is NEVER
 * re-pinned.
 */
export const NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD = 0.002;

/**
 * The ON-ROAD PUBLISHED-SLIP CEILING (rad) - the DERIVED bound the
 * published slip is ASSERTED against in the ROAD regime.
 *
 * DERIVED, NEVER A LITERAL: the ceiling is the COMMITTED slide budget
 * (`NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD` = 0.45 rad) plus the chase-lag
 * overshoot allowance
 * (`NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD` = 0.002 rad), so a
 * re-pin of EITHER export moves the bound with it (0.452 rad today).
 *
 * IT IS AN ASSERTION BOUND, NEVER A CLIP: nothing in the service clamps
 * the published slip to this value. The published slip keeps its
 * physical value, and the bound is the single source the jest pins and
 * the browser fixture mirror assert against.
 *
 * THE REGIME (why the bound is road-scoped): the published slip is the
 * committed slide carried through the first-order chase, and the chase
 * LAGS a target that shrinks as the speed scrub bleeds speed, so the
 * published slip peaks slightly above the committed budget. On the road
 * the measured peak is 0.4511617369362305 rad - 0.00084 rad of headroom
 * under this ceiling. OFF the road the off-road drag drives the same lag
 * from a FALLING denominator, and the measured raw peak reaches
 * 0.4872167352892035 rad (the post-RC-1 off-road envelope measurement),
 * so the OFF-ROAD regime carries its own DECLARED named ceiling instead
 * of this one.
 *
 * @example
 * ```ts
 * const withinOnRoadBound =
 *   Math.abs(car.slipAngle) <= NEATRACING_DRIFT_PUBLISHED_SLIP_MAX_RAD;
 * ```
 */
export const NEATRACING_DRIFT_PUBLISHED_SLIP_MAX_RAD =
  NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD +
  NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD;

/**
 * The OFF-ROAD PUBLISHED-SLIP CEILING (rad) - the DECLARED named bound
 * the published slip is ASSERTED against in the OFF-ROAD regime (grass,
 * curb and the road/grass transitions).
 *
 * WHY A SECOND, LOOSER BOUND: the off-road drag drives the same
 * chase-lag overshoot from a FALLING denominator, so the off-road
 * published slip peaks well above the ON-ROAD ceiling. The published
 * slip keeps its PHYSICAL value - this is an ASSERTION bound, NEVER a
 * clip - so the off-road regime is asserted against its own named
 * ceiling instead of being silently clipped into the on-road one.
 *
 * DERIVED FROM A MEASUREMENT (the ratified deterministic rule: the
 * ceiling is the smallest 0.001-grid value at or above the measured raw
 * off-road peak plus a 0.005 rad margin): the W2a post-RC-1 off-road
 * envelope peak is 0.4872167352892035 rad (the jest lane, seed 42, the
 * top-speed start, a full-throttle full-steer drift input, swept over
 * the seven grass/curb/transition lanes), so 0.4872167352892035 + 0.005
 * = 0.4922167352892035 and the declared ceiling is 0.493 rad.
 *
 * THE MARGIN absorbs the chase-lag drift of a lane that is not the
 * measured one. A re-measurement that moves the raw peak by more than
 * the margin FORCES a re-declaration of this value - a RECORDED
 * decision, never a silent clip.
 *
 * @example
 * ```ts
 * const withinOffRoadBound =
 *   Math.abs(car.slipAngle) <= NEATRACING_DRIFT_PUBLISHED_SLIP_OFFROAD_MAX_RAD;
 * ```
 */
export const NEATRACING_DRIFT_PUBLISHED_SLIP_OFFROAD_MAX_RAD = 0.493;

/**
 * The slip (rad) at which the tire grip budget starts to be consumed. Below
 * it the tire keeps its full lateral grip (`factor = 1`).
 */
export const NEATRACING_DRIFT_GRIP_FALLOFF_THRESHOLD_RAD = 0.2;

/**
 * The rate at which the grip budget is consumed per radian of slip past the
 * threshold. The budget is a PROPORTIONAL law of the SLIP demand scaled by
 * the TRACTION SEVERITY (Step 14Q, AC-088): the deeper the slide AND the
 * faster the car, the less of the tire's grip budget is left for the chase.
 * The speed demand is ALSO charged by the SPEED SCRUB (stage 6) - the two
 * terms are independent (the tire budget vs the longitudinal loss).
 */
export const NEATRACING_DRIFT_GRIP_FALLOFF_RATE_PER_RAD = 2;

/**
 * The minimum longitudinal speed (m/s) at which the drift model's VISIBLE
 * slide starts: below it the lateral velocity is held at exactly 0
 * (AC-047D).
 *
 * Step 14Q (AC-089) SOFTENS the gate on the COMMITMENT: the committed
 * depth and the entry rate are PROPORTIONAL to the speed below the gate
 * (they are 0 at a standstill), so the slide ramps in instead of freezing
 * on a hard cliff. The gate still holds the VISIBLE slide at exactly 0.
 */
export const NEATRACING_DRIFT_MIN_SPEED_MPS = 15;

/**
 * The TRACTION TOP SPEED (m/s) the speed-proportional laws are normalized
 * to: the arcade top speed (89 m/s).
 *
 * The model is PURE: it does not import the service, so the reference speed
 * is DECLARED here and MUST stay equal to
 * `NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S`. Every speed-proportional law
 * in this file (the entry rate, the chase response rate, the severity and
 * the slide depth budget) is normalized by it, so a re-pin of the arcade top
 * speed FORCES a re-derivation of those laws - a RECORDED decision (a
 * ledger row), never a silent drift.
 */
export const NEATRACING_DRIFT_TRACTION_TOP_SPEED_MPS = 89;

/**
 * The speed (m/s) at which the tire reaches the FULL slide budget: the
 * committed depth is `MAX_SLIP_ANGLE_RAD * min(1, speed / SATURATION_SPEED)`.
 *
 * DERIVED (Step 14Q, AC-088): the slide depth is PROPORTIONAL to the speed
 * (a slow car slides less) and SATURATES at the declared budget, so the
 * committed slide is bounded by the AC-087 budget at every speed.
 *
 * THE VALUE IS MEASURED, NOT CARRIED: it is the knee that puts the settled
 * EQUILIBRIUM speed deficit inside the ratified 14Q band. The equilibrium is
 * a MEASURED value and is never a carried constant.
 *
 * THE MEASUREMENT (the pinned RED-7 lane: the road-only resolver, the full
 * throttle, the full LEFT steer, the arcade top speed as the entry speed,
 * dt 1/60): the deficit reads 32.91154453692649 km/h at 3000 ticks AND at
 * 3600 ticks (the two readings are BIT-IDENTICAL - the lane has settled),
 * 32.91154453591922 km/h at 2400 ticks, 32.91018739912641 km/h at 1200
 * ticks and the 600-tick TRANSIENT 32.556920153495916 km/h. The settled
 * value is 0.00054 km/h from the pinned 32.911 km/h equilibrium (inside the
 * 0.002 km/h band) and the 600-tick transient is 0.354 km/h outside it, so
 * the anti-transient guard holds. The knee is sensitive: the settled
 * deficit moves about 319 km/h per rad of settled depth, so the knee is
 * re-derived (never carried) if the depth law or the scrub changes.
 *
 * The measurement and the resulting pin are recorded in the plan's ledger.
 */
export const NEATRACING_DRIFT_SLIDE_SATURATION_SPEED_MPS = 79.86;

/**
 * The steering-axis magnitude that ENGAGES a drift (band 0.4-0.6).
 */
export const NEATRACING_DRIFT_STEER_ENGAGE_THRESHOLD = 0.5;

/**
 * The steering-axis magnitude below which the wheel counts as NEUTRAL and the
 * RE-ARM token is latched again (band 0.1-0.2).
 */
export const NEATRACING_DRIFT_STEER_NEUTRAL_THRESHOLD = 0.15;

/**
 * The RAW FULL-STEER DEMANDED TURN RATE at the TRACTION TOP SPEED (rad/s).
 *
 * Step 14S (AC-091) makes the demanded turn rate an INPUT of the drift law:
 * the drift is caused by the MOMENTUM the commanded turn demands, so the
 * commitment CONSUMES the demanded turn rate. The demanded turn rate is a
 * RATE (rad/s) and NOT the per-tick heading delta (rad/tick) the yaw law
 * resolves - the two are NOT interchangeable (a unit mismatch is a silent
 * factor-of-sixty error).
 *
 * THE VALUE IS THE YAW LAW'S OWN QUANTITY: the yaw law builds its demanded
 * rate from the steer-axis product `steer * NEATRACING_PHYSICS_STEERING_RATE
 * * speedPercent` (rad per tick once it is multiplied with `dt`), so the
 * SAME product WITHOUT `dt` is the demanded rate (rad/s).
 *
 * MIRRORED, NOT IMPORTED: the model is PURE (it must not import the
 * service), so the steering rate is DECLARED here and MUST stay equal to
 * `NEATRACING_PHYSICS_STEERING_RATE` in `physics.step.service.ts` - the
 * same mirrored-constant precedent as
 * `NEATRACING_DRIFT_TRACTION_TOP_SPEED_MPS`.
 *
 * THE VALUE IS MEASURED, NOT CARRIED: the Step 14S RED lane measured
 * exactly 2 rad/s for a full steer at the arcade top speed (the speed
 * percent is exactly 1 there). That MEASUREMENT is the RAW, FACTOR-FREE
 * steer-axis rate this constant names: the service softens its user rate
 * with the speed factor above the 10 m/s band, so the rate the SERVICE
 * resolves at the arcade top speed is 1.6 rad/s - the raw rate stays the
 * mirrored constant, and the drift model's ABSENT-rate fallback uses it.
 */
export const NEATRACING_DRIFT_FULL_STEER_TURN_RATE_RAD_PER_S = 2;

/**
 * The LATERAL ACCELERATION (m/s^2) a SLIDING tire can HOLD: about 0.2 g.
 *
 * Step 14S (AC-091): the momentum excess is the lateral acceleration the
 * demanded turn DEMANDS minus the lateral acceleration the tires can HOLD.
 * The hold is the SLIDING tire's lateral capacity - a FRACTION of the ~1 g
 * peak grip, because a tire that is sliding cannot build its peak lateral
 * force - and it does NOT scale with the speed: a tire holds about the same
 * lateral acceleration at every speed.
 *
 * CALIBRATED INSIDE THE MEASURED WINDOW: the Step 14S RED lane MEASURED the
 * feasible demand bounds and this hold is the calibration that lands the law
 * inside them (the value is never carried as a literal):
 *
 *   - the 89 m/s reference lane (the service-resolved full-steer 1.6 rad/s
 *     at the full grip) demands 0.4436797752808989 rad, UNDER the 0.45 rad
 *     depth budget;
 *   - the 56 m/s AC-047A lane (the service-resolved 1.1118759778125444
 *     rad/s) demands 0.3054164438299156 rad, ABOVE the ratified 0.2 rad
 *     floor;
 *   - the 25 m/s strict-deeper pair (the service-resolved 0.5404636609301664
 *     rad/s and its double 1.0809273218603328 rad/s) demands
 *     0.1200195737277077 rad and 0.13044554944837675 rad, both strictly
 *     INSIDE the 0.14087152516904583 rad depth budget of that lane
 *     (otherwise the depth clamp would MASK the strict-deeper relation);
 *   - the 10 m/s below-gate lane (the service-resolved
 *     0.2247191011235955 rad/s) demands 0.006198347107438023 rad - a REAL
 *     (non-zero) commitment, so the ratified AC-089 softened gate keeps
 *     holding below the MIN_SPEED gate;
 *   - the standstill lane and the zero-rate lane demand EXACTLY 0 rad.
 *
 * THE MEASURED COMMITTED SLIDES (the 180-tick model lanes): 0.44745539952559543
 * rad at the 89 m/s reference lane, 0.3087228648902781 rad at the 56 m/s
 * AC-047A lane, 0.1208114199849737 rad and 0.13084147257700976 rad at the
 * 25 m/s strict-deeper pair and 0.006198347107438023 rad at the 10 m/s
 * below-gate lane.
 */
export const NEATRACING_DRIFT_MOMENTUM_HOLD_ACCEL_MPS2 = 2;

/**
 * The rate (rad/s) at which the committed slip unwinds under a countersteer
 * (144 deg/s).
 *
 * RECORDED DEVIATION (Step 14N, tune-model): the MERGE constant set and the
 * integration artifact name 2.0 rad/s, but 2.0 is INFEASIBLE under the RED
 * contract - the pose then settles after 35 ticks, OUTSIDE the ratified
 * 30..34-tick band. The feasible band is 2.25..3.65 rad/s and this value
 * sits at its centre (33 ticks). The RED contract wins over the constant
 * set; the deviation is reported to the green slice.
 */
export const NEATRACING_DRIFT_EXIT_RATE_RAD_PER_S = 2.5;

/**
 * The time constant (s) of the committed chase's exponential response at the
 * TRACTION TOP SPEED.
 *
 * Step 14Q (AC-088) makes the response a RELAXATION-LENGTH law: the real
 * time constant is `TAU * (TRACTION_TOP_SPEED_MPS / speed)`, so the chase
 * covers a FIXED DISTANCE per second and a fast car reacts FASTER (a slower
 * car lags more). The declared constant stays the TOP-SPEED reference value.
 */
export const NEATRACING_DRIFT_SLIP_RESPONSE_TAU_S = 0.045;

/**
 * The countersteer authority (m/s^2) applied AGAINST the slide. It is an
 * ACCELERATION: it is multiplied by `dt` to yield a velocity delta (REC-4).
 */
export const NEATRACING_DRIFT_COUNTERSTEER_AUTHORITY_MPS2 = 70;

/**
 * The minimum committed slip (rad) at which the countersteer authority is
 * applied - below it there is no slide left to catch.
 */
export const NEATRACING_DRIFT_COUNTERSTEER_MIN_SLIP_RAD = 0.02;

/**
 * The minimum countersteer steering magnitude (band 0.85-0.95): the in-model
 * fallback for the producer-declared countersteer intent (D2.3).
 */
export const NEATRACING_DRIFT_COUNTERSTEER_MIN_STEER = 0.9;

/**
 * The number of ticks the NEUTRAL wheel HOLDS the committed slide before the
 * latch RELEASES it (Step 14Q, AC-089).
 *
 * The countersteer-hold constant this replaces was DEAD: it was exported and
 * read by nobody, and the ratified `DriftCommittedState {slip, turnSign,
 * reArmed}` has no home for a debounce counter. Step 14Q deletes it and
 * replaces it with a constant that HAS a reader: the NEUTRAL release
 * unwinds the committed slide from the full slide budget to the hard zero
 * over this window, so the release rate is
 * `(MAX_SLIP_ANGLE_RAD - SNAP_BACK_ZERO_SLIP_RAD) / (REARM_HOLD_TICKS * dt)`.
 *
 * The latch therefore RE-ARMS: the turn sign returns to 0, the slide unwinds
 * to the hard zero and the RE-ARM token becomes reachable again, instead of
 * freezing on the `Math.sign(0) === -turnSign` defect.
 */
export const NEATRACING_DRIFT_REARM_HOLD_TICKS = 8;

/**
 * The committed slip (rad) at or below which the slide is a HARD ZERO: the
 * lateral velocity is held at exactly 0 there.
 */
export const NEATRACING_DRIFT_SNAP_BACK_ZERO_SLIP_RAD = 0.02;

/**
 * The longitudinal speed-scrub coefficient (per rad of slip, per m/s of
 * speed, per unit of the speed percentile). It replaces the retired
 * proportional traction loss and it is zero on a straight line.
 *
 * DERIVATION (the GAP-N4 D1.3 escalation decision; MEASURED, not carried):
 * the engine (17.8 m/s2 = MAX_SPEED / 5) WINS while c <= 4/9 = 0.444444 -
 * that is the exact clamp-bound crossover, and the landed 0.25 coefficient sat
 * inside that regime, so the 0.25 lane was green only because the D1.3 pin
 * was VACUOUS. Above the crossover the scrub wins and the interior
 * equilibrium is s* = sqrt(3520.4444 / c).
 *
 * At c = 0.55 the terminal speed is 79.85792723069594 m/s (287.49 km/h) and
 * the D1.3 speed deficits are 32.55473 km/h @600 ticks and 32.91146 km/h
 * @1200 ticks - a 63 % margin over the NEVER-RELAXED
 * `NEATRACING_DRIFT_MIN_SPEED_DEFICIT_KMH` = 20 km/h floor, inside the
 * projected 28-46 km/h band. The scrub max is 22.0275 m/s2, which is above
 * the 17.8 m/s2 engine, so the scrub is the binding constraint (the engine
 * can NOT hold ~320 km/h through the turn).
 *
 * 0.6 was the historical MK64 value and was REJECTED: it sits at the top of
 * the band and eats the slip-tolerance headroom down to 0.00029.
 */
export const NEATRACING_DRIFT_SPEED_SCRUB_COEFF_PER_S = 0.55;

/**
 * The NAMED slide-prominence floor (rad) on the deterministic unit proxy
 * `|slip| * |GAIN - 1|` (AC-087).
 */
export const NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD = 0.2;

/**
 * The drift-engagement gate (rad of proxy): the slide must be visible
 * before the drift is announced.
 */
export const NEATRACING_DRIFT_PROXY_ENGAGEMENT_MIN_RAD = 0.01;

/**
 * The number of CONSECUTIVE frames the engagement gate must survive.
 */
export const NEATRACING_DRIFT_PROXY_MIN_FRAMES = 3;

/**
 * The pose-spring stiffness (rad/s^2): omega = 18 rad/s (2.86 Hz).
 */
export const NEATRACING_DRIFT_POSE_STIFFNESS_RAD_PER_S2 = 324;

/**
 * The pose-spring damping (rad/s): zeta = 0.35 (about 31 % first overshoot).
 */
export const NEATRACING_DRIFT_POSE_DAMPING_RAD_PER_S = 12.6;

/**
 * The articulation amplitude (rad): the steer-yaw term of the pose target.
 */
export const NEATRACING_DRIFT_POSE_STEER_YAW_RAD = 0.08;

/**
 * The articulation ARM (rad of slip): the slip at which the steer-yaw term
 * reaches its full amplitude.
 */
export const NEATRACING_DRIFT_POSE_STEER_YAW_FULL_SLIP_RAD = 0.01;

/**
 * The articulation slew (rad/s, band 15-30): the ONLY genuinely new constant
 * of the visual half, and the bound the per-tick pose-target step rides on.
 */
export const NEATRACING_DRIFT_POSE_ARTICULATION_SLEW_PER_S = 20;

/**
 * The visual yaw gain: the base gain of the pose target. It owns the body
 * yaw (the pose target is `committedSlip * GAIN + POSE_STEER_YAW_RAD *
 * articulation`).
 *
 * RE-DERIVED (Step 14Q, AC-087): 0.5 -> 0.45. The body yaw SHAPE is
 * retained (the gain still multiplies the committed slip) and the pose is
 * NOT clamped.
 *
 * WHY DOWN AND NOT UP: the plan's 0.75 recommendation is INFEASIBLE against
 * the ratified pins. The slide-prominence proxy is `|slip| * |GAIN - 1|` and
 * it is V-shaped in the gain with a NULL at GAIN = 1.0, so the ratified
 * anti-null guard (`|GAIN - 1| >= 0.5`) forces `GAIN <= 0.5`. At 0.75 the
 * largest reachable proxy is `0.45 * 0.25 = 0.1125` rad, BELOW the
 * ratified 0.2 rad prominence floor - the drift would stop being announced,
 * a semantic break, not a re-pin. At 0.45 the guard still holds
 * (`|0.45 - 1| = 0.55 >= 0.5`) and the floor stays reachable
 * (`0.45 * 0.55 = 0.2475` rad, 24 % above the floor).
 *
 * THE MEASURED FEEL (the pinned SCN-DRIFT-01 scenario, the 180-tick build
 * hold and the 120-tick countersteer, dt 1/60, the top speed): the far-side
 * pop is 9.0289 deg, the pose SETTLES in 33 ticks (inside the ratified
 * 30..34 band), the per-tick pose-target step peaks at 2.0435 deg (under
 * the 2.2 deg bound), the per-tick articulation step peaks at 1.5279 deg
 * (exactly at the slew bound) and the prominence proxy peaks at 0.2475 rad
 * and stays above the 0.2 rad floor for 174 CONSECUTIVE ticks. The slide
 * never mirrors: the largest signed slip after the countersteer onset is 0.
 *
 * The invalidated literal (`expect(GAIN).toBe(0.5)`) is reported to the
 * green slice as ONE ledger row.
 */
export const NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD = 0.45;

/**
 * The DECLARED POSE ENVELOPE (rad): the largest body yaw the drift model
 * can declare.
 *
 * RE-DERIVED (Step 14Q, AC-087) from the PUBLISHED slip ceiling - the OLD
 * declaration was derived from the COMMITTED ceiling (0.45 * 0.5 + 0.08 =
 * 0.3050 rad), so the PUBLISHED slip (the committed slip plus the
 * chase-lag overshoot) could BREACH the declared bound. The pose target is
 * `publishedSlip * GAIN` plus the steer-yaw term, and the published slip is
 * what the pose spring actually reads (the pose is driven by the
 * PUBLISHED slip, never by the committed command), so the ceiling the
 * envelope is derived from MUST be the PUBLISHED one.
 *
 * THE REGIME-INDEPENDENT CEILING: the published slip carries TWO named
 * ceilings - the ON-ROAD one (0.452 rad, the committed budget plus the
 * 0.002 rad chase-lag overshoot) and the LOOSER OFF-ROAD one (0.493 rad,
 * the measured off-road peak plus its margin). The off-road drag drives the
 * same lag from a FALLING denominator, so the off-road published slip
 * reaches a DEEPER slide than the on-road one and the pose target can
 * exceed the on-road bound. The declaration is therefore derived from the
 * MAXIMUM of the two published ceilings, so it holds in EVERY regime:
 * `0.493 * 0.45 + 0.08` = 0.30185 rad.
 *
 * The pose is NOT clamped to the envelope: the envelope DECLARES the bound,
 * it does not enforce it (a clamp would fight the spring and flatten the
 * pop, which IS the feature).
 */
export const NEATRACING_DRIFT_POSE_ENVELOPE_RAD =
  Math.max(
    NEATRACING_DRIFT_PUBLISHED_SLIP_MAX_RAD,
    NEATRACING_DRIFT_PUBLISHED_SLIP_OFFROAD_MAX_RAD,
  ) *
    NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD +
  NEATRACING_DRIFT_POSE_STEER_YAW_RAD;

/**
 * The D1.3 speed-deficit FLOOR (km/h): the drift must bleed at least this
 * much speed across a 600-tick drift. It ACCOMPANIES (and never replaces)
 * the re-pinned `TRACTION_LOSS_MAX_SPEED_KMH` ceiling - the ceiling is a
 * SPEED bound, this export is a SCRUB-RATE bound, and the D1.3 check needs
 * both.
 */
export const NEATRACING_DRIFT_MIN_SPEED_DEFICIT_KMH = 20;

/**
 * The ABSOLUTE rounding guard (rad) subtracted from the per-tick articulation
 * slew ceiling.
 *
 * The 7a contract asserts
 * `maxArticulationStepDeg <= (ART * SLEW * dt) * (180 / PI)`. The measured
 * difference of the ROUNDED products (`ART * articulation`) can exceed that
 * analytic bound by about one double ULP: the excess comes from the
 * accumulation rounding (`art + delta`), the product rounding and the degree
 * conversion - and every one of them scales with `|articulation|` (bounded
 * by 1 rad) and NOT with the step size.
 *
 * A RELATIVE slack on the ceiling therefore under-corrects at small `dt`
 * (measured: the natural accumulation and a 0.5-ULP relative slack both
 * violate the bound at dt = 1/120, 1/90 and 1/144, while a 4-ULP relative
 * slack still violates it at dt = 1/144). An ABSOLUTE guard does not: with
 * the guard below the bound holds at dt = 1/60, 1/120, 1/90, 1/144, 1/30 and
 * 1/1000.
 *
 * The margin is measured OFFLINE (the RED contract only exercises
 * dt = 1/60, so no in-repo test covers a second `dt` - the multi-`dt` sweep
 * is a 05-green-testing follow-up): the margin is expressed in ULPs of the
 * bound (`ULP(bound) = Number.EPSILON * bound`) and is at least 5 ULP at
 * dt = 1/60 and above.
 *
 * The guard is 8e-16 rad per tick, i.e. 4.8e-14 rad/s of the 20 rad/s slew
 * (a relative change of 2.4e-15) - far below the float64 noise of the pose
 * and invisible in the feel. At a degenerate `dt <= 4e-17 s` the guard
 * zeroes the ceiling, so the articulation FREEZES instead of slewing: that
 * is deterministic and harmless, and it is documented rather than silently
 * clamped.
 */
const ARTICULATION_SLEW_ABSOLUTE_GUARD_RAD = 8e-16;

/** Clamps a value into the inclusive `[lo, hi]` range. */
const clamp = (value: number, lo: number, hi: number): number =>
  value < lo ? lo : value > hi ? hi : value;

/**
 * The committed physical drift state: the slide, the latched TURN direction
 * and the RE-ARM token.
 */
export interface DriftCommittedState {
  /** The committed slip angle (rad); NEGATIVE in a LEFT turn. */
  slip: number;
  /** The latched TURN direction (+1 = LEFT, -1 = RIGHT, 0 = uncommitted). */
  turnSign: number;
  /** The RE-ARM token: true while a fresh engagement is allowed. */
  reArmed: boolean;
}

/**
 * The visual pose accumulator - the ONLY visible-yaw source of the 14N
 * channel.
 */
export interface DriftPoseSpringState {
  /** The visible body yaw (rad). */
  pose: number;
  /** The visible body yaw rate (rad/s). */
  poseRate: number;
  /** The steer-yaw articulation (rad). */
  articulation: number;
}

/**
 * The nested drift channel carried by the physics car state: the committed
 * physical state and the visual pose accumulator.
 */
export interface DriftChannelState {
  /** The committed physical state. */
  committed: DriftCommittedState;
  /** The visual pose accumulator. */
  pose: DriftPoseSpringState;
}

/**
 * One physical drift tick: the tick-START sample the model reads.
 */
export interface DriftStepInput {
  /** Steering axis in `[-1, 1]`; positive steers counter-clockwise (LEFT). */
  steer: number;
  /** Longitudinal speed in m/s at tick start. */
  speed: number;
  /** Normalized speed in `[0, 1]`. */
  speedPercent: number;
  /** Fixed timestep in seconds (1/60). */
  dt: number;
  /** Lateral velocity at tick start (m/s). */
  lateralVelocity: number;
  /** Slip angle at tick start (rad), SLIP convention. */
  slipAngle: number;
  /**
   * The DEMANDED turn rate (rad/s) - OPTIONAL (Step 14S, AC-091).
   *
   * An EXPLICIT rate (including an explicit 0) is AUTHORITATIVE: it is the
   * turn the driver demands and the momentum law is computed from it. An
   * ABSENT rate falls back to the steering axis (see
   * `resolveDemandedTurnRateRadPerS`), so the callers that do not carry the
   * rate keep the steer-axis behaviour.
   */
  demandedTurnRateRadPerS?: number;
}

/**
 * One visual pose tick: the tick-END sample the spring reads.
 */
export interface DriftPoseStepInput {
  /** Steering axis in `[-1, 1]`; positive = LEFT. */
  steer: number;
  /** Normalized speed in `[0, 1]`. */
  speedPercent: number;
  /** The slip that feeds the pose target (rad), SLIP convention. */
  slipAngle: number;
  /** Fixed timestep in seconds (1/60). */
  dt: number;
}

/**
 * Creates the zeroed drift channel - the SPAWN contract.
 *
 * The RE-ARM TOKEN starts TRUE (A-4): a fresh car may engage a drift as soon
 * as the driver turns the wheel. The KEY ORDER is part of the AC-047D
 * byte-identity contract - the serialized channel must stay byte-identical to
 * this factory's output.
 *
 * The COLLISION BOUNCE reset is NOT this factory: a bounce is a RELEASE, not
 * a spawn, so it CLEARS the token via `zeroBouncedDriftChannel` in
 * `collision.utils.ts` (the committed slide and the pose are zeroed and
 * `reArmed` is false). A bounced car must not be able to re-latch on an
 * opposite steer alone.
 */
export function createZeroDriftChannelState(): DriftChannelState {
  return {
    committed: { slip: 0, turnSign: 0, reArmed: true },
    pose: { pose: 0, poseRate: 0, articulation: 0 },
  };
}

/**
 * The TRACTION SEVERITY: the ONE speed-and-slide scalar of the drift law.
 *
 * The severity is 0 on a straight line, lives in the closed unit interval
 * and is MONOTONIC in the slide depth. It is the shared "how much traction
 * is this car spending" measure: the deeper the slide AND the faster the
 * car, the higher the severity, and at the full slide budget on the traction
 * top speed it saturates at 1.
 *
 * @param slipAngleRad - The slip angle (rad); the sign is irrelevant.
 * @param speedMps - The absolute longitudinal speed (m/s).
 * @returns The severity in `[0, 1]`.
 */
export function computeTractionSeverity(
  slipAngleRad: number,
  speedMps: number,
): number {
  const slipShare =
    Math.abs(slipAngleRad) / NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD;
  const speedShare =
    speedMps > 0 ? speedMps / NEATRACING_DRIFT_TRACTION_TOP_SPEED_MPS : 0;
  return clamp(slipShare * speedShare, 0, 1);
}

/**
 * The committed slide DEPTH BUDGET (rad) at a speed: the speed-proportional
 * depth law (Step 14Q, AC-088).
 *
 * The depth is PROPORTIONAL to the speed below the saturation speed and
 * SATURATES at the AC-087 budget (`MAX_SLIP_ANGLE_RAD`) above it, so a slow
 * car slides proportionally less and no car can exceed the declared budget.
 *
 * THE FLOOR (the SOFTENED MIN_SPEED gate, AC-089): the budget never falls
 * to 0. It is FLOORED at the hard zero (`SNAP_BACK_ZERO_SLIP_RAD`), so the
 * softened gate can never HARD-ZERO the commitment: a car that coasts to a
 * standstill keeps its latch (the committed slip survives at the floor)
 * instead of losing it on the speed cliff. The VISIBLE slide is unaffected -
 * the CHASE keeps its exact-zero gate below `MIN_SPEED_MPS`, so the
 * published slide stays exactly 0 there (AC-047D).
 *
 * The floor only binds below about 3.6 m/s (where the proportional budget
 * drops under it), so every speed the published lanes measure is untouched.
 *
 * @param speedMps - The absolute longitudinal speed (m/s).
 * @returns The depth budget (rad) in `[SNAP_BACK_ZERO_SLIP_RAD,
 * MAX_SLIP_ANGLE_RAD]`.
 */
export function computeDriftSlideDepthBudgetRad(speedMps: number): number {
  const proportionalRad =
    (NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD * Math.max(0, speedMps)) /
    NEATRACING_DRIFT_SLIDE_SATURATION_SPEED_MPS;
  return Math.max(
    NEATRACING_DRIFT_SNAP_BACK_ZERO_SLIP_RAD,
    Math.min(NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD, proportionalRad),
  );
}

/**
 * The MOMENTUM SLIDE DEMAND (rad): the drift depth the MOMENTUM demands
 * (Step 14S, AC-091).
 *
 * THE CAUSE: the drift is produced by the MOMENTUM EXCESS - the lateral
 * acceleration the demanded turn DEMANDS minus the lateral acceleration the
 * tires can HOLD at the current speed - and NOT by the wheel command alone.
 * The excess is a SHARE in `[0, 1]` of the declared depth budget, so the
 * demand can never exceed the budget (the depth clamp never MASKS it) and
 * the demand is EXACTLY 0 at a standstill and EXACTLY 0 for a zero demanded
 * turn rate.
 *
 * THE BASIS: the POST-traction `car.speed` (m/s) - the SAME basis the
 * traction law reads (`computeTractionSeverity` and
 * `computeDriftSlideDepthBudgetRad` both read the absolute speed) - and the
 * demanded turn rate in rad/s. The demanded rate is a RATE: the yaw law
 * resolves `steer * NEATRACING_PHYSICS_STEERING_RATE * speedPercent * dt`
 * (rad per tick), so the SAME product WITHOUT `dt` is the rate.
 *
 * THE SHAPE: the demanded lateral acceleration is `speed * |rate|` (the
 * turn's own acceleration demand) and the holdable lateral acceleration is
 * `gripShare * NEATRACING_DRIFT_MOMENTUM_HOLD_ACCEL_MPS2` (the tires' grip
 * limit). The excess share is `1 - hold / demand`, clamped to `[0, 1]`:
 * below the grip limit the demand is 0 (the tires hold the turn), above it
 * the demand scales with the excess. The share is STRICTLY increasing in the
 * speed and in the rate, so the SAME wheel command slides DEEPER the faster
 * the car goes - the drift is a momentum effect, not a wheel command.
 *
 * THE CROSSOVER (a DERIVED property, never a literal): the share reaches 0
 * where `speed * |rate| == gripShare * hold`, so the momentum stops carrying
 * the turn BELOW that speed. With the service-resolved rate (the rate
 * scales with the speed percent) and the full grip the crossover sits at
 * about 9.43 m/s; the commitment therefore opens a real slide from 10 m/s up
 * and the latch HOLDS the committed depth below the crossover (the RELEASE
 * law, not the momentum, unwinds a latched slide).
 *
 * @param speedMps - The post-traction longitudinal speed (m/s).
 * @param demandedTurnRateRadPerS - The demanded turn rate (rad/s); the sign
 *   is irrelevant (the magnitude drives the demand).
 * @param availableGripShare - The available grip share in `(0, 1]` (1 = the
 *   full grip).
 * @returns The momentum slide demand (rad) in
 *   `[0, NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD]`.
 * @example
 * // The 56 m/s AC-047A lane (the service-resolved 1.1118759778125444 rad/s)
 * // demands 0.3054164438299156 rad - above the ratified 0.2 rad floor.
 * computeMomentumSlideDemandRad(56, 1.1118759778125444, 1);
 *
 * @example
 * // The below-gate lane (10 m/s, the service-resolved 0.2247191011235955
 * // rad/s) demands 0.006198347107438023 rad - a REAL (non-zero) commitment
 * // below the MIN_SPEED gate, so the AC-089 softened gate keeps holding.
 * computeMomentumSlideDemandRad(10, 0.2247191011235955, 1);
 */
export function computeMomentumSlideDemandRad(
  speedMps: number,
  demandedTurnRateRadPerS: number,
  availableGripShare: number,
): number {
  const rateMagnitude = Math.abs(demandedTurnRateRadPerS);
  if (!(speedMps > 0) || !(rateMagnitude > 0)) {
    return 0;
  }
  const gripShare = Math.min(1, Math.max(0, availableGripShare));
  const demandedAccelMps2 = speedMps * rateMagnitude;
  const holdableAccelMps2 = gripShare * NEATRACING_DRIFT_MOMENTUM_HOLD_ACCEL_MPS2;
  const excessShare = Math.min(
    1,
    Math.max(0, 1 - holdableAccelMps2 / demandedAccelMps2),
  );
  return computeDriftSlideDepthBudgetRad(speedMps) * excessShare;
}

/**
 * The committed chase RESPONSE time constant (s) at a speed.
 *
 * The chase is a RELAXATION-LENGTH law (Step 14Q, AC-088): the time
 * constant scales with `TRACTION_TOP_SPEED_MPS / speed`, so the chase covers
 * a fixed DISTANCE per second and a fast car reacts FASTER (a slow car lags
 * more). The declared constant is the traction-top-speed reference value.
 *
 * @param speedMps - The absolute longitudinal speed (m/s).
 * @returns The response time constant (s).
 */
export function computeDriftSlipResponseTauS(speedMps: number): number {
  if (!(speedMps > 0)) {
    return NEATRACING_DRIFT_SLIP_RESPONSE_TAU_S;
  }
  return (
    (NEATRACING_DRIFT_SLIP_RESPONSE_TAU_S *
      NEATRACING_DRIFT_TRACTION_TOP_SPEED_MPS) /
    speedMps
  );
}

/**
 * The NEUTRAL release RATE (rad/s): the rate at which the committed slide
 * unwinds while the wheel is NEUTRAL (Step 14Q, AC-089).
 *
 * The rate is DERIVED from the named hold window: the slide unwinds from
 * the full slide budget to the hard zero over
 * `NEATRACING_DRIFT_REARM_HOLD_TICKS` ticks, so the hold constant has a
 * real reader.
 *
 * @param dt - The tick duration (s).
 * @returns The release rate (rad/s).
 */
export function computeDriftReArmReleaseRateRadPerS(dt: number): number {
  if (!(dt > 0)) {
    return 0;
  }
  return (
    (NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD -
      NEATRACING_DRIFT_SNAP_BACK_ZERO_SLIP_RAD) /
    (NEATRACING_DRIFT_REARM_HOLD_TICKS * dt)
  );
}

/**
 * Resolves the DEMANDED turn rate (rad/s) of a tick (Step 14S, AC-091).
 *
 * An EXPLICIT rate on the input is AUTHORITATIVE: an explicit 0 is a real
 * decision (the driver demands NO turn, so the momentum demands NO slide),
 * and an explicit non-zero rate is the turn the service resolved.
 *
 * An ABSENT rate falls back to the STEERING AXIS: the RAW, FACTOR-FREE
 * steer-axis product `steer * NEATRACING_DRIFT_FULL_STEER_TURN_RATE_RAD_PER_S
 * * speedPercent` (rad/s) - the mirrored constant names the same raw
 * full-steer rate the yaw law builds its steer-axis product from. The
 * service softens its user rate with the speed factor above the 10 m/s
 * band, so this fallback is NOT the service's rate there; the PRODUCTION
 * call site always supplies the resolved rate, so the fallback only serves
 * a caller that carries no rate at all.
 *
 * @param input - The tick input.
 * @returns The demanded turn rate (rad/s); the sign follows the steer axis.
 */
function resolveDemandedTurnRateRadPerS(input: DriftStepInput): number {
  const explicitRateRadPerS = input.demandedTurnRateRadPerS;
  if (typeof explicitRateRadPerS === 'number' && Number.isFinite(explicitRateRadPerS)) {
    return explicitRateRadPerS;
  }
  // The RAW, FACTOR-FREE steer-axis rate (no speed factor): the production
  // call site always supplies the resolved rate, so this fallback only
  // serves a caller that carries none.
  return (
    input.steer *
    NEATRACING_DRIFT_FULL_STEER_TURN_RATE_RAD_PER_S *
    input.speedPercent
  );
}

/**
 * D1 stages 0-1: the SOFTENED MIN_SPEED gate and the MOMENTUM commitment.
 *
 * The latch is a three-state machine. ENGAGE needs the steer engaged, a
 * committed slip of exactly 0 and the RE-ARM token; the commitment is the
 * MOMENTUM DEMAND (Step 14S, AC-091): `slip = -turnSign * demand` (a LEFT
 * turn drives the slip NEGATIVE) and the committed depth is bounded by the
 * speed-proportional DEPTH BUDGET. RELEASE (at the hard zero, or when the
 * countersteer or the NEUTRAL wheel unwinds the slide) clears the token, so
 * an opposite steer ALONE can never re-latch and no phantom mirrored drift
 * can appear.
 *
 * THE DRIFT IS CAUSED BY THE MOMENTUM, NOT COMMANDED BY THE WHEEL: the
 * committed slide is the MOMENTUM EXCESS of the tick (the lateral
 * acceleration the demanded turn demands minus the lateral acceleration the
 * tires can hold, scaled by the depth budget). The OLD speed-proportional
 * ENTRY-RATE COMMAND is DELETED (its named constant and its named function
 * are gone) - there is NO deferred cleanup.
 *
 * The MIN_SPEED gate is SOFTENED, never removed (Step 14Q, AC-089): below
 * the gate the commitment is still allowed, but the momentum demand makes
 * the slide ramp in with the weight at speed instead of freezing on a hard
 * cliff. The VISIBLE slide stays exactly 0 below the gate - the CHASE keeps
 * the exact-zero gate (AC-047D).
 *
 * @param prev - The committed state at tick start (read-only).
 * @param input - The tick input.
 * @returns The committed state at tick end.
 */
export function stepDriftCommitment(
  prev: DriftCommittedState,
  input: DriftStepInput,
): DriftCommittedState {
  const { steer, speed, dt } = input;
  let { slip, turnSign, reArmed } = prev;

  // The MOMENTUM DEMAND (Step 14S, AC-091): the demanded turn rate of the
  // tick, the grip the tires still have and the speed-proportional DEPTH
  // BUDGET (Step 14Q, AC-088).
  const depthBudgetRad = computeDriftSlideDepthBudgetRad(speed);
  const momentumDemandRad = computeMomentumSlideDemandRad(
    speed,
    resolveDemandedTurnRateRadPerS(input),
    computeGripBudgetFactor(slip, speed),
  );

  // The countersteer is recognized against the LATCHED TURN direction, never
  // against sign(slip). The NEUTRAL wheel is a RELEASE too (Step 14Q,
  // AC-089): the old `Math.sign(0) === -turnSign` defect routed the neutral
  // wheel into the drifting branch and FROZE the slide at its full depth.
  const countersteering = turnSign !== 0 && Math.sign(steer) === -turnSign;
  const neutralWheel =
    Math.abs(steer) < NEATRACING_DRIFT_STEER_NEUTRAL_THRESHOLD;
  const releasing = turnSign !== 0 && (countersteering || neutralWheel);
  const drifting = turnSign !== 0 && !releasing;

  if (turnSign === 0) {
    // Stage 0 - ENGAGE. The MIN_SPEED gate is SOFTENED: the engagement is
    // allowed below it, and the MOMENTUM DEMAND opens the slide to the
    // depth the weight at that speed demands. The engagement needs a
    // POSITIVE demand: a wheel command the momentum does NOT carry opens
    // NO slide (AC-091), so the latch is never spent on a zero slide.
    if (
      reArmed &&
      Math.abs(steer) >= NEATRACING_DRIFT_STEER_ENGAGE_THRESHOLD &&
      slip === 0 &&
      momentumDemandRad > 0
    ) {
      turnSign = Math.sign(steer);
      slip = -turnSign * momentumDemandRad;
    } else if (neutralWheel) {
      // The wheel is NEUTRAL: the RE-ARM token is latched again.
      reArmed = true;
    }
  } else if (drifting) {
    // Stage 1 IN - the MOMENTUM DEMAND (the slide IS the momentum excess).
    //
    // THE LATCH HOLDS THE DEPTH WHEN THE MOMENTUM STOPS CARRYING THE TURN:
    // while the demand is positive the committed slide IS the demand (the
    // slide is PRODUCED by the momentum excess), and when the demand is
    // EXACTLY 0 (the tires can hold the commanded turn again, e.g. the car
    // has slowed below the momentum crossover) the committed depth is HELD
    // and only the RELEASE law (the countersteer or the NEUTRAL wheel)
    // unwinds it. The slide therefore never SNAPS to zero under a live
    // latch - the ratified AC-089 softened-gate behaviour (a non-zero
    // commitment below the MIN_SPEED gate) keeps holding.
    if (momentumDemandRad > 0) {
      slip = -turnSign * momentumDemandRad;
    }
  } else {
    // Stage 1 OUT - the RELEASE: the countersteer or the NEUTRAL wheel
    // unwinds the slide. The release check comes FIRST, on the slip as it
    // ENTERS the tick: there the slide is a hard zero, the latch is
    // released and the token is cleared.
    if (Math.abs(slip) <= NEATRACING_DRIFT_SNAP_BACK_ZERO_SLIP_RAD) {
      slip = 0;
      turnSign = 0;
      reArmed = false;
    } else {
      // The countersteer unwinds at the EXIT rate; the NEUTRAL wheel
      // unwinds over the named RE-ARM hold window, so the hold constant
      // has a real reader.
      const unwindRateRadPerS = countersteering
        ? NEATRACING_DRIFT_EXIT_RATE_RAD_PER_S
        : computeDriftReArmReleaseRateRadPerS(dt);
      slip =
        Math.sign(slip) *
        Math.max(
          NEATRACING_DRIFT_SNAP_BACK_ZERO_SLIP_RAD,
          Math.abs(slip) - unwindRateRadPerS * dt,
        );
    }
  }

  // The COMMITTED budget: the depth is PROPORTIONAL to the speed and
  // saturates at the AC-087 budget, so the committed slide can never exceed
  // the declared envelope at any speed.
  slip = clamp(slip, -depthBudgetRad, depthBudgetRad);
  return { slip, turnSign, reArmed };
}

/**
 * D1 stage 2: the tire saturation as a SPEED-AWARE BUDGET.
 *
 * The budget is a function of the SLIP and the SPEED (Step 14Q, AC-088):
 * the same slide at a higher speed is a HARDER traction state, because the
 * TRACTION SEVERITY (the ONE shared scalar) grows with both. The speed
 * demand is charged by the SPEED SCRUB (stage 6) as well.
 *
 * @param slipAngle - The committed slip (rad).
 * @param speedMps - The absolute longitudinal speed (m/s).
 * @returns The grip budget factor in `(0, 1]`.
 */
export function computeGripBudgetFactor(
  slipAngle: number,
  speedMps: number,
): number {
  const demand = Math.max(
    0,
    Math.abs(slipAngle) - NEATRACING_DRIFT_GRIP_FALLOFF_THRESHOLD_RAD,
  );
  const severity = computeTractionSeverity(slipAngle, speedMps);
  return (
    1 /
    (1 +
      NEATRACING_DRIFT_GRIP_FALLOFF_RATE_PER_RAD * demand * (1 + severity))
  );
}

/**
 * D1 stage 4: the countersteer authority, applied AGAINST the slide.
 *
 * The authority is an ACCELERATION (m/s^2) toward zero: it is multiplied by
 * `dt` to yield a velocity delta. The gate is the LATCHED TURN DIRECTION
 * (never `sign(slip)`), so the countersteer can never mirror the slide, and
 * the intent magnitude gate is the in-model fallback of D2.3.
 *
 * @param committed - The committed state (the slide and the latched turn).
 * @param input - The tick input (the steer and the speed percentile).
 * @returns The countersteer acceleration (m/s^2); 0 when the gate is closed.
 */
export function computeCountersteerAuthority(
  committed: DriftCommittedState,
  input: DriftStepInput,
): number {
  const { steer, speedPercent } = input;
  const { slip, turnSign } = committed;
  if (turnSign === 0) {
    return 0;
  }
  if (Math.sign(steer) !== -turnSign) {
    return 0;
  }
  if (Math.abs(slip) <= NEATRACING_DRIFT_COUNTERSTEER_MIN_SLIP_RAD) {
    return 0;
  }
  if (Math.abs(steer) < NEATRACING_DRIFT_COUNTERSTEER_MIN_STEER) {
    return 0;
  }
  return (
    turnSign *
    NEATRACING_DRIFT_COUNTERSTEER_AUTHORITY_MPS2 *
    Math.abs(steer) *
    speedPercent
  );
}

/**
 * D1 stage 6: the longitudinal speed scrub.
 *
 * The AC-079 law is re-homed from the retired traction loss onto the
 * saturating SPEED SCRUB: a sliding tire gives up the forward speed it
 * cannot hold. The term is zero on a straight line and grows with the slide
 * and with the speed, so the straight-line and low-speed handling stay
 * unchanged (AC-047D). The SERVICE owns the speed, so the SERVICE applies
 * it.
 *
 * @param slipAngle - The tick-start slip (rad).
 * @param speed - The longitudinal speed (m/s).
 * @param speedPercent - The normalized speed in `[0, 1]`.
 * @returns The forward speed the tires cannot hold, in m/s per second.
 */
export function computeLongitudinalSpeedScrub(
  slipAngle: number,
  speed: number,
  speedPercent: number,
): number {
  return (
    NEATRACING_DRIFT_SPEED_SCRUB_COEFF_PER_S *
    Math.abs(slipAngle) *
    speed *
    speedPercent
  );
}

/**
 * The deterministic slide-prominence proxy (rad): `|slip| * |GAIN - 1|`.
 *
 * The proxy is the NAMED prominence floor's unit system (AC-087). It is
 * V-shaped in the gain with a NULL at GAIN = 1.0, so the gain must never
 * approach 1.0 (the 7g anti-null guard).
 *
 * @param slipAngleRad - The slip angle (rad).
 * @returns The slide proxy (rad).
 */
export function computeDriftSlideProxyRad(slipAngleRad: number): number {
  return (
    Math.abs(slipAngleRad) *
    Math.abs(NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD - 1)
  );
}

/**
 * D1 stages 3-5: the committed chase, the countersteer authority and the
 * snap-back at the zero crossing.
 *
 * @param committed - The committed state AFTER the commitment step.
 * @param input - The tick input.
 * @returns The lateral velocity at tick end (m/s).
 */
export function computeNextLateralVelocity(
  committed: DriftCommittedState,
  input: DriftStepInput,
): number {
  const { speed, dt, lateralVelocity } = input;
  const { slip, turnSign } = committed;

  // Stage 0 (the chase side): below the MIN_SPEED gate there is no lateral
  // velocity at all - EXACTLY 0 (AC-047D). The gate is a VISIBLE-slide
  // gate: the commitment below it is the SOFTENED, proportional ramp
  // (Step 14Q, AC-089), and the published slide stays exactly 0.
  if (!(speed > NEATRACING_DRIFT_MIN_SPEED_MPS)) {
    return 0;
  }

  // The countersteer is recognized against the LATCHED TURN direction, never
  // against sign(slip); the NEUTRAL wheel is a release too (Step 14Q,
  // AC-089) - the same law the commitment uses, so the two stages can never
  // disagree about what the driver is doing.
  const countersteering = turnSign !== 0 && Math.sign(input.steer) === -turnSign;
  const neutralWheel =
    Math.abs(input.steer) < NEATRACING_DRIFT_STEER_NEUTRAL_THRESHOLD;
  const drifting = turnSign !== 0 && !countersteering && !neutralWheel;

  // Stage 2 - the grip budget (the speed-aware law).
  const gripFactor = computeGripBudgetFactor(slip, speed);

  // Stage 3 - the committed chase: the lateral velocity chases the committed
  // slide's `tan(slip) * speed` target with an exponential response whose
  // time constant is the RELAXATION-LENGTH law (Step 14Q, AC-088): the
  // chase covers a fixed DISTANCE per second, so a fast car reacts FASTER.
  const responseTauS = computeDriftSlipResponseTauS(speed);
  const target = Math.tan(slip) * speed;
  let nextLateralVelocity =
    lateralVelocity +
    (target - lateralVelocity) *
      (1 - Math.exp((-dt * gripFactor) / responseTauS));

  // Stage 4 - the countersteer authority, applied against the slide.
  nextLateralVelocity += computeCountersteerAuthority(committed, input) * dt;

  // Stage 5 - the hard zero: with no committed slide the lateral velocity is
  // held at exactly 0. The boundary is INCLUSIVE (`<=`), the same edge stage 1
  // releases on, so the two stages can never disagree about a slide that sits
  // exactly on the snap band.
  if (
    !drifting &&
    Math.abs(slip) <= NEATRACING_DRIFT_SNAP_BACK_ZERO_SLIP_RAD
  ) {
    nextLateralVelocity = 0;
  }

  return nextLateralVelocity;
}

/**
 * D1 stage 7: the merged visual pose spring.
 *
 * The pose target is `committedSlip * GAIN + POSE_STEER_YAW_RAD *
 * articulation`. The sign is `+committedSlip` - the upstream `-committed` is
 * a SIGN ERROR under the pinned LEFT = negative-slip convention - and the
 * articulation is the slew-limited steer-yaw term. The pose is NOT clamped:
 * a clamp would fight the spring and flatten the pop, which IS the feature.
 *
 * @param prev - The pose state at tick start (read-only).
 * @param input - The pose tick input (the tick-END slip).
 * @returns The pose state at tick end.
 */
export function stepDriftPoseSpring(
  prev: DriftPoseSpringState,
  input: DriftPoseStepInput,
): DriftPoseSpringState {
  const { steer, speedPercent, slipAngle, dt } = input;

  // The articulation target: the steer yaw, armed by the slip. On a straight
  // line every factor is exactly 0, so the straight-line identity is EXACT.
  const articulationTarget =
    steer *
    speedPercent *
    Math.min(
      1,
      Math.abs(slipAngle) / NEATRACING_DRIFT_POSE_STEER_YAW_FULL_SLIP_RAD,
    );

  // The articulation slews at a bounded rate: the slew bound is the whole
  // point of the 14N visual half. The absolute guard keeps the MEASURED
  // step inside the analytic bound (see the guard's JSDoc).
  const ceiling = Math.max(
    0,
    NEATRACING_DRIFT_POSE_ARTICULATION_SLEW_PER_S * dt -
      ARTICULATION_SLEW_ABSOLUTE_GUARD_RAD,
  );
  const articulation =
    prev.articulation +
    clamp(articulationTarget - prev.articulation, -ceiling, ceiling);

  // The pose target and the second-order spring.
  const poseTarget =
    slipAngle * NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD +
    NEATRACING_DRIFT_POSE_STEER_YAW_RAD * articulation;
  const poseRate =
    prev.poseRate +
    (NEATRACING_DRIFT_POSE_STIFFNESS_RAD_PER_S2 * (poseTarget - prev.pose) -
      NEATRACING_DRIFT_POSE_DAMPING_RAD_PER_S * prev.poseRate) *
      dt;
  const pose = prev.pose + poseRate * dt;

  return { pose, poseRate, articulation };
}

/**
 * Resolves the bounded slip angle via `atan2(lateralVelocity, speed)`.
 *
 * A stopped car reports zero slip so a bare `atan2` with zero speed
 * cannot create a phantom drift state.
 *
 * @param lateralVelocity - Lateral velocity in m/s.
 * @param speed - Longitudinal speed in m/s.
 * @returns Slip angle in `(-π/2, π/2)` rad, or `0` when stopped.
 */
export function computeSlipAngle(
  lateralVelocity: number,
  speed: number,
): number {
  if (speed <= 0) {
    return 0;
  }
  return Math.atan2(lateralVelocity, speed);
}
