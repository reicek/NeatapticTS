# NeatRacing Frame Composer Data Flow

> Permanent trace for the host-owned nine-pass composition path (slice `P1-frame-composer`, AC-002 extension).

## Sequence: one composed frame

```mermaid
sequenceDiagram
    autonumber
    participant RAF as requestAnimationFrame
    participant RS as RaceSession.handleFrame
    participant BEF as buildComposerFrame
    participant FC as createFrameComposer.drawFrame
    participant GL as WebGL2 context

    RAF->>RS: frame time (ms)
    RS->>RS: fixed-step accumulator @ 60 Hz
    RS->>BEF: assemble composer frame
    BEF->>BEF: perspectiveInto(projection, fov, aspect, near, far)
    BEF->>BEF: lookAtInto(view, eye, target, up)
    BEF->>BEF: multiplyViewProjectionInto(viewProjection, view, projection)
    Note over BEF: Computes projection * view (column-major) so GLSL `uViewProjection * world` is correct.
    BEF->>FC: NeatRacingFrameComposerFrame
    FC->>FC: refreshDrawStateBags(bags, gl, frame)
    FC->>GL: bindFramebuffer(scene.fbo), viewport(scene.size), clear(RGBA 0,0,0,0)
    FC->>GL: passes.starfield.draw(cameraPose)
    FC->>GL: passes.grid.draw(bags.grid)
    FC->>GL: passes.pulse.draw(bags.pulse)
    FC->>GL: passes.track.draw(bags.track)
    FC->>GL: passes.car.draw(bags.car)
    FC->>GL: passes.glass.draw(bags.glass)
    opt dev-only
        FC->>GL: readPixels sample -> Uint8Array
        GL-->>FC: per-pass non-black pixel count
    end
    FC->>FC: assertSceneFrameBufferHasNonEmptyContent (first frame)
    Note over FC: Center plus lower-center samples so empty sky no longer throws
    FC->>GL: bindFramebuffer(null), viewport(default), clear to sky/background
    FC->>GL: passes.bloom.draw(bloomFrame)
    Note over GL: Bloom threshold -> blur ping-pong -> additive composite to default FB
    FC->>GL: bindFramebuffer(null) (restore baseline)
```

## State ownership

| Layer | Owns | Does NOT own |
|-------|------|--------------|
| Browser entry (`browser-entry.ts`) | Live camera matrices (`viewProjection`), RAF loop, keyboard/hud wiring | Pass handles, scene FBO |
| Frame composer (`host/frame-composer.ts`) | Scene FBO + texture, draw-state bags, per-frame viewport/baseline | Individual pass internals |
| Pass modules (`engine/passes/*`) | Program compiles, pass-local GL state, geometry uploads | Framebuffer bindings after return |
| Bloom pass (`engine/passes/bloom.pass.ts`) | Internal half-res targets, three programs, additive composite | Scene texture (borrowed) |

## Critical invariants

1. **Matrix order**: `multiplyViewProjectionInto` writes `projection * view` (column-major) because every world-pass vertex shader applies it as `gl_Position = uViewProjection * vec4(world, 1.0)`.
2. **Single scene transfer**: The world is rendered exactly once into the composer-owned RGBA8 scene FBO; the bloom additive composite is the only draw that samples that texture and transfers it to the default framebuffer.
3. **Zero per-frame allocations**: Bags, `Float32Array` scratch, and the readback `Uint8Array` are allocated once at construction and refreshed in place.
4. **Framebuffer baseline**: The composer binds its scene FBO before every world pass and restores `gl.FRAMEBUFFER` to `null` after the bloom composite, so no pass can leak a bound FBO into the next frame.
5. **Default-framebuffer sky clear**: before the additive bloom composite the host clears the default framebuffer to a sky/background color so unlit pixels remain opaque and the dark page background no longer shows through.
6. **Dual-region empty-frame tripwire**: the dev-only `assertComposedFrameNonEmpty` guard samples the center and a lower-center region of the scene FBO. With the chase camera, the center is empty sky while the lower-center covers the ground grid and car, so the tripwire catches a truly blank frame without firing on a valid scene.
