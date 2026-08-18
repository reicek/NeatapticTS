/**
 * GLSL ES 3.00 shader source for the Neatenstein per-pixel floor-casting pass.
 *
 * This module exports the fragment shader source as a string so the GPU tier
 * can compile it at runtime. The shader implements per-pixel floor casting
 * using the same `fract(worldCoord)` procedural grid detection as the CPU-tier
 * `castNeatensteinFloorPerPixel` function.
 *
 * The shader source is a **forward-looking scaffold** for a future GPU
 * (WebGL2/WebGPU) tier. It is not compiled or dispatched by the current
 * CPU-tier renderer. The source string is exported so that alignment tests
 * can verify the GLSL matches the CPU-tier algorithm.
 *
 * @module
 */

import { NEATENSTEIN_FOG_START_DISTANCE } from '../framebuffer';

/**
 * Fragment shader source for the floor-casting pass.
 *
 * Uses `highp` precision and procedural `fract()`-based integer grid detection.
 * The fog start distance is derived from the shared
 * {@link NEATENSTEIN_FOG_START_DISTANCE} constant so the GPU and CPU tiers
 * use identical fog curves.
 */
export const NEATENSTEIN_FLOOR_CASTER_SHADER_SOURCE = `#version 300 es
precision highp float;
precision highp int;

uniform vec4 uCameraPos;    // (cameraX, cameraY, 0, 0)
uniform vec4 uCameraDir;    // (cosYaw, sinYaw, 0, 0)
uniform vec4 uCameraPlane;  // (planeX, planeY, 0, 0)
uniform float uFocalLength;
uniform float uCameraHeight;
uniform float uHorizonY;
uniform float uRenderDistanceCap;
uniform vec3 uGridColor;
uniform vec3 uBackgroundColor;

out vec4 fragColor;

void main() {
  // Per-pixel floor casting: compute rowDistance from camera height and
  // vertical pixel offset from the horizon.
  float rowDistance = uCameraHeight * uFocalLength / (gl_FragCoord.y - uHorizonY);

  if (rowDistance > uRenderDistanceCap) {
    fragColor = vec4(uBackgroundColor, 1.0);
    return;
  }

  // Ray direction interpolation: dir + plane * screenOffset
  float screenOffset = (gl_FragCoord.x / uFocalLength) * 2.0 - 1.0;
  vec2 rayDir = uCameraDir.xy + uCameraPlane.xy * screenOffset;

  // World coordinates via ray-direction interpolation.
  vec2 worldCoord = uCameraPos.xy + rowDistance * rayDir;

  // Procedural integer grid detection via fract(worldCoord).
  vec2 fractCoord = fract(worldCoord);
  vec2 distToLine = min(fractCoord, 1.0 - fractCoord);
  float minDist = min(distToLine.x, distToLine.y);

  // Grid line width in world space at this depth.
  float pixelWorldSize = rowDistance / uFocalLength;
  float glowWidth = pixelWorldSize * 3.0;  // GLOW_WIDTH_PX = 3

  if (minDist > glowWidth) {
    fragColor = vec4(uBackgroundColor, 1.0);
    return;
  }

  // Fog factor via smoothstep(FOG_START, CAP, rowDistance).
  float fogFactor = smoothstep(${NEATENSTEIN_FOG_START_DISTANCE.toFixed(1)}, uRenderDistanceCap, rowDistance);
  float alpha = 0.58 * (1.0 - fogFactor);

  // Blend grid color toward background based on fog alpha.
  vec3 color = mix(uBackgroundColor, uGridColor, alpha);
  fragColor = vec4(color, 1.0);
}
`;