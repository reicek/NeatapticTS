/**
 * Chapter-local `Network` anchor for the network boundary.
 *
 * This file gives the network chapter a local entrypoint before the public
 * `Network` class itself is relocated out of the architecture root. Keeping
 * this anchor narrow lets helper chapters start depending on the chapter-owned
 * path without reopening the full public and test surface in one pass.
 *
 * Read this file as a migration seam:
 *
 * 1. today it forwards the existing root implementation,
 * 2. nearby chapter files can retarget to this local anchor first,
 * 3. a later pass can move the class implementation here and leave the root
 *    file as the public compatibility facade.
 */

export { default } from '../network';
