/**
 * @module png.utils.constants
 *
 * PNG codec constants for the dependency-free PNG encoder/decoder used by
 * the enemy sprite generator. Centralizes the PNG file signature, CRC-32
 * polynomial, and bitmask constants so they have a single source of truth.
 */

/** Standard 8-byte PNG file signature. */
export const PNG_SIGNATURE = Buffer.from([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
]);

/** CRC-32 polynomial used by the PNG specification. */
export const PNG_CRC_POLYNOMIAL = 0xedb88320;

/** Number of entries in the CRC-32 lookup table. */
export const PNG_HASH_TABLE_SIZE = 256;

/** Number of bits processed per CRC-32 table entry. */
export const PNG_INITIAL_HASH = 8;

/** 32-bit unsigned mask used in CRC-32 computation. */
export const PNG_HASH_MASK = 0xffffffff;

/** 8-bit unsigned byte mask. */
export const PNG_BYTE_MASK = 0xff;
