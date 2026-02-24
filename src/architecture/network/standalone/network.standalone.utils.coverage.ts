import {
  COVERAGE_CALL_REGEX,
  COVERAGE_COUNTER_REGEX,
  COVERAGE_REPLACEMENT,
  EMPTY_TOKEN_REGEX,
  ISTANBUL_IGNORE_BLOCK_REGEX,
  REPEATED_SEMICOLON_REGEX,
  SOLITARY_SEMICOLON_REGEX,
  SOURCE_MAP_REGEX,
  STRAY_COMMA_CLOSE_REGEX,
  STRAY_COMMA_OPEN_REGEX,
} from './network.standalone.utils.types';

/**
 * Remove instrumentation artifacts and formatting detritus from function sources.
 *
 * @param code Source text potentially containing coverage wrappers.
 * @returns Cleaned source text suitable for deterministic standalone emission.
 */
export function stripCoverage(code: string): string {
  let cleanedCode = code;
  cleanedCode = cleanedCode.replace(
    ISTANBUL_IGNORE_BLOCK_REGEX,
    COVERAGE_REPLACEMENT,
  );
  cleanedCode = cleanedCode.replace(
    COVERAGE_COUNTER_REGEX,
    COVERAGE_REPLACEMENT,
  );
  cleanedCode = cleanedCode.replace(COVERAGE_CALL_REGEX, COVERAGE_REPLACEMENT);
  cleanedCode = cleanedCode.replace(SOURCE_MAP_REGEX, COVERAGE_REPLACEMENT);
  cleanedCode = cleanedCode.replace(STRAY_COMMA_OPEN_REGEX, '( ');
  cleanedCode = cleanedCode.replace(STRAY_COMMA_CLOSE_REGEX, ' )');
  cleanedCode = cleanedCode.trim();
  cleanedCode = cleanedCode.replace(
    SOLITARY_SEMICOLON_REGEX,
    COVERAGE_REPLACEMENT,
  );
  cleanedCode = cleanedCode.replace(REPEATED_SEMICOLON_REGEX, ';');
  cleanedCode = cleanedCode.replace(EMPTY_TOKEN_REGEX, COVERAGE_REPLACEMENT);
  return cleanedCode;
}
