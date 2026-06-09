/**
 * Converts a JSON object to a Markdown string.
 * Each top-level key becomes a section with a title (Title Case).
 * Values are rendered as Markdown lists (for arrays), plain text (for strings/numbers), or recursively formatted (for nested objects).
 *
 * @param {Object} obj - The input JSON object to convert.
 * @returns {string} The generated Markdown string.
 *
 * @example
 * const json = {
 *   mission: "Maintain agent/skill system usability.",
 *   constraints: ["No session log unless requested.", "Do not edit global user settings."]
 * };
 * const md = jsonToMarkdown(json);
 * // Output:
 * // ## Mission
 * //
 * // Maintain agent/skill system usability.
 * //
 * // ## Constraints
 * //
 * // - No session log unless requested.
 * // - Do not edit global user settings.
 */
export function jsonToMarkdown(obj) {
  /**
   * Converts a string from snake_case or camelCase to Title Case.
   * @param {string} str - The string to convert.
   * @returns {string} The Title Case string.
   */
  function toTitleCase(str) {
    return (
      str
        // Replace underscores with spaces
        .replace(/_/g, ' ')
        // Add space before capital letters (for camelCase)
        .replace(/([a-z])([A-Z])/g, '$1 $2')
        // Capitalize the first letter of each word
        .replace(/\b\w/g, (c) => c.toUpperCase())
        .trim()
    );
  }

  /**
   * Formats a value for Markdown output.
   * - Arrays become Markdown bullet lists.
   * - Nested objects are recursively formatted with bolded keys.
   * - Strings/numbers are returned as-is.
   * @param {*} value - The value to format.
   * @returns {string} The formatted Markdown string.
   */
  function formatContent(value) {
    if (Array.isArray(value)) {
      // Render arrays as Markdown bullet lists
      return value.map((item) => `- ${item}`).join('\n');
    } else if (typeof value === 'object' && value !== null) {
      // Recursively format nested objects
      return Object.entries(value)
        .map(([k, v]) => `**${toTitleCase(k)}:** ${formatContent(v)}`)
        .join('\n');
    } else {
      // Render strings, numbers, etc. as plain text
      return String(value);
    }
  }

  // Build the Markdown string by iterating over each key-value pair
  return Object.entries(obj)
    .map(([key, value]) => {
      const sectionTitle = toTitleCase(key); // Format key as section title
      const content = formatContent(value); // Format value as section content
      // Each section starts with a level-2 heading
      return `## ${sectionTitle}\n\n${content}\n`;
    })
    .join('\n'); // Separate sections with blank lines
}
