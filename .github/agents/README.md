Agents footer requirement

Every .agent.md file in this directory MUST end with the canonical `## Output format` footer and a trailing fenced `structured-v1` block (the repository canonical template).

Run the repository validator:

- Check: `npm run agents:validate-quality`
- Auto-fix: `npm run agents:validate-quality -- --fix`
