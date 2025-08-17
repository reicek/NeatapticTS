## Contributing to NeatapticTS

Thank you for helping improve NeatapticTS — this project is educational and community-focused, and contributions that improve clarity, examples, tests, or API ergonomics are especially welcome.

### Quick checklist

- Fork the repo and open a branch for your work
- Run `npm install`
- Make code or docs changes
- Run tests and `npm run docs` if you changed JSDoc
- Open a pull request that links an issue (recommended)

---

### Setup (local)

1. Fork this repository and clone your fork.
2. Install dependencies:

```powershell
npm install
```

---

### Running tests and docs

- Run unit tests:

```powershell
npm test
```

- Generate the documentation and mirrored `src` READMEs (run this if you changed JSDoc comments):

```powershell
npm run docs
```

---

### Code style and quality

- Keep changes small and focused. Prefer many small PRs over one large PR.
- Follow TypeScript types and avoid `any` unless necessary; include type updates where relevant.
- Add or update tests for behavioral changes. Tests live in `test/` and run with `npm test`.

---

### Documentation

- Update JSDoc comments in the `src/` files when changing public APIs.
- Run `npm run docs` to regenerate per-folder `README.md` files (these are mirrored into `src/*/README.md` and the `docs/` site).

---

### Reporting bugs

Open an issue with the following minimal information:

- Repro steps or a small example
- Error/log output and Node version (or browser/OS when applicable)
- The expected vs actual behavior

If possible, include a small runnable snippet that reproduces the problem.

---

### Submitting a pull request

1. Create a descriptive branch name (e.g. `fix/activation-pool-bug` or `feat/multiobjective-telemetry`).
2. Open an issue first for non-trivial changes and note the issue number in your PR.
3. Ensure tests pass and run `npm run docs` if you changed JSDoc or public APIs.
4. In your PR description include:
   - A short summary of the change
   - The motivation and any user-visible API changes
   - Links to related issues
   - Test and docs status (e.g., `tests: pass`, `docs regenerated`)

---

### Review and CI

The project runs tests and basic checks on PRs. Address review comments promptly. Small style or linter failures are typically fixed during the PR review.

---

### Attribution & license

This project is released under the MIT License. Core ideas and portions of code are derived from the original Neataptic (Thomas Wagenaar) and Synaptic (Juan Cazala). See `LICENSE` for details.

---

### Code of conduct

We follow a community-friendly code of conduct. Be respectful in discussions and PR reviews.

---

### Need help?

Open an issue describing what you'd like to change and we can advise on implementation approach and scope.

Thank you for contributing!
