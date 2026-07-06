# Releasing NeatapticTS

## Migration: Demo-agnostic public library refactor

A recent refactor removed demo-specific names from the public library surface in `src/`. This is a **breaking change** for any external code that imported the old racing-named GPU module or symbols. No backward-compatibility aliases are provided.

| Old                                                  | New                                                            | Notes                                                                                |
| ---------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `src/architecture/network/gpu/network.gpu.racing.ts` | `src/architecture/network/gpu/network.gpu.batch-evaluation.ts` | File renamed to describe the actual capability: batch evaluation of networks/agents. |
| `RacingBatchOptions`                                 | `BatchEvaluationOptions`                                       | Exported options interface.                                                          |
| `evaluateRacingGeneration`                           | `evaluateBatchGeneration`                                      | Per-generation batch evaluator.                                                      |
| `RacingAgentRequest`                                 | `AgentEvaluationRequest`                                       | Per-agent evaluation request shape.                                                  |
| `evaluateConcurrentRacingAgents`                     | `evaluateConcurrentAgents`                                     | Concurrent agent evaluator.                                                          |

Internal-only renames in the two-population NGE collective module also removed racing framing:

- `raceState` → `episodeState`
- `car` → `agent`
- `race-pack` → `episode-pack`
- other `race*` identifiers → `episode*` equivalents

JSDoc across the GPU batch-evaluation module, NGE collective/evolution modules, export helpers, visualization utilities, and worker-payload types was sanitized so generated `src/**/README.md` documentation no longer references `flappy_bird`, `racing_curriculum`, `neatchat`, `asciiMaze`, `ant-hive`, or other demo names. The `examples/` and `docs/browser-tests/` directories keep their demo-specific names by design.

To migrate, update every import and usage from the old racing names to the new batch-evaluation/agent names listed above.

---

This repository provides a small release flow using a dedicated manual pipeline and release-triggered publishers.

Primary workflow used for releases

- Manual release pipeline (Actions → "Manual release pipeline") — preferred manual entrypoint.
  - Inputs:
    - `level`: `patch` | `minor` | `major` (default: `patch`)
    - `branch`: branch to release from (default: `release`)
  - What it does:
    1. Checks out the specified branch.
    2. Runs `npm version <level>` which updates `package.json`, creates a commit and a tag.
    3. Pushes the commit and tag to the remote.
    4. Exits; the repository `release` event (tag/release published) will trigger the publishing workflows.

Release-triggered workflows

- `publish.yml` (trigger: `release: published`) — builds and publishes packages to:

  - npmjs.org (uses `NPM_TOKEN` secret)
  - GitHub Packages (scoped package: `@reicek/neataptic-ts`)

- `deploy-pages.yml` (trigger: `release: published`) — runs `npm run docs` and publishes `docs/` to the `gh-pages` branch.

Required repository secrets

- `NPM_TOKEN` — npm auth token with publish rights (add at Settings → Secrets → Actions).
- `GITHUB_TOKEN` — automatically provided to Actions (no manual setup required) and used for GitHub Packages and Pages deploy.

How to publish a new release (recommended)

1. Ensure your release branch (default `release`) contains the changes you want to publish.
2. Go to the repository Actions tab, choose "Manual release pipeline" and click "Run workflow".
3. Select `level` (patch/minor/major) and `branch` (e.g., `release`), then run.
4. The workflow will create a tag and push it. After the tag is pushed, GitHub will fire the `release` event and the `publish` and `deploy-pages` workflows will run automatically.

Verification after release

- npm: check https://www.npmjs.com/package/@reicek/neataptic-ts for the published version.
- GitHub Packages: check the Packages tab on the repository.
- Docs: check https://reicek.github.io/NeatapticTS/ after the `gh-pages` deploy completes.

Troubleshooting & notes

- Branch protection: if the chosen branch has protection rules that prevent the Actions bot from pushing commits/tags, the manual pipeline may fail to push. In that case you can:

  - Temporarily allow Actions to push, or
  - Use a PAT (personal access token) stored as a secret and update the workflow to use it for pushing, or
  - Create the tag locally and push it from your machine.

- Avoid duplicate publishers: the manual pipeline intentionally only bumps/tags/pushes and does not publish directly — the `publish` and `deploy-pages` workflows run on the `release` event and perform the actual publishing steps once.

- If you'd like a single workflow to perform bump+publish+deploy in one run (instead of splitting by event), tell me and I can convert the manual pipeline to do everything and remove the release-triggered workflows.

If you want I can also add automatic changelog generation (from commits) and include the generated release notes in the GitHub Release body.
