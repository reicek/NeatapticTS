/**
 * Root bridge for NEAT maintenance and structural-viability surfaces.
 *
 * Maintenance is the conservative counterpart to mutation. Mutation explores
 * new structure on purpose. Maintenance keeps a concrete network usable after
 * those edits, after imports, or after other topology-changing operations leave
 * the graph too sparse or locally broken.
 *
 * That distinction matters in practice. A NEAT controller spends much of its
 * time inventing new topology, but users also pause and resume runs, import old
 * checkpoints, hand-edit networks, and apply pruning or crossover in ways that
 * can leave a graph technically valid yet structurally awkward. This chapter is
 * the small repair-policy bridge for those moments when the goal is not to
 * explore harder, but to restore a stable baseline so later evaluation and
 * mutation stages still have a usable network to work with.
 *
 * This boundary is intentionally narrow. It does not choose mutation operators,
 * drive speciation, or reshape the broader controller loop. Instead it answers
 * one smaller question: once a network already exists, what is the minimal
 * policy surface for keeping that network structurally viable?
 *
 * Read the maintenance root in three passes:
 *
 * - start here when you want the policy distinction between conservative repair
 *   and exploratory mutation,
 * - continue into `facade/` when you need the stable `Neat` wrappers,
 * - drop into `../mutation/repair/` only when you want the lower-level graph
 *   edits that actually reconnect nodes or add hidden capacity.
 *
 * The current maintenance subtree is organized around one public facade:
 *
 * - `facade/` keeps the stable `Neat` entrypoints that expose the hidden-node
 *   floor, enforce that floor, and repair obvious dead ends without asking the
 *   caller to drop into the broader mutation chapter.
 *
 * Read this chapter when you want the policy story first. Jump to `facade/`
 * when you need the concrete public wrappers and the narrow host contract they
 * depend on.
 *
 * ```mermaid
 * flowchart TD
 *   Network[Existing network] --> Policy[Maintenance policy surface]
 *   Policy --> Target[Read hidden-node floor]
 *   Policy --> Floor[Enforce minimum hidden capacity]
 *   Policy --> DeadEnds[Repair obvious connectivity dead ends]
 *   Floor --> Facade[facade and stable Neat wrappers]
 *   DeadEnds --> Facade
 *   Target --> Facade
 * ```
 *
 * The decision ladder is deliberately conservative:
 *
 * ```mermaid
 * flowchart TD
 *   Start[Imported edited or recently repaired network] --> Inspect{What do you need?}
 *   Inspect --> Read[Only inspect the hidden floor]
 *   Inspect --> Capacity[Enforce minimum hidden capacity]
 *   Inspect --> Connectivity[Repair obvious dead ends]
 *   Read --> FacadeRead[maintenance facades read target]
 *   Capacity --> FacadeFloor[maintenance facade delegates hidden-floor repair]
 *   Connectivity --> FacadeDeadEnds[maintenance facade delegates dead-end repair]
 *   FacadeFloor --> Repair[mutation repair mechanics]
 *   FacadeDeadEnds --> Repair
 * ```
 *
 * Historically this kind of boundary is easy to under-document because it does
 * not feel as exciting as mutation or speciation. But for real users it often
 * answers the first practical question after loading or editing a network:
 * "do I need another search step, or just a conservative repair pass before I
 * continue?"
 *
 * Recommended reading inside maintenance:
 * - `./facade/README.md` for the public wrapper semantics and host contract
 * - `../mutation/repair/README.md` for the lower-level repair mechanics that
 *   the facade delegates to when work is actually needed
 *
 * @module maintenance
 */

export {};
