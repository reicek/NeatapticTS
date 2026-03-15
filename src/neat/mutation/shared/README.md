# neat/mutation/shared

Shared contracts for the NEAT mutation chapter.

These types keep the mutation subtree decoupled from the full controller and
network implementations while the direct-path split continues. Every chapter
under `mutation/` depends on this file as a leaf contract surface.

## neat/mutation/shared/mutation.types.ts

### ConnectionWithMetadata

Runtime interface for a connection within a genome.

### GenomeWithMetadata

Runtime interface for a genome with mutation-related metadata.
Avoids circular dependencies by defining only the properties accessed in mutation modules.

### MutationMethod

Runtime interface for a mutation method descriptor.

### NeatControllerForMutation

Runtime interface for the NEAT controller used in mutation operations.
Avoids circular dependencies by defining only properties accessed in mutation modules.

### NodeSplitRecord

Runtime interface for node-split innovation records.

### NodeWithMetadata

Runtime interface for a node within a genome.

### OperatorStats

Runtime interface for operator statistics tracking.
