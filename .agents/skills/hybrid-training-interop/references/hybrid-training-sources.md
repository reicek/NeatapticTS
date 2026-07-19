# Hybrid Training Reference Notes

This file stores paraphrased reference notes for the
`hybrid-training-interop` skill.

These notes summarize upstream material instead of copying it verbatim. Use the
linked sources for canonical wording and details.

## Source Map

### 1. PyTorch parameter flatten and restore helpers

- Upstream repo: https://github.com/pytorch/pytorch
- Relevant implementation:
  - `torch/nn/utils/convert_parameters.py`
  - `torch/csrc/api/include/torch/nn/utils/convert_parameters.h`
- Why it matters:
  - PyTorch makes the flattening contract explicit: flatten parameters in the
    provided iterable order, concatenate them, and restore by slicing the vector
    back in the same order.
  - The implementation also checks device consistency, which is a useful reminder
    that a vector contract needs environment assumptions spelled out.

### 2. Wikipedia overview of backpropagation

- URL: https://en.wikipedia.org/wiki/Backpropagation
- Authors: Wikipedia contributors
- License note: Wikipedia text is available under CC BY-SA 4.0.
- Why it matters:
  - Backpropagation is the gradient-computation mechanism, not the entire hybrid
    policy.
  - The distinction between computing gradients and deciding whether trained
    weights persist is central to this repo's contract design.
  - The standard update equation is a concise way to document learning-rate and
    mutation-policy separation.

## Practical Notes

### Ordering is the contract

- A parameter vector is only meaningful if the order is stable and documented.
- Layout metadata is what turns a flat numeric buffer into a reusable contract.

### Import is a mutation boundary

- Writing a vector back into a network is a mutating action.
- That mutation must be isolated and deliberate, especially when many
  candidates share an evaluation context.

### Training policy is separate from gradient math

- Gradient math says how to update parameters.
- Policy says whether to train, when to train, and whether trained parameters are
  kept.
- Keeping those separate avoids hidden Lamarckian behavior.

## Working Heuristics For This Repo

- Export values plus layout metadata together.
- Make mismatch detection explicit before import.
- Prefer vector-based isolation when interoperability with workers or
  checkpoints is the long-term target.
- Make persistence policy visible in the public API, not buried in downstream
  application code.
