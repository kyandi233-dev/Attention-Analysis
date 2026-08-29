# Local/Remote Repository Synchronization Policy

This repository is the producer/runtime repository for the NIR/RGB modality.

## Canonical identity

- Remote: `https://github.com/kyandi233-dev/Attention-Analysis.git`
- Default controlled branch: `nvidia-cuda`
- Alternate branch: `amd-DirectML`, only when explicitly selected by a current handoff
- Workspace registry: `D:\Project\PROJECT_INDEX.md`
- Central integration truth: `greenboo26/focuswave-multimodal-attention-analysis`

## Required synchronization rules

1. Verify repository path, remote URL, branch, upstream, local HEAD and remote HEAD before any write.
2. Routine synchronization is fast-forward-only. Never use force push, hard reset, clean, or detached HEAD as a normal working state.
3. Do not create a branch or worktree for convenience. A new ref requires an explicit task name, purpose, owner, base ref and cleanup/retention decision.
4. Preserve uncommitted files, runtime models, credentials and machine-local state. Do not stage them implicitly.
5. Keep one active checkout per branch unless Git worktree registration is intentional and recorded.
6. Treat local-only, gone-upstream and diverged branches as review states; do not delete or rewrite them automatically.
7. Producer outputs must cross into the central analysis repository only through declared provenance and acceptance contracts.

Before reporting completion, record the evidence for the local/remote ref mapping and the remaining dirty or historical states.
