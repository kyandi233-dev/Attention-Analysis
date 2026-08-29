# Local/Remote Repository Synchronization Policy

## Canonical identity

- Remote: `https://github.com/kyandi233-dev/attention-pipeline-v2.git`
- Controlled branch: `codex/v2-YOLO+Tracking+RInet`
- Workspace registry: `D:\Project\PROJECT_INDEX.md`
- Central integration truth: `greenboo26/focuswave-multimodal-attention-analysis`

## Required rules

1. Verify path, remote, branch, upstream, local HEAD and remote HEAD before writes.
2. Use fast-forward-only routine synchronization; never force push, hard reset or clean a worktree containing user changes.
3. Do not create branches or worktrees for convenience. Any new ref requires an explicit task, purpose, owner, base ref and retention decision.
4. Preserve uncommitted runtime changes, sampled configurations, models, credentials and machine-local outputs; stage only named policy or code files.
5. Treat local-only, gone-upstream and diverged branches as review states, not automatic cleanup targets.
6. Pipeline outputs become central scientific evidence only after the central repository's provenance and acceptance gates pass.

Completion reports must include the actual local/remote ref evidence and any remaining dirty state.
