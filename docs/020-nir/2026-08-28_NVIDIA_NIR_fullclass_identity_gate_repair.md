# 2026-08-28 NVIDIA NIR full-class identity gate repair

## Scope and safety boundary

- Target: the final RITnet full-class completion gate only.
- Existing `sub-056` output is retained without deletion or marker edits.
- No YOLO stage, RITnet numeric rerun, or multi-subject queue is started by this repair.

## Root cause

The final completion validator compared the entire `work_identity` object. The
object includes both immutable identity (`git_commit`, configuration/model/data
SHA-256 values, and source identity) and `git_branch`. A result produced from
the same commit in a detached acceptance worktree records `git_branch: HEAD`,
whereas the formal checkout records `git_branch: nvidia-cuda`; this ref-name
difference incorrectly invalidated an otherwise identical result.

## Repair contract

- `scientific_identity` is the strict gate: result-affecting source-file
  hashes, configuration/model/source hashes, and algorithm/schema versions.
- `provenance_identity` records the Git commit/ref and is retained for audit,
  but documentation, tests, validator and launcher changes do not force a
  numerical rerun when `scientific_identity` is equal.
- Legacy manifests are not rewritten. Their result-affecting file hashes are
  reconstructed from their recorded Git commit and compared with the current
  checkout; source-line endings are canonicalized before hashing so Git's
  Windows CRLF checkout filter does not create a false algorithm change.
- Regression coverage verifies detached-HEAD versus branch provenance does not
  affect equivalence, while a result input hash remains strict.

## Verification handoff

Run runtime tests with `D:\Project\厚粲杯\08_算法\.venv_nir_gpu\Scripts\python.exe`
and prepend that environment's `torch\\lib` directory to `PATH`. Then invoke
the existing `sub-056` single-subject full-class command without `--force`.
Expected result: `skipped_valid_completion`; any other result must be reported
with its validator reason and must not trigger deletion or a queue launch.

## Evidence recorded on 2026-08-28

- Targeted regression: 2 passed.
- Full `runtime/nir-formal/tests` suite: 36 passed using `.venv_nir_gpu` with
  the required temporary PyTorch DLL `PATH` entry.
- The official final validator accepted existing `sub-056` when its recorded
  `a00dd08` provenance was requested from the later `4d55514` checkout. The
  result-affecting source set is identical after canonical source newline
  hashing; the marker remains `complete`.
- After committing the repair, the canonical one-subject launcher was run with
  `--output <formal-output-root> --subjects sub-056 --device 0` and emitted
  `skipped_valid_completion`. It selected exactly one validated historical
  source, did not enter the numeric core or YOLO, and only refreshed the normal
  batch selection summary at the output root.

## Remaining execution condition

The live checkout was `4d55514`, a later documentation-only commit. The
two-layer gate now permits a strict skip only after it proves that the recorded
commit and live checkout have identical result-affecting source content and
identical scientific inputs. A changed result-affecting source file, config,
model/data hash, source hash, algorithm version, or schema version remains a
hard rejection. No queue was started to bridge this validation.
