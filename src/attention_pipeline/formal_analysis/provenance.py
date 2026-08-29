from __future__ import annotations

from pathlib import Path
import re
import subprocess

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def _git(repo: Path, *args: str, allow_empty: bool = False) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(
            f"无法解析 Git provenance: repo={repo}, command={' '.join(args)}, error={message}"
        )
    value = proc.stdout.strip()
    if not value and not allow_empty:
        raise RuntimeError(
            f"无法解析 Git provenance: repo={repo}, command={' '.join(args)} 返回空值"
        )
    return value


def resolve_git_checkout(
    repo: str | Path,
    *,
    role: str,
    require_clean: bool = True,
) -> dict[str, object]:
    """Resolve one runtime checkout to an auditable full commit SHA.

    The function never substitutes a configured or historical commit. If the
    checkout is absent, HEAD cannot be resolved, or the working tree is dirty
    while ``require_clean`` is enabled, execution fails closed.
    """
    root = Path(repo).expanduser().resolve()
    if not root.exists():
        raise RuntimeError(f"{role} Git checkout 不存在: {root}")

    top_level = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    commit = _git(top_level, "rev-parse", "--verify", "HEAD").lower()
    if not _FULL_SHA.fullmatch(commit):
        raise RuntimeError(f"{role} Git HEAD 不是完整40位SHA: {commit!r}")

    status = _git(top_level, "status", "--porcelain", allow_empty=True)
    clean = not bool(status)
    if require_clean and not clean:
        raise RuntimeError(
            f"{role} Git checkout 存在未提交改动；commit SHA 无法完整代表实际运行代码/证据"
        )

    origin = None
    try:
        origin = _git(top_level, "config", "--get", "remote.origin.url")
    except RuntimeError:
        origin = None

    return {
        "status": "resolved",
        "role": role,
        "commit_sha": commit,
        "repo_root": str(top_level),
        "origin_url": origin,
        "worktree_clean": clean,
        "resolution_method": "git-rev-parse-HEAD",
    }


def collect_runtime_provenance(
    *,
    code_repo: str | Path,
    evidence_repo: str | Path,
    evidence_repository: str | None = None,
    require_clean: bool = True,
) -> dict[str, object]:
    """Resolve code and evidence commits from the checkouts used at runtime."""
    code = resolve_git_checkout(code_repo, role="code", require_clean=require_clean)
    evidence = resolve_git_checkout(
        evidence_repo, role="evidence", require_clean=require_clean
    )
    if evidence_repository:
        evidence["declared_repository"] = str(evidence_repository)
    return {
        "code": code,
        "evidence": evidence,
        "policy": {
            "fixed_commit_fallback_allowed": False,
            "unresolved_commit_behavior": "fail_closed",
            "dirty_checkout_behavior": "fail_closed" if require_clean else "record",
        },
    }
