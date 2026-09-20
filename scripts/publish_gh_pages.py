"""Publish the HTML reports in data/ to the gh-pages branch as a local commit.

Run this after regenerating one or more reports:
    uv run python3 scripts/publish_gh_pages.py

What it does:
  1. Checks out the gh-pages branch into a dedicated worktree at
     .gh-pages-worktree/ (created on first run, reused afterwards).
  2. Copies every top-level *.html file from data/ into that worktree,
     renaming similarity_explorer.html -> index.html (the site root) and
     keeping the other report filenames as-is.
  3. Removes any previously-published report file that no longer has a
     source in data/, so the branch mirrors the current reports exactly.
  4. Commits the result on gh-pages if anything changed.

It never pushes - review the commit (e.g. `git -C .gh-pages-worktree log -p
-1` or `git -C .gh-pages-worktree show --stat`) and push it yourself:
    git push origin gh-pages

Pass --dry-run to see what would change without touching the worktree, or
--push to push to origin automatically after committing.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
WORKTREE_DIR = REPO_ROOT / ".gh-pages-worktree"
BRANCH = "gh-pages"

# Reports whose published name differs from the source filename in data/.
# Anything else in data/*.html is published under its own name unchanged.
RENAMES = {
    "similarity_explorer.html": "index.html",
}

# Files in the worktree that are never touched by the sync (branch metadata,
# not a generated report).
PROTECTED = {".nojekyll", ".git"}


def run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def ensure_worktree() -> None:
    if WORKTREE_DIR.exists():
        return

    existing = run(["git", "worktree", "list", "--porcelain"], cwd=REPO_ROOT)
    if str(WORKTREE_DIR) in existing:
        raise RuntimeError(
            f"{WORKTREE_DIR} is registered as a worktree but missing on disk. "
            f"Run `git worktree prune` and retry."
        )

    branches = run(["git", "branch", "--list", BRANCH], cwd=REPO_ROOT)
    if branches:
        run(["git", "worktree", "add", str(WORKTREE_DIR), BRANCH], cwd=REPO_ROOT)
        return

    remote_branches = run(
        ["git", "ls-remote", "--heads", "origin", BRANCH], cwd=REPO_ROOT
    )
    if not remote_branches:
        raise RuntimeError(
            f"No local or remote '{BRANCH}' branch found - create it first."
        )
    run(["git", "fetch", "origin", BRANCH], cwd=REPO_ROOT)
    run(
        ["git", "worktree", "add", "-b", BRANCH, str(WORKTREE_DIR), f"origin/{BRANCH}"],
        cwd=REPO_ROOT,
    )


def target_name(source: Path) -> str:
    return RENAMES.get(source.name, source.name)


def sync_reports(dry_run: bool) -> list[str]:
    sources = sorted(DATA_DIR.glob("*.html"))
    if not sources:
        raise RuntimeError(f"No *.html files found in {DATA_DIR}")

    wanted = {target_name(src): src for src in sources}
    changes = []

    for name, src in wanted.items():
        dest = WORKTREE_DIR / name
        if dest.exists() and dest.read_bytes() == src.read_bytes():
            continue
        changes.append(f"update {name}" if dest.exists() else f"add {name}")
        if not dry_run:
            shutil.copyfile(src, dest)

    for existing in WORKTREE_DIR.glob("*.html"):
        if existing.name in PROTECTED or existing.name in wanted:
            continue
        changes.append(f"remove {existing.name}")
        if not dry_run:
            existing.unlink()

    return changes


def commit(changes: list[str]) -> bool:
    run(["git", "add", "-A"], cwd=WORKTREE_DIR)
    status = run(["git", "status", "--porcelain"], cwd=WORKTREE_DIR)
    if not status:
        return False

    message_lines = ["Update HTML reports", ""] + [f"- {c}" for c in changes]
    run(["git", "commit", "-m", "\n".join(message_lines)], cwd=WORKTREE_DIR)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would change, do nothing"
    )
    parser.add_argument(
        "--push", action="store_true", help="Push gh-pages to origin after committing"
    )
    args = parser.parse_args()

    ensure_worktree()
    changes = sync_reports(args.dry_run)

    if not changes:
        print("No changes - gh-pages already matches data/*.html")
        return

    print("Changes:")
    for change in changes:
        print(f"  {change}")

    if args.dry_run:
        print("\n(dry run - nothing written)")
        return

    committed = commit(changes)
    if not committed:
        print("\nNo net changes to commit (files matched after copy).")
        return

    commit_hash = run(["git", "rev-parse", "--short", "HEAD"], cwd=WORKTREE_DIR)
    print(f"\nCommitted {commit_hash} on {BRANCH}.")

    if args.push:
        run(["git", "push", "origin", BRANCH], cwd=WORKTREE_DIR)
        print(f"Pushed {BRANCH} to origin.")
    else:
        print(f"Review it, then push with: git push origin {BRANCH}")


if __name__ == "__main__":
    main()
