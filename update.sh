#!/usr/bin/env bash
# One-click GitHub sync script: autosave -> pull --rebase -> commit -> push
# Usage:
#   ./update.sh "Your commit message"
# If no message is given, the current timestamp will be used.
# run chmod +x update.sh first.

set -euo pipefail

# 1) Ensure we are inside a Git repository
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "❌ Not inside a Git repository."
  exit 1
fi

# 2) Detect current branch; fallback to 'main'
branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
if [[ "$branch" == "HEAD" || -z "$branch" ]]; then
  branch="main"
fi

# 3) Check if remote 'origin' exists
if ! git remote get-url origin >/dev/null 2>&1; then
  echo "❌ No remote 'origin' found. Please set it first, e.g.:"
  echo "   git remote add origin git@github.com:<your-username>/<your-repo>.git"
  exit 1
fi

# 4) Autosave uncommitted changes to avoid losing work
if ! git diff --quiet || ! git diff --cached --quiet; then
  git add -A
  git commit -m "WIP: autosave before sync" || true
fi

# 5) Fetch and rebase onto the latest remote branch
git fetch origin
# If no upstream is set, link this branch to origin
if ! git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
  git branch --set-upstream-to="origin/$branch" "$branch" 2>/dev/null || true
fi
git pull --rebase origin "$branch"

# 6) Commit actual changes with user message (or timestamp)
msg="${1:-"update: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"}"
if ! git diff --quiet || ! git diff --cached --quiet; then
  git add -A
  git commit -m "$msg" || true
fi

# 7) Push to GitHub
git push origin "$branch"

echo "✅ Sync complete: branch '$branch' pushed to origin."