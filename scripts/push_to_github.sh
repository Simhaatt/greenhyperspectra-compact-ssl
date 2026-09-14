#!/usr/bin/env bash
# One-time publication of this repository to GitHub.
#
#   bash scripts/push_to_github.sh <github-username> [repo-name]
#
# Requires the GitHub CLI (https://cli.github.com) authenticated once with
#   gh auth login
# or a git credential helper that can push over HTTPS.
set -euo pipefail

USER_NAME="${1:?usage: push_to_github.sh <github-username> [repo-name]}"
REPO_NAME="${2:-greenhyperspectra-compact-ssl}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Substituting placeholders for $USER_NAME/$REPO_NAME"
for f in README.md CITATION.cff; do
    sed -i.bak "s|OWNER/greenhyperspectra-compact-ssl|$USER_NAME/$REPO_NAME|g" "$f"
    rm -f "$f.bak"
done

if [ ! -d .git ]; then
    git init -b main
    git add -A
    git commit -m "Reproducibility package for band- and label-efficient hyperspectral plant trait prediction"
else
    git add -A
    git diff --cached --quiet || git commit -m "Set repository URL"
fi

echo "==> Creating remote repository"
if command -v gh >/dev/null 2>&1; then
    gh repo create "$USER_NAME/$REPO_NAME" --public --source=. --remote=origin --push \
        --description "Code, selected wavelengths and result tables for band- and label-efficient hyperspectral plant trait prediction (GreenHyperSpectra)"
else
    echo "    gh not found - create $REPO_NAME on github.com first, then:"
    git remote add origin "https://github.com/$USER_NAME/$REPO_NAME.git" 2>/dev/null || \
        git remote set-url origin "https://github.com/$USER_NAME/$REPO_NAME.git"
    git push -u origin main
fi

echo
echo "Done: https://github.com/$USER_NAME/$REPO_NAME"
echo
echo "Next, to mint a DOI:"
echo "  1. Sign in at https://zenodo.org with GitHub and switch this repo ON"
echo "  2. Fill the article DOI into .zenodo.json"
echo "  3. git tag -a v1.0.0 -m 'Release accompanying the manuscript' && git push --tags"
echo "  4. Create a GitHub Release from that tag - Zenodo archives it and issues the DOI"
