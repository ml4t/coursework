#!/usr/bin/env bash
# Qualify a release and push its tag. GitHub Actions publishes to PyPI through OIDC.
set -euo pipefail

cd "$(dirname "$0")"

version=$(grep -m1 '^version' pyproject.toml | cut -d'"' -f2)
tag="v${version}"
origin=$(git remote get-url origin 2>/dev/null || true)

case "$origin" in
    *ml4t/coursework*) ;;
    *)
        echo "origin is '${origin:-unset}', not ml4t/coursework"
        exit 1
        ;;
esac

if [ "$(git branch --show-current)" != "main" ]; then
    echo "release from main, not $(git branch --show-current)"
    exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
    echo "the working tree is dirty"
    exit 1
fi
if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
    echo "HEAD does not match origin/main"
    exit 1
fi
if ! curl -Lsf -o /dev/null "https://github.com/ml4t/coursework"; then
    echo "github.com/ml4t/coursework is not publicly reachable"
    exit 1
fi
if git show-ref --verify --quiet "refs/tags/${tag}" ||
    git ls-remote --exit-code --tags origin "refs/tags/${tag}" >/dev/null 2>&1; then
    echo "${tag} already exists"
    exit 1
fi
if curl -Lsf -o /dev/null "https://pypi.org/pypi/ml4t-coursework/${version}/json"; then
    echo "ml4t-coursework ${version} already exists on PyPI"
    exit 1
fi

uv sync --all-extras --locked
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run pytest -q
uv run ruff check --isolated --select E4,E7,E9,F src tests

rm -rf dist
uv build
uv run twine check dist/*

echo "About to push ${tag}. GitHub Actions will publish it to PyPI."
read -r -p "Type the version to confirm: " confirm
if [ "$confirm" != "$version" ]; then
    echo "mismatch, stopped"
    exit 1
fi

git tag -a "$tag" -m "ml4t-coursework ${version}"
git push origin "$tag"
