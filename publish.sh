#!/usr/bin/env bash
# Upload ml4t-coursework to PyPI. Run by Stefan, not by an agent: the token is his and the
# first upload of a name cannot be undone.
#
#   bash ~/ml4t/libraries/ml4t-coursework/publish.sh
#
# Needs a PyPI API token. Either export it first:
#   export UV_PUBLISH_TOKEN=pypi-...
# or let uv prompt for it.
set -euo pipefail

cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# The package moved out of foundations-dev on 2026-08-29 and is its own repo
# here. The hold that blocked the upload until it did is gone; what is left is
# the reason that hold existed.
#
# The order cannot be redone. Claiming a PyPI name is permanent, and the first
# release's metadata is what the project page shows forever after. pyproject.toml
# points Homepage, Repository, Issues and Changelog at github.com/ml4t/coursework,
# so that repository has to exist and this checkout has to be it before the first
# upload freezes a link to a 404.
# ---------------------------------------------------------------------------
echo "== source =="
origin=$(git remote get-url origin 2>/dev/null || true)
case "$origin" in
    *ml4t/coursework*) ;;
    *)
        echo "  origin is '${origin:-unset}', not ml4t/coursework."
        echo "  Refusing to upload: the metadata names a repository this checkout is not."
        exit 1
        ;;
esac
if ! git ls-remote --exit-code origin HEAD >/dev/null 2>&1; then
    echo "  ${origin} does not exist yet, or is unreachable."
    echo "  Create and push it first. The first release freezes a link to it and a release"
    echo "  cannot be redone, so a source link that 404s on day one stays wrong."
    exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
    echo "  the working tree is dirty. Upload what is committed, not what is on disk."
    exit 1
fi
if [ "$(git rev-parse HEAD)" != "$(git rev-parse @{u} 2>/dev/null)" ]; then
    echo "  HEAD is not what origin has. Push first: the release points at a commit"
    echo "  nobody else can fetch otherwise."
    exit 1
fi
echo "  origin ${origin} exists and matches HEAD"

# The source moved; these check that the metadata moved with it. The two can come
# apart, and only one of them is visible on PyPI.
echo "== metadata =="
metadata_fault=0
for key in Homepage Repository Issues Documentation Changelog; do
    if ! grep -q "^${key} = " pyproject.toml; then
        echo "  MISSING [project.urls] ${key}"
        metadata_fault=1
    fi
done
if grep -q "Proprietary" pyproject.toml; then
    echo "  pyproject.toml still says Proprietary. The licence is MIT, like every sibling."
    metadata_fault=1
fi
if ! grep -q 'license = { text = "MIT" }' pyproject.toml; then
    echo "  MISSING the MIT licence declaration"
    metadata_fault=1
fi
if [ ! -f LICENSE ]; then
    echo "  MISSING the LICENSE file"
    metadata_fault=1
fi
if grep -q "ml4trading.io\"*$" <(grep "^Homepage" pyproject.toml); then
    echo "  Homepage points at the marketing site. House pattern points it at the repo."
    metadata_fault=1
fi
if [ "$metadata_fault" -ne 0 ]; then
    echo
    echo "Refusing to upload. The first release freezes all of this and a release cannot be redone."
    exit 1
fi
echo "  urls, licence and LICENSE file all present"

echo "== tests =="
uv run pytest -q

echo
echo "== is the name still free? =="
if curl -sf -o /dev/null "https://pypi.org/pypi/ml4t-coursework/json"; then
    echo "  ml4t-coursework already EXISTS on PyPI."
    echo "  This would be a new release of an existing project, not a first claim."
    echo "  Check the version in pyproject.toml before continuing; an existing version cannot"
    echo "  be overwritten and a wrong one cannot be withdrawn."
    read -r -p "  Continue? [y/N] " reply
    [[ "$reply" == "y" || "$reply" == "Y" ]] || { echo "  stopped"; exit 1; }
else
    echo "  ml4t-coursework is unclaimed. This upload claims it."
fi

echo
echo "== build =="
rm -rf dist
uv build
ls -la dist/

echo
VERSION=$(grep -m1 '^version' pyproject.toml | cut -d'"' -f2)
echo "== about to upload version ${VERSION} to PyPI =="
echo "  This is public and permanent. A version number cannot be reused, even after deletion."
read -r -p "  Type the version to confirm: " confirm
[[ "$confirm" == "$VERSION" ]] || { echo "  mismatch, stopped"; exit 1; }

uv publish

echo
echo "== verify the published artifact installs =="
rm -rf /tmp/ml4t-pypi-check
uv venv --python 3.11 /tmp/ml4t-pypi-check -q
echo "  waiting for the index to catch up before installing"
until uv pip install --python /tmp/ml4t-pypi-check -q "ml4t-coursework==${VERSION}" 2>/dev/null; do
    sleep 5
done
/tmp/ml4t-pypi-check/bin/python -c "
import ml4t_coursework as m
from ml4t_coursework import contracts
print('installed', m.__version__, 'with', len(contracts.load_all()), 'components')"
echo
echo "Done. Change the first cell of the setup notebook and every unit notebook to:"
echo "    %pip install -q \"ml4t-coursework[data]\""
