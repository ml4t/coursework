#!/usr/bin/env bash
# Upload ml4t-coursework to PyPI. Run by Stefan, not by an agent: the token is his and the
# first upload of a name cannot be undone.
#
#   bash ~/ml4t/courses/foundations-dev/helper/publish.sh
#
# Needs a PyPI API token. Either export it first:
#   export UV_PUBLISH_TOKEN=pypi-...
# or let uv prompt for it.
set -euo pipefail

cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# HOLD until the package has moved to its own repo, ~/ml4t/libraries/coursework
# (Stefan, 2026-08-25). Do not upload from inside foundations-dev.
#
# The order cannot be redone. Claiming a PyPI name is permanent, and the first
# release's metadata is what the project page shows forever after, so uploading
# from the path the package is leaving publishes a source link that is wrong on
# day one. Move first, then upload.
#
# Settled and already in pyproject.toml: MIT licence, and the house URL pattern
# pointing at github.com/ml4t/coursework. The repo is ml4t/coursework public
# with an ml4t/coursework-dev private sidecar.
#
# Once the move has landed, delete this block. Nothing else in the script changes.
# ---------------------------------------------------------------------------
if [ "${1:-}" != "--move-is-done" ]; then
    cat <<'HOLD'
Refusing to upload: the package has not moved yet.

ml4t-coursework becomes its own repo, ml4t/coursework, alongside the other
published ml4t-* libraries. A PyPI name is permanent and the first release's
metadata is frozen with it, so uploading from inside foundations-dev publishes
a source link that is wrong on day one.

Re-run with --move-is-done once the extraction has landed.
HOLD
    exit 1
fi

# The flag says the source moved. These check that the metadata moved with it,
# because those two can come apart and only one of them is visible on PyPI.
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
echo "    %pip install -q ml4t-coursework"
