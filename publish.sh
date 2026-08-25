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
